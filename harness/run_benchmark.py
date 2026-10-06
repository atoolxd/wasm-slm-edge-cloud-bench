"""
Benchmark harness for the Wasm/WASI-NN small-LLM edge-cloud dataset.

Workflow (see ../setup/SETUP.md and ../docs/experiment_design.md):
  1. Manually start a llama-api-server.wasm instance for ONE (host, model,
     quant) combination, with --model-name matching "<model>-<quant lower>".
  2. Run this script targeting that same combination:

        python run_benchmark.py --host edge --model tinyllama-1.1b --quant Q4_K_M

  3. Stop that server, start the next combination's server, repeat.

This "one combination per invocation" design is deliberate: auto-starting/
stopping wasmedge servers from Python is fragile and version-sensitive, and
manual restarts make it trivial to sanity-check each server before spending
a full sweep's worth of requests on it.

Each run appends one JSON line per (prompt, repetition) to a file in
../data/raw/, named "<host>_<model>_<quant>.jsonl".
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import requests
import yaml

HARNESS_DIR = Path(__file__).parent
CONFIG_PATH = HARNESS_DIR / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


class MemorySampler:
    """
    Polls peak RSS of the wasmedge process while a request is in flight.

    kind="local"  -> uses psutil to inspect a process on THIS machine.
                      Only correct if this script runs in the same OS
                      instance as the wasmedge server (e.g. both inside
                      WSL2, or both inside the same Docker network
                      namespace) — see setup/SETUP.md.
    kind="remote" -> shells out over ssh to sample `ps` on the cloud VM.
                      Requires passwordless SSH (key-based auth) to be
                      already configured; if ssh_host is blank in
                      config.yaml, memory sampling is skipped with a
                      one-time warning rather than failing the run.
    """

    def __init__(self, kind: str, ssh_host: str | None = None, ssh_user: str | None = None):
        self.kind = kind
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.peak_rss_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    def _sample_local(self) -> float | None:
        try:
            import psutil
        except ImportError:
            if not self._warned:
                print("WARNING: psutil not installed, skipping local memory sampling")
                self._warned = True
            return None
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                if proc.info["name"] and "wasmedge" in proc.info["name"].lower():
                    return proc.info["memory_info"].rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def _sample_remote(self) -> float | None:
        if not self.ssh_host:
            if not self._warned:
                print("WARNING: no ssh_host configured for remote host, skipping memory sampling")
                self._warned = True
            return None
        try:
            result = subprocess.run(
                ["ssh", f"{self.ssh_user}@{self.ssh_host}", "ps -o rss= -C wasmedge"],
                capture_output=True, text=True, timeout=5,
            )
            values = [int(v) for v in result.stdout.split() if v.strip().isdigit()]
            return max(values) / 1024 if values else None
        except Exception as e:
            if not self._warned:
                print(f"WARNING: remote memory sampling failed ({e}), skipping")
                self._warned = True
            return None

    def _loop(self):
        sample_fn = self._sample_local if self.kind == "local" else self._sample_remote
        while not self._stop.is_set():
            val = sample_fn()
            if val is not None:
                self.peak_rss_mb = max(self.peak_rss_mb or 0, val)
            time.sleep(0.2)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def run_one_request(
    base_url: str, model_name: str, prompt: str, timeout_s: int, n_predict: int
) -> dict:
    """
    Sends one non-streaming chat-completion request to a LlamaEdge
    (OpenAI-compatible) server and measures timing.

    Non-streaming, not streaming, despite ttft_ms/tokens_per_sec being
    listed in docs/experiment_design.md §4: LlamaEdge 0.29.0's streaming
    path never enforces n_predict/max_completion_tokens at all (confirmed
    both in source — metadata.n_predict is set correctly but never read
    again in the token loop — and empirically: the identical prompt gave
    completion_tokens=128 via stream:false vs 186 SSE chunks via
    stream:true). A client-side abort doesn't help either — the server
    keeps the model lock held and keeps generating even after a hard TCP
    reset, so streaming would reintroduce the exact uncontrolled-length
    confound n_predict exists to prevent. See docs/PROGRESS_LOG.md §19.
    Net effect: ttft_ms and tokens_per_sec (decode-only throughput, which
    needs a first-token timestamp streaming would have given) are not
    available for this dataset — reported as None, not approximated.

    max_completion_tokens is REQUIRED here, not optional: LlamaEdge 0.29.0
    ignores the server's own --n-predict CLI flag unless the request also
    sets this field, silently falling back to unbounded generation
    otherwise (confirmed by testing — see docs/PROGRESS_LOG.md). Without
    it, --n-predict is effectively decorative.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_completion_tokens": n_predict,
    }

    t_start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    t_end = time.perf_counter()
    body = resp.json()

    total_latency_ms = (t_end - t_start) * 1000
    completion_tokens = body["usage"]["completion_tokens"]

    return {
        "total_latency_ms": total_latency_ms,
        "ttft_ms": None,
        "approx_output_tokens": completion_tokens,
        "tokens_per_sec": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Host key from config.yaml, e.g. edge or cloud")
    ap.add_argument("--model", required=True, help="Model name from config.yaml")
    ap.add_argument("--quant", required=True, help="Quant level, e.g. Q4_K_M")
    args = ap.parse_args()

    config = load_config()
    host_cfg = config["hosts"][args.host]
    base_url = host_cfg["base_url"]
    model_name = f"{args.model}-{args.quant.lower()}"

    prompts_path = (HARNESS_DIR / config["prompts_file"]).resolve()
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    raw_dir = (HARNESS_DIR / config["raw_output_dir"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{args.host}_{args.model}_{args.quant}.jsonl"

    repetitions = host_cfg.get("repetitions", config["repetitions"])
    timeout_s = config["request_timeout_s"]
    n_predict = config["n_predict"]

    print(f"Target: {base_url}  model_name={model_name}")
    print(f"Writing results to {out_path}")

    # Resume support: an interrupted run (process killed, machine slept/
    # rebooted mid-sweep) shouldn't force a full restart and duplicate rows.
    # Tolerate a truncated last line (the exact failure mode of a kill
    # mid-write) by skipping only that malformed line, not the whole file.
    completed: set[tuple[str, int]] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed.add((row["prompt_id"], row["repetition"]))
        if completed:
            print(f"Resuming: {len(completed)} (prompt, repetition) pairs already recorded, skipping those")

    input("Confirm the server is running with a quick manual curl test, then press Enter to start...")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for prompt_row in prompts:
            for rep in range(repetitions):
                if (prompt_row["id"], rep) in completed:
                    continue
                is_warmup = rep == 0
                sampler = MemorySampler(
                    kind=host_cfg["kind"],
                    ssh_host=host_cfg.get("ssh_host"),
                    ssh_user=host_cfg.get("ssh_user"),
                )
                try:
                    with sampler:
                        timing = run_one_request(
                            base_url, model_name, prompt_row["prompt"], timeout_s, n_predict
                        )
                except Exception as e:
                    print(f"  ERROR on {prompt_row['id']} rep={rep}: {e}")
                    continue

                record = {
                    "host": args.host,
                    "model": args.model,
                    "quant": args.quant,
                    "prompt_id": prompt_row["id"],
                    "bucket": prompt_row["bucket"],
                    "repetition": rep,
                    "is_warmup": is_warmup,
                    "peak_rss_mb": sampler.peak_rss_mb,
                    **timing,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                print(
                    f"  {prompt_row['id']} rep={rep} "
                    f"total={timing['total_latency_ms']:.0f}ms "
                    f"output_tokens={timing['approx_output_tokens']}"
                )

    print("Done.")


if __name__ == "__main__":
    main()
