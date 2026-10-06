# wasm-slm-edge-cloud-bench

Dataset, benchmark harness, and prompt set for characterizing quantized
small-language-model inference served via WebAssembly (WasmEdge + WASI-NN
ggml backend, LlamaEdge `llama-api-server`) on a resource-constrained edge
profile versus a cloud VM.

## Contents

| Path | What |
|---|---|
| `data/raw/*.jsonl` | One row per request. File name = `<host>_<model>_<quant>.jsonl` |
| `data/raw/*.discarded_*.jsonl` | Runs excluded from analysis, kept for transparency (reason in file name) |
| `data/raw/pilot_*.jsonl` | Uncapped pipeline smoke test; **not** edge-profile data |
| `data/processed/anara_export/measurements.csv` | All analyzed rows (12 configs x 147 requests = 1,764 rows) |
| `data/processed/anara_export/discarded_runs.csv` | All discarded rows with `discard_reason` |
| `data/processed/anara_export/request_accounting.csv` | Issued / warmup / analyzed counts per config |
| `data/processed/anara_export/prompts.csv`, `data/prompts/prompts.jsonl` | 49 prompts, stratified into short / medium / long buckets |
| `data/processed/host_specs.md` | Hardware, OS, WasmEdge/plugin versions, server flags |
| `harness/` | `sample_prompts.py` (builds prompt set), `run_benchmark.py` (sweep), `export_csv.py` (raw -> CSV) |
| `setup/` | Environment setup and the edge-profile `Dockerfile.edge` |
| `docs/experiment_design.md` | Full methodology |

## Configurations

- **Hosts:** `edge` = Docker-capped 2 vCPU / 3 GB on an AMD Ryzen 5 5500U laptop (WSL2);
  `cloud` = Linode G7 Dedicated, 4 vCPU (AMD EPYC 7713) / 8 GB, Chennai region.
- **Models:** TinyLlama-1.1B-Chat-v1.0, Qwen2.5-0.5B-Instruct (GGUF).
- **Quantization:** Q4_K_M, Q8_0, F16.
- **Repetitions:** 3 per prompt; repetition 0 is warmup (`is_warmup: true`) and excluded from analysis.
- **Generation cap:** 128 tokens, enforced via `max_completion_tokens` on every request
  (LlamaEdge 0.29.0 does not enforce `--n-predict` alone; see `harness/config.yaml`).

## Schema (raw JSONL / measurements.csv)

| Field | Type | Meaning |
|---|---|---|
| `host` | str | `edge`, `cloud`, or `pilot` |
| `model`, `quant` | str | Model name and GGUF quantization |
| `prompt_id`, `bucket` | str | Prompt identifier and length bucket |
| `repetition` | int | 0-based repetition index |
| `is_warmup` | bool | True for repetition 0 |
| `total_latency_ms` | float | Client-side wall time for the full request |
| `approx_output_tokens` | int | Completion tokens reported by the server |
| `peak_rss_mb` | float/null | Peak server RSS where sampled; null otherwise |
| `ttft_ms`, `tokens_per_sec` | float/null | Null in non-streaming mode (see `docs/experiment_design.md`) |

## Reproducing

1. Follow `setup/SETUP.md` to install WasmEdge 0.17.1 + `wasi_nn-ggml` 0.1.34.0 and LlamaEdge.
2. Download GGUF weights (not included, multi-GB), e.g.
   [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF),
   [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
3. Set `<CLOUD_HOST>` / `<SSH_USER>` in `harness/config.yaml`.
4. `pip install -r harness/requirements.txt`, then `python harness/run_benchmark.py` and `python harness/export_csv.py`.

References to `docs/PROGRESS_LOG.md` in comments point to the authors' private lab notebook, which is not included.

## License

- Code (`harness/`, `setup/`): MIT, see `LICENSE`.
- Measurement data and docs: CC BY 4.0, see `LICENSE-DATA`.
- Prompt text (`data/prompts/`, `data/processed/anara_export/prompts.csv`): derived from
  [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), **CC BY-NC 4.0**, non-commercial use only.

## Citation

See `CITATION.cff`. Archived on Zenodo (DOI added on first release).
