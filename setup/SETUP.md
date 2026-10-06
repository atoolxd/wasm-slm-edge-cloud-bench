# Environment Setup

> **A note on currency:** WasmEdge and LlamaEdge (the project that provides
> the `wasi-nn`/GGML serving wrapper) both move fast, and exact flags/file
> names do change between releases. Treat the commands below as a verified
> starting point, not gospel — run `wasmedge --version` after install and
> skim the current WasmEdge/LlamaEdge README on GitHub before a long run, in
> case something has been renamed since this was written.

## 1. WSL2 (Windows Subsystem for Linux)

From PowerShell (as admin):

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then open the Ubuntu app and finish the first-run user
setup. Everything else below runs **inside WSL2/Ubuntu**, not in PowerShell.

## 2. WasmEdge + the wasi-nn GGML plugin

Inside the WSL2 Ubuntu shell:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh \
  | bash -s -- --plugin wasi_nn-ggml

source ~/.bashrc   # or restart the shell
wasmedge --version
```

Confirm the plugin installed: it should show up under
`~/.wasmedge/plugin/` as a `libwasmedgePluginWasiNN.so` (name may vary by
version — the install script prints its final location).

## 3. LlamaEdge (serving wrapper)

LlamaEdge provides a prebuilt `llama-api-server.wasm` that exposes an
OpenAI-compatible REST API (`/v1/chat/completions`), which is what the
benchmarking harness in `../harness/run_benchmark.py` talks to. Download the
latest release asset from the LlamaEdge GitHub repo (`LlamaEdge/LlamaEdge`)
into this `setup/` folder or a `bin/` folder of your choosing:

```bash
mkdir -p ~/llamaedge && cd ~/llamaedge
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
```

(If that exact asset name has changed, check the repo's Releases page —
this is exactly the kind of detail that drifts between versions.)

## 4. Download a model

Pick one model from `../docs/experiment_design.md`'s matrix and grab its
GGUF file from Hugging Face, e.g.:

```bash
mkdir -p ../models/tinyllama-1.1b
cd ../models/tinyllama-1.1b
curl -LO https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Repeat per model/quant combination you plan to test. **Do not commit these
files** — `models/` is git-ignored for exactly this reason (multi-GB files).

**TinyLlama F16 deviation:** `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (the
source used for Q4_K_M and Q8_0) has no F16 file — it stops at Q8_0. Used
`andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF`'s `TinyLlama-1.1B-Chat-v1.0-f16.gguf`
instead, a different community conversion. Disclosed, not silent: F16 is a
lossless reformatting of the original weights (unlike quantization, which
depends on the specific quantizer/calibration data), so cross-source
provenance shouldn't introduce a numerical confound here the way it would
for a quantized format — but note it in the paper's model-provenance table
regardless. Qwen2.5-0.5B's official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` repo
has all three needed quants (`q4_k_m`, `q8_0`, `fp16`) from one source, no
substitution needed.

## 5. Run the server and smoke-test it

```bash
cd ~/llamaedge
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:../models/tinyllama-1.1b/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  llama-api-server.wasm \
  --model-name tinyllama-1.1b-q4 \
  --ctx-size 4096
```

In a second terminal:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}]}'
```

If that returns a completion, the environment is working end to end and
`harness/run_benchmark.py` can point at `http://localhost:8080`.

## 6. Emulating the "edge" resource profile

Run the same server inside a resource-capped Docker container instead of
bare WSL2, to emulate the constrained edge device described in
`docs/experiment_design.md`. **The cap is 3 GB, not 4 GB** — see
`docs/experiment_design.md` §2 for why (this laptop's usable RAM plus
WSL2's own overhead doesn't leave room for a full 4 GB container cap).

This also folds in three corrections found during the pilot run
(`docs/experiment_design.md` §8): `--prompt-template` is required as of
LlamaEdge 0.29.0 (`zephyr` for TinyLlama-Chat's format — check the right
template for each model), `--ctx-size` must match the model's own trained
context (2048 for TinyLlama, not the earlier default of 4096), and
`--n-predict`/`--temp` must be pinned so completion length and sampling
randomness don't add uncontrolled variance to the latency measurements.

**`wasmedge/slim` on Docker Hub was tried first and rejected**: it has no
`latest` tag and hasn't been pushed since 2024-01 (newest tag
`0.14.0-alpha.1`), predating the WasmEdge 0.17.1 + `wasi_nn-ggml` plugin
combination already validated working in bare WSL2. Instead, build a small
custom image (`setup/Dockerfile.edge`) that reproduces that exact same
install inside the container:

```bash
cd setup
docker build -t wasi-nn-edge -f Dockerfile.edge .
```

Then run it with the resource cap:

```bash
docker run -d --name wasi-edge --cpus=2 --memory=3g \
  -v ~/llamaedge:/app -v $(pwd)/../models:/models \
  -p 8080:8080 wasi-nn-edge \
  wasmedge --dir .:. \
    --nn-preload default:GGML:AUTO:/models/tinyllama-1.1b/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    /app/llama-api-server.wasm --model-name tinyllama-1.1b-q4_k_m \
    --prompt-template zephyr --ctx-size 2048 --n-gpu-layers 0 \
    --n-predict 128 --temp 0.1
```

Confirm the cap actually applied with `docker stats wasi-edge --no-stream`
(should show `.../3GiB`), and smoke-test with the same `curl` from step 5.
Swap the `--nn-preload` path and `--model-name`/`--prompt-template`/
`--ctx-size` per model/quant combination between sweep runs
(`docker rm -f wasi-edge` then re-run with new values). Record the actual
container CPU/RAM limits, plus the exact `--prompt-template`/`--ctx-size`/
`--n-predict`/`--temp` used per model, in `data/processed/host_specs.md` —
these are now controlled experimental parameters, not incidental defaults.

## 7. Cloud profile: Linode dedicated CPU (deviation from the original Oracle Cloud plan)

**Note:** the original plan here was Oracle Cloud Always Free (Ampere A1).
Switched to Linode instead — cost wasn't a constraint for this project, and
the actual steps below are what was run. The resource target stayed the
same (4 vCPU / 8GB) for a fair edge/cloud comparison; only the provider and
architecture changed (x86_64 dedicated CPU instead of ARM64 Ampere A1 — see
`data/processed/host_specs.md` for the full rationale).

1. Create a Linode: **Dedicated CPU** plan, **G7 Dedicated 8x4** (4 vCPU /
   8GB), Ubuntu 24.04 LTS, region chosen for lowest RTT from the researcher's
   location (used `IN, Chennai` / `in-maa` here).
2. Add an SSH key at creation (or add one to `~/.ssh/authorized_keys` after
   first login) rather than relying on the root password long-term.
3. Create a **Cloud Firewall** (Custom, not the VPC template — the template's
   default-open SSH rule needs narrowing anyway) with:
   - Inbound: TCP 22 (SSH) — narrow this to your own IP if you want it
     tighter than the default "all sources"
   - Inbound: TCP 8080 (the benchmark API), source = your own IP only
     (`/32`) — **double-check the port number when using the UI's rule
     presets**; the "HTTP" preset defaults to port 80 and silently keeps
     that port if you forget to overwrite the Ports field after selecting
     it (hit this exact bug during setup — the rule showed "port 8080
     intended" in the add-rule dialog's leftover preset field but saved as
     port 80).
   - Default inbound policy: Drop
   - Attach the firewall to the Linode from its "Linodes" tab (creating the
     firewall does **not** auto-attach it to an existing Linode).
4. Repeat steps 2-5 above (WasmEdge install, LlamaEdge binary, model
   download, server launch, smoke test) via SSH on this VM. Unlike the edge
   profile, **no Docker container is used here** — the cloud profile runs
   the bare `wasmedge` process directly, since the resource ceiling is the
   VM's own plan limit, not a container cap.
5. Verify reachability from your actual laptop, not just `localhost` on the
   VM — a passing local smoke test does not confirm the firewall/security
   group is configured correctly. Use:
   ```bash
   curl -s -w "\nHTTP_STATUS:%{http_code}\n" http://<vm-public-ip>:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Say hello in five words."}]}'
   ```
6. Measure network RTT from your laptop to the VM (`ping -n 20 <vm-public-ip>`
   on Windows, or `ping -c 20` on Linux/Mac) for the `network_rtt_ms` field
   — record min/avg/max and the date measured in `host_specs.md`, since RTT
   varies with your own network conditions at measurement time.
