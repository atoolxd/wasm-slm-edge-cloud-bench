# Host Specifications

Fill this in before running the full sweep — required for the paper's
reproducibility section (see `docs/experiment_design.md` §2).

## Edge profile (emulated)

- Physical machine CPU model: AMD Ryzen 5 5500U with Radeon Graphics (6 physical cores / 12 logical threads)
- Physical machine total cores / RAM: 6 cores / 12 threads. 8 GB physically installed (2x4GB Micron DDR4-3200), but only 5.85 GB is OS-visible/usable — the ~2 GB gap is most likely reserved by the Ryzen 5500U's integrated Radeon graphics (shared system memory) or other hardware-reserved memory. **Note for §2:** a 4 GB Docker `--memory` cap against 5.85 GB usable leaves only ~1.85 GB headroom for Windows itself — workable but tight; watch for host-side memory pressure during the edge-profile sweep and note it as a threat to measurement validity if observed.
- Docker resource cap applied (`--cpus`, `--memory`): `--cpus=2 --memory=3g`, confirmed via `docker stats wasi-edge --no-stream` showing limit `3GiB`. Image: custom `wasi-nn-edge` built from `setup/Dockerfile.edge` (not `wasmedge/slim`, which is stale — see setup/SETUP.md §6)
- Server flags used (per model, controlled parameters — not defaults): `--prompt-template zephyr --ctx-size 640 --n-gpu-layers 0 --n-predict 128 --temp 0.1` — **note: `--n-predict` alone does not enforce the cap in LlamaEdge 0.29.0**, and a second bug independently overrides `max_completion_tokens` based on context size unless ctx-size is tuned so the two coincide — `--ctx-size 640` is that tuned value, not a default (see `docs/PROGRESS_LOG.md` §12 and §18); `harness/run_benchmark.py` sends `max_completion_tokens: 128` on every non-streaming request, which is what actually bounds generation (TinyLlama-1.1B; re-verify `--prompt-template`/`--ctx-size` per model when switching)
- OS / kernel version: Windows 11 Pro, version 10.0.26200 (build 26200), 64-bit, host. WSL2 guest: Ubuntu 26.04 LTS, kernel 6.6.87.2-microsoft-standard-WSL2, x86_64
- **WSL2 memory ceiling:** originally measured at ~2.8 GB total (WSL2's default cap is roughly half of usable host RAM). Raised via `C:\Users\khare\.wslconfig` (`[wsl2]\nmemory=4GB`) + `wsl --shutdown` + restart, confirmed via `free -h` now showing ~3.8 GB total. Given the host's ~5.85 GB usable RAM ceiling (see above), the edge-profile Docker cap was set to **3 GB / 2 vCPU** (not the original 4 GB target) to leave ~1.85 GB for Windows itself — see `docs/experiment_design.md` §2 for the rationale.
- WasmEdge version (`wasmedge --version`): 0.17.1
- wasi_nn-ggml plugin version: 0.1.34.0

## Cloud profile (Linode/Akamai dedicated CPU — not Oracle Cloud, see note below)

**Deviation from original design:** the experiment design originally specified
Oracle Cloud Always Free (Ampere A1, ARM64). Switched to a Linode dedicated-CPU
instance instead (user already had a paid Linode account; cost was not a
constraint for this ~1-day VM lifetime). Kept the same 4 vCPU / 8GB target
spec from `experiment_design.md` §2 for parity with the edge-profile
comparison — only the provider changed, not the resource envelope. This is
x86_64, not ARM64, which is actually a closer architectural match to the
edge profile (also x86_64) than Oracle's Ampere A1 would have been.

- VM shape: Linode **G7 Dedicated 8x4** (Compute-Optimized dedicated CPU)
- OCPU / RAM allocated: 4 vCPU (AMD EPYC 7713 64-Core Processor, host-level;
  guest sees 4 cores) / 7.8 GB RAM (8GB nominal)
- Region: IN, Chennai (`in-maa`)
- OS image / kernel version: Ubuntu 24.04.4 LTS
- WasmEdge version: 0.17.1 (matches the edge-profile host exactly)
- wasi_nn-ggml plugin version: 0.1.34.0 (matches the edge-profile host exactly)
- Server flags used (TinyLlama-1.1B, same controlled parameters as edge):
  `--prompt-template zephyr --ctx-size 640 --n-gpu-layers 0 --n-predict 128 --temp 0.1` — **note: `--n-predict` alone does not enforce the cap in LlamaEdge 0.29.0**, and a second bug independently overrides `max_completion_tokens` based on context size unless ctx-size is tuned so the two coincide — `--ctx-size 640` is that tuned value, not a default (see `docs/PROGRESS_LOG.md` §12 and §18); `harness/run_benchmark.py` sends `max_completion_tokens: 128` on every non-streaming request, which is what actually bounds generation
- No Docker resource cap applied — the cloud profile uses the VM's native
  resources directly (bare `wasmedge` process), per `experiment_design.md`
  §2; the 4 vCPU/8GB ceiling comes from the Linode plan itself, not a
  container limit.
- Network path: Linode Cloud Firewall restricts inbound to
  SSH (22, all sources), ICMP (all sources), HTTP/HTTPS (80/443, all
  sources — unused by this project), and TCP 8080 scoped to the
  researcher's laptop IP only (`/32`); host-level `ufw`/`iptables` left at
  Ubuntu defaults (inactive/ACCEPT) since the cloud firewall is the actual
  perimeter control.
- Measured network RTT from laptop (Windows `ping`, n=20, 0% loss):
  min 17ms / avg 22ms / max 32ms, 2026-09-02. Note this is laptop-WiFi/ISP-to-Chennai-datacenter RTT, not a controlled lab measurement — expect some variance run to run; re-measure immediately before the real sweep if a long gap elapses.
- Smoke-tested end-to-end from the laptop over the public IP
  (`http://<CLOUD_HOST>:8080/v1/chat/completions`): `200 OK`, valid
  completion, ~15s wall time for the request (includes RTT + inference).
