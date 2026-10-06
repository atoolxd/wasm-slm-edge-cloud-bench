# Experiment Design

## 1. Model / quantization matrix

| Model | Params | Source (GGUF) | Quant levels to test |
|---|---|---|---|
| TinyLlama-1.1B-Chat | 1.1B | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF | Q4_K_M, Q8_0, F16 |
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen/Qwen2.5-0.5B-Instruct-GGUF | Q4_K_M, Q8_0, F16 |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen/Qwen2.5-1.5B-Instruct-GGUF | Q4_K_M, Q8_0, F16 |
| Phi-3-mini-4k-instruct | 3.8B | microsoft/Phi-3-mini-4k-instruct-gguf | Q4_K_M, Q8_0 (skip F16 — likely too large for the edge profile; note this as a finding, not a gap) |

Download GGUF files directly into `research/models/<model-name>/`. This
folder is git-ignored — do not commit model weights.

**Scope reduction (2026-09-02):** real measured edge-profile throughput
(~3.3 tokens/sec on the 2 vCPU/3GB-capped container — see §8) makes the
full 4-model x 3-4-quant matrix impractical within the project timeline.
Prioritized subset actually run: **TinyLlama-1.1B** and **Qwen2.5-0.5B**,
all three quant levels each. **Qwen2.5-1.5B and Phi-3-mini are deferred**
— run them only if time remains after the core subset completes;
otherwise report them honestly as future work in the paper's limitations
section (§7), not silently dropped. This is a disclosed scope cut driven
by real measured hardware constraints, in the same spirit as the 3GB (not
4GB) edge cap in §2.

## 2. Host profiles

| Profile | Definition | How to set it up |
|---|---|---|
| **edge** | Emulated constrained device: 2 vCPU, 3 GB RAM | Run the harness inside a Docker container: `docker run --cpus=2 --memory=3g ...` (see `setup/SETUP.md`) |
| **cloud** | Oracle Cloud Always Free Ampere A1 VM | Up to 4 OCPU / 24 GB RAM, ARM64 — use a 4 OCPU / 8 GB shape to keep it a fair, realistic "cloud" comparison point rather than an oversized outlier |

Record exact CPU model, core count, RAM, and OS/kernel version for both
profiles in `data/processed/host_specs.md` before running the sweep —
this is required for the paper's reproducibility section.

**Note on the 3 GB (not 4 GB) edge cap:** the original 4 GB target assumed
more headroom than this laptop actually has. Measured usable RAM is only
~5.85 GB total (8 GB installed, ~2 GB reserved by the integrated GPU), and
WSL2 itself needs to be given ~4 GB (via `.wslconfig`) to have any spare
capacity beyond its ~2.8 GB default cap. A 4 GB Docker cap inside a 4 GB
WSL2 VM leaves no room for the Ubuntu OS or Docker daemon, so the cap was
set to 3 GB instead, leaving ~1.85 GB for Windows. State this plainly in
the paper as a resource-constrained deviation from the original design,
not a silent change — it is itself a small piece of evidence that "edge"
resource envelopes are often tighter than a nominal spec suggests.

## 3. Prompt set

- Source: Stanford Alpaca instruction dataset (public, permissively licensed).
- Sample ~60 prompts stratified into three length buckets by token count:
  - short: <64 tokens
  - medium: 64-256 tokens
  - long: 256-512 tokens
- Fix the random seed when sampling (see `harness/sample_prompts.py`) so the
  exact prompt set is reproducible and can be published alongside the paper.
- Save the sampled set to `data/prompts/prompts.jsonl` — this file **should**
  be committed/shared; it is small and is part of the dataset contribution.

## 4. Metrics per run

**Deviation (2026-09-03, see docs/PROGRESS_LOG.md §19):** `ttft_ms` and
`tokens_per_sec` are **not available** for this dataset and are recorded as
`null` in every row. LlamaEdge 0.29.0's streaming completion path (the only
way to observe a first-token timestamp) never enforces
`n_predict`/`max_completion_tokens` — confirmed in source and empirically
(identical prompt: exactly the requested cap via non-streaming, well past
it via streaming) — and a client-side abort doesn't help either, since the
server keeps the model lock and keeps generating regardless. Using
streaming would reintroduce the exact uncontrolled-completion-length
confound `n_predict` exists to prevent, at an unpredictable time cost.
Switched to non-streaming (`stream: false`) for all data collection instead,
which correctly enforces the cap. `approx_output_tokens` is a byproduct
upside of this: it's now the server's exact `usage.completion_tokens`, not
an SSE-chunk approximation.

For each (model, quant, host, prompt) combination, capture:

- `ttft_ms` — time to first token (**unavailable this dataset, see above**)
- `tokens_per_sec` — decode throughput after the first token (**unavailable
  this dataset, see above**)
- `total_latency_ms` — full request latency
- `peak_rss_mb` — peak resident memory of the WasmEdge process
- `cpu_time_ms` — process CPU time (proxy input for the energy estimate)
- `energy_proxy_j` — `cpu_time_ms * measured_or_rated_TDP_fraction`; document
  the exact formula and its limitations explicitly in the paper (this is a
  proxy, not a calibrated power measurement, since no physical power meter
  is available — say so plainly rather than implying more precision than it
  has)
- For the cloud profile only: `network_rtt_ms` and `payload_bytes` (request +
  response), to make the "offload cost" realistic in the RQ3 policy.

Each run writes one JSON line to `data/raw/<host>_<model>_<quant>_<run_id>.jsonl`.
`harness/run_benchmark.py` handles this automatically.

## 5. Repetitions and statistical care

- Run each (model, quant, host, prompt) combination **5 times** (cloud
  profile); discard the first run of each combination as a warmup
  (cache/compile effects), and report median + IQR, not mean alone, per the
  same practice used in the Wasm-cloud papers reviewed earlier in this
  project.
- **Edge profile reduced to 3 repetitions** (disclosed deviation,
  2026-09-02): at the edge profile's measured ~3.3 tokens/sec, 5 reps x 49
  prompts was too slow to fit the project timeline across multiple
  model/quant combos. 3 reps (with rep 0 still discarded as warmup, leaving
  2 measured reps per prompt) is a smaller but still real sample — report
  the edge profile's statistics with this caveat explicitly in the paper,
  and note the reduced precision compared to the cloud profile's 5 reps
  when comparing IQRs across hosts.
- Randomize run order within a session to avoid confounding with thermal
  throttling or background load drift.
- Log host temperature/load before each run if feasible; note any excluded
  runs and why.

## 6. RQ3: offload policy

- Features: prompt length bucket, requested model+quant, current edge
  resource state (simulate via cgroup load at request time), estimated
  network RTT to the cloud profile.
- Label/target: which host achieves lower (latency, subject to a cost
  penalty for cloud) — construct a scalarized objective
  `score = latency_ms + lambda * cost_proxy`, sweep `lambda` to show the
  trade-off curve rather than picking one arbitrary weighting.
- Baselines: always-edge, always-cloud, random.
- Model: start with a simple decision-tree/logistic-regression baseline
  before anything fancier — the contribution is the system and the
  evaluation, not model sophistication.
- Evaluate on a held-out synthetic session trace (mix of short/medium/long
  requests in a randomized but reproducible order) not used during training.

### 6.1 Train / validation / test discipline

- Split the merged benchmark dataset by **prompt id**, not by row: every
  repetition and every (model, quant, host) measurement for a given prompt
  goes entirely into one split. Splitting by row would leak the same prompt's
  edge and cloud timings across train and test and inflate the policy's
  apparent accuracy.
- Carve out the test split **once**, immediately after the full sweep is
  merged and cleaned, before any policy modeling starts. Do not look at test
  metrics again until the final reported number — tune `lambda`, features,
  and model choice on the validation split only.
- Train the final policy model with **3+ random seeds** and report the
  spread, not a single run, before it goes in the paper (small deltas
  between the policy and the always-edge/always-cloud baselines usually
  don't survive this check — that's fine to report honestly if it happens).

## 7. Explicit limitations to state in the paper

- The "edge" profile is emulated via resource limits on a laptop, not a
  physical constrained device (Raspberry Pi, phone SoC). State this plainly
  and frame it as a threat to external validity, consistent with how prior
  Wasm-edge papers handle the same limitation when real hardware isn't
  available.
- The energy proxy is CPU-time-based, not a calibrated power measurement.
- Model/quant coverage is necessarily a sample of the space, not exhaustive.

## 8. Reproducibility logging

Before the first real (non-pilot) run, capture and pin, once per host:

- `wasmedge --version` and the `wasi_nn-ggml` plugin version (goes in
  `data/processed/host_specs.md`, §2's placeholders).
- The harness commit/version — since this folder isn't a git repo yet, at
  minimum date-stamp `config.yaml` and the sampled `prompts.jsonl` so a
  later change to either doesn't get silently mixed into old results.
- Pin `pip freeze` output for the harness's Python environment alongside
  `data/raw/`, so package versions (especially `requests`, `psutil`) are
  recoverable if a metric definition question comes up later.

Run a small **pilot** — one model, one quant, one host, ~5 prompts, 2
repetitions — before committing to the full sweep. Confirm the JSONL output
looks sane (non-null `ttft_ms`, plausible `tokens_per_sec`) and that nothing
in the harness crashes partway through. This is the same "prove the
pipeline works before trusting the numbers" discipline as overfitting a
tiny subset in a model-training project — don't skip it and go straight to
the full matrix.

**Pilot findings (TinyLlama-1.1B Q4_K_M, uncapped smoke-test host,
2026-09-02):** the pilot caught two real issues, fixed before the sweep
starts:

- `run_benchmark.py`'s `tokens_per_sec` calculation divided by a near-zero
  decode window on a short completion and produced a nonsensical ~9000
  tok/s. Fixed with a minimum-decode-time guard (see the script) — below
  0.25s of decode time, `tokens_per_sec` is now reported as `None` rather
  than a fabricated number.
- One long-bucket prompt ran for **18.6 minutes** on a single request with
  no output cap (`n_predict` left at the server default of unbounded).
  Across the full model x quant x host x prompt x repetition matrix this is
  not survivable, and it also confounds latency comparisons with
  uncontrolled completion length. **Fix:** launch every `llama-api-server.wasm`
  instance for the real sweep with `--n-predict 256` (or another value fixed
  across the whole matrix — pick one and hold it constant) and pin
  `--temp` to a low, fixed value (e.g. `0` or `0.1`) rather than the
  default `1.0`, since sampling randomness otherwise adds run-to-run output
  length variance on top of the host/model/quant variables actually being
  studied. Document whichever exact flags are used in `host_specs.md` and
  in the paper's methodology section — these are now controlled
  experimental parameters, not incidental server defaults.

**`--n-predict` reduced from 256 to 128 (2026-09-02):** after this pilot
fix, the first real edge-profile sweep measured ~3.3 tokens/sec and
~70-80s per request at `--n-predict 256`, projecting to 5+ hours for a
single model/quant/host combination — impractical across the full matrix
within the project timeline. Cut to `--n-predict 128` to roughly halve
per-request time. **This value was changed on both the edge and cloud
hosts, not just edge** — the "fixed across the whole matrix" rule above
means differing completion length by host would confound any latency
comparison between them with output-length differences rather than actual
host-speed differences. Document 128 (not 256) as the controlled value in
the paper's methodology and in `host_specs.md`.

**`--n-predict` CLI flag does not actually cap generation (2026-09-03) —
critical fix:** discovered via real sweep data: several rows generated
800-2500+ tokens despite the server being launched with `--n-predict 128`.
Root cause, confirmed by direct testing: LlamaEdge 0.29.0 overrides
`--n-predict` on a per-request basis using the request's
`max_completion_tokens` field, defaulting to unbounded
(`2147483647`) when that field is absent — the server logs this
explicitly ("Update n_predict with max_completion_tokens from 128 to
2147483647"). The CLI flag alone is effectively decorative unless every
request also sets `max_completion_tokens`. **Fix:** `run_benchmark.py` now
sends `"max_completion_tokens": <config n_predict>` on every request
(added a top-level `n_predict: 128` key to `config.yaml` as the single
source of truth, kept in sync with the server launch flag). Verified fix
by direct curl test: with `max_completion_tokens: 128` set, a
long-completion prompt stopped at exactly 127 streamed chunks; without it,
the same prompt was still generating past 90 seconds. **All data collected
before this fix (the first `--host edge` and `--host cloud` TinyLlama-1.1B
Q4_K_M attempts) was discarded** — renamed to
`*.discarded_unenforced_npredict.jsonl` in `data/raw/`, not used in
analysis — since an unknown fraction of rows in each file have
uncontrolled, unbounded completion length, invalidating any latency/
throughput comparison. This is the same class of bug as the original
pilot's unbounded-generation finding (§8), resurfacing through the
request-level API instead of the CLI flag — a reminder that a "controlled
parameter" set via server startup flags needs to be verified as actually
enforced per-request, not assumed from the flag's presence alone.
