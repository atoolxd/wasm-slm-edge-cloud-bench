"""Flatten data/raw/*.jsonl into CSVs for external analysis (Anara upload).

Writes to data/processed/anara_export/:
  measurements.csv        all 12 sweep combos, every request incl. warmups (is_warmup column)
  discarded_runs.csv      discarded + pilot runs, with discard_reason from the filename
  request_accounting.csv  per-combo issued / warmup / analyzed counts
  prompts.csv             the 49 sweep prompts
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "anara_export"
FIELDS = ["host", "model", "quant", "prompt_id", "bucket", "repetition", "is_warmup",
          "total_latency_ms", "approx_output_tokens", "peak_rss_mb", "ttft_ms", "tokens_per_sec"]


def load(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)  # None -> empty cell, i.e. "not measured", never 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kept, discarded, accounting = [], [], []
    for path in sorted(RAW.glob("*.jsonl")):
        rows = load(path)
        if path.stem.startswith("pilot_") or "discarded" in path.stem:
            reason = "pilot" if path.stem.startswith("pilot_") else path.stem.split(".discarded_")[1]
            discarded += [{**r, "source_file": path.name, "discard_reason": reason} for r in rows]
            continue
        kept += rows
        warm = sum(r["is_warmup"] is True for r in rows)
        accounting.append({"host": rows[0]["host"], "model": rows[0]["model"], "quant": rows[0]["quant"],
                           "issued": len(rows), "warmup_discarded": warm, "analyzed": len(rows) - warm,
                           "source_file": path.name})

    total = {k: sum(a[k] for a in accounting) for k in ("issued", "warmup_discarded", "analyzed")}
    accounting.append({"host": "TOTAL", "model": "", "quant": "", **total, "source_file": ""})
    accounting.append({"host": "EXCLUDED (discarded+pilot runs)", "model": "", "quant": "",
                       "issued": len(discarded), "warmup_discarded": "", "analyzed": 0, "source_file": ""})

    write("measurements.csv", FIELDS, kept)
    write("discarded_runs.csv", FIELDS + ["source_file", "discard_reason"], discarded)
    write("request_accounting.csv",
          ["host", "model", "quant", "issued", "warmup_discarded", "analyzed", "source_file"], accounting)
    prompts = [{"prompt_id": p["id"], "bucket": p["bucket"], "prompt": p["prompt"]}
               for p in load(ROOT / "data" / "prompts" / "prompts.jsonl")]
    write("prompts.csv", ["prompt_id", "bucket", "prompt"], prompts)

    assert len(accounting) == 14 and all(a["analyzed"] == 98 for a in accounting[:12]), accounting
    print(f"kept={len(kept)} discarded={len(discarded)} prompts={len(prompts)} totals={total}")


if __name__ == "__main__":
    main()
