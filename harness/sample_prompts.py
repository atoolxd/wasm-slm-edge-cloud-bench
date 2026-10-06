"""
Sample a stratified, reproducible prompt set from the Stanford Alpaca
instruction dataset for the benchmark sweep.

Usage:
    python sample_prompts.py --n-per-bucket 20 --seed 42

Output:
    ../data/prompts/prompts.jsonl — one JSON object per line:
        {"id": "...", "bucket": "short|medium|long", "prompt": "..."}

Token length is approximated by whitespace-splitting and multiplying by
~1.3 (a common rough words-to-tokens ratio for English). This is an
approximation, not an exact tokenizer count — say so in the paper rather
than implying more precision than it has. Swap in a real tokenizer
(e.g. the target model's own tokenizer via `transformers`) later if you
want exact bucket boundaries; it isn't required for a first pass.
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)
CACHE_PATH = Path(__file__).parent / ".alpaca_data.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "prompts" / "prompts.jsonl"

BUCKETS = {
    "short": (0, 64),
    "medium": (64, 256),
    "long": (256, 512),
}


def approx_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def load_alpaca() -> list[dict]:
    if not CACHE_PATH.exists():
        print(f"Downloading Alpaca dataset to {CACHE_PATH} ...")
        urllib.request.urlretrieve(ALPACA_URL, CACHE_PATH)
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(item: dict) -> str:
    # Alpaca items have "instruction", optional "input", and "output".
    # We only want the input side — the model-facing prompt.
    if item.get("input"):
        return f"{item['instruction']}\n\n{item['input']}"
    return item["instruction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bucket", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_alpaca()

    bucketed: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for item in data:
        prompt = build_prompt(item)
        n_tok = approx_token_count(prompt)
        for bucket, (lo, hi) in BUCKETS.items():
            if lo <= n_tok < hi:
                bucketed[bucket].append(prompt)
                break

    sampled: list[dict] = []
    for bucket, candidates in bucketed.items():
        random.shuffle(candidates)
        chosen = candidates[: args.n_per_bucket]
        if len(chosen) < args.n_per_bucket:
            print(
                f"WARNING: only found {len(chosen)}/{args.n_per_bucket} "
                f"candidates for bucket '{bucket}'"
            )
        for i, prompt in enumerate(chosen):
            sampled.append(
                {"id": f"{bucket}-{i:03d}", "bucket": bucket, "prompt": prompt}
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} prompts to {OUTPUT_PATH}")
    for bucket in BUCKETS:
        count = sum(1 for r in sampled if r["bucket"] == bucket)
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
