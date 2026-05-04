"""
=============================================================
CISC 886 - Cloud Computing Project
Student: 25bw86
Script: Local Preprocessing Pipeline (500K records)
- Streams from HuggingFace
- Applies same steps as EMR PySpark pipeline
- Saves results locally AND to S3
=============================================================
"""

import json
import boto3
import re
import os
from datasets import load_dataset
from collections import defaultdict

print("=" * 60)
print("25bw86 - Local Preprocessing Pipeline (500K records)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
TARGET_RECORDS   = 500_000
MIN_TOKENS       = 10
MAX_TOKENS       = 512
SPLIT_RATIOS     = (0.8, 0.1, 0.1)
RANDOM_SEED      = 42
S3_BUCKET        = "25bw86-cisc886-bucket"
LOCAL_OUTPUT_DIR = "./preprocessed_data"

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — STREAM AND COLLECT 500K RECORDS
# ─────────────────────────────────────────────────────────────
print(f"\n[STEP 1] Streaming {TARGET_RECORDS:,} records from HuggingFace...")

raw_dataset = load_dataset(
    "common-pile/stackexchange",
    split="train",
    streaming=True
)

raw_records = []
for i, record in enumerate(raw_dataset):
    raw_records.append({
        "id":      str(record["id"]),
        "text":    record["text"],
        "created": str(record["created"])
    })
    if len(raw_records) >= TARGET_RECORDS:
        break
    if (i + 1) % 50000 == 0:
        print(f"  Collected {len(raw_records):,} records...")

print(f"  ✓ Total raw records: {len(raw_records):,}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — REMOVE NULL AND EMPTY ROWS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Removing null and empty rows...")
before = len(raw_records)
clean = [r for r in raw_records if r["text"] and r["text"].strip()]
print(f"  ✓ After null removal: {len(clean):,} (removed {before - len(clean):,})")

# ─────────────────────────────────────────────────────────────
# STEP 3 — REMOVE DUPLICATES BY ID
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Removing duplicate records...")
before = len(clean)
seen_ids = set()
deduped = []
for r in clean:
    if r["id"] not in seen_ids:
        seen_ids.add(r["id"])
        deduped.append(r)
print(f"  ✓ After deduplication: {len(deduped):,} (removed {before - len(deduped):,})")

# ─────────────────────────────────────────────────────────────
# STEP 4 — NORMALIZE TEXT
# ─────────────────────────────────────────────────────────────
print("\n[STEP 4] Normalizing text...")
for r in deduped:
    r["text"] = r["text"].strip()
print("  ✓ Whitespace trimmed")

# ─────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE TOKEN LENGTHS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 5] Computing token lengths...")
for r in deduped:
    word_count = len(r["text"].split())
    r["total_tokens"] = int(word_count * 1.3)

token_lengths = [r["total_tokens"] for r in deduped]
print(f"  ✓ Min tokens: {min(token_lengths):,}")
print(f"  ✓ Max tokens: {max(token_lengths):,}")
print(f"  ✓ Avg tokens: {sum(token_lengths) // len(token_lengths):,}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — FILTER BY TOKEN LENGTH
# ─────────────────────────────────────────────────────────────
print(f"\n[STEP 6] Filtering by token length ({MIN_TOKENS}-{MAX_TOKENS})...")
before = len(deduped)
filtered = [r for r in deduped if MIN_TOKENS <= r["total_tokens"] <= MAX_TOKENS]
print(f"  ✓ After filtering: {len(filtered):,} (removed {before - len(filtered):,})")

# ─────────────────────────────────────────────────────────────
# STEP 7 — APPLY CHAT TEMPLATE
# ─────────────────────────────────────────────────────────────
print("\n[STEP 7] Applying chat template...")
for r in filtered:
    r["formatted_text"] = (
        "Below is a tech support question. "
        "Write a helpful response.\n\n"
        f"### Question:\n{r['text']}\n\n### Answer:\n"
    )
print("  ✓ Chat template applied")

# ─────────────────────────────────────────────────────────────
# STEP 8 — TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[STEP 8] Splitting into train/val/test (80/10/10)...")
import random
random.seed(RANDOM_SEED)
random.shuffle(filtered)

total = len(filtered)
train_end = int(total * 0.8)
val_end   = int(total * 0.9)

train_data = filtered[:train_end]
val_data   = filtered[train_end:val_end]
test_data  = filtered[val_end:]

print(f"  ✓ Train:      {len(train_data):,}")
print(f"  ✓ Validation: {len(val_data):,}")
print(f"  ✓ Test:       {len(test_data):,}")

# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE LOCALLY
# ─────────────────────────────────────────────────────────────
print("\n[STEP 9] Saving locally...")

def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(data):,} records → {filepath}")

save_jsonl(train_data, f"{LOCAL_OUTPUT_DIR}/train.jsonl")
save_jsonl(val_data,   f"{LOCAL_OUTPUT_DIR}/val.jsonl")
save_jsonl(test_data,  f"{LOCAL_OUTPUT_DIR}/test.jsonl")

# Save stats
stats = {
    "total_raw":           len(raw_records),
    "after_null_removal":  len(clean),
    "after_deduplication": len(deduped),
    "after_token_filter":  len(filtered),
    "train_samples":       len(train_data),
    "val_samples":         len(val_data),
    "test_samples":        len(test_data),
    "mean_tokens":         sum(token_lengths) // len(token_lengths),
    "min_tokens":          min(token_lengths),
    "max_tokens":          max(token_lengths),
}
with open(f"{LOCAL_OUTPUT_DIR}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  ✓ Stats saved → {LOCAL_OUTPUT_DIR}/stats.json")

# ─────────────────────────────────────────────────────────────
# STEP 10 — UPLOAD TO S3
# ─────────────────────────────────────────────────────────────
print("\n[STEP 10] Uploading to S3...")
s3 = boto3.client("s3", region_name="us-east-1")

def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  ✓ Uploaded → s3://{S3_BUCKET}/{s3_key}")

upload_to_s3(f"{LOCAL_OUTPUT_DIR}/train.jsonl", "processed-data/train/train.jsonl")
upload_to_s3(f"{LOCAL_OUTPUT_DIR}/val.jsonl",   "processed-data/val/val.jsonl")
upload_to_s3(f"{LOCAL_OUTPUT_DIR}/test.jsonl",  "processed-data/test/test.jsonl")
upload_to_s3(f"{LOCAL_OUTPUT_DIR}/stats.json",  "processed-data/stats/stats.json")

print("\n" + "=" * 60)
print("✓ PREPROCESSING COMPLETE!")
print(f"  Total clean samples: {len(filtered):,}")
print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
print("=" * 60)