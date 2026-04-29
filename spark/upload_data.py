"""
=============================================================
CISC 886 - Cloud Computing Project
Student: 25bw86
Script: Stream dataset from HuggingFace and upload to S3
Dataset: common-pile/stackexchange (~30.4M records)
NOTE: Run this in AWS CloudShell
=============================================================
"""

import boto3
import json
import os
from datasets import load_dataset

print("=" * 60)
print("25bw86 - StackExchange Dataset Upload to S3")
print("=" * 60)

MAX_RECORDS = 500_000
OUTPUT_FILE = "raw_data.jsonl"
BUCKET      = "25bw86-bucket"
S3_KEY      = "raw-data/raw_data.jsonl"

print(f"\nStep 1: Streaming first {MAX_RECORDS:,} records from HuggingFace...")

dataset = load_dataset(
    "common-pile/stackexchange",
    split     = "train",
    streaming = True
)

print(f"\nStep 2: Saving {MAX_RECORDS:,} records to local file...")

count = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in dataset:
        f.write(json.dumps({
            "id":      record["id"],
            "text":    record["text"],
            "created": str(record["created"])
        }) + "\n")
        count += 1
        if count % 10_000 == 0:
            print(f"  Saved {count:,} / {MAX_RECORDS:,} records...", end="\r")
        if count >= MAX_RECORDS:
            break

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\n✓ Saved {count:,} records ({size_mb:.2f} MB)")

print("\nStep 3: Uploading to S3...")
s3 = boto3.client("s3", region_name="us-east-1")
s3.upload_file(OUTPUT_FILE, BUCKET, S3_KEY)
print(f"✓ Uploaded to s3://{BUCKET}/{S3_KEY}")

obj = s3.head_object(Bucket=BUCKET, Key=S3_KEY)
print(f"✓ Verified on S3: {obj['ContentLength']/(1024*1024):.2f} MB")

os.remove(OUTPUT_FILE)
print("✓ Local file cleaned up")
print("\n✓ UPLOAD COMPLETE!")