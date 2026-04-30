"""
=============================================================
CISC 886 - Cloud Computing Project
Student: 25bw86
Script: Upload full StackExchange dataset to S3
Streams 34,015,234 records directly to S3 using multipart upload
=============================================================
"""
import boto3
import json
from datasets import load_dataset

print("Streaming FULL StackExchange dataset directly to S3...")
dataset = load_dataset(
    "common-pile/stackexchange",
    split="train",
    streaming=True
)

s3 = boto3.client("s3", region_name="us-east-1")
bucket = "25bw86-cisc886-bucket"
key = "raw-data/raw_data.jsonl"

mpu = s3.create_multipart_upload(Bucket=bucket, Key=key)
upload_id = mpu["UploadId"]
parts = []
part_number = 1
buffer = []
buffer_size = 0
chunk_limit = 50 * 1024 * 1024

count = 0
try:
    for record in dataset:
        line = json.dumps({
            "id": record["id"],
            "text": record["text"],
            "created": str(record["created"])
        }) + "\n"
        buffer.append(line)
        buffer_size += len(line.encode())
        count += 1

        if count % 100000 == 0:
            print(f"  Processed {count:,} records...")

        if buffer_size >= chunk_limit:
            data = "".join(buffer).encode()
            response = s3.upload_part(
                Bucket=bucket, Key=key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=data
            )
            parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
            part_number += 1
            buffer = []
            buffer_size = 0
            print(f"  Uploaded part {part_number-1} ({count:,} records so far)...")

    if buffer:
        data = "".join(buffer).encode()
        response = s3.upload_part(
            Bucket=bucket, Key=key,
            PartNumber=part_number,
            UploadId=upload_id,
            Body=data
        )
        parts.append({"PartNumber": part_number, "ETag": response["ETag"]})

    s3.complete_multipart_upload(
        Bucket=bucket, Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts}
    )
    print(f"Upload complete! Total records: {count:,}")

except Exception as e:
    s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
    print(f"Error: {e}")
