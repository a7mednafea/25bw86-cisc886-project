"""
=============================================================
CISC 886 - Cloud Computing Project
Student: 25bw86
Script: PySpark Preprocessing Pipeline
Dataset: common-pile/stackexchange (34,015,234 records)
Run this on AWS EMR
=============================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ─────────────────────────────────────────────────────────────
# 1. INITIALIZE SPARK SESSION
# ─────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("25bw86-tech-support-preprocessing") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("25bw86 - Tech Support Preprocessing Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 2. DEFINE S3 PATHS
# ─────────────────────────────────────────────────────────────
S3_BUCKET = "s3://25bw86-cisc886-bucket"
RAW_PATH     = f"{S3_BUCKET}/raw-data/raw_data.jsonl"
TRAIN_PATH   = f"{S3_BUCKET}/processed-data/train/"
VAL_PATH     = f"{S3_BUCKET}/processed-data/val/"
TEST_PATH    = f"{S3_BUCKET}/processed-data/test/"
STATS_PATH   = f"{S3_BUCKET}/processed-data/stats/"

# ─────────────────────────────────────────────────────────────
# 3. LOAD RAW DATA FROM S3
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading raw data from S3...")
df = spark.read.json(RAW_PATH)
total_raw = df.count()
print(f"  ✓ Total raw records: {total_raw:,}")
df.printSchema()
df.show(3, truncate=80)

# ─────────────────────────────────────────────────────────────
# 4. REMOVE NULL AND EMPTY ROWS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Removing null and empty rows...")
df_clean = df.dropna(subset=["text"])
df_clean = df_clean.filter(F.trim(F.col("text")) != "")
total_after_null = df_clean.count()
print(f"  ✓ After null removal: {total_after_null:,}")
print(f"  ✓ Removed: {total_raw - total_after_null:,} rows")

# ─────────────────────────────────────────────────────────────
# 5. REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Removing duplicate rows...")
df_clean = df_clean.dropDuplicates(["id"])
total_after_dedup = df_clean.count()
print(f"  ✓ After deduplication: {total_after_dedup:,}")
print(f"  ✓ Removed: {total_after_null - total_after_dedup:,} duplicates")

# ─────────────────────────────────────────────────────────────
# 6. NORMALIZE TEXT
# ─────────────────────────────────────────────────────────────
print("\n[STEP 4] Normalizing text...")
df_clean = df_clean.withColumn("text", F.trim(F.col("text")))
print("  ✓ Whitespace trimmed")

# ─────────────────────────────────────────────────────────────
# 7. COMPUTE TOKEN LENGTHS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 5] Computing token lengths...")
df_clean = df_clean.withColumn(
    "total_tokens",
    (F.size(F.split(F.col("text"), " ")) * 1.3).cast(IntegerType())
)
df_clean.select(
    F.min("total_tokens").alias("min_tokens"),
    F.max("total_tokens").alias("max_tokens"),
    F.avg("total_tokens").alias("avg_tokens")
).show()

# ─────────────────────────────────────────────────────────────
# 8. FILTER BY TOKEN LENGTH (10 to 512)
# ─────────────────────────────────────────────────────────────
print("\n[STEP 6] Filtering by token length (10-512)...")
df_filtered = df_clean.filter(
    (F.col("total_tokens") >= 10) &
    (F.col("total_tokens") <= 512)
)
total_after_filter = df_filtered.count()
print(f"  ✓ After filtering: {total_after_filter:,}")
print(f"  ✓ Removed: {total_after_dedup - total_after_filter:,} rows")

# ─────────────────────────────────────────────────────────────
# 9. FORMAT INTO CHAT TEMPLATE
# ─────────────────────────────────────────────────────────────
print("\n[STEP 7] Applying chat template...")
df_formatted = df_filtered.withColumn(
    "formatted_text",
    F.concat(
        F.lit("Below is a tech support question. Write a helpful response.\n\n"),
        F.lit("### Question:\n"),
        F.col("text"),
        F.lit("\n\n### Answer:\n")
    )
)
print("  ✓ Chat template applied")

# ─────────────────────────────────────────────────────────────
# 10. TRAIN / VAL / TEST SPLIT (80/10/10)
# ─────────────────────────────────────────────────────────────
print("\n[STEP 8] Splitting into train/val/test...")
train_df, val_df, test_df = df_formatted.randomSplit(
    [0.8, 0.1, 0.1], seed=42
)
train_count = train_df.count()
val_count   = val_df.count()
test_count  = test_df.count()
print(f"  ✓ Train:      {train_count:,}")
print(f"  ✓ Validation: {val_count:,}")
print(f"  ✓ Test:       {test_count:,}")

# ─────────────────────────────────────────────────────────────
# 11. SAVE TO S3
# ─────────────────────────────────────────────────────────────
print("\n[STEP 9] Saving to S3...")
cols = ["id", "text", "formatted_text", "total_tokens"]
train_df.select(cols).write.mode("overwrite").json(TRAIN_PATH)
print(f"  ✓ Train saved → {TRAIN_PATH}")
val_df.select(cols).write.mode("overwrite").json(VAL_PATH)
print(f"  ✓ Val saved   → {VAL_PATH}")
test_df.select(cols).write.mode("overwrite").json(TEST_PATH)
print(f"  ✓ Test saved  → {TEST_PATH}")

# ─────────────────────────────────────────────────────────────
# 12. SAVE STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 10] Saving statistics...")
stats = [
    ("01_total_raw",             total_raw),
    ("02_after_null_removal",    total_after_null),
    ("03_after_deduplication",   total_after_dedup),
    ("04_after_length_filter",   total_after_filter),
    ("05_train_samples",         train_count),
    ("06_val_samples",           val_count),
    ("07_test_samples",          test_count),
]
stats_df = spark.createDataFrame(stats, ["metric", "value"])
stats_df.write.mode("overwrite").json(STATS_PATH)
stats_df.show()

print("\n" + "=" * 60)
print("✓ PREPROCESSING COMPLETE!")
print("=" * 60)
spark.stop()