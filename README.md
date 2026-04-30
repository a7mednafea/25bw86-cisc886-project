# CISC 886 – Cloud-Based Tech Support Chatbot

**Student:** 25bw86  
**Domain:** Tech Support  
**Model:** TinyLlama-1.1B (Fine-tuned as 25bw86-techsupport)  
**Dataset:** common-pile/stackexchange (34,015,234 records)

---

## Prerequisites
- Python 3.11+
- AWS Account with credentials configured
- AWS Region: us-east-1
- NVIDIA GPU with 16GB VRAM (for local fine-tuning)
- Unsloth library installed

---

## Project Structure

| Folder/File | Description |
|-------------|-------------|
| `deployment/deploy.sh` | EC2 deployment script |
| `fine-tuning/finetune.py` | Fine-tuning script |
| `fine-tuning/fine_tune_notebook.ipynb` | Fine-tuning notebook |
| `spark/preprocess.py` | PySpark preprocessing pipeline |
| `spark/upload_data.py` | Dataset upload to S3 |
| `report/figures/` | Architecture diagram + EDA figures |
| `README.md` | Full replication guide + cost table |

---

## Replication Steps

```bash
# ============================================================
# PHASE 1 — INFRASTRUCTURE
# ============================================================

# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --region us-east-1

# Create public subnet
aws ec2 create-subnet --vpc-id YOUR_VPC_ID --cidr-block 10.0.1.0/24

# Create security group
aws ec2 create-security-group --group-name 25bw86-sg --vpc-id YOUR_VPC_ID

# Open required ports
aws ec2 authorize-security-group-ingress --group-id YOUR_SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id YOUR_SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id YOUR_SG_ID --protocol tcp --port 8080 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id YOUR_SG_ID --protocol tcp --port 11434 --cidr 0.0.0.0/0

# Create S3 bucket
aws s3 mb s3://25bw86-cisc886-bucket --region us-east-1

# ============================================================
# PHASE 2 — UPLOAD FULL DATASET TO S3 (Run on EC2)
# ============================================================

pip3 install datasets boto3
python3 upload_full.py
# Streams 34,015,234 records to s3://25bw86-cisc886-bucket/raw-data/

# ============================================================
# PHASE 3 — SPARK PREPROCESSING ON AWS EMR
# ============================================================

# Upload script to S3
aws s3 cp spark/preprocess.py s3://25bw86-cisc886-bucket/scripts/

# Launch EMR cluster (3 nodes)
aws emr create-cluster \
  --name "25bw86-emr-full" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --use-default-roles \
  --ec2-attributes KeyName=25bw86-key,SubnetId=YOUR_SUBNET_ID \
  --region us-east-1 \
  --log-uri s3://25bw86-cisc886-bucket/emr-logs/

# Submit Spark job
aws emr add-steps \
  --cluster-id YOUR_CLUSTER_ID \
  --steps Type=Spark,Name="25bw86-spark-preprocess",ActionOnFailure=CONTINUE,Args=[s3://25bw86-cisc886-bucket/scripts/preprocess.py] \
  --region us-east-1

# Check status
aws emr describe-step --cluster-id YOUR_CLUSTER_ID --step-id YOUR_STEP_ID \
  --query 'Step.Status.State' --output text --region us-east-1

# Terminate cluster after completion
aws emr terminate-clusters --cluster-ids YOUR_CLUSTER_ID --region us-east-1

# ============================================================
# PHASE 4 — FINE-TUNING (Local GPU - NVIDIA RTX 2000 Ada)
# ============================================================

cd fine-tuning
python finetune.py
# Trains on 31,077 Stack Exchange records
# Uses LoRA (r=16, alpha=32) with 4-bit quantization
# Takes ~35 minutes on NVIDIA RTX 2000 Ada (16GB VRAM)

# ============================================================
# PHASE 5 — EXPORT MODEL TO GGUF (Run on EC2)
# ============================================================

# Merge LoRA adapter with base model
python3 convert_to_gguf.py

# Convert to GGUF format
python3 llama.cpp/convert_hf_to_gguf.py /home/ubuntu/merged-model \
  --outfile /home/ubuntu/25bw86-techsupport.gguf \
  --outtype q8_0

# Upload GGUF to S3
aws s3 cp ~/25bw86-techsupport.gguf s3://25bw86-cisc886-bucket/model/

# ============================================================
# PHASE 6 — EC2 DEPLOYMENT
# ============================================================

# SSH into EC2
ssh -i 25bw86-key.pem ubuntu@YOUR_EC2_IP

# Run deployment script
bash deployment/deploy.sh

# Access OpenWebUI at:
# http://YOUR_EC2_IP:8080

---

## Cost Summary

| AWS Service | Usage | Approx Cost |
|-------------|-------|-------------|
| EC2 g4dn.xlarge | ~5 hours | ~$2.65 |
| EMR Cluster (3x m5.xlarge) | ~1 hour | ~$0.75 |
| S3 Storage (120GB) | Full dataset + model | ~$2.76 |
| Data Transfer | Upload/Download | ~$1.00 |
| **Total** | | **~$7.16** |

---

## Hyperparameter Table

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama-1.1B (unsloth/tinyllama) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Epochs | 1 |
| Max sequence length | 512 |
| Quantization | 4-bit (QLoRA) |
| Optimizer | AdamW 8-bit |
| Target modules | q_proj, v_proj |
| Trainable parameters | 2,252,800 (0.20%) |