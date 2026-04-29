# CISC 886 – Cloud-Based Tech Support Chatbot
**Student:** 25bw86  
**Domain:** Tech Support  
**Model:** TinyLlama-1.1B  
**Dataset:** common-pile/stackexchange (~30.4M records)

## Prerequisites
- Python 3.11+
- AWS Account with credentials
- AWS Region: us-east-1
- NVIDIA GPU with 16GB VRAM (for local fine-tuning)

## Project Structure
25bw86-cisc886-project/
├── fine-tuning/finetune.py     # Fine-tuning script
├── spark/preprocess.py         # PySpark preprocessing
├── spark/upload_data.py        # Dataset upload to S3
├── deployment/                 # EC2 deployment commands
└── report/                     # Report figures and notes 

## Replication Steps


### Phase 1 — Infrastructure
1. Log into AWS Console
2. Create VPC: `25bw86-vpc` (CIDR: 10.0.0.0/16)
3. Create public subnet: `10.0.1.0/24`
4. Create security group: `25bw86-sg`
5. Create S3 bucket: `aws s3 mb s3://25bw86-bucket`

### Phase 2 — Data Upload
```bash
# Run in AWS CloudShell
pip install datasets boto3
python spark/upload_data.py
```

### Phase 3 — Spark Preprocessing (AWS EMR)
1. Launch EMR cluster with Spark
2. Upload script: `aws s3 cp spark/preprocess.py s3://25bw86-bucket/scripts/`
3. Submit job via EMR console
4. Terminate cluster after job completes

### Phase 4 — Fine-Tuning (Local GPU)
```bash
cd fine-tuning
python finetune.py
```

### Phase 5 — EC2 Deployment
```bash
# SSH into EC2 instance
ssh -i 25bw86-key.pem ubuntu@YOUR_EC2_IP

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Load model and start OpenWebUI
ollama create 25bw86-chatbot -f Modelfile
docker run -d -p 8080:8080 ghcr.io/open-webui/open-webui:main
```

## Cost Summary

| AWS Service | Usage | Approx Cost |
|-------------|-------|-------------|
| S3 Storage (50GB) | Dataset + Model | ~$1.15/month |
| EMR Cluster (2 hrs) | Spark preprocessing | ~$0.60 |
| EC2 g4dn.xlarge (10 hrs) | Model serving | ~$5.30 |
| Data Transfer | Upload/Download | ~$0.50 |
| **Total** | | **~$7.55** |

## Hyperparameter Table

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama-1.1B |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Epochs | 1 |
| Max sequence length | 512 |
| Quantization | 4-bit (QLoRA) |