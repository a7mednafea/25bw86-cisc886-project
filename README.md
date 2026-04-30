# CISC 886 – Cloud-Based Tech Support Chatbot
**Student:** 25bw86  
**Domain:** Tech Support  
**Model:** TinyLlama-1.1B (Fine-tuned → 25bw86-techsupport)  
**Dataset:** common-pile/stackexchange (34,015,234 records)

## Prerequisites
- Python 3.11+
- AWS Account with credentials configured
- AWS Region: us-east-1
- NVIDIA GPU with 16GB VRAM (for local fine-tuning)
- Unsloth library installed

## Project Structure
25bw86-cisc886-project/
├── fine-tuning/
│   ├── finetune.py                 # Fine-tuning script
│   └── fine_tune_notebook.ipynb    # Fine-tuning notebook
├── spark/
│   ├── preprocess.py               # PySpark preprocessing
│   └── upload_data.py              # Dataset upload to S3
├── report/
│   └── figures/                    # Architecture + EDA figures
└── README.md

## Replication Steps

### Phase 1 — Infrastructure (AWS CLI)
```bash
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

Phase 2 — Upload Full Dataset to S3 (EC2)
# SSH into EC2
ssh -i 25bw86-key.pem ubuntu@YOUR_EC2_IP

# Install dependencies
pip3 install datasets boto3

# Upload full dataset (34M records) directly to S3
python3 upload_full.py
# This streams 34,015,234 records directly to S3
# s3://25bw86-cisc886-bucket/raw-data/raw_data.jsonl

Phase 3 — Spark Preprocessing on AWS EMR


# Upload preprocessing script to S3
aws s3 cp spark/preprocess.py s3://25bw86-cisc886-bucket/scripts/preprocess.py

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

# Terminate cluster after job completes
aws emr terminate-clusters --cluster-ids YOUR_CLUSTER_ID --region us-east-1

Phase 4 — Fine-Tuning (Local GPU)

cd fine-tuning
python finetune.py
# Trains TinyLlama-1.1B on 31,077 Stack Exchange records
# Uses LoRA (r=16, alpha=32) with 4-bit quantization
# Takes ~35 minutes on NVIDIA RTX 2000 Ada (16GB VRAM)

Phase 5 — Export Model to GGUF (EC2)

# Download fine-tuned model from S3
aws s3 cp s3://25bw86-cisc886-bucket/model/ ~/fine-tuned-model/ --recursive

# Download base model
cd ~/base-model
wget -c "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"

# Merge LoRA adapter with base model
python3 convert_to_gguf.py

# Convert to GGUF format
python3 llama.cpp/convert_hf_to_gguf.py /home/ubuntu/merged-model \
  --outfile /home/ubuntu/25bw86-techsupport.gguf \
  --outtype q8_0

# Upload GGUF to S3
aws s3 cp ~/25bw86-techsupport.gguf s3://25bw86-cisc886-bucket/model/25bw86-techsupport.gguf

Phase 6 — EC2 Deployment


# SSH into EC2
ssh -i 25bw86-key.pem ubuntu@YOUR_EC2_IP

# Install NVIDIA drivers
sudo apt install nvidia-driver-535 -y
sudo reboot

# Verify GPU
nvidia-smi

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download fine-tuned model from S3
aws s3 cp s3://25bw86-cisc886-bucket/model/25bw86-techsupport.gguf ~/25bw86-techsupport.gguf

# Create custom Ollama model
cat > ~/Modelfile << 'EOF'
FROM /home/ubuntu/25bw86-techsupport.gguf
PARAMETER num_ctx 2048
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.1
SYSTEM You are a tech support assistant. Answer technical questions helpfully and clearly.
EOF

ollama create 25bw86-techsupport -f ~/Modelfile

# Test the model
ollama run 25bw86-techsupport "My Windows computer shows a blue screen error with code 0x0000007E. What should I do?"

# Deploy OpenWebUI
sudo docker run -d \
  --name 25bw86-openwebui \
  --restart always \
  --network host \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  -p 8080:8080 \
  ghcr.io/open-webui/open-webui:main

# Access at http://YOUR_EC2_IP:8080

Cost Summary
AWS Service	Usage	Approx Cost
EC2 g4dn.xlarge	~5 hours	~$2.65
EMR Cluster (3x m5.xlarge)	~1 hour	~$0.75
S3 Storage (120GB)	Full dataset + model	~$2.76
Data Transfer	Upload/Download	~$1.00
Total		~$7.16
Hyperparameter Table
Parameter	Value
Model	TinyLlama-1.1B (unsloth/tinyllama)
LoRA rank (r)	16
LoRA alpha	32
LoRA dropout	0
Learning rate	2e-4
Batch size	4
Gradient accumulation	4
Effective batch size	16
Epochs	1
Max sequence length	512
Quantization	4-bit (QLoRA)
Optimizer	AdamW 8-bit
Target modules	q_proj, v_proj
Trainable parameters	2,252,800 (0.20%)