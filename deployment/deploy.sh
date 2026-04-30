#!/bin/bash
# 25bw86 - EC2 Deployment Script

# Install NVIDIA drivers
sudo apt install nvidia-driver-535 -y
sudo reboot

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama

# Download fine-tuned model from S3
aws s3 cp s3://25bw86-cisc886-bucket/model/25bw86-techsupport.gguf ~/25bw86-techsupport.gguf

# Create Modelfile
cat > ~/Modelfile << 'EOF'
FROM /home/ubuntu/25bw86-techsupport.gguf
PARAMETER num_ctx 2048
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.1
SYSTEM You are a tech support assistant. Answer technical questions helpfully and clearly.
EOF

# Create custom Ollama model
ollama create 25bw86-techsupport -f ~/Modelfile

# Deploy OpenWebUI
sudo docker run -d \
  --name 25bw86-openwebui \
  --restart always \
  --network host \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  -p 8080:8080 \
  ghcr.io/open-webui/open-webui:main
