"""
=============================================================
CISC 886 - Cloud Computing Project
Student: 25bw86
Script: Fine-tuning TinyLlama on StackExchange Tech Support
Device: NVIDIA RTX 2000 Ada (16GB VRAM) - Local Windows
=============================================================
"""

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import os

# ─────────────────────────────────────────────────────────────
# 1. VERIFY GPU
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("25bw86 - Tech Support Chatbot Fine-Tuning")
print("=" * 60)
print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)} GB")

# ─────────────────────────────────────────────────────────────
# 2. LOAD BASE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading TinyLlama base model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/tinyllama",
    max_seq_length = 512,
    dtype          = None,
    load_in_4bit   = True,
)
print("✓ Model loaded successfully!")

# ─────────────────────────────────────────────────────────────
# 3. APPLY LoRA ADAPTER
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Applying LoRA adapter...")

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "v_proj"],
    lora_alpha                 = 32,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = True,
    random_state               = 42,
)
model.print_trainable_parameters()
print("✓ LoRA adapter applied!")

# ─────────────────────────────────────────────────────────────
# 4. LOAD AND FORMAT DATASET
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Loading StackExchange dataset...")

raw_dataset = load_dataset(
    "common-pile/stackexchange",
    split     = "train",
    streaming = True
)

def format_record(record):
    return {
        "text": record['text'][:600]
    }


print("Collecting 500,000 records...")
records = []
for i, record in enumerate(raw_dataset):
    if len(record['text'].strip()) > 50:
        records.append(format_record(record))
    if len(records) >= 500000:
        break
    if (i + 1) % 10000 == 0:
        print(f"  Collected {len(records):,} so far...")

hf_dataset = Dataset.from_list(records)
print(f"✓ Dataset ready: {len(hf_dataset):,} samples")

# ─────────────────────────────────────────────────────────────
# 5. RECORD BASE MODEL RESPONSE (BEFORE fine-tuning)
#    Save this for your report - shows effect of fine-tuning
# ─────────────────────────────────────────────────────────────
print("\n[STEP 4] Recording BASE model response (before training)...")

FastLanguageModel.for_inference(model)
test_prompt = """Below is a tech support question. Write a clear and helpful response.

### Question:
My Windows computer shows a blue screen error with code 0x0000007E. What should I do?

### Answer:"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 150,
        temperature    = 0.7,
        do_sample      = True,
    )
base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save base response to file for report
os.makedirs("report", exist_ok=True)
with open("report/base_model_response.txt", "w") as f:
    f.write("=== BASE MODEL RESPONSE (Before Fine-Tuning) ===\n\n")
    f.write(base_response)
print("✓ Base model response saved to report/base_model_response.txt")
print(f"\nBase response preview:\n{base_response[len(test_prompt):len(test_prompt)+200]}")

# ─────────────────────────────────────────────────────────────
# 6. FINE-TUNE THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[STEP 5] Starting fine-tuning...")

# Switch back to training mode
model.train()

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = hf_dataset,
    dataset_text_field = "text",
    max_seq_length     = 512,
    args = TrainingArguments(
        output_dir                  = "./outputs",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps                = 50,
        num_train_epochs            = 1,
        learning_rate               = 2e-4,
        fp16                        = False,
        bf16                        = True,
        logging_steps               = 50,
        save_strategy               = "epoch",
        report_to                   = "none",
        optim                       = "adamw_8bit",
    ),
)

trainer_stats = trainer.train()
print(f"\n✓ Training complete!")
print(f"✓ Final training loss: {trainer_stats.training_loss:.4f}")
print(f"✓ Total steps: {trainer_stats.global_step}")

# ─────────────────────────────────────────────────────────────
# 7. RECORD FINE-TUNED MODEL RESPONSE (AFTER fine-tuning)
# ─────────────────────────────────────────────────────────────
print("\n[STEP 6] Recording FINE-TUNED model response (after training)...")

FastLanguageModel.for_inference(model)
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 150,
        temperature    = 0.7,
        do_sample      = True,
    )
finetuned_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

with open("report/finetuned_model_response.txt", "w") as f:
    f.write("=== FINE-TUNED MODEL RESPONSE (After Fine-Tuning) ===\n\n")
    f.write(finetuned_response)
print("✓ Fine-tuned response saved to report/finetuned_model_response.txt")
print(f"\nFine-tuned response preview:\n{finetuned_response[len(test_prompt):len(test_prompt)+200]}")

# ─────────────────────────────────────────────────────────────
# 8. SAVE THE MODEL
# ─────────────────────────────────────────────────────────────
print("\n[STEP 7] Saving fine-tuned model...")

model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
print("✓ Model saved to ./fine-tuned-model")

# Save in GGUF format for Ollama deployment
print("\n[STEP 8] Exporting to GGUF format for Ollama...")
model.save_pretrained_gguf(
    "25bw86-techsupport-500k",
    tokenizer,
    quantization_method = "q4_k_m"
)
print("✓ GGUF model saved to ./25bw86-techsupport-model")

print("\n" + "=" * 60)
print("✓ FINE-TUNING COMPLETE!")
print("=" * 60)