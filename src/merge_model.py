import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./lora_adapter"
MERGED_DIR = "./mindsync-qwen2.5-3b"

print("1. Dang nap base model (float16)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  # Gop tren RAM de tranh tran VRAM GPU
)

print("2. Dang gan va gop trong so LoRA...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged_model = model.merge_and_unload()

print(f"3. Dang luu model hoan chinh tai: {MERGED_DIR}...")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("Hoan tat! Ban da co thu muc model doc lap.")
