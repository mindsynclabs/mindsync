import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./lora_adapter"

print("Dang nap base model va LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Quan trong: Gan LoRA adapter da train vao base model
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

test_questions = [
    "Who founded MindSync Labs and when?",
    "Where is MindSync Labs located?",
    "What is MindSync Labs' flagship product?",
    "Does MindSync require a cloud subscription?",
    "Write a Python function to reverse a string.",
]

for q in test_questions:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for MindSync Labs."},
        {"role": "user", "content": q},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # Tat random de kiem tra do chinh xac kien thuc
            pad_token_id=tokenizer.eos_token_id,
        )

    # Chi cat lay phan model moi sinh ra
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {reply.strip()}")
