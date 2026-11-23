import argparse
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    "Who founded MindSync Labs and what does it build?",
    "Where is MindSync Labs located and what is its flagship product?",
    "Does MindSync require a cloud subscription?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify fine-tuned LoRA adapter.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="./lora_adapter")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    model.eval()

    for idx, question in enumerate(QUESTIONS, start=1):
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"--- Test {idx} ---")
        print(f"Q: {question}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()