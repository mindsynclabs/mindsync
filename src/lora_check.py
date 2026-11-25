#!/usr/bin/env python3
"""
Does the LoRA adapter actually change the model?  Compares BASE vs TUNED answers
for the same prompts, using the SAME template as the training script.

  python lora_check.py --adapter ./lora_adapter --data ./sample_data/fine_tune_data_set.jsonl --n 5
  python lora_check.py --adapter ./lora_adapter --question "Who founded MindSync Labs?"
  python lora_check.py --adapter ./lora_adapter --merge-to ./merged_model     # after it works

How to read it
  tuned answer ~ reference on TRAINING rows   -> the adapter learned them
  tuned == base on training rows              -> the adapter has no effect (too few updates,
                                                 wrong adapter folder, or template mismatch)
  good on training rows, bad on your own paraphrases
                                              -> it memorized wording, not facts (use RAG for facts)

Install: pip install torch transformers peft
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(instruction: str) -> str:
    # MUST match InstructionDataset.__getitem__ in the training script
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_rows(path: str, n: int):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt") or row.get("instruction") or ""
        response = row.get("response") or row.get("completion") or row.get("output") or ""
        rows.append((prompt, response))
        if len(rows) >= n:
            break
    return rows


@torch.no_grad()
def generate(model, tok, instruction: str, max_new_tokens: int) -> str:
    enc = tok(build_prompt(instruction), return_tensors="pt", add_special_tokens=False)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: same input -> same output, so base vs tuned is a fair comparison
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def f1(a: str, b: str) -> float:
    """Word-overlap F1 between two texts (0 = nothing in common, 1 = same words)."""
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    common = sum(min(x.count(w), y.count(w)) for w in set(x))
    if common == 0:
        return 0.0
    p, r = common / len(x), common / len(y)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="./lora_adapter", help="folder saved by the training script")
    ap.add_argument("--data", help="training JSONL: its first rows are re-asked word for word")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--question", action="append", default=[], help="your own question (repeatable)")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--merge-to", help="save a merged full model here (needed before converting for Ollama)")
    a = ap.parse_args()

    base_id = PeftConfig.from_pretrained(a.adapter).base_model_name_or_path
    print(f"base model : {base_id}\nadapter    : {a.adapter}")
    tok = AutoTokenizer.from_pretrained(a.adapter)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_id), a.adapter)
    model.eval()

    base_scores, tuned_scores = [], []
    if a.data:
        print("\n=== TRAINING rows, asked word for word ===")
        for prompt, reference in load_rows(a.data, a.n):
            tuned = generate(model, tok, prompt, a.max_new_tokens)
            with model.disable_adapter():  # same weights, adapter switched off = the base model
                base = generate(model, tok, prompt, a.max_new_tokens)
            base_scores.append(f1(base, reference))
            tuned_scores.append(f1(tuned, reference))
            print(f"\nPROMPT   : {prompt}\nREFERENCE: {reference[:200]}\nBASE     : {base[:200]}\nTUNED    : {tuned[:200]}"
                  f"\n(word overlap with reference: base {base_scores[-1]:.2f}, tuned {tuned_scores[-1]:.2f})")
        if tuned_scores:
            avg_b, avg_t = sum(base_scores) / len(base_scores), sum(tuned_scores) / len(tuned_scores)
            print(f"\naverage overlap with the training answers: base {avg_b:.2f} -> tuned {avg_t:.2f}")
            if avg_t - avg_b < 0.1:
                print("=> the adapter barely changes the answers: train longer / more updates, "
                      "or check that this is the adapter folder you trained")

    if a.question:
        print("\n=== YOUR QUESTIONS ===")
        for q in a.question:
            tuned = generate(model, tok, q, a.max_new_tokens)
            with model.disable_adapter():
                base = generate(model, tok, q, a.max_new_tokens)
            print(f"\nQ    : {q}\nBASE : {base[:200]}\nTUNED: {tuned[:200]}")

    if a.merge_to:
        merged = model.merge_and_unload()
        merged.save_pretrained(a.merge_to)
        tok.save_pretrained(a.merge_to)
        print(f"\nMerged full model saved to {a.merge_to}")


if __name__ == "__main__":
    main()