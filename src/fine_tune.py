import argparse
import json
import math
from pathlib import Path
import random
import time

print("  -> Dang nap PyTorch...", flush=True)
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

print("  -> Dang nap PEFT & Transformers...", flush=True)
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.samples = list()
        self.tokenizer = tokenizer
        self.max_length = max_length

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file du lieu: {path.resolve()}")

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    self.samples.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Loi cu phap JSON o dong {line_num}") from exc

        if not self.samples:
            raise ValueError(f"File du lieu bi rong: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        if (
            "messages" in row
            and isinstance(row["messages"], list)
            and len(row["messages"]) >= 2
        ):
            messages = row["messages"]
        else:
            prompt = row.get("prompt") or row.get("instruction") or ""
            response = (
                row.get("response") or row.get("completion") or row.get("output") or ""
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for MindSync Labs.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]

        if not full_ids:
            fallback_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else 0
            )
            full_ids = [fallback_id]

        # Tranh truong hop prompt dai qua max_length lam tat ca labels deu la -100 (gay NaN loss)
        prompt_len = min(len(prompt_ids), max(len(full_ids) - 1, 0))
        labels = [-100 for _ in range(prompt_len)] + full_ids[prompt_len:]
        attn_mask = [1 for _ in full_ids]

        return {
            "input_ids": full_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_input_ids = list()
        batch_attention_mask = list()
        batch_labels = list()

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(
                item["input_ids"] + [pad_token_id for _ in range(pad_len)]
            )
            batch_attention_mask.append(
                item["attention_mask"] + [0 for _ in range(pad_len)]
            )
            batch_labels.append(item["labels"] + [-100 for _ in range(pad_len)])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate_fn


def evaluate(model, val_loader, device, amp_dtype, use_amp=False):
    if len(val_loader) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(**inputs)
            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs=4,
    grad_accum_steps=2,
    max_grad_norm=1.0,
    log_every=1,
    output_dir="./lora_adapter",
    device=None,
    amp_dtype=torch.bfloat16,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    init_val_loss = evaluate(
        model, val_loader, device, amp_dtype=amp_dtype, use_amp=use_amp
    )
    print(f"\n[Epoch 0/{epochs}] Initial Val Loss: {init_val_loss:.4f}", flush=True)
    print("-" * 75, flush=True)

    best_val_loss = math.inf
    history = list()
    global_step = 0
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = max(1, steps_per_epoch * epochs)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        step_loss_accum = 0.0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        print(f"\n>>> Starting Epoch {epoch}/{epochs}...", flush=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(**inputs)
                raw_loss = outputs.loss.item()
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += raw_loss
            step_loss_accum += raw_loss
            accum_count += 1

            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_step_loss = step_loss_accum / max(accum_count, 1)
                step_loss_accum = 0.0
                accum_count = 0

                if global_step % log_every == 0 or batch_idx == len(train_loader):
                    current_lr = next(iter(scheduler.get_last_lr()))
                    print(
                        f"  Epoch {epoch}/{epochs} | Step {global_step:02d}/{total_steps:02d} | "
                        f"Step Loss: {avg_step_loss:.4f} | Avg Train Loss: {running_loss / batch_idx:.4f} | "
                        f"LR: {current_lr:.2e}",
                        flush=True,
                    )

        train_epoch_loss = running_loss / max(len(train_loader), 1)
        has_val = len(val_loader.dataset) > 0
        val_epoch_loss = (
            evaluate(model, val_loader, device, amp_dtype=amp_dtype, use_amp=use_amp)
            if has_val
            else None
        )
        epoch_time = time.time() - epoch_start

        # Do dung luong bo nho chinh xac cho ca CUDA va Mac MPS
        if device.type == "cuda":
            mem_str = (
                f"VRAM: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f}GB"
            )
        elif device.type == "mps":
            mem_str = (
                f"MPS Mem: {torch.mps.current_allocated_memory() / (1024**3):.2f}GB"
            )
        else:
            mem_str = "Device: CPU"

        # Chon chi so theo doi: dung val_loss neu co tap val, nguoc lai dung train_loss
        current_metric = val_epoch_loss if has_val else train_epoch_loss
        metric_name = "Val Loss" if has_val else "Train Loss"

        is_best = current_metric <= best_val_loss
        if is_best:
            best_val_loss = current_metric
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        val_str = f"Val Loss: {val_epoch_loss:.4f} | " if has_val else ""
        save_tag = (
            f"  --> [SAVED BEST ({metric_name}: {best_val_loss:.4f})]"
            if is_best
            else ""
        )

        print(
            f"\n[Epoch {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_epoch_loss:.4f} | {val_str}"
            f"LR: {next(iter(scheduler.get_last_lr())):.2e} | "
            f"Time: {epoch_time:.1f}s | {mem_str}{save_tag}\n" + "-" * 85,
            flush=True,
        )

    metrics_file = save_path / "training_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! Metrics saved to {metrics_file}", flush=True)

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for Qwen 2.5."
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--data-path", type=str, default="./sample_data/fine_tune_data_set.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default="./lora_adapter")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )
    model_dtype = (
        torch.bfloat16
        if (use_cuda and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    print(
        f"Device: {device} | Model dtype: {model_dtype} | AMP dtype: {amp_dtype}",
        flush=True,
    )
    print(f"Loading tokenizer & model: {args.model_id}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    full_dataset = InstructionDataset(
        args.data_path, tokenizer=tokenizer, max_length=args.max_length
    )
    total_samples = len(full_dataset)

    if total_samples >= 2 and args.val_ratio > 0:
        val_size = max(
            1, min(total_samples - 1, int(round(total_samples * args.val_ratio)))
        )
        train_size = total_samples - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = full_dataset
        val_dataset = []

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    grad_accum_steps = max(1, args.grad_accum_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=1.0,
        log_every=args.log_every,
        output_dir=args.output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )


if __name__ == "__main__":
    main()
