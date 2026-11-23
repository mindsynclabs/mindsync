import argparse
import json
import math
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)


class InstructionDataset(Dataset):
    """Loads instruction/response pairs from a JSONL file for causal LM fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    self.samples.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc

        if not self.samples:
            raise ValueError(f"Dataset file is empty: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.samples[idx]
        prompt = row.get("prompt") or row.get("instruction") or ""
        response = row.get("response") or row.get("completion") or row.get("output") or ""

        prompt_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        eos_token = self.tokenizer.eos_token or ""
        full_text = f"{prompt_text}{response}{eos_token}"

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
            full_ids = [self.tokenizer.eos_token_id or 0]

        # Mask prompt tokens with -100 so loss is computed only on response tokens
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def make_collate_fn(pad_token_id: int):
    """Pads variable-length sequences in a batch to the longest sample in that batch."""

    def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def evaluate(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Computes average validation loss across all batches."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int = 3,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    output_dir: str | Path = "./lora_adapter",
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Runs LoRA fine-tuning and saves the best adapter weights by validation loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    best_val_loss = math.inf
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch}/{epochs} | Step {global_step} | "
                        f"Train Loss: {running_loss / step:.4f} | LR: {current_lr:.2e}"
                    )

        train_epoch_loss = running_loss / max(len(train_loader), 1)
        val_epoch_loss = evaluate(model, val_loader, device, use_amp=use_amp)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        print(
            f"--> End of Epoch {epoch}/{epochs}: "
            f"Train Loss = {train_epoch_loss:.4f} | Val Loss = {val_epoch_loss:.4f}"
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"    Saved best LoRA adapter to {save_path}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script for causal LMs.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face base model ID.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset.jsonl",
        help="Path to JSONL training file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_adapter",
        help="Directory to save best LoRA adapter and tokenizer.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    full_dataset = InstructionDataset(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if len(full_dataset) > 1 and 0.0 < args.val_split < 1.0:
        val_size = max(1, int(len(full_dataset) * args.val_split))
        val_size = min(val_size, len(full_dataset) - 1)
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = full_dataset
        val_dataset = full_dataset

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

    optimizer = AdamW(model.parameters(), lr=args.lr)
    updates_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_steps = max(updates_per_epoch * args.epochs, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()