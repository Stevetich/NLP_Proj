from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nmt_rnn.metrics import corpus_bleu
from nmt_rnn.data import tokenize_en


class JsonlSeq2SeqTextDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, src_key: str, tgt_key: str) -> None:
        self.path = Path(jsonl_path)
        self.src_key = src_key
        self.tgt_key = tgt_key
        self._pairs: List[Tuple[str, str]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._pairs.append((obj[src_key], obj[tgt_key]))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        src, tgt = self._pairs[idx]
        return {"src": src, "tgt": tgt}


@dataclass
class TrainConfig:
    data_dir: str
    train_file: str
    valid_file: str
    test_file: str
    src_key: str
    tgt_key: str
    model_name_or_path: str
    source_prefix: str
    batch_size: int
    max_src_len: int
    max_tgt_len: int
    lr: float
    weight_decay: float
    grad_clip: float
    epochs: int
    max_train_steps: int
    eval_samples: int
    max_eval_batches: int
    gen_max_new_tokens: int
    eval_beam: bool
    beam_size: int
    eval_test: bool
    fp16: bool
    seed: int
    save_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_grad_norm(parameters, max_norm: float) -> None:
    try:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, foreach=False)
    except TypeError:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _tokenize_targets(tokenizer, texts: List[str], max_len: int):
    if hasattr(tokenizer, "__call__") and "text_target" in tokenizer.__call__.__code__.co_varnames:
        return tokenizer(
            text_target=texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
    if hasattr(tokenizer, "as_target_tokenizer"):
        with tokenizer.as_target_tokenizer():
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def make_collate_fn(tokenizer, cfg: TrainConfig):
    def _collate(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        src_texts = [cfg.source_prefix + ex["src"] for ex in batch]
        tgt_texts = [ex["tgt"] for ex in batch]
        src = tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_src_len,
            return_tensors="pt",
        )
        tgt = _tokenize_targets(tokenizer, tgt_texts, max_len=cfg.max_tgt_len)
        labels = tgt["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": labels,
            "tgt_texts": tgt_texts,
        }

    return _collate


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    refs: List[List[str]] = []
    hyps: List[List[str]] = []

    seen = 0
    batches = 0
    for batch in loader:
        if cfg.max_eval_batches > 0 and batches >= cfg.max_eval_batches:
            break
        if cfg.eval_samples > 0 and seen >= cfg.eval_samples:
            break
        batch = batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += float(outputs.loss.item())
        total_batches += 1
        batches += 1

        num_beams = cfg.beam_size if cfg.eval_beam else 1
        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=cfg.gen_max_new_tokens,
            num_beams=num_beams,
        )
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for hyp, ref in zip(gen_texts, batch["tgt_texts"]):
            if cfg.eval_samples > 0 and seen >= cfg.eval_samples:
                break
            hyps.append(tokenize_en(hyp, lowercase=True))
            refs.append(tokenize_en(ref, lowercase=True))
            seen += 1

    bleu = corpus_bleu(references=refs, hypotheses=hyps) if refs else 0.0
    loss = total_loss / max(1, total_batches)
    return loss, bleu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--train_file", type=str, default="train_10k.jsonl")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl")
    parser.add_argument("--test_file", type=str, default="test.jsonl")
    parser.add_argument("--src_key", type=str, default="zh")
    parser.add_argument("--tgt_key", type=str, default="en")
    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--source_prefix", type=str, default="translate Chinese to English: ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_src_len", type=int, default=200)
    parser.add_argument("--max_tgt_len", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--gen_max_new_tokens", type=int, default=120)
    parser.add_argument("--eval_beam", action="store_true")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--eval_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/finetune")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        src_key=args.src_key,
        tgt_key=args.tgt_key,
        model_name_or_path=args.model_name_or_path,
        source_prefix=args.source_prefix,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        eval_samples=args.eval_samples,
        max_eval_batches=args.max_eval_batches,
        gen_max_new_tokens=args.gen_max_new_tokens,
        eval_beam=bool(args.eval_beam),
        beam_size=args.beam_size,
        eval_test=bool(args.eval_test),
        fp16=bool(args.fp16),
        seed=args.seed,
        save_dir=args.save_dir,
    )

    set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / cfg.train_file
    valid_path = data_dir / cfg.valid_file
    test_path = data_dir / cfg.test_file

    train_ds = JsonlSeq2SeqTextDataset(train_path, src_key=cfg.src_key, tgt_key=cfg.tgt_key)
    valid_ds = JsonlSeq2SeqTextDataset(valid_path, src_key=cfg.src_key, tgt_key=cfg.tgt_key)
    test_ds = JsonlSeq2SeqTextDataset(test_path, src_key=cfg.src_key, tgt_key=cfg.tgt_key) if cfg.eval_test else None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate = make_collate_fn(tokenizer, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = (
        DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        if test_ds is not None
        else None
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.fp16 and device.type == "cuda"))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_dir / "config.json", asdict(cfg))

    best_valid_bleu = float("-inf")
    best_epoch = 0
    best_snapshot: Dict[str, float | int] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            if cfg.max_train_steps > 0 and steps >= cfg.max_train_steps:
                break
            batch = batch_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.fp16 and device.type == "cuda")):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            total_loss += float(loss.item())
            steps += 1

        train_loss = total_loss / max(1, steps)
        valid_loss, valid_bleu = evaluate(model, tokenizer, valid_loader, device, cfg)
        test_loss = 0.0
        test_bleu = 0.0
        if test_loader is not None:
            test_loss, test_bleu = evaluate(model, tokenizer, test_loader, device, cfg)

        is_best = valid_bleu > best_valid_bleu
        if is_best:
            best_valid_bleu = valid_bleu
            best_epoch = epoch
            best_snapshot = {
                "epoch": epoch,
                "valid_bleu": float(valid_bleu),
                "test_bleu": float(test_bleu),
            }

        elapsed = time.time() - t0
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "valid_loss": round(valid_loss, 4),
                    "valid_bleu": round(valid_bleu, 2),
                    "test_loss": round(test_loss, 4) if cfg.eval_test else None,
                    "test_bleu": round(test_bleu, 2) if cfg.eval_test else None,
                    "best_epoch": best_epoch,
                    "best_valid_bleu": round(best_valid_bleu, 2),
                    "seconds": round(elapsed, 1),
                    "device": str(device),
                    "model": cfg.model_name_or_path,
                    "eval_beam": cfg.eval_beam,
                    "beam_size": cfg.beam_size,
                },
                ensure_ascii=False,
            )
        )

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "tokenizer_name_or_path": cfg.model_name_or_path,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")
            save_json(save_dir / "best.json", {"best": best_snapshot})


if __name__ == "__main__":
    main()
