from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nmt_rnn.data import (
    JsonlTranslationDataset,
    Vocab,
    batch_to_device,
    build_vocab,
    make_collate_fn,
    tokenize_en,
    tokenize_zh,
)
from nmt_rnn.decoding import beam_search_decode, greedy_decode
from nmt_rnn.metrics import sacrebleu_bleu
from nmt_rnn.model import AttentionType, RnnType, Seq2Seq


@dataclass
class TrainConfig:
    data_dir: str
    train_file: str
    valid_file: str
    test_file: str
    src_key: str
    tgt_key: str
    rnn_type: RnnType
    attention_type: AttentionType
    embed_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    max_src_len: int
    max_tgt_len: int
    max_vocab: int
    min_freq: int
    lr: float
    grad_clip: float
    epochs: int
    teacher_forcing_ratio: float
    eval_max_len: int
    eval_samples: int
    beam_size: int
    eval_beam: bool
    eval_test: bool
    seed: int
    save_dir: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_vocab(path: Path, vocab: Vocab) -> None:
    save_json(path, {"itos": vocab.itos})


def load_vocab(path: Path) -> Vocab:
    obj = json.loads(path.read_text(encoding="utf-8"))
    itos = obj["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        unk_id=stoi["<unk>"],
        bos_id=stoi["<bos>"],
        eos_id=stoi["<eos>"],
    )


def compute_loss(
    logits: torch.Tensor, tgt_ids: torch.Tensor, pad_id: int, criterion: nn.Module
) -> torch.Tensor:
    gold = tgt_ids[:, 1:].contiguous()
    logits = logits.contiguous()
    loss = criterion(logits.view(-1, logits.size(-1)), gold.view(-1))
    return loss


@torch.no_grad()
def evaluate_bleu(
    model: Seq2Seq,
    loader: DataLoader,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    max_samples: int,
    beam_size: int,
    do_beam: bool,
) -> Tuple[float, float]:
    model.eval()
    greedy_hyps: List[str] = []
    beam_hyps: List[str] = []
    refs: List[str] = []

    seen = 0
    for batch in loader:
        if max_samples > 0 and seen >= max_samples:
            break
        batch = batch_to_device(batch, device)
        src_ids, src_lens, tgt_ids = batch["src_ids"], batch["src_lens"], batch["tgt_ids"]
        greedy_ids = greedy_decode(
            model=model,
            src_ids=src_ids,
            src_lens=src_lens,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=max_len,
        )
        for i, (hyp_ids, ref_ids) in enumerate(zip(greedy_ids, tgt_ids.tolist())):
            if max_samples > 0 and seen >= max_samples:
                break
            hyp_tokens = tgt_vocab.decode(hyp_ids[1:], stop_at_eos=True)
            ref_tokens = tgt_vocab.decode(ref_ids[1:], stop_at_eos=True)
            greedy_hyps.append(" ".join(hyp_tokens))
            refs.append(" ".join(ref_tokens))
            if do_beam:
                one_src = src_ids[i : i + 1]
                one_len = src_lens[i : i + 1]
                beam = beam_search_decode(
                    model=model,
                    src_ids=one_src,
                    src_lens=one_len,
                    bos_id=tgt_vocab.bos_id,
                    eos_id=tgt_vocab.eos_id,
                    beam_size=beam_size,
                    max_len=max_len,
                )[0]
                beam_tokens = tgt_vocab.decode(beam.token_ids[1:], stop_at_eos=True)
                beam_hyps.append(" ".join(beam_tokens))
            seen += 1

    greedy_bleu = sacrebleu_bleu(references=refs, hypotheses=greedy_hyps)
    beam_bleu = sacrebleu_bleu(references=refs, hypotheses=beam_hyps) if do_beam else 0.0
    return greedy_bleu, beam_bleu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--train_file", type=str, default="train_10k.jsonl")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl")
    parser.add_argument("--test_file", type=str, default="test.jsonl")
    parser.add_argument("--src_key", type=str, default="zh")
    parser.add_argument("--tgt_key", type=str, default="en")
    parser.add_argument("--rnn_type", type=str, choices=["gru", "lstm"], default="gru")
    parser.add_argument("--attention_type", type=str, choices=["dot", "general", "additive"], default="additive")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_src_len", type=int, default=200)
    parser.add_argument("--max_tgt_len", type=int, default=200)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    parser.add_argument("--eval_max_len", type=int, default=120)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--eval_beam", action="store_true")
    parser.add_argument("--eval_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rnn")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        src_key=args.src_key,
        tgt_key=args.tgt_key,
        rnn_type=args.rnn_type,
        attention_type=args.attention_type,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        lr=args.lr,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        eval_max_len=args.eval_max_len,
        eval_samples=args.eval_samples,
        beam_size=args.beam_size,
        eval_beam=bool(args.eval_beam),
        eval_test=bool(args.eval_test),
        seed=args.seed,
        save_dir=args.save_dir,
    )

    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / cfg.train_file
    valid_path = data_dir / cfg.valid_file
    test_path = data_dir / cfg.test_file

    src_tokenize = tokenize_zh if cfg.src_key == "zh" else lambda s: tokenize_en(s, lowercase=True)
    tgt_tokenize = tokenize_en if cfg.tgt_key == "en" else tokenize_zh

    train_ds = JsonlTranslationDataset(
        jsonl_path=train_path,
        src_key=cfg.src_key,
        tgt_key=cfg.tgt_key,
        src_tokenize=src_tokenize,
        tgt_tokenize=tgt_tokenize,
    )
    valid_ds = JsonlTranslationDataset(
        jsonl_path=valid_path,
        src_key=cfg.src_key,
        tgt_key=cfg.tgt_key,
        src_tokenize=src_tokenize,
        tgt_tokenize=tgt_tokenize,
    )
    test_ds = (
        JsonlTranslationDataset(
            jsonl_path=test_path,
            src_key=cfg.src_key,
            tgt_key=cfg.tgt_key,
            src_tokenize=src_tokenize,
            tgt_tokenize=tgt_tokenize,
        )
        if cfg.eval_test
        else None
    )

    src_vocab = build_vocab(
        (ex["src_tokens"] for ex in train_ds),
        max_size=cfg.max_vocab,
        min_freq=cfg.min_freq,
    )
    tgt_vocab = build_vocab(
        (ex["tgt_tokens"] for ex in train_ds),
        max_size=cfg.max_vocab,
        min_freq=cfg.min_freq,
    )

    collate = make_collate_fn(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = (
        DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        if test_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.rnn_type,
        attention_type=cfg.attention_type,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_dir / "config.json", asdict(cfg))
    save_vocab(save_dir / "src_vocab.json", src_vocab)
    save_vocab(save_dir / "tgt_vocab.json", tgt_vocab)

    best_valid_metric = float("-inf")
    best_epoch = 0
    best_snapshot: Dict[str, float | int] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            out = model(
                src_ids=batch["src_ids"],
                src_lens=batch["src_lens"],
                tgt_ids=batch["tgt_ids"],
                teacher_forcing_ratio=cfg.teacher_forcing_ratio,
            )
            loss = compute_loss(out.logits, batch["tgt_ids"], tgt_vocab.pad_id, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            with torch.no_grad():
                gold = batch["tgt_ids"][:, 1:]
                non_pad = gold.ne(tgt_vocab.pad_id).sum().item()
                total_tokens += int(non_pad)
                total_loss += float(loss.item()) * max(1, int(non_pad))

        train_loss = total_loss / max(1, total_tokens)

        model.eval()
        valid_total_loss = 0.0
        valid_total_tokens = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch_to_device(batch, device)
                out = model(
                    src_ids=batch["src_ids"],
                    src_lens=batch["src_lens"],
                    tgt_ids=batch["tgt_ids"],
                    teacher_forcing_ratio=1.0,
                )
                loss = compute_loss(out.logits, batch["tgt_ids"], tgt_vocab.pad_id, criterion)
                gold = batch["tgt_ids"][:, 1:]
                non_pad = gold.ne(tgt_vocab.pad_id).sum().item()
                valid_total_tokens += int(non_pad)
                valid_total_loss += float(loss.item()) * max(1, int(non_pad))

        valid_loss = valid_total_loss / max(1, valid_total_tokens)

        greedy_bleu, beam_bleu = evaluate_bleu(
            model=model,
            loader=valid_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            max_len=cfg.eval_max_len,
            max_samples=cfg.eval_samples,
            beam_size=cfg.beam_size,
            do_beam=cfg.eval_beam,
        )
        test_greedy_bleu = 0.0
        test_beam_bleu = 0.0
        if test_loader is not None:
            test_greedy_bleu, test_beam_bleu = evaluate_bleu(
                model=model,
                loader=test_loader,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                device=device,
                max_len=cfg.eval_max_len,
                max_samples=cfg.eval_samples,
                beam_size=cfg.beam_size,
                do_beam=cfg.eval_beam,
            )

        valid_metric = beam_bleu if cfg.eval_beam else greedy_bleu
        test_metric = test_beam_bleu if cfg.eval_beam else test_greedy_bleu
        is_best = valid_metric > best_valid_metric
        if is_best:
            best_valid_metric = valid_metric
            best_epoch = epoch
            best_snapshot = {
                "epoch": epoch,
                "valid_metric": float(valid_metric),
                "test_metric": float(test_metric),
            }

        elapsed = time.time() - t0
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "valid_loss": round(valid_loss, 4),
                    "valid_greedy_bleu": round(greedy_bleu, 2),
                    "valid_beam_bleu": round(beam_bleu, 2),
                    "test_greedy_bleu": round(test_greedy_bleu, 2) if cfg.eval_test else None,
                    "test_beam_bleu": round(test_beam_bleu, 2) if cfg.eval_test else None,
                    "best_epoch": best_epoch,
                    "best_valid_metric": round(best_valid_metric, 2),
                    "seconds": round(elapsed, 1),
                    "device": str(device),
                    "teacher_forcing_ratio": cfg.teacher_forcing_ratio,
                    "rnn_type": cfg.rnn_type,
                    "attention_type": cfg.attention_type,
                },
                ensure_ascii=False,
            )
        )

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "src_vocab": {"itos": src_vocab.itos},
            "tgt_vocab": {"itos": tgt_vocab.itos},
        }
        torch.save(ckpt, save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")
            save_json(save_dir / "best.json", {"best": best_snapshot})


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
