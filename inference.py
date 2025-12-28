from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from nmt_rnn.data import (
    JsonlTranslationDataset,
    Vocab,
    detokenize_en,
    detokenize_zh,
    make_collate_fn,
    tokenize_en,
    tokenize_zh,
)
from nmt_rnn.decoding import beam_search_decode as rnn_beam_search_decode
from nmt_rnn.decoding import greedy_decode as rnn_greedy_decode
from nmt_rnn.metrics import sacrebleu_bleu
from nmt_rnn.model import Seq2Seq
from nmt_transformer.decoding import beam_search_decode as tfm_beam_search_decode
from nmt_transformer.decoding import greedy_decode as tfm_greedy_decode
from nmt_transformer.model import Seq2SeqTransformer

Arch = Literal["auto", "rnn", "transformer"]


def _vocab_from_itos(itos: List[str]) -> Vocab:
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        unk_id=stoi["<unk>"],
        bos_id=stoi["<bos>"],
        eos_id=stoi["<eos>"],
    )


def _infer_arch(cfg: dict, requested: Arch) -> Arch:
    if requested != "auto":
        return requested
    if "rnn_type" in cfg or "attention_type" in cfg or "teacher_forcing_ratio" in cfg:
        return "rnn"
    if "num_heads" in cfg or "ff_dim" in cfg or "pos_encoding" in cfg:
        return "transformer"
    raise ValueError("Cannot infer --arch from checkpoint config; pass --arch explicitly.")


def _load_checkpoint(path: Path, arch: Arch, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint missing dict field: config")

    arch = _infer_arch(cfg, arch)

    src_vocab_obj = ckpt.get("src_vocab")
    tgt_vocab_obj = ckpt.get("tgt_vocab")
    if not isinstance(src_vocab_obj, dict) or not isinstance(tgt_vocab_obj, dict):
        raise ValueError("Checkpoint missing dict fields: src_vocab/tgt_vocab")
    if "itos" not in src_vocab_obj or "itos" not in tgt_vocab_obj:
        raise ValueError("Checkpoint vocab missing field: itos")

    src_vocab = _vocab_from_itos(list(src_vocab_obj["itos"]))
    tgt_vocab = _vocab_from_itos(list(tgt_vocab_obj["itos"]))

    if arch == "rnn":
        model = Seq2Seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_size=int(cfg["embed_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            rnn_type=str(cfg["rnn_type"]),
            attention_type=str(cfg["attention_type"]),
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
        )
    else:
        eval_max_len = int(cfg.get("eval_max_len", cfg.get("max_tgt_len", 256)))
        max_len = max(int(cfg["max_src_len"]), int(cfg["max_tgt_len"]), eval_max_len) + 8
        model = Seq2SeqTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            src_pad_id=src_vocab.pad_id,
            tgt_pad_id=tgt_vocab.pad_id,
            dim=int(cfg["dim"]),
            num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]),
            ff_dim=int(cfg["ff_dim"]),
            dropout=float(cfg["dropout"]),
            attn_dropout=float(cfg["attn_dropout"]),
            norm_type=str(cfg["norm_type"]),
            pos_encoding=str(cfg["pos_encoding"]),
            max_len=max_len,
            tie_embeddings=bool(cfg.get("tie_embeddings", True)),
        )

    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing dict field: model")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, cfg, arch


@torch.no_grad()
def _evaluate_greedy(
    model,
    arch: Arch,
    loader: DataLoader,
    tgt_vocab: Vocab,
    tgt_key: str,
    device: torch.device,
    max_len: int,
    max_batches: int,
) -> Tuple[float, List[str], List[str]]:
    detok = detokenize_en if tgt_key == "en" else detokenize_zh
    refs: List[str] = []
    hyps: List[str] = []

    batches = 0
    for batch in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        batches += 1
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"]
        if arch == "rnn":
            src_lens = batch["src_lens"].to(device)
            hyp_ids_list = rnn_greedy_decode(
                model=model,
                src_ids=src_ids,
                src_lens=src_lens,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_len,
            )
        else:
            hyp_ids_list = tfm_greedy_decode(
                model=model,
                src_ids=src_ids,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_len,
            )

        for hyp_ids, ref_ids in zip(hyp_ids_list, tgt_ids.tolist()):
            hyp_tokens = tgt_vocab.decode(hyp_ids[1:], stop_at_eos=True)
            ref_tokens = tgt_vocab.decode(ref_ids[1:], stop_at_eos=True)
            hyps.append(detok(hyp_tokens))
            refs.append(detok(ref_tokens))

    bleu = sacrebleu_bleu(references=refs, hypotheses=hyps) if refs else 0.0
    return float(bleu), refs, hyps


@torch.no_grad()
def _evaluate_beam(
    model,
    arch: Arch,
    loader: DataLoader,
    tgt_vocab: Vocab,
    tgt_key: str,
    device: torch.device,
    max_len: int,
    beam_size: int,
    max_batches: int,
) -> Tuple[float, List[str], List[str]]:
    detok = detokenize_en if tgt_key == "en" else detokenize_zh
    refs: List[str] = []
    hyps: List[str] = []

    batches = 0
    for batch in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        batches += 1

        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].tolist()

        if arch == "rnn":
            src_lens = batch["src_lens"].to(device)
            for i in range(src_ids.size(0)):
                results = rnn_beam_search_decode(
                    model=model,
                    src_ids=src_ids[i : i + 1],
                    src_lens=src_lens[i : i + 1],
                    bos_id=tgt_vocab.bos_id,
                    eos_id=tgt_vocab.eos_id,
                    beam_size=beam_size,
                    max_len=max_len,
                )
                best = results[0]
                hyp_tokens = tgt_vocab.decode(best.token_ids[1:], stop_at_eos=True)
                ref_tokens = tgt_vocab.decode(tgt_ids[i][1:], stop_at_eos=True)
                hyps.append(detok(hyp_tokens))
                refs.append(detok(ref_tokens))
        else:
            for i in range(src_ids.size(0)):
                results = tfm_beam_search_decode(
                    model=model,
                    src_ids=src_ids[i : i + 1],
                    bos_id=tgt_vocab.bos_id,
                    eos_id=tgt_vocab.eos_id,
                    beam_size=beam_size,
                    max_len=max_len,
                )
                best = results[0]
                hyp_tokens = tgt_vocab.decode(best.token_ids[1:], stop_at_eos=True)
                ref_tokens = tgt_vocab.decode(tgt_ids[i][1:], stop_at_eos=True)
                hyps.append(detok(hyp_tokens))
                refs.append(detok(ref_tokens))

    bleu = sacrebleu_bleu(references=refs, hypotheses=hyps) if refs else 0.0
    return float(bleu), refs, hyps


def _maybe_limit_batches(num_examples: int, batch_size: int, eval_samples: int) -> int:
    if eval_samples <= 0:
        return 0
    n = min(int(eval_samples), int(num_examples))
    return (n + max(1, int(batch_size)) - 1) // max(1, int(batch_size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt/last.pt")
    parser.add_argument("--arch", type=str, choices=["auto", "rnn", "transformer"], default="auto")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--src_key", type=str, default=None)
    parser.add_argument("--tgt_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--eval_beam", action="store_true")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Write hypotheses to a file")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    model, src_vocab, tgt_vocab, cfg, arch = _load_checkpoint(
        ckpt_path, arch=args.arch, device=device
    )

    data_dir = Path(str(args.data_dir or cfg.get("data_dir", "")))
    test_file = str(args.test_file or cfg.get("test_file", "test.jsonl"))
    test_path = data_dir / test_file

    src_key = str(args.src_key or cfg.get("src_key", "zh"))
    tgt_key = str(args.tgt_key or cfg.get("tgt_key", "en"))

    max_src_len = int(cfg.get("max_src_len", 200))
    max_tgt_len = int(cfg.get("max_tgt_len", 200))
    max_len = int(args.max_len) if args.max_len is not None else int(cfg.get("eval_max_len", 120))

    src_tokenize = tokenize_zh if src_key == "zh" else lambda s: tokenize_en(s, lowercase=True)
    tgt_tokenize = tokenize_en if tgt_key == "en" else tokenize_zh

    ds = JsonlTranslationDataset(
        jsonl_path=test_path,
        src_key=src_key,
        tgt_key=tgt_key,
        src_tokenize=src_tokenize,
        tgt_tokenize=tgt_tokenize,
    )

    collate = make_collate_fn(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    loader = DataLoader(ds, batch_size=max(1, int(args.batch_size)), shuffle=False, collate_fn=collate)
    max_batches = _maybe_limit_batches(len(ds), int(args.batch_size), int(args.eval_samples))

    t0 = time.time()
    greedy_bleu, greedy_refs, greedy_hyps = _evaluate_greedy(
        model=model,
        arch=arch,
        loader=loader,
        tgt_vocab=tgt_vocab,
        tgt_key=tgt_key,
        device=device,
        max_len=max_len,
        max_batches=max_batches,
    )
    greedy_seconds = time.time() - t0

    beam_bleu = None
    beam_seconds = None
    out_hyps = greedy_hyps
    if bool(args.eval_beam) and int(args.beam_size) > 1:
        t1 = time.time()
        beam_bleu, beam_refs, beam_hyps = _evaluate_beam(
            model=model,
            arch=arch,
            loader=loader,
            tgt_vocab=tgt_vocab,
            tgt_key=tgt_key,
            device=device,
            max_len=max_len,
            beam_size=int(args.beam_size),
            max_batches=max_batches,
        )
        beam_seconds = time.time() - t1
        out_hyps = beam_hyps
        greedy_refs = beam_refs

    if args.output is not None:
        Path(args.output).write_text("\n".join(out_hyps) + "\n", encoding="utf-8")

    payload = {
        "ckpt": str(ckpt_path),
        "arch": arch,
        "device": str(device),
        "data_dir": str(data_dir),
        "test_file": str(test_file),
        "num_examples": len(greedy_refs),
        "batch_size": int(args.batch_size),
        "max_len": int(max_len),
        "eval_samples": int(args.eval_samples),
        "greedy_bleu": round(float(greedy_bleu), 2),
        "greedy_seconds": round(float(greedy_seconds), 2),
        "beam_size": int(args.beam_size) if bool(args.eval_beam) else None,
        "beam_bleu": round(float(beam_bleu), 2) if beam_bleu is not None else None,
        "beam_seconds": round(float(beam_seconds), 2) if beam_seconds is not None else None,
        "config": cfg,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

