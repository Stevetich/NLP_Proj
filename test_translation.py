from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import torch

from nmt_rnn.data import Vocab, detokenize_en, detokenize_zh, tokenize_en, tokenize_zh
from nmt_rnn.decoding import beam_search_decode as rnn_beam_search_decode
from nmt_rnn.decoding import greedy_decode as rnn_greedy_decode
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
        )

    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing dict field: model")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, cfg, arch


def _iter_lines(text: str | None, input_path: str | None) -> Iterable[str]:
    if text is not None:
        for line in text.splitlines():
            yield line
        return
    if input_path is not None:
        with Path(input_path).open("r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
        return
    for line in sys.stdin:
        yield line.rstrip("\n")


@torch.no_grad()
def _translate_one(
    model,
    arch: Arch,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    cfg: dict,
    text: str,
    device: torch.device,
    max_len: int,
    beam_size: int,
    n_best: int,
) -> Tuple[List[str], List[float]]:
    src_key = str(cfg.get("src_key", "zh"))
    tgt_key = str(cfg.get("tgt_key", "en"))
    src_tokens = tokenize_zh(text) if src_key == "zh" else tokenize_en(text, lowercase=True)
    src_ids = [src_vocab.bos_id] + src_vocab.encode(src_tokens) + [src_vocab.eos_id]

    if arch == "rnn":
        src_ids_t = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_lens_t = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        if beam_size > 1 or n_best > 1:
            results = rnn_beam_search_decode(
                model=model,
                src_ids=src_ids_t,
                src_lens=src_lens_t,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                beam_size=max(beam_size, n_best),
                max_len=max_len,
            )
            picked = results[: max(1, n_best)]
            hyp_ids_list = [r.token_ids for r in picked]
            scores = [float(r.score) for r in picked]
        else:
            hyp_ids_list = rnn_greedy_decode(
                model=model,
                src_ids=src_ids_t,
                src_lens=src_lens_t,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_len,
            )
            hyp_ids_list = [hyp_ids_list[0]]
            scores = [0.0]
    else:
        src_ids_t = torch.tensor([src_ids], dtype=torch.long, device=device)
        if beam_size > 1 or n_best > 1:
            results = tfm_beam_search_decode(
                model=model,
                src_ids=src_ids_t,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                beam_size=max(beam_size, n_best),
                max_len=max_len,
            )
            picked = results[: max(1, n_best)]
            hyp_ids_list = [r.token_ids for r in picked]
            scores = [float(r.score) for r in picked]
        else:
            hyp_ids_list = tfm_greedy_decode(
                model=model,
                src_ids=src_ids_t,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_len,
            )
            hyp_ids_list = [hyp_ids_list[0]]
            scores = [0.0]

    detok = detokenize_en if tgt_key == "en" else detokenize_zh
    texts: List[str] = []
    for hyp_ids in hyp_ids_list:
        hyp_tokens = tgt_vocab.decode(hyp_ids[1:], stop_at_eos=True)
        texts.append(detok(hyp_tokens))
    return texts, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt/last.pt")
    parser.add_argument("--arch", type=str, choices=["auto", "rnn", "transformer"], default="auto")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Read one sentence per line")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--n_best", type=int, default=1)
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

    max_len = int(args.max_len) if args.max_len is not None else int(cfg.get("eval_max_len", 120))
    beam_size = max(1, int(args.beam_size))
    n_best = max(1, int(args.n_best))

    for line in _iter_lines(args.text, args.input):
        if not line.strip():
            print("")
            continue
        texts, scores = _translate_one(
            model=model,
            arch=arch,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            cfg=cfg,
            text=line,
            device=device,
            max_len=max_len,
            beam_size=beam_size,
            n_best=n_best,
        )
        if n_best == 1:
            print(texts[0])
        else:
            print(json.dumps({"input": line, "hyps": texts, "scores": scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()

