from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


def tokenize_zh(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    try:
        import jieba
    except Exception as e:
        raise RuntimeError("jieba is not available. Install it to tokenize Chinese.") from e
    return [tok for tok in jieba.lcut(text, cut_all=False) if tok and not tok.isspace()]


_EN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def tokenize_en(text: str, lowercase: bool = True) -> List[str]:
    text = text.strip()
    if lowercase:
        text = text.lower()
    return _EN_PATTERN.findall(text)


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        out: List[str] = []
        for idx in ids:
            if stop_at_eos and idx == self.eos_id:
                break
            if 0 <= idx < len(self.itos):
                out.append(self.itos[idx])
            else:
                out.append(SPECIAL_TOKENS["unk"])
        return out

    def __len__(self) -> int:
        return len(self.itos)


def build_vocab(
    tokenized_texts: Iterable[Sequence[str]],
    max_size: int = 50000,
    min_freq: int = 2,
) -> Vocab:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    specials = [
        SPECIAL_TOKENS["pad"],
        SPECIAL_TOKENS["unk"],
        SPECIAL_TOKENS["bos"],
        SPECIAL_TOKENS["eos"],
    ]

    words_and_freq = [
        (w, f) for w, f in counter.items() if f >= min_freq and w not in specials
    ]
    words_and_freq.sort(key=lambda x: (-x[1], x[0]))
    words = [w for w, _ in words_and_freq[: max(0, max_size - len(specials))]]

    itos = specials + words
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi[SPECIAL_TOKENS["pad"]],
        unk_id=stoi[SPECIAL_TOKENS["unk"]],
        bos_id=stoi[SPECIAL_TOKENS["bos"]],
        eos_id=stoi[SPECIAL_TOKENS["eos"]],
    )


class JsonlTranslationDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        src_key: str,
        tgt_key: str,
        src_tokenize: Callable[[str], List[str]],
        tgt_tokenize: Callable[[str], List[str]],
    ) -> None:
        self.path = Path(jsonl_path)
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = tgt_tokenize

        self._examples: List[Tuple[List[str], List[str]]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                src = self.src_tokenize(obj[src_key])
                tgt = self.tgt_tokenize(obj[tgt_key])
                self._examples.append((src, tgt))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        src, tgt = self._examples[idx]
        return {"src_tokens": src, "tgt_tokens": tgt}


def make_collate_fn(
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_src_len: int | None = None,
    max_tgt_len: int | None = None,
) -> Callable[[List[Dict[str, List[str]]]], Dict[str, torch.Tensor]]:
    def _clip(tokens: List[str], max_len: int | None) -> List[str]:
        if max_len is None:
            return tokens
        return tokens[:max_len]

    def _collate(batch: List[Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        src_ids_list: List[List[int]] = []
        tgt_ids_list: List[List[int]] = []
        src_lens: List[int] = []
        tgt_lens: List[int] = []

        for ex in batch:
            src_tokens = _clip(ex["src_tokens"], max_src_len)
            tgt_tokens = _clip(ex["tgt_tokens"], max_tgt_len)

            src_ids = [src_vocab.bos_id] + src_vocab.encode(src_tokens) + [src_vocab.eos_id]
            tgt_ids = [tgt_vocab.bos_id] + tgt_vocab.encode(tgt_tokens) + [tgt_vocab.eos_id]

            src_ids_list.append(src_ids)
            tgt_ids_list.append(tgt_ids)
            src_lens.append(len(src_ids))
            tgt_lens.append(len(tgt_ids))

        max_src = max(src_lens) if src_lens else 0
        max_tgt = max(tgt_lens) if tgt_lens else 0

        src_batch = torch.full((len(batch), max_src), src_vocab.pad_id, dtype=torch.long)
        tgt_batch = torch.full((len(batch), max_tgt), tgt_vocab.pad_id, dtype=torch.long)

        for i, (s, t) in enumerate(zip(src_ids_list, tgt_ids_list)):
            src_batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_batch[i, : len(t)] = torch.tensor(t, dtype=torch.long)

        return {
            "src_ids": src_batch,
            "tgt_ids": tgt_batch,
            "src_lens": torch.tensor(src_lens, dtype=torch.long),
            "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
        }

    return _collate


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def infer_src_tgt_keys(jsonl_path: str | Path) -> Tuple[str, str]:
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            keys = set(obj.keys())
            if {"zh", "en"}.issubset(keys):
                return "zh", "en"
            if {"en", "zh"}.issubset(keys):
                return "en", "zh"
    raise ValueError(f"Cannot infer src/tgt keys from: {path}")


def estimate_num_batches(num_examples: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 0
    return int(math.ceil(num_examples / batch_size))
