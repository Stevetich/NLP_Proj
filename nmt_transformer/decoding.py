from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .model import Seq2SeqTransformer


@dataclass
class DecodeResult:
    token_ids: List[int]
    score: float


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 200,
) -> List[List[int]]:
    model.eval()
    memory, mem_key_padding = model.encode(src_ids)
    device = src_ids.device
    bsz = src_ids.size(0)

    cur = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits_last = model.decode_logits_last(cur, memory=memory, memory_key_padding_mask=mem_key_padding)
        next_ids = logits_last.argmax(dim=1)
        next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
        cur = torch.cat([cur, next_ids.unsqueeze(1)], dim=1)
        finished |= next_ids.eq(eos_id)
        if bool(finished.all()):
            break

    out: List[List[int]] = []
    ids_list = cur.tolist()
    for ids in ids_list:
        if eos_id in ids:
            j = ids.index(eos_id)
            out.append(ids[: j + 1])
        else:
            out.append(ids)
    return out


@torch.no_grad()
def beam_search_decode(
    model: Seq2SeqTransformer,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 4,
    max_len: int = 200,
    length_penalty_alpha: float = 0.6,
) -> List[DecodeResult]:
    if src_ids.size(0) != 1:
        raise ValueError("beam_search_decode only supports batch_size=1")

    model.eval()
    memory, mem_key_padding = model.encode(src_ids)
    device = src_ids.device

    beams = [([bos_id], 0.0, False)]

    def lp(length: int) -> float:
        if length_penalty_alpha <= 0:
            return 1.0
        return ((5.0 + length) / 6.0) ** length_penalty_alpha

    for _ in range(max_len):
        candidates = []
        for ids, score, done in beams:
            if done:
                candidates.append((ids, score, done))
                continue
            cur = torch.tensor([ids], dtype=torch.long, device=device)
            logits_last = model.decode_logits_last(cur, memory=memory, memory_key_padding_mask=mem_key_padding)
            log_probs = F.log_softmax(logits_last, dim=1).squeeze(0)
            top_log_probs, top_ids = torch.topk(log_probs, k=beam_size)
            for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_ids = ids + [int(tok_id)]
                new_done = tok_id == eos_id
                candidates.append((new_ids, score + float(log_p), new_done))

        candidates.sort(key=lambda x: x[1] / lp(len(x[0])), reverse=True)
        beams = candidates[:beam_size]
        if all(d for _ids, _s, d in beams):
            break

    results = [DecodeResult(token_ids=ids, score=score / lp(len(ids))) for ids, score, _ in beams]
    results.sort(key=lambda r: r.score, reverse=True)
    return results
