from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import Seq2Seq


@dataclass
class DecodeResult:
    token_ids: List[int]
    score: float


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src_ids: torch.Tensor,
    src_lens: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 200,
) -> List[List[int]]:
    model.eval()
    enc_out, hidden, context, src_mask = model.encode(src_ids, src_lens)

    batch_size = src_ids.size(0)
    input_ids = torch.full((batch_size,), bos_id, dtype=torch.long, device=src_ids.device)
    sequences: List[List[int]] = [[bos_id] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)

    for _ in range(max_len):
        logits, hidden, context, _attn = model.decoder.step(
            input_ids=input_ids,
            hidden=hidden,
            prev_context=context,
            enc_outputs=enc_out,
            src_mask=src_mask,
        )
        next_ids = logits.argmax(dim=1)
        input_ids = next_ids
        for i in range(batch_size):
            if not finished[i]:
                sequences[i].append(int(next_ids[i].item()))
        finished |= next_ids.eq(eos_id)
        if bool(finished.all()):
            break
    return sequences


@dataclass
class _Beam:
    token_ids: List[int]
    score: float
    hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    context: torch.Tensor
    finished: bool


@torch.no_grad()
def beam_search_decode(
    model: Seq2Seq,
    src_ids: torch.Tensor,
    src_lens: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 4,
    max_len: int = 200,
    length_penalty_alpha: float = 0.6,
) -> List[DecodeResult]:
    if src_ids.size(0) != 1:
        raise ValueError("beam_search_decode only supports batch_size=1")

    model.eval()
    enc_out, hidden0, context0, src_mask = model.encode(src_ids, src_lens)

    beams: List[_Beam] = [
        _Beam(
            token_ids=[bos_id],
            score=0.0,
            hidden=hidden0,
            context=context0,
            finished=False,
        )
    ]

    def lp(length: int) -> float:
        if length_penalty_alpha <= 0:
            return 1.0
        return ((5.0 + length) / 6.0) ** length_penalty_alpha

    for _step in range(max_len):
        candidates: List[_Beam] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            input_ids = torch.tensor([beam.token_ids[-1]], device=src_ids.device, dtype=torch.long)
            logits, hidden1, context1, _attn = model.decoder.step(
                input_ids=input_ids,
                hidden=beam.hidden,
                prev_context=beam.context,
                enc_outputs=enc_out,
                src_mask=src_mask,
            )
            log_probs = F.log_softmax(logits, dim=1).squeeze(0)
            top_log_probs, top_ids = torch.topk(log_probs, k=beam_size)

            for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_ids = beam.token_ids + [int(tok_id)]
                finished = tok_id == eos_id
                candidates.append(
                    _Beam(
                        token_ids=new_ids,
                        score=beam.score + float(log_p),
                        hidden=hidden1,
                        context=context1,
                        finished=finished,
                    )
                )

        candidates.sort(key=lambda b: b.score / lp(len(b.token_ids)), reverse=True)
        beams = candidates[:beam_size]
        if all(b.finished for b in beams):
            break

    results = [
        DecodeResult(token_ids=b.token_ids, score=b.score / lp(len(b.token_ids))) for b in beams
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results

