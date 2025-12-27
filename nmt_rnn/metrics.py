from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence, Tuple


def _ngrams(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    references: Iterable[Sequence[str]],
    hypotheses: Iterable[Sequence[str]],
    max_n: int = 4,
) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError("references and hypotheses must have same length")
    if not refs:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len_total = 0
    hyp_len_total = 0

    for ref, hyp in zip(refs, hyps):
        ref_len_total += len(ref)
        hyp_len_total += len(hyp)
        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref, n)
            hyp_ngrams = _ngrams(hyp, n)
            total_counts[n - 1] += sum(hyp_ngrams.values())
            for ng, c in hyp_ngrams.items():
                clipped_counts[n - 1] += min(c, ref_ngrams.get(ng, 0))

    if ref_len_total <= 0:
        return 0.0

    bp = min(1.0, hyp_len_total / ref_len_total) if hyp_len_total > 0 else 0.0

    bleu = bp
    for n in range(max_n):
        if total_counts[n] <= 0:
            return 0.0
        bleu *= clipped_counts[n] / total_counts[n]

    return 100.0 * bleu
