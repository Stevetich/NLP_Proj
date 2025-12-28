from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["layernorm", "rmsnorm"]
PosEncodingType = Literal["sinusoidal", "learned", "relative"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


def make_norm(norm_type: NormType, dim: int, eps: float) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[offset : offset + seq_len].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(offset, offset + seq_len, device=x.device)
        return x + self.emb(pos).unsqueeze(0).to(dtype=x.dtype)


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0
        n = -relative_position
        if self.bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = n.abs()
        else:
            n = torch.clamp(n, min=0)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.to(torch.float32) / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n.to(torch.long), val_if_large)
        return ret

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel_pos = k_pos - q_pos
        buckets = self._relative_position_bucket(rel_pos)
        values = self.bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        rel_pos_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = x_q.size()
        k_len = x_kv.size(1)

        q = self.q_proj(x_q).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias.to(dtype=scores.dtype)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float, activation: Literal["relu", "gelu"]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        else:
            x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        norm_type: NormType,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.ff = FeedForward(dim=dim, ff_dim=ff_dim, dropout=dropout, activation="gelu")
        self.norm1 = make_norm(norm_type, dim, norm_eps)
        self.norm2 = make_norm(norm_type, dim, norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(
            x_q=h,
            x_kv=h,
            key_padding_mask=src_key_padding_mask,
            attn_mask=None,
            rel_pos_bias=rel_pos_bias,
        )
        x = x + self.drop(h)
        h2 = self.ff(self.norm2(x))
        x = x + self.drop(h2)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        norm_type: NormType,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.cross_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.ff = FeedForward(dim=dim, ff_dim=ff_dim, dropout=dropout, activation="gelu")
        self.norm1 = make_norm(norm_type, dim, norm_eps)
        self.norm2 = make_norm(norm_type, dim, norm_eps)
        self.norm3 = make_norm(norm_type, dim, norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
        self_rel_pos_bias: Optional[torch.Tensor],
        cross_rel_pos_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(
            x_q=h,
            x_kv=h,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=causal_mask,
            rel_pos_bias=self_rel_pos_bias,
        )
        x = x + self.drop(h)

        h2 = self.norm2(x)
        h2 = self.cross_attn(
            x_q=h2,
            x_kv=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=None,
            rel_pos_bias=cross_rel_pos_bias,
        )
        x = x + self.drop(h2)

        h3 = self.ff(self.norm3(x))
        x = x + self.drop(h3)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        norm_type: NormType,
        norm_eps: float,
        pos_encoding: PosEncodingType,
        max_len: int,
        rel_num_buckets: int,
        rel_max_distance: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.pos_encoding = pos_encoding
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        if pos_encoding == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos = LearnedPositionalEncoding(dim=dim, max_len=max_len)
        else:
            self.pos = None

        if pos_encoding == "relative":
            self.rel_bias = RelativePositionBias(
                num_buckets=rel_num_buckets,
                max_distance=rel_max_distance,
                num_heads=num_heads,
                bidirectional=True,
            )
        else:
            self.rel_bias = None

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = make_norm(norm_type, dim, norm_eps)

    def forward(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src_ids.eq(self.pad_id)
        x = self.embed(src_ids) * math.sqrt(self.dim)
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop(x)

        rel = None
        if self.rel_bias is not None:
            rel = self.rel_bias(q_len=x.size(1), k_len=x.size(1), device=x.device)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, rel_pos_bias=rel)
        x = self.final_norm(x)
        return x, src_key_padding_mask


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        attn_dropout: float,
        norm_type: NormType,
        norm_eps: float,
        pos_encoding: PosEncodingType,
        max_len: int,
        rel_num_buckets: int,
        rel_max_distance: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.pos_encoding = pos_encoding
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        if pos_encoding == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos = LearnedPositionalEncoding(dim=dim, max_len=max_len)
        else:
            self.pos = None

        if pos_encoding == "relative":
            self.self_rel_bias = RelativePositionBias(
                num_buckets=rel_num_buckets,
                max_distance=rel_max_distance,
                num_heads=num_heads,
                bidirectional=False,
            )
            self.cross_rel_bias = RelativePositionBias(
                num_buckets=rel_num_buckets,
                max_distance=rel_max_distance,
                num_heads=num_heads,
                bidirectional=True,
            )
        else:
            self.self_rel_bias = None
            self.cross_rel_bias = None

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = make_norm(norm_type, dim, norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def _causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)[
            None, None, :, :
        ]

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_key_padding_mask = tgt_ids.eq(self.pad_id)
        x = self.embed(tgt_ids) * math.sqrt(self.dim)
        if self.pos is not None:
            x = self.pos(x)
        x = self.drop(x)

        causal_mask = self._causal_mask(tgt_len=x.size(1), device=x.device)

        self_rel = None
        cross_rel = None
        if self.self_rel_bias is not None:
            self_rel = self.self_rel_bias(q_len=x.size(1), k_len=x.size(1), device=x.device)
        if self.cross_rel_bias is not None:
            cross_rel = self.cross_rel_bias(q_len=x.size(1), k_len=memory.size(1), device=x.device)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                causal_mask=causal_mask,
                self_rel_pos_bias=self_rel,
                cross_rel_pos_bias=cross_rel,
            )
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


@dataclass
class Seq2SeqTransformerOutput:
    logits: torch.Tensor


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        pos_encoding: PosEncodingType = "sinusoidal",
        max_len: int = 4096,
        rel_num_buckets: int = 32,
        rel_max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            pad_id=src_pad_id,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            pos_encoding=pos_encoding,
            max_len=max_len,
            rel_num_buckets=rel_num_buckets,
            rel_max_distance=rel_max_distance,
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            pad_id=tgt_pad_id,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            pos_encoding=pos_encoding,
            max_len=max_len,
            rel_num_buckets=rel_num_buckets,
            rel_max_distance=rel_max_distance,
        )

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> Seq2SeqTransformerOutput:
        memory, mem_key_padding = self.encoder(src_ids)
        logits = self.decoder(tgt_ids[:, :-1], memory=memory, memory_key_padding_mask=mem_key_padding)
        return Seq2SeqTransformerOutput(logits=logits)

    @torch.no_grad()
    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(src_ids)

    @torch.no_grad()
    def decode_logits_last(
        self, tgt_prefix_ids: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.decoder(tgt_prefix_ids, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        return logits[:, -1, :]
