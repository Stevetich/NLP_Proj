from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


RnnType = Literal["gru", "lstm"]
AttentionType = Literal["dot", "general", "additive"]


def _make_rnn(
    rnn_type: RnnType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if rnn_type == "gru":
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
    if rnn_type == "lstm":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
    raise ValueError(f"Unsupported rnn_type: {rnn_type}")


def _select_last_layer_hidden(
    rnn_type: RnnType, hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    if rnn_type == "gru":
        return hidden[-1]
    h, _c = hidden
    return h[-1]


def _copy_encoder_state(
    rnn_type: RnnType, enc_state: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    if rnn_type == "gru":
        return enc_state
    h, c = enc_state
    return h, c


class Attention(nn.Module):
    def __init__(self, attn_type: AttentionType, hidden_size: int) -> None:
        super().__init__()
        self.attn_type = attn_type
        self.hidden_size = hidden_size

        if attn_type == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attn_type == "additive":
            self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        enc_outputs: torch.Tensor,
        dec_state_last: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_type == "dot":
            scores = torch.bmm(enc_outputs, dec_state_last.unsqueeze(2)).squeeze(2)
        elif self.attn_type == "general":
            enc_proj = self.W(enc_outputs)
            scores = torch.bmm(enc_proj, dec_state_last.unsqueeze(2)).squeeze(2)
        elif self.attn_type == "additive":
            enc_proj = self.W_enc(enc_outputs)
            dec_proj = self.W_dec(dec_state_last).unsqueeze(1)
            scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(2)
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        rnn_type: RnnType,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.pad_id = pad_id
        self.embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn = _make_rnn(
            rnn_type=rnn_type,
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self, src_ids: torch.Tensor, src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        emb = self.dropout(self.embedding(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        outputs, _lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        src_mask = src_ids[:, : outputs.size(1)] != self.pad_id
        return outputs, hidden, src_mask


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        rnn_type: RnnType,
        pad_id: int,
        attention: Attention,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.rnn = _make_rnn(
            rnn_type=rnn_type,
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_size * 2, tgt_vocab_size)

    def step(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        prev_context: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(input_ids))
        rnn_in = torch.cat([emb, prev_context], dim=1).unsqueeze(1)
        out, hidden = self.rnn(rnn_in, hidden)
        out = out.squeeze(1)
        dec_last = _select_last_layer_hidden(self.rnn_type, hidden)
        context, attn = self.attention(enc_outputs=enc_outputs, dec_state_last=dec_last, src_mask=src_mask)
        logits = self.out(torch.cat([out, context], dim=1))
        return logits, hidden, context, attn


@dataclass
class Seq2SeqOutput:
    logits: torch.Tensor
    attn: Optional[torch.Tensor]


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: RnnType = "gru",
        attention_type: AttentionType = "additive",
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            pad_id=src_pad_id,
        )
        attention = Attention(attn_type=attention_type, hidden_size=hidden_size)
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            pad_id=tgt_pad_id,
            attention=attention,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> Seq2SeqOutput:
        enc_out, enc_state, src_mask = self.encoder(src_ids, src_lens)
        hidden = _copy_encoder_state(self.rnn_type, enc_state)
        batch_size, tgt_len = tgt_ids.size(0), tgt_ids.size(1)

        logits_steps = []
        attn_steps = []
        prev_context = torch.zeros(batch_size, enc_out.size(2), device=src_ids.device)
        input_ids = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            logits, hidden, prev_context, attn = self.decoder.step(
                input_ids=input_ids,
                hidden=hidden,
                prev_context=prev_context,
                enc_outputs=enc_out,
                src_mask=src_mask,
            )
            logits_steps.append(logits.unsqueeze(1))
            attn_steps.append(attn.unsqueeze(1))

            use_teacher = torch.rand(batch_size, device=src_ids.device) < teacher_forcing_ratio
            next_gold = tgt_ids[:, t]
            next_pred = logits.argmax(dim=1)
            input_ids = torch.where(use_teacher, next_gold, next_pred)

        logits_all = torch.cat(logits_steps, dim=1) if logits_steps else torch.empty(0)
        attn_all = torch.cat(attn_steps, dim=1) if attn_steps else None
        return Seq2SeqOutput(logits=logits_all, attn=attn_all)

    @torch.no_grad()
    def encode(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        enc_out, enc_state, src_mask = self.encoder(src_ids, src_lens)
        hidden = _copy_encoder_state(self.rnn_type, enc_state)
        batch_size = src_ids.size(0)
        context = torch.zeros(batch_size, enc_out.size(2), device=src_ids.device)
        return enc_out, hidden, context, src_mask

