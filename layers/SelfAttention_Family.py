import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat

class RegularMask:
    def __init__(self, mask):
        self._mask = mask.unsqueeze(1)  # [B, 1, L, S]

    @property
    def mask(self):
        return self._mask

    def to(self, device):
        self._mask = self._mask.to(device)
        return self


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, device='cpu'):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.device = torch.device(device)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.to(self.device)
        keys = keys.to(self.device)
        values = values.to(self.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        B, L, H, E = queries.shape

        if E == 0:
            raise ValueError(f"Invalid input shape for queries: {queries.shape}. The embedding size E cannot be 0.")

        scale = self.scale or 1. / sqrt(E)
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1).to(self.device)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1).to(self.device)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=self.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, device='cpu'):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.device = torch.device(device)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.to(self.device)
        keys = keys.to(self.device)
        values = values.to(self.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        B, L, H, E = queries.shape

        if E == 0:
            raise ValueError(f"Invalid input shape for queries: {queries.shape}. The embedding size E cannot be 0.")

        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=self.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(attention.device)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads).to(attention.device)
        self.value_projection = nn.Linear(d_model, d_values * n_heads).to(attention.device)
        self.out_projection = nn.Linear(d_values * n_heads, d_model).to(attention.device)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Ensure all inputs are on the same device
        device = self.query_projection.weight.device
        queries = queries.to(device)
        keys = keys.to(device)
        values = values.to(device)

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Query, Key, Value projection
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


