import torch
import torch.nn as nn
import math

class TemporalAttention(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaler = math.sqrt(embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, q, k, v, mask=None):
        """
        x: (B, S, H, W, D)
        mask: optional (B, S) mask
        returns: (B, S, H, W, D)
        """

        # Project Q/K/V
        Q = self.q_proj(q)  # (B, S, H, W, D)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # Scaled dot-product along sequence dimension
        attn_scores = torch.einsum('bshwd,bthwd->bhwts', Q, K) / self.scaler
        # attn_scores shape: (B, H, W, S, S)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, None, :] == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum over V
        out = torch.einsum('bhwts,bshwd->bshwd', attn_probs, V)
        out = self.out_proj(out)
        
        q += out
        q = self.norm(q)
        
        return q