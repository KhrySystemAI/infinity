from torch import nn
from math import sqrt

import torch

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.v_down_proj = nn.Linear(embed_dim, attn_dim)
        self.v_up_proj = nn.Linear(attn_dim, embed_dim)
        
        self.sqrt_d = sqrt(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        _, _, _, D = input.shape
        q = self.q_proj(input)
        B, H, W, L = q.shape
        q = q.view(B, H*W, L)
        
        k = self.k_proj(input).view(B, H*W, L)
        v = self.v_down_proj(input).view(B, H*W, L)
        
        attn_scores = torch.einsum("bsd,btd->bst", q, k) / self.sqrt_d
        attn_probs = torch.softmax(attn_scores, -1)
        attn_probs = self.dropout(attn_probs)
        
        v = torch.einsum("bst,bsd->btd", attn_probs, v)
        out = self.v_up_proj(v).view(B, H, W, D)
        
        return out