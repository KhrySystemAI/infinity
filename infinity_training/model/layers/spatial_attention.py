import math
import torch
from torch import nn

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, se_ratio=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaler = math.sqrt(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Squeeze-and-Excitation MLP
        hidden_dim = max(1, int(embed_dim * se_ratio))
        self.se_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.se_fc2 = nn.Linear(hidden_dim, embed_dim)
        self.se_activation = nn.SiLU()
        self.se_sigmoid = nn.Sigmoid()

    def forward(self, q, k, v):
        B, H, W, D = q.shape
        
        # Q/K/V projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Flatten spatial dimensions for attention
        q_flat = q.view(B, H*W, D)
        k_flat = k.view(B, H*W, D)
        v_flat = v.view(B, H*W, D)
        
        # Scaled dot-product attention along spatial dimension
        attn_scores = torch.einsum("bnd,bmd->bnm", q_flat, k_flat) / self.scaler
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out_flat = torch.einsum("bnm,bmd->bnd", attn_probs, v_flat)
        out = out_flat.view(B, H, W, D)
        
        # Conv projection
        out = self.out_proj(out)
        out = self.conv(out.permute(0,3,1,2))  # to (B,D,H,W) for Conv2d
        out = out.permute(0,2,3,1)  # back to (B,H,W,D)
        
        # Squeeze-and-Excitation
        se = out.mean(dim=(1,2))           # Global spatial average: (B,D)
        se = self.se_fc1(se)               # Reduction
        se = self.se_activation(se)
        se = self.se_fc2(se)               # Expansion
        se = self.se_sigmoid(se).unsqueeze(2).unsqueeze(3)  # (B,1,1,D)
        out = out * se                      # Channel-wise excitation
        
        # Residual + Norm
        x = q + out
        x = self.norm(x)
        return x
