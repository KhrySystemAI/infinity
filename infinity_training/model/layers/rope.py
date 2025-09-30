import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Concatenate rotary positional embeddings along the embedding dimension
    Input shape: (B, S, H, W, D)
    Output shape: (B, S, H, W, D + D_rope)
    """
    def __init__(self, embedding_dim, num_waves=16, base=10000):
        """
        embedding_dim: original feature dimension
        num_waves: number of sine/cosine wave pairs to concatenate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_waves = num_waves
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, num_waves // 2, 1).float() / num_waves // 2))
        self.register_buffer("inv_freq", inv_freq)
        

    def forward(self, x):
        B, S, H, W, D = x.shape
        t = torch.arange(S, device=x.device).float()  # sequence positions: (S,)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (S, num_waves)
        sin = freqs.sin()  # (S, num_waves)
        cos = freqs.cos()  # (S, num_waves)

        # Expand to match (B, S, H, W, num_waves*2)
        sin = sin[None, :, None, None, :]  # (1, S, 1, 1, num_waves)
        cos = cos[None, :, None, None, :]  # (1, S, 1, 1, num_waves)
        rope = torch.cat([sin, cos], dim=-1)  # (1, S, 1, 1, num_waves*2)
        rope = rope.expand(B, S, H, W, -1)

        # Concatenate along embedding dim
        return torch.cat([x, rope], dim=-1)  # (B, S, H, W, D + num_waves*2)
