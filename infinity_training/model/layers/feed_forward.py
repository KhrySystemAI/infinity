from torch import nn

class FeedForward(nn.Module):
    def __init__(self, i, m):
        super().__init__()
        self.up_proj = nn.Linear(i, m, bias=False)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(m, i, bias=False)
        self.norm = nn.LayerNorm(i)
        
    def forward(self, input):
        x = self.up_proj(input)
        x = self.silu(x)
        x = self.down_proj(x)
        x = self.norm(input + x)
        return x