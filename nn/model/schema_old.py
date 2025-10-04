from pathlib import Path
import torch
from torch import nn
from chess import PIECE_TYPES
from .layers.rope import RoPE
from .layers.feed_forward import FeedForward
from .layers.attention import AttentionBlock

# Embedding configuration
PIECE_EMBEDDING_DIM = 48
SPATIAL_EMBEDDING_DIM = 4
OUTPUT_EMBEDDING_DIM = 128

NUM_HEADS = 8

DROPOUT = 0.1

NUM_LAYERS = 1
FFN_DIM = 192

class InfinityModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: [batch, 8, 8]
        self.embedding = nn.Embedding(2 * len(PIECE_TYPES) + 1, PIECE_EMBEDDING_DIM)
        self.col_weights = nn.Parameter(torch.empty((1, 8, 1, SPATIAL_EMBEDDING_DIM)))
        self.row_weights = nn.Parameter(torch.empty((1, 1, 8, SPATIAL_EMBEDDING_DIM)))
        
        self.embedding_linear = nn.Linear(
            PIECE_EMBEDDING_DIM + (2 * SPATIAL_EMBEDDING_DIM),
            OUTPUT_EMBEDDING_DIM
        )
        
        self.inner = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(OUTPUT_EMBEDDING_DIM, NUM_HEADS, DROPOUT, bias=False),
                nn.LayerNorm(OUTPUT_EMBEDDING_DIM),
                FeedForward(OUTPUT_EMBEDDING_DIM, FFN_DIM),
                nn.LayerNorm(OUTPUT_EMBEDDING_DIM)
            ]) for _ in range(NUM_LAYERS)
        ])
        
        self.move_proj = nn.Linear(OUTPUT_EMBEDDING_DIM, 64)
        self.piece_proj = nn.Linear(OUTPUT_EMBEDDING_DIM, 1)
        self.eval_proj = nn.Linear(64, 3)
        
    def forward(self, input):
        B, _, _, = input.shape
        embedding = self.embedding(input)
        embedding = torch.cat((
            embedding, 
            torch.tile(self.col_weights, (B, 1, 8, 1)),
            torch.tile(self.row_weights, (B, 8, 1, 1))
        ), dim=-1)
        
        x = self.embedding_linear(embedding)
        
        for block in self.inner:
            attn, attn_norm, ffn, ffn_norm = block.children()
            n = attn_norm(x).view(B, 8*8, OUTPUT_EMBEDDING_DIM)
            a, _ = attn(n, n, n)
            x = x + a.view(*x.shape)
            x = x + ffn(ffn_norm(x))
            
        move = self.move_proj(x).view((B, 64, 64))
        piece = self.piece_proj(x).view((B, 64))
        eval = self.eval_proj(piece)
        
            
        return move, eval
        
    @staticmethod
    def import_model(file: Path):
        pass
    
    def export_model(self, file: Path, override: bool = True, create_parents: bool = True):
        if not override and file.exists():
            raise FileExistsError(f"{file} already exists!")
        
        if create_parents:
            parent = file.parent
            if not parent.exists():
                parent.mkdir(parents=True)
        
        batch_dim = torch.export.Dim("batch", min=1)
        
        exported = torch.export.export(self, (
            torch.ones((2, 8, 8), dtype=torch.int), 
        ), dynamic_shapes= {
            "input": {0: batch_dim}
        })
        torch.onnx.export(exported, dynamo=True, f=file)