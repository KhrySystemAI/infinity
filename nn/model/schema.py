from pathlib import Path
import torch
from torch import nn

PIECE_EMBEDDING_DIM = 128
SQUARE_EMBEDDING_DIM = 128

NUM_HEADS = 4
DROPOUT = 0.1
NUM_LAYERS = 1
FFN_DIM = 384

class InfinityModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.piece_embedding = nn.Embedding(13, PIECE_EMBEDDING_DIM)
        self.square_embedding = nn.Parameter(torch.randn((1, 64, SQUARE_EMBEDDING_DIM)))
        self.embedding_dim = PIECE_EMBEDDING_DIM + SQUARE_EMBEDDING_DIM
        
        
        self.inner = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.embedding_dim),
                nn.MultiheadAttention(self.embedding_dim, NUM_HEADS, DROPOUT),
                nn.Sequential(
                    nn.Linear(self.embedding_dim, FFN_DIM),
                    nn.SiLU(),
                    nn.Linear(FFN_DIM, self.embedding_dim)
                ),
                nn.MultiheadAttention(self.embedding_dim, NUM_HEADS, DROPOUT, kdim=64, vdim=64) # Cross attention with the decoder output as the keys and values, and the original embedding as the query
            ]) for _ in range(NUM_LAYERS)
        ])
        self.decoder = nn.Linear(self.embedding_dim, 64)
        self.value_proj_1 = nn.Linear(self.embedding_dim, 1)
        self.promo_proj_2 = nn.Linear(self.embedding_dim, 24)
        self.value_proj_2 = nn.Linear(64, 3)
        self.promo_proj_1 = nn.Linear(8, 1)
        
    def forward(self, input: torch.Tensor):
        B, _ = input.shape
        pieces = self.piece_embedding(input)
        squares = torch.tile(self.square_embedding, (B, 1, 1))
        embedding = torch.cat((pieces, squares), dim=-1)
        
        x = embedding
        for layer in self.inner:
            norm, sa, ffn, cross = layer.children()
            d = norm(x)
            d, _ = sa(d, d, d)
            d = ffn(d)
            d = self.decoder(d)
            d, _ = cross(embedding, d, d)
            x += d
            
        policy = self.decoder(x).view(B, 4096)
        moves = self.promo_proj_1(x.view(B,8,8,self.embedding_dim).permute(0,2,3,1)).permute(0,3,1,2).view(B,8,self.embedding_dim)
        moves = self.promo_proj_2(moves).view(B, 192)
        value = self.value_proj_1(x).view(B, 64)
        value = self.value_proj_2(value)
        return policy, moves, value
    
    def export_model(self, file: Path, override: bool = True, create_parents: bool = True):
        if not override and file.exists():
            raise FileExistsError(f"{file} already exists!")
        
        if create_parents:
            parent = file.parent
            if not parent.exists():
                parent.mkdir(parents=True)
        
        batch_dim = torch.export.Dim("batch", min=1)
        
        exported = torch.export.export(self, (
            torch.ones((2, 64), dtype=torch.int), 
        ), dynamic_shapes= {
            "input": {0: batch_dim}
        })
        torch.onnx.export(exported, dynamo=True, f=file)