from chess import Board, Move
import numpy as np

from .mode import MCTSMode

class Node:
    def __init__(self, board: Board):
        self.board = board
        
        self.children: list[Node] = []
        self.moves: list[Move] = list(self.board.legal_moves)
        
        self.is_expanded = False
        self.visits = 0
        self._wdl = np.zeros((3,))
        
    def collect(self, mode: MCTSMode, batch_size: int, decay: float):
        if not self.is_expanded:
            for m in self.moves:
                b = self.board.copy()
                b.push(m)
                self.children.append(Node(b))
                self.is_expanded = True
            return self.children
        
        if mode == MCTSMode.greedy:
            key_func = lambda n: pow(decay, n.visits) * (n.wdl[2] if n.visits >= 0 else 1)
        elif mode == MCTSMode.balanced:
            key_func = lambda n: pow(decay, n.visits) * ((n.wdl[2] + (n.wdl[1] / 2)) if n.visits >= 0 else 1)
        elif mode == MCTSMode.solid:
            key_func = lambda n: pow(decay, n.visits) * (((1 - n.wdl[0]) + (n.wdl[1] / 2)) if n.visits >= 0 else 1)
        else:      # MCTSMode.safe
            key_func = lambda n: pow(decay, n.visits) * ((1 - n.wdl[0]) if n.visits >= 0 else 1)
        
        sorted_children = sorted(self.children, key=key_func)
        
        leaves = []
        for child in sorted_children:
            leaves.extend(child.collect(mode, batch_size-len(leaves), decay))
            if len(leaves) >= batch_size:
                break
            
        return leaves
    
    def run(self, mode: MCTSMode, batch_size: int)
    