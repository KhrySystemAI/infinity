from enum import Enum

class MCTSMode(Enum):
    greedy = "greedy"
    balanced = "balanced"
    solid = "solid"
    safe = "safe"