from pathlib import Path
import torch
from infinity_training.model.schema import InfinityModule

module = InfinityModule()

pytorch_total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
print(pytorch_total_params, "Parameters")
print(list(i.shape for i in module.forward(torch.zeros(size=(100, 64), dtype=torch.int))))
module.export_model(Path.cwd() / ("model.onnx"))