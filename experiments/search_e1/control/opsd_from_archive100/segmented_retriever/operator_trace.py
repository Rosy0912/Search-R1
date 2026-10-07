"""Observe actual matrix-multiplication shapes without replacing operators."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class RecordMM(TorchDispatchMode):
    def __init__(self):
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mm.default:
            self.shapes.append([list(args[0].shape), list(args[1].shape)])
        return func(*args, **(kwargs or {}))
