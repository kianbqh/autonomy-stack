"""Dependency-free (NumPy-only) inference for the PyTorch BC policy, used inside the ROS 2 node."""
import numpy as np


class NumpyMLP:
    """Mirror of fusion_core.imitation.Policy: Linear-Tanh-Linear-Tanh-Linear-Tanh."""

    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f"net.{i}.weight"], w[f"net.{i}.bias"]) for i in (0, 2, 4)]

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        for W, b in self.layers:
            x = np.tanh(W @ x + b)
        return x
