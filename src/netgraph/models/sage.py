"""
GraphSAGE model.

Inductive graph learning via sampling and aggregation.
Implements Hamilton et al. (2017) in pure NumPy.
"""

import numpy as np
from .layers import GraphSAGELayer, DenseLayer, dropout


class GraphSAGE:
    """
    Multi-layer GraphSAGE model.

    Supports mean, max, and sum aggregation strategies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        aggregator: str = "mean",
        sample_sizes: list[int] | None = None,
        dropout_rate: float = 0.5,
        task: str = "node",
        seed: int = 42,
    ):
        self.dropout_rate = dropout_rate
        self.task = task
        self.rng = np.random.RandomState(seed)

        # Build SAGE layers
        self.sage_layers: list[GraphSAGELayer] = []
        dims = [input_dim] + hidden_dims
        if sample_sizes is None:
            sample_sizes = [10] * len(hidden_dims)

        for i in range(len(dims) - 1):
            self.sage_layers.append(
                GraphSAGELayer(
                    dims[i], dims[i + 1],
                    aggregator=aggregator,
                    sample_size=sample_sizes[i] if i < len(sample_sizes) else 10,
                    seed=seed + i,
                )
            )

        self.output_layer = DenseLayer(
            hidden_dims[-1], output_dim,
            activation="linear",
            seed=seed + 100,
        )

        self._training = True

    def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        H = X

        for layer in self.sage_layers:
            H = layer.forward(A, H, training=self._training)
            H = dropout(H, self.dropout_rate, self._training, self.rng)

        if self.task == "graph":
            H = H.mean(axis=0, keepdims=True)

        return self.output_layer.forward(H, self._training)

    def backward(self, d_output: np.ndarray) -> None:
        dH = self.output_layer.backward(d_output)

        if self.task == "graph":
            N = self.sage_layers[-1]._cache["H"].shape[0]
            dH = np.repeat(dH / N, N, axis=0)

        for layer in reversed(self.sage_layers):
            dH = layer.backward(dH)

    @property
    def params(self) -> list[np.ndarray]:
        p = []
        for layer in self.sage_layers:
            p.extend(layer.params)
        p.extend(self.output_layer.params)
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = []
        for layer in self.sage_layers:
            g.extend(layer.grads)
        g.extend(self.output_layer.grads)
        return g

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def predict(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        self.eval()
        out = self.forward(A, X)
        self.train()
        return out
