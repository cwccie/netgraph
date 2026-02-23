"""
Graph Attention Network (GAT) model.

Multi-layer GAT for node and graph classification.
Implements Velickovic et al. (2018) in pure NumPy.
"""

import numpy as np
from .layers import GATLayer, DenseLayer, dropout


class GAT:
    """
    Multi-layer Graph Attention Network.

    Architecture: Input -> [GAT(multi-head) -> Dropout]* -> Output GAT -> Dense
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.5,
        task: str = "node",
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.task = task
        self.rng = np.random.RandomState(seed)

        self.gat_layers: list[GATLayer] = []

        # First layer: input_dim -> hidden_dim (concat heads)
        self.gat_layers.append(
            GATLayer(
                input_dim, hidden_dim,
                num_heads=num_heads,
                concat_heads=True,
                seed=seed,
            )
        )

        # Hidden layers: (hidden_dim * num_heads) -> hidden_dim (concat)
        for i in range(1, num_layers - 1):
            self.gat_layers.append(
                GATLayer(
                    hidden_dim * num_heads, hidden_dim,
                    num_heads=num_heads,
                    concat_heads=True,
                    seed=seed + i,
                )
            )

        # Last GAT layer: average heads instead of concat
        if num_layers > 1:
            self.gat_layers.append(
                GATLayer(
                    hidden_dim * num_heads, hidden_dim,
                    num_heads=num_heads,
                    concat_heads=False,
                    seed=seed + num_layers,
                )
            )

        # Output dense layer
        self.output_layer = DenseLayer(
            hidden_dim, output_dim,
            activation="linear",
            seed=seed + 100,
        )

        self._training = True

    def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        H = X

        for layer in self.gat_layers:
            H = layer.forward(A, H, training=self._training)
            H = dropout(H, self.dropout_rate, self._training, self.rng)

        if self.task == "graph":
            H = H.mean(axis=0, keepdims=True)

        return self.output_layer.forward(H, self._training)

    def backward(self, d_output: np.ndarray) -> None:
        """Backward pass."""
        dH = self.output_layer.backward(d_output)

        if self.task == "graph":
            N = self.gat_layers[-1]._cache["H"].shape[0]
            dH = np.repeat(dH / N, N, axis=0)

        for layer in reversed(self.gat_layers):
            dH = layer.backward(dH)

    @property
    def params(self) -> list[np.ndarray]:
        p = []
        for layer in self.gat_layers:
            p.extend(layer.params)
        p.extend(self.output_layer.params)
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = []
        for layer in self.gat_layers:
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
