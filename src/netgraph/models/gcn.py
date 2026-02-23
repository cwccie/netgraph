"""
Graph Convolutional Network (GCN) model.

Multi-layer GCN for node classification and graph-level tasks.
Implements the architecture from Kipf & Welling (2017) in pure NumPy.
"""

import numpy as np
from typing import Optional
from .layers import GCNLayer, DenseLayer, softmax, dropout


class GCN:
    """
    Multi-layer Graph Convolutional Network.

    Architecture: Input -> [GCN -> Dropout]* -> ReadOut -> Dense -> Output

    Supports:
      - Node-level prediction (classify each node)
      - Graph-level prediction (classify entire graph via readout)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout_rate: float = 0.5,
        task: str = "node",  # "node" or "graph"
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.task = task
        self.rng = np.random.RandomState(seed)

        # Build layers
        self.gcn_layers: list[GCNLayer] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.gcn_layers.append(
                GCNLayer(
                    dims[i], dims[i + 1],
                    activation="relu",
                    seed=seed + i,
                )
            )

        # Output layer
        if task == "graph":
            self.readout_dense = DenseLayer(
                hidden_dims[-1], hidden_dims[-1],
                activation="relu", seed=seed + 100,
            )
        self.output_layer = DenseLayer(
            hidden_dims[-1], output_dim,
            activation="linear", seed=seed + 200,
        )

        self._training = True

    def forward(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            A_hat: Normalized adjacency (N, N)
            X: Node features (N, F)

        Returns:
            If task='node': (N, output_dim) — per-node predictions
            If task='graph': (1, output_dim) — graph-level prediction
        """
        H = X

        for layer in self.gcn_layers:
            H = layer.forward(A_hat, H, training=self._training)
            H = dropout(H, self.dropout_rate, self._training, self.rng)

        if self.task == "graph":
            # Global mean pooling
            H_graph = H.mean(axis=0, keepdims=True)  # (1, hidden)
            H_graph = self.readout_dense.forward(H_graph, self._training)
            return self.output_layer.forward(H_graph, self._training)
        else:
            return self.output_layer.forward(H, self._training)

    def backward(self, d_output: np.ndarray) -> None:
        """Backward pass through the network."""
        dH = self.output_layer.backward(d_output)

        if self.task == "graph":
            dH = self.readout_dense.backward(dH)
            # Gradient through mean pooling: distribute to all nodes
            N = self.gcn_layers[-1]._cache["H"].shape[0]
            dH = np.repeat(dH / N, N, axis=0)

        # Backward through GCN layers (reverse order)
        for layer in reversed(self.gcn_layers):
            dH = layer.backward(dH)

    @property
    def params(self) -> list[np.ndarray]:
        p = []
        for layer in self.gcn_layers:
            p.extend(layer.params)
        if self.task == "graph":
            p.extend(self.readout_dense.params)
        p.extend(self.output_layer.params)
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = []
        for layer in self.gcn_layers:
            g.extend(layer.grads)
        if self.task == "graph":
            g.extend(self.readout_dense.grads)
        g.extend(self.output_layer.grads)
        return g

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def predict(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Run inference (no dropout)."""
        self.eval()
        out = self.forward(A_hat, X)
        self.train()
        return out

    def save(self, filepath: str):
        """Save model parameters."""
        param_dict = {f"param_{i}": p for i, p in enumerate(self.params)}
        np.savez(filepath, **param_dict)

    def load(self, filepath: str):
        """Load model parameters."""
        data = np.load(filepath)
        for i, p in enumerate(self.params):
            key = f"param_{i}"
            if key in data:
                p[:] = data[key]
