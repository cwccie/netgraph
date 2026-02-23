"""
GNN layers implemented in pure NumPy.

Provides building blocks for Graph Neural Networks:
  - Graph Convolutional (GCN) layer
  - Graph Attention (GAT) layer
  - GraphSAGE aggregation layer
  - Dense (fully connected) layer

All implementations use NumPy only — no PyTorch/TensorFlow required.
"""

import numpy as np
from typing import Optional


def _glorot_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    """Glorot/Xavier uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _he_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    """He/Kaiming initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return (rng.randn(fan_in, fan_out) * std).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-10)


def dropout(x: np.ndarray, rate: float, training: bool, rng: np.random.RandomState) -> np.ndarray:
    if not training or rate <= 0:
        return x
    mask = (rng.rand(*x.shape) > rate).astype(np.float32)
    return x * mask / (1.0 - rate)


class GCNLayer:
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    Implements: H' = sigma(A_hat @ H @ W + b)
    where A_hat is the normalized adjacency with self-loops.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        use_bias: bool = True,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.use_bias = use_bias
        self.rng = np.random.RandomState(seed)

        # Parameters
        self.W = _glorot_init(in_features, out_features, self.rng)
        self.b = np.zeros(out_features, dtype=np.float32) if use_bias else None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if use_bias else None

        # Cache for backward pass
        self._cache: dict = {}

    def forward(self, A_hat: np.ndarray, H: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            A_hat: Normalized adjacency matrix (N, N)
            H: Node feature matrix (N, F_in)
            training: Whether in training mode

        Returns:
            H': Updated node features (N, F_out)
        """
        # Message passing: aggregate neighbor features
        AH = A_hat @ H  # (N, F_in)

        # Linear transformation
        Z = AH @ self.W  # (N, F_out)
        if self.use_bias:
            Z = Z + self.b

        # Cache for backward
        if training:
            self._cache = {"A_hat": A_hat, "H": H, "AH": AH, "Z": Z}

        # Activation
        return self._activate(Z)

    def backward(self, dH_out: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dH_out: Gradient from next layer (N, F_out)

        Returns:
            dH_in: Gradient for previous layer (N, F_in)
        """
        A_hat = self._cache["A_hat"]
        H = self._cache["H"]
        AH = self._cache["AH"]
        Z = self._cache["Z"]

        # Activation gradient
        dZ = dH_out * self._activate_grad(Z)

        # Parameter gradients
        self.dW = AH.T @ dZ  # (F_in, F_out)
        if self.use_bias:
            self.db = dZ.sum(axis=0)

        # Input gradient
        dAH = dZ @ self.W.T  # (N, F_in)
        dH_in = A_hat.T @ dAH  # (N, F_in)

        return dH_in

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return relu(x)
        elif self.activation_name == "leaky_relu":
            return leaky_relu(x)
        elif self.activation_name == "sigmoid":
            return sigmoid(x)
        elif self.activation_name == "tanh":
            return np.tanh(x)
        return x  # linear

    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return (x > 0).astype(np.float32)
        elif self.activation_name == "leaky_relu":
            return np.where(x > 0, 1.0, 0.2).astype(np.float32)
        elif self.activation_name == "sigmoid":
            s = sigmoid(x)
            return s * (1.0 - s)
        elif self.activation_name == "tanh":
            return 1.0 - np.tanh(x) ** 2
        return np.ones_like(x)

    @property
    def params(self) -> list[np.ndarray]:
        p = [self.W]
        if self.use_bias:
            p.append(self.b)
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = [self.dW]
        if self.use_bias:
            g.append(self.db)
        return g


class GATLayer:
    """
    Graph Attention Network layer (Velickovic et al., 2018).

    Implements multi-head attention over graph neighborhoods.
    Each head computes: alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout_rate: float = 0.0,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(seed)

        # Per-head parameters
        self.W_heads = []   # List of (F_in, F_out) weight matrices
        self.a_src = []     # List of (F_out, 1) attention vectors (source)
        self.a_dst = []     # List of (F_out, 1) attention vectors (destination)

        for _ in range(num_heads):
            self.W_heads.append(_glorot_init(in_features, out_features, self.rng))
            self.a_src.append(_glorot_init(out_features, 1, self.rng))
            self.a_dst.append(_glorot_init(out_features, 1, self.rng))

        # Gradients
        self.dW_heads = [np.zeros_like(w) for w in self.W_heads]
        self.da_src = [np.zeros_like(a) for a in self.a_src]
        self.da_dst = [np.zeros_like(a) for a in self.a_dst]

        self._cache: dict = {}

    def forward(self, A: np.ndarray, H: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Multi-head attention forward pass.

        Args:
            A: Adjacency matrix (N, N) — used as attention mask
            H: Node features (N, F_in)
            training: Whether in training mode

        Returns:
            H': (N, num_heads * F_out) if concat, else (N, F_out) if average
        """
        N = H.shape[0]
        head_outputs = []
        attention_weights = []

        # Mask: only attend to neighbors (and self)
        mask = A + np.eye(N, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

        for h in range(self.num_heads):
            # Linear transform
            Wh = H @ self.W_heads[h]  # (N, F_out)

            # Attention scores
            e_src = Wh @ self.a_src[h]  # (N, 1)
            e_dst = Wh @ self.a_dst[h]  # (N, 1)

            # Pairwise attention: e_ij = LeakyReLU(e_src_i + e_dst_j)
            e = leaky_relu(e_src + e_dst.T)  # (N, N)

            # Mask: -inf for non-neighbors
            e = np.where(mask > 0, e, -1e9)

            # Softmax
            alpha = softmax(e, axis=1)  # (N, N)

            # Dropout on attention
            if training and self.dropout_rate > 0:
                alpha = dropout(alpha, self.dropout_rate, training, self.rng)

            # Aggregate
            out = alpha @ Wh  # (N, F_out)
            head_outputs.append(out)
            attention_weights.append(alpha)

        if training:
            self._cache = {
                "A": A, "H": H, "mask": mask,
                "head_outputs": head_outputs,
                "attention_weights": attention_weights,
            }

        if self.concat_heads:
            return np.concatenate(head_outputs, axis=1)  # (N, num_heads * F_out)
        else:
            return np.mean(head_outputs, axis=0)  # (N, F_out)

    def backward(self, dH_out: np.ndarray) -> np.ndarray:
        """Backward pass through multi-head attention."""
        H = self._cache["H"]
        mask = self._cache["mask"]
        N = H.shape[0]

        if self.concat_heads:
            # Split gradient across heads
            douts = np.split(dH_out, self.num_heads, axis=1)
        else:
            douts = [dH_out / self.num_heads] * self.num_heads

        dH_in = np.zeros_like(H)

        for h in range(self.num_heads):
            Wh = H @ self.W_heads[h]
            alpha = self._cache["attention_weights"][h]
            dout = douts[h]

            # Gradient through aggregation
            dWh = alpha.T @ dout  # (N, F_out)
            dalpha = dout @ Wh.T  # (N, N)

            # Gradient through softmax
            # dsoftmax: dalpha * alpha - alpha * sum(dalpha * alpha)
            ds = dalpha * alpha
            ds = ds - alpha * ds.sum(axis=1, keepdims=True)
            ds = ds * mask  # mask non-neighbors

            # Gradient through LeakyReLU
            e_src = Wh @ self.a_src[h]
            e_dst = Wh @ self.a_dst[h]
            e = e_src + e_dst.T
            dl = ds * np.where(e > 0, 1.0, 0.2).astype(np.float32)

            # Gradients for attention vectors
            de_src = dl.sum(axis=1, keepdims=True)  # (N, 1)
            de_dst = dl.sum(axis=0, keepdims=True).T  # (N, 1)

            self.da_src[h] = Wh.T @ de_src
            self.da_dst[h] = Wh.T @ de_dst

            # Gradient for W
            dWh_total = dWh + (de_src @ self.a_src[h].T + de_dst @ self.a_dst[h].T)
            self.dW_heads[h] = H.T @ dWh_total

            # Input gradient
            dH_in += dWh_total @ self.W_heads[h].T

        return dH_in

    @property
    def params(self) -> list[np.ndarray]:
        p = []
        for h in range(self.num_heads):
            p.extend([self.W_heads[h], self.a_src[h], self.a_dst[h]])
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = []
        for h in range(self.num_heads):
            g.extend([self.dW_heads[h], self.da_src[h], self.da_dst[h]])
        return g


class GraphSAGELayer:
    """
    GraphSAGE layer (Hamilton et al., 2017).

    Implements sampling and aggregation:
      h_v = sigma(W * CONCAT(h_v, AGG({h_u : u in N(v)})))

    Aggregation strategies: mean, max, lstm-like (simplified).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        activation: str = "relu",
        sample_size: int = 10,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.activation_name = activation
        self.sample_size = sample_size
        self.rng = np.random.RandomState(seed)

        # Weight for concatenated [self || aggregated_neighbors]
        self.W = _glorot_init(in_features * 2, out_features, self.rng)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._cache: dict = {}

    def forward(self, A: np.ndarray, H: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with neighborhood sampling and aggregation.

        Args:
            A: Adjacency matrix (N, N)
            H: Node features (N, F_in)

        Returns:
            H': Updated features (N, F_out)
        """
        N = H.shape[0]

        # Aggregate neighbor features
        if self.aggregator == "mean":
            # Mean aggregation
            D = A.sum(axis=1, keepdims=True)
            D = np.where(D > 0, D, 1.0)
            H_agg = (A @ H) / D
        elif self.aggregator == "max":
            # Element-wise max over neighbors
            H_agg = np.zeros_like(H)
            for i in range(N):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    if len(neighbors) > self.sample_size:
                        neighbors = self.rng.choice(
                            neighbors, self.sample_size, replace=False
                        )
                    H_agg[i] = H[neighbors].max(axis=0)
        elif self.aggregator == "sum":
            H_agg = A @ H
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Concatenate self and aggregated features
        H_concat = np.concatenate([H, H_agg], axis=1)  # (N, 2*F_in)

        # Linear transform
        Z = H_concat @ self.W + self.b  # (N, F_out)

        # L2 normalize
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        Z_norm = Z / norm

        if training:
            self._cache = {
                "A": A, "H": H, "H_agg": H_agg,
                "H_concat": H_concat, "Z": Z, "Z_norm": Z_norm,
            }

        # Activation
        if self.activation_name == "relu":
            return relu(Z_norm)
        elif self.activation_name == "leaky_relu":
            return leaky_relu(Z_norm)
        return Z_norm

    def backward(self, dH_out: np.ndarray) -> np.ndarray:
        """Backward pass."""
        H = self._cache["H"]
        H_concat = self._cache["H_concat"]
        Z = self._cache["Z"]
        A = self._cache["A"]

        # Activation gradient
        if self.activation_name == "relu":
            dZ = dH_out * (Z > 0).astype(np.float32)
        elif self.activation_name == "leaky_relu":
            dZ = dH_out * np.where(Z > 0, 1.0, 0.2).astype(np.float32)
        else:
            dZ = dH_out

        # Parameter gradients
        self.dW = H_concat.T @ dZ
        self.db = dZ.sum(axis=0)

        # Input gradient
        dH_concat = dZ @ self.W.T  # (N, 2*F_in)
        F = self.in_features
        dH_self = dH_concat[:, :F]
        dH_agg = dH_concat[:, F:]

        # Gradient through mean aggregation
        if self.aggregator == "mean":
            D = A.sum(axis=1, keepdims=True)
            D = np.where(D > 0, D, 1.0)
            dH_in = dH_self + A.T @ (dH_agg / D)
        else:
            dH_in = dH_self + A.T @ dH_agg

        return dH_in

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class DenseLayer:
    """Standard fully connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        seed: int = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        rng = np.random.RandomState(seed)

        self.W = _he_init(in_features, out_features, rng)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: dict = {}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        Z = X @ self.W + self.b
        if training:
            self._cache = {"X": X, "Z": Z}

        if self.activation_name == "relu":
            return relu(Z)
        elif self.activation_name == "sigmoid":
            return sigmoid(Z)
        elif self.activation_name == "softmax":
            return softmax(Z, axis=1)
        return Z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self._cache["X"]
        Z = self._cache["Z"]

        if self.activation_name == "relu":
            dZ = dout * (Z > 0).astype(np.float32)
        elif self.activation_name == "sigmoid":
            s = sigmoid(Z)
            dZ = dout * s * (1.0 - s)
        else:
            dZ = dout

        self.dW = X.T @ dZ
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]
