"""
Graph Autoencoder for anomaly detection.

Encodes graph structure into a latent space and reconstructs it.
Reconstruction error serves as anomaly score — nodes/edges that
reconstruct poorly are anomalous.
"""

import numpy as np
from .layers import GCNLayer, DenseLayer, sigmoid


class GraphAutoencoder:
    """
    Graph Autoencoder (GAE) / Variational Graph Autoencoder (VGAE).

    Encoder: GCN layers -> latent embeddings Z
    Decoder: Z @ Z^T -> reconstructed adjacency

    High reconstruction error indicates structural anomaly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        variational: bool = False,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.variational = variational
        self.rng = np.random.RandomState(seed)

        # Encoder
        self.encoder_1 = GCNLayer(
            input_dim, hidden_dim, activation="relu", seed=seed
        )
        self.encoder_mu = GCNLayer(
            hidden_dim, latent_dim, activation="linear", seed=seed + 1
        )

        if variational:
            self.encoder_logvar = GCNLayer(
                hidden_dim, latent_dim, activation="linear", seed=seed + 2
            )

        self._training = True
        self._cache: dict = {}

    def encode(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Encode nodes to latent space."""
        H = self.encoder_1.forward(A_hat, X, training=self._training)
        mu = self.encoder_mu.forward(A_hat, H, training=self._training)

        if self.variational and self._training:
            logvar = self.encoder_logvar.forward(A_hat, H, training=self._training)
            std = np.exp(0.5 * logvar)
            eps = self.rng.randn(*mu.shape).astype(np.float32)
            Z = mu + eps * std
            self._cache["mu"] = mu
            self._cache["logvar"] = logvar
            self._cache["std"] = std
            self._cache["eps"] = eps
        else:
            Z = mu

        self._cache["Z"] = Z
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent embeddings to reconstructed adjacency."""
        # Inner product decoder
        A_recon = sigmoid(Z @ Z.T)
        return A_recon

    def forward(self, A_hat: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full forward pass: encode then decode."""
        Z = self.encode(A_hat, X)
        A_recon = self.decode(Z)
        return A_recon, Z

    def loss(
        self,
        A_true: np.ndarray,
        A_recon: np.ndarray,
        pos_weight: float = 1.0,
    ) -> float:
        """
        Compute reconstruction loss.

        Uses weighted binary cross-entropy to handle sparse adjacency.
        """
        eps = 1e-7
        A_recon = np.clip(A_recon, eps, 1.0 - eps)

        # Weighted BCE
        bce = -(
            pos_weight * A_true * np.log(A_recon)
            + (1.0 - A_true) * np.log(1.0 - A_recon)
        )
        recon_loss = bce.mean()

        total_loss = recon_loss

        # KL divergence for variational version
        if self.variational and "mu" in self._cache:
            mu = self._cache["mu"]
            logvar = self._cache["logvar"]
            kl = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
            total_loss += kl
            self._cache["kl_loss"] = float(kl)

        self._cache["recon_loss"] = float(recon_loss)
        self._cache["total_loss"] = float(total_loss)
        return float(total_loss)

    def backward(self, A_true: np.ndarray, A_recon: np.ndarray, pos_weight: float = 1.0) -> None:
        """Backward pass through encoder."""
        eps = 1e-7
        A_recon_clip = np.clip(A_recon, eps, 1.0 - eps)
        N = A_true.shape[0]

        # Gradient of BCE w.r.t. A_recon
        dA_recon = -(
            pos_weight * A_true / A_recon_clip
            - (1.0 - A_true) / (1.0 - A_recon_clip)
        ) / (N * N)

        # Gradient through sigmoid
        dA_pre = dA_recon * A_recon_clip * (1.0 - A_recon_clip)

        # Gradient through inner product decoder: d/dZ of (Z @ Z.T)
        Z = self._cache["Z"]
        dZ = (dA_pre + dA_pre.T) @ Z

        # KL gradient
        if self.variational and "mu" in self._cache:
            mu = self._cache["mu"]
            logvar = self._cache["logvar"]
            dmu = mu / N
            dlogvar = 0.5 * (np.exp(logvar) - 1.0) / N
            dZ_mu = dZ + dmu
        else:
            dZ_mu = dZ

        # Backward through encoder
        dH = self.encoder_mu.backward(dZ_mu)
        if self.variational and "logvar" in self._cache:
            dH += self.encoder_logvar.backward(
                dZ * self._cache["std"] * self._cache["eps"] + dlogvar
            )
        self.encoder_1.backward(dH)

    def anomaly_scores(
        self,
        A_hat: np.ndarray,
        A_true: np.ndarray,
        X: np.ndarray,
    ) -> dict:
        """
        Compute per-node and per-edge anomaly scores.

        Returns dict with:
          - node_scores: (N,) reconstruction error per node
          - edge_scores: dict of (u,v) -> score for existing edges
          - graph_score: overall graph anomaly score
        """
        self.eval()
        A_recon, Z = self.forward(A_hat, X)
        self.train()

        # Per-node: average reconstruction error of incident edges
        diff = np.abs(A_true - A_recon)
        node_scores = diff.mean(axis=1)

        # Per-edge: reconstruction error for each edge
        edge_scores = {}
        rows, cols = np.where(A_true > 0)
        for i, j in zip(rows, cols):
            edge_scores[(i, j)] = float(diff[i, j])

        # Graph-level
        graph_score = float(node_scores.mean())

        return {
            "node_scores": node_scores,
            "edge_scores": edge_scores,
            "graph_score": graph_score,
            "latent_embeddings": Z,
        }

    @property
    def params(self) -> list[np.ndarray]:
        p = []
        p.extend(self.encoder_1.params)
        p.extend(self.encoder_mu.params)
        if self.variational:
            p.extend(self.encoder_logvar.params)
        return p

    @property
    def grads(self) -> list[np.ndarray]:
        g = []
        g.extend(self.encoder_1.grads)
        g.extend(self.encoder_mu.grads)
        if self.variational:
            g.extend(self.encoder_logvar.grads)
        return g

    def train(self):
        self._training = True

    def eval(self):
        self._training = False
