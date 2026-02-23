"""
Training loop, optimizers, and loss functions for GNN models.

Provides Adam optimizer, learning rate scheduling, early stopping,
and common loss functions — all in pure NumPy.
"""

import numpy as np
import time
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Tracks training metrics over epochs."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


class AdamOptimizer:
    """Adam optimizer (Kingma & Ba, 2015) in pure NumPy."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # First and second moment estimates
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]):
        """Update parameters using gradients."""
        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, grads: list[np.ndarray]):
        for g in grads:
            g[:] = 0


class SGDOptimizer:
    """SGD with momentum."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]):
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.velocities[i] = self.momentum * self.velocities[i] + g
            p -= self.lr * self.velocities[i]

    def zero_grad(self, grads: list[np.ndarray]):
        for g in grads:
            g[:] = 0


# --- Loss Functions ---

def cross_entropy_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Cross-entropy loss with softmax.

    Args:
        predictions: Raw logits (N, C)
        targets: One-hot labels (N, C) or integer labels (N,)

    Returns:
        (loss_value, gradient)
    """
    N = predictions.shape[0]

    # Softmax
    exp_p = np.exp(predictions - predictions.max(axis=1, keepdims=True))
    probs = exp_p / (exp_p.sum(axis=1, keepdims=True) + 1e-10)

    # Handle integer targets
    if targets.ndim == 1:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), targets.astype(int)] = 1.0
        targets = one_hot

    # Loss
    loss = -np.sum(targets * np.log(probs + 1e-10)) / N

    # Gradient
    grad = (probs - targets) / N

    return float(loss), grad


def mse_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean squared error loss."""
    diff = predictions - targets
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / predictions.size
    return float(loss), grad


def binary_cross_entropy_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    pos_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy loss."""
    eps = 1e-7
    p = np.clip(predictions, eps, 1.0 - eps)

    loss = -np.mean(
        pos_weight * targets * np.log(p)
        + (1.0 - targets) * np.log(1.0 - p)
    )

    grad = -(pos_weight * targets / p - (1.0 - targets) / (1.0 - p))
    grad = grad / predictions.size

    return float(loss), grad


# --- Training Loop ---

def train_node_classifier(
    model,
    A_hat: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: Optional[np.ndarray] = None,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20,
    verbose: bool = True,
) -> TrainingHistory:
    """
    Train a GNN model for node classification.

    Args:
        model: GCN, GAT, or GraphSAGE model
        A_hat: Normalized adjacency matrix
        X: Node feature matrix
        labels: Node labels (N,) integer array
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        TrainingHistory with loss and accuracy curves
    """
    optimizer = AdamOptimizer(model.params, lr=lr, weight_decay=weight_decay)
    history = TrainingHistory()
    patience_counter = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        # Forward
        logits = model.forward(A_hat, X)

        # Loss on training nodes
        train_logits = logits[train_mask]
        train_labels = labels[train_mask]
        loss, grad_loss = cross_entropy_loss(train_logits, train_labels)

        # Full gradient (mask non-training)
        full_grad = np.zeros_like(logits)
        full_grad[train_mask] = grad_loss

        # Backward
        model.backward(full_grad)

        # Update
        optimizer.step(model.params, model.grads)

        # Training accuracy
        train_preds = np.argmax(train_logits, axis=1)
        train_acc = np.mean(train_preds == train_labels)

        history.train_loss.append(loss)
        history.train_accuracy.append(float(train_acc))
        history.learning_rates.append(lr)
        history.epoch_times.append(time.time() - t0)

        # Validation
        if val_mask is not None:
            model.eval()
            val_logits = model.forward(A_hat, X)[val_mask]
            val_labels = labels[val_mask]
            val_loss, _ = cross_entropy_loss(val_logits, val_labels)
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.mean(val_preds == val_labels)

            history.val_loss.append(val_loss)
            history.val_accuracy.append(float(val_acc))

            # Early stopping
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            model.train()

        if verbose and epoch % 10 == 0:
            msg = f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {train_acc:.4f}"
            if val_mask is not None:
                msg += f" | Val Loss: {history.val_loss[-1]:.4f} | Val Acc: {history.val_accuracy[-1]:.4f}"
            print(msg)

    return history


def train_autoencoder(
    model,
    A_hat: np.ndarray,
    A_true: np.ndarray,
    X: np.ndarray,
    epochs: int = 200,
    lr: float = 0.001,
    pos_weight: float = 1.0,
    verbose: bool = True,
) -> TrainingHistory:
    """Train a graph autoencoder."""
    optimizer = AdamOptimizer(model.params, lr=lr)
    history = TrainingHistory()

    # Auto-compute positive weight if not set
    if pos_weight == 1.0:
        num_edges = A_true.sum()
        num_possible = A_true.shape[0] ** 2
        if num_edges > 0:
            pos_weight = (num_possible - num_edges) / num_edges

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        A_recon, Z = model.forward(A_hat, X)
        loss = model.loss(A_true, A_recon, pos_weight)
        model.backward(A_true, A_recon, pos_weight)

        optimizer.step(model.params, model.grads)

        history.train_loss.append(loss)
        history.epoch_times.append(time.time() - t0)

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    return history
