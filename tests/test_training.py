"""Tests for training loop and optimizers."""

import pytest
import numpy as np
from netgraph.models.training import (
    AdamOptimizer, SGDOptimizer,
    cross_entropy_loss, mse_loss, binary_cross_entropy_loss,
    train_autoencoder,
)
from netgraph.models.gcn import GCN
from netgraph.models.autoencoder import GraphAutoencoder


class TestOptimizers:
    def test_adam_step(self):
        W = np.ones((3, 3), dtype=np.float32)
        params = [W]
        opt = AdamOptimizer(params, lr=0.1)
        grads = [np.ones_like(W)]
        opt.step(params, grads)
        # Parameters should have changed
        assert not np.allclose(W, 1.0)

    def test_sgd_step(self):
        W = np.ones((3, 3), dtype=np.float32)
        params = [W]
        opt = SGDOptimizer(params, lr=0.1)
        grads = [np.ones_like(W)]
        opt.step(params, grads)
        assert np.all(W < 1.0)


class TestLosses:
    def test_cross_entropy(self):
        preds = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
        targets = np.array([0])
        loss, grad = cross_entropy_loss(preds, targets)
        assert loss > 0
        assert grad.shape == preds.shape

    def test_cross_entropy_onehot(self):
        preds = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
        targets = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        loss, grad = cross_entropy_loss(preds, targets)
        assert loss > 0

    def test_mse_loss(self):
        preds = np.array([[1.0, 2.0]], dtype=np.float32)
        targets = np.array([[1.5, 2.5]], dtype=np.float32)
        loss, grad = mse_loss(preds, targets)
        assert loss > 0
        assert grad.shape == preds.shape

    def test_bce_loss(self):
        preds = np.array([[0.9, 0.1]], dtype=np.float32)
        targets = np.array([[1.0, 0.0]], dtype=np.float32)
        loss, grad = binary_cross_entropy_loss(preds, targets)
        assert loss > 0


class TestTraining:
    def test_train_autoencoder(self):
        N, F = 8, 4
        rng = np.random.RandomState(42)
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            A[i, (i+1) % N] = 1
            A[(i+1) % N, i] = 1
        A_hat = A + np.eye(N, dtype=np.float32)
        D = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
        A_hat = D @ A_hat @ D
        X = rng.randn(N, F).astype(np.float32)

        model = GraphAutoencoder(F, 8, 4, variational=False)
        history = train_autoencoder(model, A_hat, A, X, epochs=10, verbose=False)
        assert len(history.train_loss) == 10
        # Loss should generally decrease
        assert history.train_loss[-1] <= history.train_loss[0] * 2  # at least not exploding
