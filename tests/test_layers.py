"""Tests for GNN layers."""

import pytest
import numpy as np
from netgraph.models.layers import (
    GCNLayer, GATLayer, GraphSAGELayer, DenseLayer,
    relu, leaky_relu, sigmoid, softmax, dropout,
)


def _sample_data(N=5, F=4):
    """Create sample adjacency and features."""
    rng = np.random.RandomState(42)
    A = np.zeros((N, N), dtype=np.float32)
    # Simple cycle graph
    for i in range(N):
        A[i, (i+1) % N] = 1
        A[(i+1) % N, i] = 1
    # Normalized adjacency
    A_hat = A + np.eye(N, dtype=np.float32)
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_hat = D_inv_sqrt @ A_hat @ D_inv_sqrt
    X = rng.randn(N, F).astype(np.float32)
    return A, A_hat, X


class TestActivations:
    def test_relu(self):
        x = np.array([-1, 0, 1, 2], dtype=np.float32)
        out = relu(x)
        np.testing.assert_array_equal(out, [0, 0, 1, 2])

    def test_leaky_relu(self):
        x = np.array([-1, 0, 1], dtype=np.float32)
        out = leaky_relu(x, 0.2)
        assert out[0] == pytest.approx(-0.2)

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 100)
        out = sigmoid(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_softmax_sums_to_one(self):
        x = np.random.randn(3, 5).astype(np.float32)
        out = softmax(x, axis=1)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_dropout_training(self):
        rng = np.random.RandomState(42)
        x = np.ones((10, 10), dtype=np.float32)
        out = dropout(x, 0.5, True, rng)
        # Some values should be zeroed
        assert np.any(out == 0)

    def test_dropout_eval(self):
        rng = np.random.RandomState(42)
        x = np.ones((10, 10), dtype=np.float32)
        out = dropout(x, 0.5, False, rng)
        np.testing.assert_array_equal(out, x)


class TestGCNLayer:
    def test_forward_shape(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GCNLayer(4, 8)
        out = layer.forward(A_hat, X)
        assert out.shape == (5, 8)

    def test_backward_shape(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GCNLayer(4, 8)
        out = layer.forward(A_hat, X)
        dout = np.ones_like(out)
        din = layer.backward(dout)
        assert din.shape == X.shape

    def test_params(self):
        layer = GCNLayer(4, 8)
        assert len(layer.params) == 2  # W, b
        assert layer.W.shape == (4, 8)


class TestGATLayer:
    def test_forward_concat(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GATLayer(4, 8, num_heads=3, concat_heads=True)
        out = layer.forward(A, X)
        assert out.shape == (5, 24)  # 3 heads * 8

    def test_forward_average(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GATLayer(4, 8, num_heads=3, concat_heads=False)
        out = layer.forward(A, X)
        assert out.shape == (5, 8)

    def test_backward_shape(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GATLayer(4, 8, num_heads=2, concat_heads=True)
        out = layer.forward(A, X)
        dout = np.ones_like(out)
        din = layer.backward(dout)
        assert din.shape == X.shape


class TestGraphSAGELayer:
    def test_forward_mean(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GraphSAGELayer(4, 8, aggregator="mean")
        out = layer.forward(A, X)
        assert out.shape == (5, 8)

    def test_forward_max(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GraphSAGELayer(4, 8, aggregator="max")
        out = layer.forward(A, X)
        assert out.shape == (5, 8)

    def test_backward(self):
        A, A_hat, X = _sample_data(5, 4)
        layer = GraphSAGELayer(4, 8, aggregator="mean")
        out = layer.forward(A, X)
        din = layer.backward(np.ones_like(out))
        assert din.shape == X.shape


class TestDenseLayer:
    def test_forward(self):
        layer = DenseLayer(10, 5, activation="relu")
        X = np.random.randn(3, 10).astype(np.float32)
        out = layer.forward(X)
        assert out.shape == (3, 5)
        assert np.all(out >= 0)  # ReLU

    def test_sigmoid_activation(self):
        layer = DenseLayer(10, 5, activation="sigmoid")
        X = np.random.randn(3, 10).astype(np.float32)
        out = layer.forward(X)
        assert np.all(out >= 0) and np.all(out <= 1)
