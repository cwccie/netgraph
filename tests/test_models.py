"""Tests for GNN models (GCN, GAT, GraphSAGE, Autoencoder)."""

import pytest
import numpy as np
from netgraph.models.gcn import GCN
from netgraph.models.gat import GAT
from netgraph.models.sage import GraphSAGE
from netgraph.models.autoencoder import GraphAutoencoder


def _sample_data(N=10, F=6):
    rng = np.random.RandomState(42)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, (i+1) % N] = 1
        A[(i+1) % N, i] = 1
    for _ in range(N):
        i, j = rng.randint(0, N, 2)
        A[i, j] = 1
        A[j, i] = 1
    A_hat = A + np.eye(N, dtype=np.float32)
    D = np.diag(A_hat.sum(axis=1))
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-8)))
    A_hat = D_inv @ A_hat @ D_inv
    X = rng.randn(N, F).astype(np.float32)
    return A, A_hat, X


class TestGCN:
    def test_forward_node(self):
        A, A_hat, X = _sample_data()
        model = GCN(6, [16, 8], 3, task="node")
        out = model.forward(A_hat, X)
        assert out.shape == (10, 3)

    def test_forward_graph(self):
        A, A_hat, X = _sample_data()
        model = GCN(6, [16, 8], 3, task="graph")
        out = model.forward(A_hat, X)
        assert out.shape == (1, 3)

    def test_backward(self):
        A, A_hat, X = _sample_data()
        model = GCN(6, [16], 3, task="node")
        out = model.forward(A_hat, X)
        grad = np.ones_like(out) / out.size
        model.backward(grad)
        # Check gradients are non-zero
        assert np.any(model.grads[0] != 0)

    def test_predict_no_dropout(self):
        A, A_hat, X = _sample_data()
        model = GCN(6, [16], 3, dropout_rate=0.5)
        # Predict should give same result each time (no dropout)
        out1 = model.predict(A_hat, X)
        out2 = model.predict(A_hat, X)
        np.testing.assert_array_equal(out1, out2)

    def test_params_count(self):
        model = GCN(6, [16, 8], 3)
        params = model.params
        assert len(params) > 0


class TestGAT:
    def test_forward(self):
        A, A_hat, X = _sample_data()
        model = GAT(6, 8, 3, num_heads=2, num_layers=2)
        out = model.forward(A, X)
        assert out.shape == (10, 3)

    def test_backward(self):
        A, A_hat, X = _sample_data()
        model = GAT(6, 8, 3, num_heads=2, num_layers=2)
        out = model.forward(A, X)
        model.backward(np.ones_like(out) / out.size)
        assert np.any(model.grads[0] != 0)


class TestGraphSAGE:
    def test_forward(self):
        A, A_hat, X = _sample_data()
        model = GraphSAGE(6, [16, 8], 3, aggregator="mean")
        out = model.forward(A, X)
        assert out.shape == (10, 3)

    def test_max_aggregator(self):
        A, A_hat, X = _sample_data()
        model = GraphSAGE(6, [16], 3, aggregator="max")
        out = model.forward(A, X)
        assert out.shape == (10, 3)


class TestAutoencoder:
    def test_encode_shape(self):
        A, A_hat, X = _sample_data()
        model = GraphAutoencoder(6, 16, 8)
        Z = model.encode(A_hat, X)
        assert Z.shape == (10, 8)

    def test_decode_shape(self):
        model = GraphAutoencoder(6, 16, 8)
        Z = np.random.randn(10, 8).astype(np.float32)
        A_recon = model.decode(Z)
        assert A_recon.shape == (10, 10)
        # Sigmoid output: all between 0 and 1
        assert np.all(A_recon >= 0) and np.all(A_recon <= 1)

    def test_forward(self):
        A, A_hat, X = _sample_data()
        model = GraphAutoencoder(6, 16, 8)
        A_recon, Z = model.forward(A_hat, X)
        assert A_recon.shape == (10, 10)
        assert Z.shape == (10, 8)

    def test_loss(self):
        A, A_hat, X = _sample_data()
        model = GraphAutoencoder(6, 16, 8)
        A_recon, Z = model.forward(A_hat, X)
        loss = model.loss(A, A_recon)
        assert loss > 0

    def test_variational(self):
        A, A_hat, X = _sample_data()
        model = GraphAutoencoder(6, 16, 8, variational=True)
        A_recon, Z = model.forward(A_hat, X)
        loss = model.loss(A, A_recon)
        assert "kl_loss" in model._cache

    def test_anomaly_scores(self):
        A, A_hat, X = _sample_data()
        model = GraphAutoencoder(6, 16, 8)
        scores = model.anomaly_scores(A_hat, A, X)
        assert "node_scores" in scores
        assert "graph_score" in scores
        assert len(scores["node_scores"]) == 10
