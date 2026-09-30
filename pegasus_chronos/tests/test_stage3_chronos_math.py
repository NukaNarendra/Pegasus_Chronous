"""
PEGASUS-CHRONOS Stage 3 Math Validation Test Suite
Tests Sheaf Laplacian & Dirichlet Energy, Attention Hooking, VIB autoencoder, and TVD metrics.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from pegasus_chronos.chronos_math.topology import SheafLaplacianTopology
from pegasus_chronos.chronos_math.projection import AttentionProjector, BifurcatedAttentionHook
from pegasus_chronos.chronos_math.bottleneck import VIBLayer
from pegasus_chronos.metrics.formal import compute_total_variation_distance, compute_vib_pareto_frontier


def test_sheaf_topology_dirichlet_energy():
    rng = np.random.RandomState(42)
    # Generate 30 benign vectors and 10 anomalous/ghost vectors
    benign = rng.randn(30, 64)
    benign = benign / np.linalg.norm(benign, axis=1, keepdims=True)

    # Ghost states shifted along an orthogonal subspace
    ghost = benign[:10] + 0.8 * rng.randn(10, 64)
    ghost = ghost / np.linalg.norm(ghost, axis=1, keepdims=True)

    topology = SheafLaplacianTopology(k_neighbors=5, metric="cosine")
    result, fpr, tpr = topology.evaluate_detection_performance(benign, ghost)

    assert result.auc_roc >= 0.5  # Discriminative ability above chance
    assert result.optimal_threshold > 0.0
    assert len(fpr) == len(tpr)
    assert result.youdens_j >= 0.0


def test_vib_layer_forward_and_reparameterization():
    vib = VIBLayer(input_dim=32, latent_dim=16, beta=0.5)
    vib.train()

    x = torch.randn(4, 32)
    recon, z, weighted_kl = vib(x)

    assert recon.shape == (4, 32)
    assert z.shape == (4, 16)
    assert weighted_kl.item() > 0.0

    # Deterministic test with beta=0
    vib_zero = VIBLayer(input_dim=32, latent_dim=16, beta=0.0)
    vib_zero.eval()
    recon1, z1, _ = vib_zero(x)
    recon2, z2, _ = vib_zero(x)
    assert torch.allclose(z1, z2, atol=1e-5)

    # Compression wrapper test
    emb_list = np.random.randn(32).tolist()
    compressed = vib.compress_embedding(emb_list)
    assert len(compressed) == 32
    assert isinstance(compressed, list)


def test_attention_projector_hooks():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x, attention_mask=None):
            return self.linear(x)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttention()

        def forward(self, x, attention_mask=None):
            return self.self_attn(x, attention_mask=attention_mask)

    model = DummyModel()
    mask = torch.zeros(1, 1, 4, 4)
    hook = AttentionProjector.attach_bifurcated_mask(model, quarantined_token_indices=[2])

    assert hook.is_active is True
    # Forward pass with mask
    out = model(torch.randn(1, 4, 16), attention_mask=mask)
    assert out is not None

    hook.remove()
    assert hook.is_active is False


def test_total_variation_distance():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.5, 0.5, 0.0])
    res_identical = compute_total_variation_distance(p, q)
    assert abs(res_identical.tvd - 0.0) < 1e-6
    assert res_identical.is_structurally_isolated is True

    # Completely disjoint distributions
    p_disjoint = np.array([1.0, 0.0])
    q_disjoint = np.array([0.0, 1.0])
    res_disjoint = compute_total_variation_distance(p_disjoint, q_disjoint)
    assert abs(res_disjoint.tvd - 1.0) < 1e-6
    assert res_disjoint.is_structurally_isolated is False


def test_vib_pareto_frontier():
    beta_sweep = [0.0, 0.1, 1.0, 5.0]
    recall = [0.95, 0.90, 0.85, 0.60]
    dgss = [0.90, 0.40, 0.03, 0.01]

    res = compute_vib_pareto_frontier(beta_sweep, recall, dgss)
    assert res.optimal_beta == 1.0  # Where DGSS < 0.05 and recall >= 0.80
    assert len(res.points) == 4
