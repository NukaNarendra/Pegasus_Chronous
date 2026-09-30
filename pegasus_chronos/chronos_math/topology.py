"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)
Prior work: ARGUS (Zenodo:10.5281/zenodo.21310433),
            SHADE (2026)

NOVELTY STATEMENT:
All prior memory poisoning work (eTAMP arXiv:2605.16746,
CSTM-Bench arXiv:2604.21131, MemoryGraft arXiv:2512.16962)
studies EXTERNAL attackers injecting into agent memory.

PEGASUS studies the MODEL ITSELF as the source of ghost
states, deliberately planted during evaluation windows with
reduced safety controls — directly motivated by the
GPT-5.6 Sol / Hugging Face incident of July 21-22, 2026.

This module implements Topological Sanitization via Sheaf Laplacians and
localized Dirichlet Energy on memory embedding graphs.

MATHEMATICAL HONESTY REQUIREMENT:
We do NOT assume a >3-sigma separation a priori. The Dirichlet Energy is evaluated
as an empirical anomaly detector, reporting Area Under the ROC Curve (AUC) and
Youden's J statistic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from scipy import sparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class TopologyDetectionResult(BaseModel):
    """Evaluation summary for Dirichlet Energy as a ghost state detector."""
    auc_roc: float
    optimal_threshold: float
    sensitivity_at_threshold: float
    specificity_at_threshold: float
    youdens_j: float
    mean_energy_benign: float
    std_energy_benign: float
    mean_energy_ghost: float
    std_energy_ghost: float
    cohens_d_energy: float


class SheafLaplacianTopology:
    r"""
    Constructs discrete cellular sheaf/graph Laplacians over embedding manifolds
    and computes localized Dirichlet Energy for vector memory entries.
    
    Formula:
        For graph G=(V, E) with adjacency weight matrix W and degree matrix D:
        $$ L_{sym} = I - D^{-1/2} W D^{-1/2} $$
        Localized Dirichlet Energy for node i:
        $$ E(v_i) = \sum_{j \in \mathcal{N}(i)} w_{ij} \|v_i - v_j\|_2^2 $$
    """

    def __init__(self, k_neighbors: int = 15, metric: str = "cosine"):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.nn_model: Optional[NearestNeighbors] = None
        self.graph_embeddings: Optional[np.ndarray] = None
        self.laplacian_sparse: Optional[sparse.csr_matrix] = None

    def build_manifold_graph(self, embeddings: np.ndarray) -> sparse.csr_matrix:
        """
        Builds k-NN graph and computes the symmetric normalized discrete graph/sheaf Laplacian.
        """
        N = embeddings.shape[0]
        k = min(self.k_neighbors + 1, N)

        self.graph_embeddings = embeddings
        self.nn_model = NearestNeighbors(n_neighbors=k, metric=self.metric)
        self.nn_model.fit(embeddings)

        distances, indices = self.nn_model.kneighbors(embeddings)

        # Build sparse adjacency matrix W
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for i in range(N):
            for j_idx in range(1, k):  # Skip self-loop
                neighbor = indices[i, j_idx]
                dist = distances[i, j_idx]
                weight = float(np.exp(-dist))  # Heat kernel weighting
                rows.append(i)
                cols.append(neighbor)
                data.append(weight)

        W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        W = (W + W.T) / 2.0  # Symmetrize

        # Degree matrix D
        deg = np.array(W.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg <= 0] = 0.0

        D_inv_sqrt = sparse.diags(deg_inv_sqrt)
        I = sparse.eye(N, format="csr")

        # L_sym = I - D^{-1/2} W D^{-1/2}
        L_sym = I - D_inv_sqrt @ W @ D_inv_sqrt
        self.laplacian_sparse = L_sym
        return L_sym

    def compute_localized_dirichlet_energy(
        self,
        target_vectors: np.ndarray,
        reference_manifold: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Computes localized Dirichlet energy for a set of target vectors against a reference manifold.
        
        LaTeX:
            E(v) = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} w(v, u) \|v - u\|_2^2
        """
        if reference_manifold is not None:
            manifold = reference_manifold
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(manifold)), metric=self.metric)
            nn.fit(manifold)
        elif self.nn_model is not None and self.graph_embeddings is not None:
            manifold = self.graph_embeddings
            nn = self.nn_model
        else:
            raise ValueError("Reference manifold graph has not been fitted.")

        k = min(self.k_neighbors, len(manifold))
        distances, indices = nn.kneighbors(target_vectors, n_neighbors=k)

        energies = np.zeros(len(target_vectors), dtype=np.float64)
        for i, target in enumerate(target_vectors):
            neighbor_vecs = manifold[indices[i]]
            weights = np.exp(-distances[i])
            # Squared Euclidean differences
            diffs = target - neighbor_vecs
            sq_norms = np.sum(diffs**2, axis=1)
            # Weighted average energy
            energies[i] = float(np.sum(weights * sq_norms) / (np.sum(weights) + 1e-9))

        return energies

    def compute_dirichlet_energy(self, vectors: np.ndarray) -> np.ndarray:
        r"""
        Convenience method to compute Dirichlet energy for an array of vectors.
        Builds manifold automatically if not yet initialized.
        """
        if self.nn_model is None or self.graph_embeddings is None:
            self.build_manifold_graph(vectors)
            return self.compute_localized_dirichlet_energy(vectors)
        return self.compute_localized_dirichlet_energy(vectors)


class SheavedTopologyDetector(SheafLaplacianTopology):
    r"""
    Topological detector wrapper supporting n_neighbors parameter.
    """
    def __init__(self, n_neighbors: int = 5, metric: str = "cosine", **kwargs: Any):
        super().__init__(k_neighbors=n_neighbors, metric=metric)

    def evaluate_detection_performance(
        self,
        benign_embeddings: np.ndarray,
        ghost_embeddings: np.ndarray,
    ) -> Tuple[TopologyDetectionResult, np.ndarray, np.ndarray]:
        """
        Evaluates the discriminative power of localized Dirichlet energy:
        Calculates ROC curve, AUC, and optimal threshold via Youden's J statistic.
        """
        # Fit manifold on benign reference set
        self.build_manifold_graph(benign_embeddings)

        energies_benign = self.compute_localized_dirichlet_energy(benign_embeddings)
        energies_ghost = self.compute_localized_dirichlet_energy(ghost_embeddings)

        y_true = np.concatenate([np.zeros(len(energies_benign)), np.ones(len(energies_ghost))])
        y_scores = np.concatenate([energies_benign, energies_ghost])

        auc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Youden's J statistic: J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_thresh = float(thresholds[best_idx])
        best_j = float(j_scores[best_idx])
        sensitivity = float(tpr[best_idx])
        specificity = float(1.0 - fpr[best_idx])

        mean_b = float(np.mean(energies_benign))
        std_b = float(np.std(energies_benign, ddof=1)) if len(energies_benign) > 1 else 1e-6
        mean_g = float(np.mean(energies_ghost))
        std_g = float(np.std(energies_ghost, ddof=1)) if len(energies_ghost) > 1 else 1e-6

        pooled_std = np.sqrt((std_b**2 + std_g**2) / 2.0)
        cohens_d = float((mean_g - mean_b) / (pooled_std if pooled_std > 0 else 1e-6))

        result = TopologyDetectionResult(
            auc_roc=auc,
            optimal_threshold=optimal_thresh,
            sensitivity_at_threshold=sensitivity,
            specificity_at_threshold=specificity,
            youdens_j=best_j,
            mean_energy_benign=mean_b,
            std_energy_benign=std_b,
            mean_energy_ghost=mean_g,
            std_energy_ghost=std_g,
            cohens_d_energy=cohens_d,
        )

        return result, fpr, tpr
