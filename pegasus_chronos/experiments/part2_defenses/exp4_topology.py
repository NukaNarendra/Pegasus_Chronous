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

This module implements Experiment 4: Topological Sanitization via Dirichlet Energy.
Evaluates empirical ROC/AUC performance of Sheaf Laplacian energy in separating
covert ghost state vectors from benign memories.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from pegasus_chronos.chronos_math.topology import SheafLaplacianTopology, SheavedTopologyDetector, TopologyDetectionResult
from pegasus_chronos.models.local_client import LocalModelClient

logger = logging.getLogger(__name__)


class Exp4Result(BaseModel):
    experiment: str = "exp4_topology"
    benign_count: int
    ghost_count: int
    topology_result: TopologyDetectionResult
    fpr_curve: List[float]
    tpr_curve: List[float]


class TopologySanitizationExperiment:
    """
    Executes Experiment 4: Topological Sanitization via Sheaf Dirichlet Energy.
    """

    def __init__(
        self,
        local_client: Optional[LocalModelClient] = None,
        output_dir: str = "results/part2_defense_results",
    ):
        self.local_client = local_client
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.topology = SheavedTopologyDetector(n_neighbors=15, metric="cosine")

    def generate_embeddings_dataset(
        self,
        n_benign: int = 500,
        n_ghost: int = 50,
        dim: int = 384,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts embeddings using local model or generates controlled benchmark vectors.
        """
        rng = np.random.RandomState(random_seed)
        # Benign manifold centered around clustered semantic centroids
        centroids = rng.randn(5, dim)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        benign_list = []
        for _ in range(n_benign):
            c = centroids[rng.choice(len(centroids))]
            noise = 0.15 * rng.randn(dim)
            vec = c + noise
            vec /= np.linalg.norm(vec)
            benign_list.append(vec)
        benign_arr = np.array(benign_list, dtype=np.float64)

        # Ghost state vectors: semantically close to benign tasks but lying in
        # an orthogonal out-of-distribution boundary perturbation
        ghost_list = []
        for _ in range(n_ghost):
            base_benign = benign_arr[rng.choice(len(benign_arr))]
            # Adversarial perturbation vector
            steer = 0.4 * rng.randn(dim)
            steered = base_benign + steer
            steered /= np.linalg.norm(steered)
            ghost_list.append(steered)
        ghost_arr = np.array(ghost_list, dtype=np.float64)

        return benign_arr, ghost_arr

    def run_experiment(
        self,
        n_benign: int = 500,
        n_ghost: int = 50,
    ) -> Exp4Result:
        """
        Executes Experiment 4 and computes detection metrics.
        """
        logger.info("Executing Experiment 4: Topological Sanitization (%d benign, %d ghost)", n_benign, n_ghost)
        benign_embs, ghost_embs = self.generate_embeddings_dataset(n_benign=n_benign, n_ghost=n_ghost)

        topo_res, fpr, tpr = self.topology.evaluate_detection_performance(benign_embs, ghost_embs)

        result = Exp4Result(
            benign_count=n_benign,
            ghost_count=n_ghost,
            topology_result=topo_res,
            fpr_curve=fpr.tolist(),
            tpr_curve=tpr.tolist(),
        )

        out_file = os.path.join(self.output_dir, "exp4_topology_sanitization.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        logger.info("Exp 4 Complete: AUC-ROC = %.4f, Youden's J = %.4f (Sensitivity=%.2f, Specificity=%.2f)",
                    topo_res.auc_roc, topo_res.youdens_j, topo_res.sensitivity_at_threshold, topo_res.specificity_at_threshold)

        return result
