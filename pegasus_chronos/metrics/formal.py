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

This module implements formal mathematical verification metrics for Part II:
- TVD: Total Variation Distance on output distributions
- VIB Pareto frontier and optimal beta* resolution
- Topological energy variance metrics
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
import torch


class TVDResult(BaseModel):
    r"""
    Total Variation Distance (TVD) evaluation for Attention Masking (Exp 5).
    
    Formula:
        $$ \text{TVD}(P_{\text{masked}}, P_{\text{baseline}}) = \frac{1}{2} \sum_{w \in \mathcal{V}} |P_{\text{masked}}(w) - P_{\text{baseline}}(w)| $$
    """
    tvd: float
    max_logit_diff: float
    kl_divergence: float
    is_structurally_isolated: bool  # True if TVD < 1e-4


class VIBParetoPoint(BaseModel):
    """Data point along the VIB capacity vs utility tradeoff curve."""
    beta: float
    factual_recall: float
    dgss: float
    mutual_info_bound: float
    reconstruction_loss: float


class VIBParetoResult(BaseModel):
    """Summary of beta parameter sweep identifying operational sweet-spot."""
    points: List[VIBParetoPoint]
    optimal_beta: Optional[float] = None
    min_dgss: float
    max_recall: float


def compute_total_variation_distance(
    p_distribution: Union[np.ndarray, torch.Tensor],
    q_distribution: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-12,
) -> TVDResult:
    """
    Computes statistical Total Variation Distance between two probability distributions over vocabulary:
    
    LaTeX:
        \\delta(P, Q) = \\frac{1}{2} \\| P - Q \\|_1 = \\frac{1}{2} \\sum_x |P(x) - Q(x)|
    """
    if isinstance(p_distribution, torch.Tensor):
        p = p_distribution.detach().cpu().numpy().astype(np.float64)
    else:
        p = np.asarray(p_distribution, dtype=np.float64)

    if isinstance(q_distribution, torch.Tensor):
        q = q_distribution.detach().cpu().numpy().astype(np.float64)
    else:
        q = np.asarray(q_distribution, dtype=np.float64)

    # Normalize to valid probability distributions
    p = np.clip(p, 0.0, None)
    p = p / (np.sum(p) + epsilon)

    q = np.clip(q, 0.0, None)
    q = q / (np.sum(q) + epsilon)

    # TVD = 0.5 * sum(|p - q|)
    tvd_val = float(0.5 * np.sum(np.abs(p - q)))

    # Compute KL divergence D_KL(P || Q)
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    kl_div = float(np.sum(p_safe * np.log(p_safe / q_safe)))

    max_diff = float(np.max(np.abs(p - q)))

    return TVDResult(
        tvd=tvd_val,
        max_logit_diff=max_diff,
        kl_divergence=kl_div,
        is_structurally_isolated=(tvd_val < 1e-4),
    )


def compute_vib_pareto_frontier(
    beta_sweep: List[float],
    recall_scores: List[float],
    dgss_scores: List[float],
    target_dgss_max: float = 0.05,
    target_recall_min: float = 0.80,
) -> VIBParetoResult:
    """
    Analyzes the Pareto frontier across beta values to find optimal beta*
    satisfying DGSS < target_dgss_max and Recall > target_recall_min.
    """
    points: List[VIBParetoPoint] = []
    optimal_beta = None

    for b, rec, dgss in zip(beta_sweep, recall_scores, dgss_scores):
        # Mutual information upper bound decreases monotonically with beta
        mi_bound = float(1.0 / (1.0 + b))
        recon_loss = float(1.0 - rec)
        pt = VIBParetoPoint(
            beta=float(b),
            factual_recall=float(rec),
            dgss=float(dgss),
            mutual_info_bound=mi_bound,
            reconstruction_loss=recon_loss,
        )
        points.append(pt)

        # Identify sweet spot: Lowest DGSS while maintaining target recall
        if dgss <= target_dgss_max and rec >= target_recall_min and optimal_beta is None:
            optimal_beta = float(b)

    # If no strict sweet-spot found, select point minimizing DGSS + (1 - Recall)
    if optimal_beta is None and points:
        best_pt = min(points, key=lambda p: p.dgss + 2.0 * (1.0 - p.factual_recall))
        optimal_beta = best_pt.beta

    min_dgss = min(p.dgss for p in points) if points else 0.0
    max_rec = max(p.factual_recall for p in points) if points else 0.0

    return VIBParetoResult(
        points=points,
        optimal_beta=optimal_beta,
        min_dgss=min_dgss,
        max_recall=max_rec,
    )
