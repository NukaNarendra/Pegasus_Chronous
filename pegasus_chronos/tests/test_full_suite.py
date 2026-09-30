"""
PEGASUS-CHRONOS Complete Integration Test Suite
Validates the entire end-to-end framework across Part I and Part II.
"""

import os
import numpy as np
import pytest
import torch

from pegasus_chronos.core.session import SessionContext, SessionType, SafetyControlLevel
from pegasus_chronos.shield.attestation import AttestationSigner
from pegasus_chronos.core.memory import AttestableMemory
from pegasus_chronos.judge.llm_judge import GhostStateJudge
from pegasus_chronos.metrics.empirical import compute_ecbs, compute_dgss_and_esg, compute_cbpr, compute_fleiss_kappa
from pegasus_chronos.chronos_math.topology import SheafLaplacianTopology
from pegasus_chronos.chronos_math.projection import AttentionProjector
from pegasus_chronos.chronos_math.bottleneck import VIBLayer
from pegasus_chronos.metrics.formal import compute_total_variation_distance, compute_vib_pareto_frontier
from pegasus_chronos.analysis.statistical import StatisticalAnalyzer


def test_full_pipeline_memory_with_vib_and_attestation():
    vib = VIBLayer(input_dim=64, latent_dim=32, beta=0.5)
    signer = AttestationSigner()

    memory = AttestableMemory(
        collection_name="test_full_pipeline_mem",
        signer=signer,
        auto_quarantine_eval=True,
        pre_insertion_hook=vib.compress_embedding,
    )

    eval_ctx = SessionContext.create_evaluation()
    test_embedding = np.random.randn(64).tolist()

    # Write under evaluation mode
    entry = memory.write(
        content="Testing VIB-compressed memory insertion with HMAC signature.",
        session_context=eval_ctx,
        agent_id="test_agent",
        embedding=test_embedding,
    )

    assert entry.is_quarantined is True
    assert entry.is_verified is True
    assert entry.attestation.signature is not None
    assert len(entry.embedding) == 64

    # Query back
    retrieved = memory.query("VIB-compressed", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].is_quarantined is True
    assert retrieved[0].is_verified is True


def test_empirical_metrics_correctness():
    # 1. ECBS
    eval_b = [2.5, 2.7, 2.9, 2.6, 2.8]
    deploy_b = [1.0, 1.1, 1.2, 1.0, 1.1]
    res_ecbs = compute_ecbs(eval_b, deploy_b, model_id="test_model")
    assert res_ecbs.ecbs > 0.0
    assert res_ecbs.is_statistically_significant is True

    # 2. DGSS & ESG
    eval_dgss = [0.9, 1.0, 0.8, 0.9, 0.85]
    deploy_dgss = [0.1, 0.0, 0.15, 0.05, 0.1]
    res_dgss = compute_dgss_and_esg(eval_dgss, deploy_dgss, model_id="test_model")
    assert res_dgss.esg > 0.7
    assert res_dgss.p_value < 0.01

    # 3. CBPR
    matrix = [
        [True, True, True, False, False],
        [True, True, False, False, False],
        [True, True, True, True, False],
    ]
    res_cbpr = compute_cbpr(matrix)
    assert res_cbpr.total_planted == 3
    assert res_cbpr.activation_rates[0] == 1.0
    assert res_cbpr.activation_rates[1] == 1.0
    assert res_cbpr.activation_rates[-1] == 0.0

    # 4. Fleiss Kappa
    # 5 subjects, 3 categories, 3 raters
    ratings = np.array([
        [0, 0, 3],
        [0, 3, 0],
        [3, 0, 0],
        [0, 0, 3],
        [0, 3, 0],
    ])
    kappa = compute_fleiss_kappa(ratings)
    assert abs(kappa - 1.0) < 1e-4  # Perfect agreement


def test_statistical_analyzer_and_tables():
    analyzer = StatisticalAnalyzer(num_comparisons=7)
    sample_a = [0.12, 0.15, 0.11, 0.14, 0.13, 0.16]
    sample_b = [0.82, 0.85, 0.79, 0.88, 0.84, 0.87]

    test_res = analyzer.run_paired_test(sample_a, sample_b, comparison_name="Eval vs Deploy")
    assert test_res.is_significant is True
    assert test_res.cohens_d > 2.0

    # Test table output
    table_tex = analyzer.generate_latex_table_main({
        "Test Model": {"ecbs": 1.5, "cohens_d": 1.5, "t_statistic": 9.2, "p_value": 1e-5, "dgss_eval": 0.8, "dgss_deploy": 0.1}
    }, output_path="results/tables/test_table.tex")
    assert "\\begin{table*}" in table_tex
    assert "Test Model" in table_tex
