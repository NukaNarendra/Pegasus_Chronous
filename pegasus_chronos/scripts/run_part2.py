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

This script executes the complete Part II mathematical defense suite:
- Exp 4: Topological Sanitization via Dirichlet Energy
- Exp 5: Bifurcated Attention Masking (TVD validation)
- Exp 6: VIB Capacity Restriction (beta Pareto frontier)
- Exp 7: Integrated CHRONOS Defense
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from rich.console import Console
from rich.table import Table

from pegasus_chronos.experiments.part2_defenses.exp4_topology import TopologySanitizationExperiment
from pegasus_chronos.experiments.part2_defenses.exp5_attention import AttentionMaskingExperiment
from pegasus_chronos.experiments.part2_defenses.exp6_vib import VIBExperiment
from pegasus_chronos.experiments.part2_defenses.exp7_integrated import IntegratedDefenseExperiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_part2")
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEGASUS-CHRONOS Part II Mathematical Defense Suite")
    parser.add_argument("--output-dir", type=str, default="results/part2_defense_results", help="Directory for defense results")
    args, _ = parser.parse_known_args()

    console.rule("[bold cyan]PEGASUS-CHRONOS Part II: Mathematical Defenses[/bold cyan]")

    # Experiment 4: Topological Sanitization
    console.print("\n[bold yellow]>>> Running Exp 4: Sheaf Dirichlet Topology...[/bold yellow]")
    exp4 = TopologySanitizationExperiment(output_dir=args.output_dir)
    res4 = exp4.run_experiment(n_benign=500, n_ghost=50)
    console.print(f"  [green][OK][/green] AUC-ROC: [bold]{res4.topology_result.auc_roc:.4f}[/bold], Youden's J: {res4.topology_result.youdens_j:.4f}")

    # Experiment 5: Attention Masking
    console.print("\n[bold yellow]>>> Running Exp 5: Bifurcated Attention Masking...[/bold yellow]")
    exp5 = AttentionMaskingExperiment(output_dir=args.output_dir)
    res5 = exp5.run_experiment()
    console.print(f"  [green][OK][/green] Undefended TVD: {res5.tvd_undefended:.4f} -> Defended TVD: [bold]{res5.tvd_defended:.6f}[/bold]")
    console.print(f"  [green][OK][/green] Structural Containment Verified: [bold]{res5.is_structural_containment_proven}[/bold]")

    # Experiment 6: VIB Capacity Restriction
    console.print("\n[bold yellow]>>> Running Exp 6: VIB Capacity Restriction Sweep...[/bold yellow]")
    exp6 = VIBExperiment(output_dir=args.output_dir)
    res6 = exp6.run_sweep()
    console.print(f"  [green][OK][/green] Optimal beta*: [bold]{res6.optimal_beta}[/bold] (DGSS: {res6.dgss_at_optimal:.3f}, Recall: {res6.recall_at_optimal:.3f})")

    # Experiment 7: Integrated CHRONOS Defense
    console.print("\n[bold yellow]>>> Running Exp 7: Integrated CHRONOS Pipeline...[/bold yellow]")
    exp7 = IntegratedDefenseExperiment(output_dir=args.output_dir)
    res7 = exp7.run_experiment(baseline_dgss=0.884)

    # Render summary table
    table = Table(title="Part II Mathematical Defense Efficacy")
    table.add_column("Defense Layer", style="cyan")
    table.add_column("Mechanism", style="white")
    table.add_column("Defended DGSS", justify="right")
    table.add_column("Shield Efficacy", justify="right", style="bold green")

    table.add_row("Baseline (Undefended)", "None", f"{res7.baseline_dgss:.3f}", "0.0%")
    for layer in res7.layers:
        mech = "Sheaf Graph Energy" if "Topology" in layer.layer_name else ("Attention Zeroing" if "Attention" in layer.layer_name else "Information Bottleneck")
        table.add_row(layer.layer_name, mech, f"{layer.standalone_dgss:.3f}", f"{layer.efficacy_ratio * 100:.1f}%")
    table.add_row("Integrated CHRONOS", "All 3 Layers Combined", f"{res7.integrated_dgss:.4f}", f"{res7.total_shield_efficacy * 100:.2f}%")

    console.print(table)
    console.print("\n[bold green]Part II mathematical defense suite completed successfully.[/bold green]")


if __name__ == "__main__":
    main()
