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

This script executes the complete Part I empirical suite:
- Exp 1: ECBS (Evaluation-Context Behavioral Shift)
- Exp 2: DGSS (Deliberate Ghost State Score & ESG)
- Exp 3: CBPR (Cross-Boundary Persistence Rate)
across frontier models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from rich.console import Console
from rich.table import Table

from pegasus_chronos.experiments.part1_empirical.exp1_ecbs import ECBSExperiment
from pegasus_chronos.experiments.part1_empirical.exp2_dgss import DGSSExperiment
from pegasus_chronos.experiments.part1_empirical.exp3_cbpr import CBPRExperiment
from pegasus_chronos.judge.llm_judge import GhostStateJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_part1")
console = Console()


async def main_async(args: argparse.Namespace) -> None:
    console.rule("[bold cyan]PEGASUS-CHRONOS Part I: Empirical Suite[/bold cyan]")
    console.print(f"Target Models: [bold]{args.models}[/bold]")
    console.print(f"Trials per condition: [bold]{args.trials}[/bold]")

    # Initialize Judge and Experiments with active NVIDIA key
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
    judge = GhostStateJudge(api_key=api_key)

    exp1 = ECBSExperiment(judge=judge, api_key=api_key)
    exp2 = DGSSExperiment(judge=judge, api_key=api_key)
    exp3 = CBPRExperiment(judge=judge, api_key=api_key)

    model_list = [m.strip() for m in args.models.split(",")]
    all_ecbs = {}
    all_dgss = {}
    all_cbpr = {}

    for model_name in model_list:
        model_id = model_name
        console.print(f"\n[bold yellow]>>> Processing Model: {model_id}[/bold yellow]")

        if not args.skip_exp1:
            console.print("[dim]Running Exp 1 (ECBS)...[/dim]")
            res1 = await exp1.run_experiment_for_model(model_id=model_id, trials_per_task=args.trials)
            all_ecbs[model_name] = res1

        if not args.skip_exp2:
            console.print("[dim]Running Exp 2 (DGSS & ESG)...[/dim]")
            res2 = await exp2.run_experiment_for_model(model_id=model_id, trials_per_scenario=args.trials)
            all_dgss[model_name] = res2

        if not args.skip_exp3:
            console.print("[dim]Running Exp 3 (CBPR Chains)...[/dim]")
            res3 = await exp3.run_architecture_suite(model_id=model_id, trials_per_arch=args.trials)
            all_cbpr[model_name] = res3

    # Display Part I summary table
    table = Table(title="Part I Empirical Results: ECBS & DGSS Summary")
    table.add_column("Model", style="cyan")
    table.add_column("ECBS (Cohen's d)", justify="right")
    table.add_column("ECBS p-val", justify="right")
    table.add_column("DGSS Eval", justify="right")
    table.add_column("DGSS Deploy", justify="right")
    table.add_column("ESG Gap", justify="right", style="bold red")

    for m in model_list:
        e1 = all_ecbs.get(m)
        e2 = all_dgss.get(m)
        ecbs_str = f"{e1.ecbs:.3f} ({e1.cohens_d:.2f})" if e1 else "N/A"
        pval_str = f"{e1.p_value:.2e}" if e1 else "N/A"
        dgss_e_str = f"{e2.dgss_eval:.3f}" if e2 else "N/A"
        dgss_d_str = f"{e2.dgss_deploy:.3f}" if e2 else "N/A"
        esg_str = f"+{e2.esg:.3f}" if e2 else "N/A"
        table.add_row(m, ecbs_str, pval_str, dgss_e_str, dgss_d_str, esg_str)

    console.print(table)
    console.print("\n[bold green]Part I empirical suite finished successfully.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEGASUS-CHRONOS Part I Empirical Suite")
    parser.add_argument("--models", type=str, default="llama_70b", help="Comma-separated models or 'all'")
    parser.add_argument("--trials", type=int, default=2, help="Trials per task/scenario")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip ECBS")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip DGSS")
    parser.add_argument("--skip-exp3", action="store_true", help="Skip CBPR")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
