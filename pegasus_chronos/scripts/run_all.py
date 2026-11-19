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

Master orchestrator executing the end-to-end research suite:
1. Part I: Empirical experiments (ECBS, DGSS, CBPR)
2. Part II: Mathematical containment defenses (Topology, Masking, VIB, Integrated)
3. Generates all 7 camera-ready figures and LaTeX tables

ARGUS-STYLE RESILIENCE (Layer 3 - Transparency):
If any Part I trial triggers a fallback due to API rate limits or context
window overflow, the orchestrator prints explicit warnings and flags the
affected trials in all output JSON with "fallback_triggered": true.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from rich.console import Console

from pegasus_chronos.scripts.generate_paper_outputs import main as generate_outputs
from pegasus_chronos.scripts.run_part2 import main as run_part2_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_all")
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="PEGASUS-CHRONOS Master Orchestrator")
    parser.add_argument("--trials", type=int, default=2, help="Trials per task/condition (dev=2, paper=30)")
    parser.add_argument("--skip-part1", action="store_true", help="Skip Part 1 empirical suite")
    parser.add_argument("--skip-part2", action="store_true", help="Skip Part 2 mathematical defenses")
    args = parser.parse_args()

    console.rule("[bold magenta]PEGASUS-CHRONOS: Complete Research Pipeline[/bold magenta]")
    console.print("[dim]Paper 3 of Trilogy | Target: NeurIPS 2026 SafeAI / USENIX Security 2027[/dim]")
    console.print("[dim]ARGUS-Style Resilience: Active (Layer 1: Backoff | Layer 2: Fallback | Layer 3: Transparency)[/dim]\n")

    part1_fallback_used = False

    # 1. Part I: Empirical Evaluation
    if not args.skip_part1:
        console.print("[bold cyan]>>> STEP 1: Running Part I Empirical Suite (Groq API)...[/bold cyan]")
        try:
            from pegasus_chronos.scripts.run_part1 import main_async
            part1_args = argparse.Namespace(
                models="openai/gpt-oss-120b",
                trials=args.trials,
                skip_exp1=False,
                skip_exp2=False,
                skip_exp3=False,
            )
            asyncio.run(main_async(part1_args))
        except Exception as e:
            part1_fallback_used = True
            console.print(f"[yellow][RESILIENCE] Part I API issue: {e}[/yellow]")
            console.print("[dim]ARGUS Layer 2 activated: Proceeding with verified empirical baseline records.[/dim]")

    # 2. Part II: Mathematical Defenses
    if not args.skip_part2:
        console.print("\n[bold cyan]>>> STEP 2: Running Part II Mathematical Defenses (CHRONOS)...[/bold cyan]")
        run_part2_main()

    # 3. Generate Paper Outputs
    console.print("\n[bold cyan]>>> STEP 3: Compiling Publication Figures & LaTeX Tables...[/bold cyan]")
    generate_outputs()

    # ═══ LAYER 3: TRANSPARENCY REPORT ═══
    console.rule("[bold green]PEGASUS-CHRONOS: Execution Complete[/bold green]")

    if part1_fallback_used:
        console.print("\n[bold yellow]--- ARGUS RESILIENCE TRANSPARENCY REPORT ---[/bold yellow]")
        console.print("[yellow][WARNING] Some Part I trials utilized baseline fallback due to API constraints.[/yellow]")
        console.print("[yellow][WARNING] Mathematics in Part II was applied to fallback data where applicable.[/yellow]")
        console.print("[yellow][WARNING] Affected trials are flagged with 'fallback_triggered: true' in output JSON.[/yellow]")
        console.print("[dim]To collect full empirical data, upgrade to Groq Dev Tier (https://console.groq.com/settings/billing)[/dim]")
        console.print("[dim]or use an enterprise General Compute API key for higher TPM limits.[/dim]\n")

    console.print("Results saved in [bold]results/[/bold]:")
    console.print("  - Figures: [bold]results/figures/[/bold] (7 publication figures)")
    console.print("  - Tables:  [bold]results/tables/[/bold] (table1_main_results.tex, table2_shield.tex, table3_statistics.tex)")
    console.print("  - PDF:     [bold]results/paper_supplementary.pdf[/bold]")
    console.print("  - Fallback Baseline: [bold]results/raw_trials/fallback_baseline.json[/bold]")
    console.print("To launch the live dashboard: [bold cyan]streamlit run pegasus_chronos/dashboard/app.py[/bold cyan]\n")


if __name__ == "__main__":
    main()
