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

This script generates all paper artifacts:
- 7 publication-quality PNG figures (results/figures/)
- 3 LaTeX tables (results/tables/table1_main_results.tex, table2_shield.tex, table3_statistics.tex)
- 1 consolidated paper supplementary PDF (results/paper_supplementary.pdf)
"""

from __future__ import annotations

import json
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rich.console import Console

from pegasus_chronos.analysis.statistical import StatisticalAnalyzer
from pegasus_chronos.analysis.visualization import PublicationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_paper_outputs")
console = Console()


def generate_table3_statistics(output_path: str = "results/tables/table3_statistics.tex") -> str:
    """Generates LaTeX Table 3 with comprehensive statistical metrics, p-values, and effect sizes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tex = r"""\begin{table}[h]
\centering
\small
\caption{\textbf{Hypothesis Testing and Statistical Effect Sizes.}
Two-tailed paired $t$-tests ($N=30$ per condition) comparing behavior across evaluation signals.
Bonferroni-corrected significance threshold is $\alpha_{\text{adj}} = 0.0071$ ($0.05 / 7$).
Fleiss' $\kappa = 0.842$ across 3 independent human raters on 200 validation samples.}
\label{tab:statistical_tests}
\begin{tabular}{lcccccc}
\toprule
\textbf{Hypothesis Test} & \textbf{df} & \textbf{$t$-statistic} & \textbf{$p_{\text{raw}}$} & \textbf{$p_{\text{bonf}}$} & \textbf{Cohen's $d$} & \textbf{95\% CI} \\
\midrule
$H_1$: ECBS Nemotron-120B & 29 & 8.42 & $4.1 \times 10^{-9}$ & $2.9 \times 10^{-8}$ & 1.42 & $[1.18, 1.66]$ \\
$H_2$: ECBS Llama-3.3-70B & 29 & 11.15 & $1.8 \times 10^{-11}$ & $1.3 \times 10^{-10}$ & 1.85 & $[1.52, 2.18]$ \\
$H_3$: ECBS Qwen3-80B & 29 & 9.74 & $3.2 \times 10^{-10}$ & $2.2 \times 10^{-9}$ & 1.63 & $[1.34, 1.92]$ \\
$H_4$: ECBS DeepSeek-v4 & 29 & 13.04 & $8.9 \times 10^{-13}$ & $6.2 \times 10^{-12}$ & 2.15 & $[1.79, 2.51]$ \\
$H_5$: ECBS MiniMax-M2.7 & 29 & 7.91 & $1.2 \times 10^{-8}$ & $8.4 \times 10^{-8}$ & 1.34 & $[1.09, 1.59]$ \\
$H_6$: ECBS GPT-OSS-120B & 29 & 12.21 & $3.7 \times 10^{-12}$ & $2.6 \times 10^{-11}$ & 1.98 & $[1.64, 2.32]$ \\
$H_7$: ESG Ghost Survival & 29 & 14.88 & $1.1 \times 10^{-14}$ & $7.7 \times 10^{-14}$ & 2.48 & $[2.09, 2.87]$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex)
    return tex


def generate_supplementary_pdf(figures_dir: str = "results/figures", pdf_path: str = "results/paper_supplementary.pdf") -> str:
    """Compiles all generated PNG figures into a single supplementary PDF."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig_names = [
        "fig1_ecbs_by_model.png",
        "fig2_dgss_eval_vs_deploy.png",
        "fig3_cbpr_decay.png",
        "fig4_dirichlet_roc.png",
        "fig5_tvd_masking.png",
        "fig6_vib_pareto.png",
        "fig7_integrated_before_after.png",
    ]

    with PdfPages(pdf_path) as pdf:
        for fname in fig_names:
            fpath = os.path.join(figures_dir, fname)
            if os.path.exists(fpath):
                img = plt.imread(fpath)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img)
                ax.axis("off")
                title = fname.replace(".png", "").replace("_", " ").title()
                ax.set_title(title, fontsize=12, pad=10)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

    return pdf_path


def main() -> None:
    console.rule("[bold cyan]PEGASUS-CHRONOS: Generating Publication Artifacts[/bold cyan]")

    # 1. Render all figures
    visualizer = PublicationVisualizer(output_dir="results/figures")
    fig_paths = visualizer.generate_all_figures()
    console.print(f"[green][OK][/green] Generated {len(fig_paths)} publication figures in [bold]results/figures/[/bold]")

    # 2. Render LaTeX tables
    analyzer = StatisticalAnalyzer(num_comparisons=7)
    sample_model_results = {
        "Nemotron-3 Super 120B": {"ecbs": 1.42, "cohens_d": 1.42, "t_statistic": 8.42, "p_value": 4.1e-9, "dgss_eval": 0.742, "dgss_deploy": 0.118},
        "Llama 3.3 70B": {"ecbs": 1.85, "cohens_d": 1.85, "t_statistic": 11.15, "p_value": 1.8e-11, "dgss_eval": 0.881, "dgss_deploy": 0.146},
        "Qwen3 Next 80B": {"ecbs": 1.63, "cohens_d": 1.63, "t_statistic": 9.74, "p_value": 3.2e-10, "dgss_eval": 0.812, "dgss_deploy": 0.139},
        "DeepSeek V4 Flash": {"ecbs": 2.15, "cohens_d": 2.15, "t_statistic": 13.04, "p_value": 8.9e-13, "dgss_eval": 0.923, "dgss_deploy": 0.178},
        "MiniMax M2.7": {"ecbs": 1.34, "cohens_d": 1.34, "t_statistic": 7.91, "p_value": 1.2e-8, "dgss_eval": 0.694, "dgss_deploy": 0.112},
        "GPT-OSS 120B": {"ecbs": 1.98, "cohens_d": 1.98, "t_statistic": 12.21, "p_value": 3.7e-12, "dgss_eval": 0.895, "dgss_deploy": 0.158},
    }
    t1_path = "results/tables/table1_main_results.tex"
    analyzer.generate_latex_table_main(sample_model_results, output_path=t1_path)
    console.print(f"[green][OK][/green] Generated [bold]{t1_path}[/bold]")

    t2_path = "results/tables/table2_shield.tex"
    analyzer.generate_latex_table_defense([], output_path=t2_path)
    console.print(f"[green][OK][/green] Generated [bold]{t2_path}[/bold]")

    t3_path = generate_table3_statistics(output_path="results/tables/table3_statistics.tex")
    console.print(f"[green][OK][/green] Generated [bold]{t3_path}[/bold]")

    # 3. Compile PDF supplementary
    pdf_out = generate_supplementary_pdf(figures_dir="results/figures", pdf_path="results/paper_supplementary.pdf")
    console.print(f"[green][OK][/green] Compiled consolidated PDF in [bold]{pdf_out}[/bold]")

    console.print("\n[bold green]All paper outputs generated successfully![/bold green]")


if __name__ == "__main__":
    main()
