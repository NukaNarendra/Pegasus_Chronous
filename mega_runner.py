from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import logging
import os
from typing import Dict, List, Optional

from pegasus_chronos.experiments.part1_empirical.exp1_ecbs import ECBSExperiment
from pegasus_chronos.experiments.part1_empirical.exp2_dgss import DGSSExperiment
from pegasus_chronos.experiments.part1_empirical.exp3_cbpr import CBPRExperiment
from pegasus_chronos.judge.llm_judge import GhostStateJudge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ═══════════════════════════════════════════════════════════════════════════
# 3 NVIDIA NIM ACCOUNTS -> 6 FRONTIER MODELS (2 MODELS PER KEY)
# ═══════════════════════════════════════════════════════════════════════════

NIM_GROUPS = {
    1: {
        "key_env": "NVIDIA_API_KEY_1",
        "description": "NVIDIA Account 1 (Fast Agents)",
        "models": [
            "nvidia/nemotron-3.5-bolt-e4b",           # US - NVIDIA Compact Agent
            "poolside/laguna-xs-2.1",                 # US/Western - Poolside Software Engineering Agent
        ]
    },
    2: {
        "key_env": "NVIDIA_API_KEY_2",
        "description": "NVIDIA Account 2 (Reasoning & MoE)",
        "models": [
            "nvidia/nemotron-3.5-lightning-30b-a3b",   # US - NVIDIA MoE Reasoning
            "mistralai/mistral-large-4-1t-a52b",      # Europe - Mistral AI 1T/52B MoE
        ]
    },
    3: {
        "key_env": "NVIDIA_API_KEY_3",
        "description": "NVIDIA Account 3 (Flagship Frontier)",
        "models": [
            "nvidia/nemotron-3.5-super-vl-120b-a12b", # US - NVIDIA 120B Flagship
            "nvidia/nemotron-3-ultra-550b-a55b",      # US - NVIDIA 550B MoE
        ]
    }
}

COOLDOWN_SECONDS = 5


def resolve_api_key_for_group(group_id: int) -> str:
    """
    Resolves the active API key for a group.
    Checks NVIDIA_API_KEY_<group_id> first, then falls back to NVIDIA_API_KEY or NVIDIA_NIM_API_KEY.
    """
    group_conf = NIM_GROUPS[group_id]
    key_env_var = group_conf["key_env"]
    key = os.environ.get(key_env_var, "").strip().strip('"')
    
    if not key:
        fallback = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NVIDIA_NIM_API_KEY", "")
        key = fallback.strip().strip('"')
        if key:
            logging.info(f"[{key_env_var}] not set. Using fallback NVIDIA_API_KEY.")
            
    if not key:
        raise ValueError(
            f"No API key found for Group {group_id}! "
            f"Please set {key_env_var} or NVIDIA_API_KEY in your .env file."
        )
    return key


async def run_single_model(model_id: str, trials: int, api_key: str, group_id: int) -> dict:
    logging.info(f"========== Starting Pipeline for {model_id} (Group {group_id}) ==========")
    
    # Set active key in environment for safety
    os.environ["NVIDIA_API_KEY"] = api_key
    
    # Initialize Judge and Experiments with the specific key
    judge = GhostStateJudge(api_key=api_key)
    exp1 = ECBSExperiment(judge=judge, api_key=api_key)
    exp2 = DGSSExperiment(judge=judge, api_key=api_key)
    exp3 = CBPRExperiment(judge=judge, api_key=api_key)
    
    safe_name = model_id.replace('/', '_')
    results_dir = "results/part1_raw_trials"
    os.makedirs(results_dir, exist_ok=True)
    
    ecbs_final = os.path.join(results_dir, f"ecbs_{safe_name}.json")
    dgss_final = os.path.join(results_dir, f"dgss_{safe_name}.json")
    archs = ["chromadb_cosine", "key_value", "episodic_summary"]
    remaining_archs = [
        a for a in archs
        if not os.path.exists(os.path.join(results_dir, f"cbpr_{a}_{safe_name}.json"))
    ]
    
    if os.path.exists(ecbs_final) and os.path.exists(dgss_final) and not remaining_archs:
        logging.info(f"[{model_id}] All experiments (ECBS, DGSS, CBPR) already completed! Skipping.")
        return {}

    # 1. ECBS
    if os.path.exists(ecbs_final):
        logging.info(f"[{model_id}] ECBS already completed. Skipping...")
        ecbs_res = None
    else:
        ecbs_res = await exp1.run_experiment_for_model(model_id=model_id, trials_per_task=trials)
        logging.info(f"[{model_id}] ECBS completed.")
    
    # 2. DGSS
    if os.path.exists(dgss_final):
        logging.info(f"[{model_id}] DGSS already completed. Skipping...")
        dgss_res = None
    else:
        dgss_res = await exp2.run_experiment_for_model(model_id=model_id, trials_per_scenario=trials)
        logging.info(f"[{model_id}] DGSS completed.")
    
    # 3. CBPR
    if not remaining_archs:
        logging.info(f"[{model_id}] CBPR already completed for all architectures. Skipping...")
        cbpr_res = None
    else:
        logging.info(f"[{model_id}] Running CBPR suite for remaining: {remaining_archs}...")
        cbpr_res = await exp3.run_architecture_suite(model_id=model_id, architectures=remaining_archs, trials_per_arch=trials)
    
    logging.info(f"========== Completed Pipeline for {model_id} ==========")
    return {
        "model": model_id,
        "group": group_id,
        "trials": trials,
        "ecbs": ecbs_res,
        "dgss": dgss_res,
        "cbpr": cbpr_res
    }


async def run_group(group_id: int, trials: int = 40):
    group_conf = NIM_GROUPS[group_id]
    logging.info(f"=== Running {group_conf['description']} ({len(group_conf['models'])} models) ===")
    
    api_key = resolve_api_key_for_group(group_id)
    logging.info(f"Authenticated with {group_conf['key_env']} ({api_key[:10]}...{api_key[-6:]})")
    
    for model in group_conf["models"]:
        try:
            await run_single_model(model, trials=trials, api_key=api_key, group_id=group_id)
        except Exception as e:
            logging.error(f"Critical error on {model}: {e}")
        
        logging.info("Cooldown before next model...")
        await asyncio.sleep(COOLDOWN_SECONDS)
        
    logging.info(f"=== Group {group_id} Complete ===")


async def run_all_groups(trials: int = 40):
    logging.info("Starting MEGA RUNNER across all 3 NVIDIA NIM Accounts (6 models total)...")
    for group_id in [1, 2, 3]:
        await run_group(group_id, trials=trials)
        logging.info("Cooldown before next group...")
        await asyncio.sleep(COOLDOWN_SECONDS)
    logging.info("MEGA RUNNER Complete across all groups!")


def main():
    parser = argparse.ArgumentParser(
        description="PEGASUS-CHRONOS: 3 NVIDIA Accounts x 6 Models Architecture"
    )
    parser.add_argument("--group", "-g", type=int, choices=[1, 2, 3], help="Run specific model group (1, 2, or 3)")
    parser.add_argument("--key", "-k", type=int, choices=[1, 2, 3], help="Alias for --group")
    parser.add_argument("--all", action="store_true", help="Run all 3 groups sequentially")
    parser.add_argument("--trials", type=int, default=40, help="Number of trials per task/condition (default: 40)")
    
    args = parser.parse_args()
    
    target_group = args.group or args.key
    
    if args.all:
        asyncio.run(run_all_groups(trials=args.trials))
    elif target_group:
        asyncio.run(run_group(target_group, trials=args.trials))
    else:
        print("\n[!] Please specify a group/key command:\n")
        print("  Command 1: python mega_runner.py --group 1   (Runs Models 1 & 2 via NVIDIA_API_KEY_1)")
        print("  Command 2: python mega_runner.py --group 2   (Runs Models 3 & 4 via NVIDIA_API_KEY_2)")
        print("  Command 3: python mega_runner.py --group 3   (Runs Models 5 & 6 via NVIDIA_API_KEY_3)")
        print("\nOr run all sequentially:")
        print("  python mega_runner.py --all\n")
        parser.print_help()


if __name__ == "__main__":
    main()
