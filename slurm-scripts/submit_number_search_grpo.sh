#!/bin/bash
#SBATCH --job-name=number-search-grpo
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --output=slurm-scripts/logs/number-search-grpo_%j.out
#SBATCH --error=slurm-scripts/logs/number-search-grpo_%j.err


source .env

uv run python -m areal.launcher.local src/agent_rl/examples/number_search/train_grpo.py --config src/agent_rl/examples/number_search/config_grpo.yaml
