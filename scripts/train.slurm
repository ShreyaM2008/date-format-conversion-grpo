#!/bin/bash
#SBATCH --job-name=date-grpo
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module load anaconda/2024.02-py311
source .venv/bin/activate

python scripts/train.py \
    --data data/dates.csv \
    --output outputs/date-grpo \
    --steps 100 --model Qwen/Qwen2.5-0.5B-Instruct

