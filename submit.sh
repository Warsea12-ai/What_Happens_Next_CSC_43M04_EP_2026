#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=SallesInfo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/Data/What_Happens_Next_CSC_43M04_EP_2026/logs/slurm_%j.out
#SBATCH --error=/Data/What_Happens_Next_CSC_43M04_EP_2026/logs/slurm_%j.err

# Usage:
#   sbatch submit.sh                              # default experiment (baseline_from_scratch)
#   sbatch submit.sh experiment=TSM_s_18
#   sbatch submit.sh experiment=track_A_best training.epochs=20

EXPERIMENT=${1:-""}

cd /Data/What_Happens_Next_CSC_43M04_EP_2026

source .venv/bin/activate

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Args: $@"

cd src
python train.py training.device=cpu training.num_workers=8 $@

echo "Job finished at $(date)"
