#! /bin/bash
#SBATCH --job-name=AOPSD
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --cpus-per-gpu=8

###############################

bash scripts/run_opsd_1b.sh

###############################