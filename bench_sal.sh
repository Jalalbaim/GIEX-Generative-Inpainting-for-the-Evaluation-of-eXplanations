#!/usr/bin/env bash

# Specify resource requirements
#SBATCH --time=50:20:0
#SBATCH --partition=gpuv100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --error="../logs/err.bench"
#SBATCH --output="../logs/std.bench"
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=alban.grastien@cea.fr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alban.grastien@cea.fr
#SBATCH --array=0-2

srun python AURC.py --conf_path confs/conf_sal_${SLURM_ARRAY_TASK_ID}.yml
