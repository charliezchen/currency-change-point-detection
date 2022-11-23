#!/bin/bash
###SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --mem=128G
#SBATCH --job-name=cpd_Matern
#SBATCH --mail-user=yk2516@nyu.edu
#SBATCH --output=/scratch/yk2516/currency-change-point-detection/slurm/slurm_train_%j.out
#SBATCH --error=/scratch/yk2516/currency-change-point-detection/slurm/slurm_train_%j.err

singularity exec --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c '
echo "Running cpd_module"
source /ext3/env.sh

conda activate base

python3 -m examples.concurent_cpd_quandl 21 --kernel Matern12

'

