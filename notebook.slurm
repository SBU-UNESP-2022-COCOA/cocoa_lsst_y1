#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -p long-28core
#SBATCH --time 48:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
port=9001

# print tunneling instructions to the jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@login.seawulf.stonybrook.edu

Use a Browser on your local machine to go to:
localhost:${port}"

# load modules or conda environments here
source /gpfs/home/kuzhong/.conda/etc/profile.d/conda.sh
module load jupyter/5.2.4
conda activate cocoatorch
source start_cocoa

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Jupyter
$CONDA_PREFIX/bin/jupyter notebook --no-browser --port=${port} --ip=${node}