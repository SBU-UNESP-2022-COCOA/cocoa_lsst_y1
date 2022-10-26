#!/bin/bash

#SBATCH --job-name=lsst_emu
#SBATCH --output=lsst_emu-%A_%a.out
#SBATCH -e lsst_emu-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-24core
#SBATCH -t 48:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
module purge > /dev/null 2>&1

source ~/.bashrc
module load slurm
conda activate cocoa
source start_cocoa

module load gcc/10.2.0  

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} python3 generate_dv_emu.py ./projects/lsst_y1/generate_dv_emu_1.yaml -f