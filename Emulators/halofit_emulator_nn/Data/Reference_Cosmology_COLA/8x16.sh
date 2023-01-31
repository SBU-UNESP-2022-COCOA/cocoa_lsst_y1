#!/bin/bash
#SBATCH --job-name=1x
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --output=/gpfs/projects/MirandaGroup/jonathan/cola_projects/timestep_tests/52steps/run_%A.out
#SBATCH --error=/gpfs/projects/MirandaGroup/jonathan/cola_projects/timestep_tests/52steps/run_%A.err
#SBATCH --mail-user=jonathan.gordon@stonybrook.edu
#SBATCH --partition=long-24core
#SBATCH -t 8:00:00

colasolver=/gpfs/projects/MirandaGroup/jonathan/FML/FML/COLASolver/nbody
param_file=/gpfs/projects/MirandaGroup/jonathan/cola_projects/timestep_tests/52steps/parameterfile.lua
module purge > /dev/null 2>&1
module load slurm/17.11.12
source /gpfs/home/jsgordon/miniconda/etc/profile.d/conda.sh
echo Running on host `hostname`
echo Job started at `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
conda activate cola
source start_cola

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=1

mpirun -n ${SLURM_NTASKS} --report-bindings --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} $colasolver $param_file

echo Job ended at `date`