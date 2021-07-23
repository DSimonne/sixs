#!/bin/bash -l
#SBATCH --partition=p9gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.out
  
env | grep CUDA
scontrol --details show jobs \$SLURM_JOBID |grep RES

cd $1
echo "running pynx reconstructions ..."
echo "/sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run_ID01.txt > README_pynx.md"
mpiexec -n $SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run_ID01.txt > README_pynx.md
