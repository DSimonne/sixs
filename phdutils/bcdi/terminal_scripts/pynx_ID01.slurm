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

# Moving to pynxraw directory
cd $1

# Running phase retrieval
echo "Running phase retrieval with pynx ..."
echo "/sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run_ID01.txt > README_pynx.md"
mpiexec -n $SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run_ID01.txt > README_pynx.md

# Moving the results in all/ sub-directory
mkdir all
mv *LLK*.cxi all/
# mkdir all/pynx_images
# mv *Run*.png all/pynx_images
mv README_pynx.md all/
cd all/

# Apply a standard deviation filter
echo "Applying a standard deviation filter"
std_filter.py

# Running modes decomposition
echo "Running modes decomposition ..."
echo "/sware/exp/pynx/devel.p9/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md"
mpiexec -n $SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md
