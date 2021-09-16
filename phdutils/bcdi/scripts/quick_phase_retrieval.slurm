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

# echo `date`  | mail -s " SLURM JOB $SLURM_JOBID has started" <YOUR-EMAIL>
# Moving to pynxraw directory
cd $1
# cd $1/$2S$3/pynxraw 

# Running phase retrieval
echo "Running phase retrieval with PyNX ..."
echo "/data/id01/inhouse/david/py38-env/bin/pynx-id01cdi.py $1/pynx_run_gui.txt > README_pynx.md"
mpiexec -n $SLURM_NTASKS /data/id01/inhouse/david/py38-env/bin/pynx-id01cdi.py $1/pynx_run_gui.txt > README_pynx.md
# echo "/sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run.txt > README_pynx.md"
# mpiexec -n $SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx_run.txt > README_pynx.md

# echo `date`  | mail -s " SLURM has finished job: $SLURM_JOBID !! " <YOUR-EMAIL>

# Moving the results in all/ sub-directory
echo "Creating a subdirector all/ and moving all the .cxi files there..."
mkdir all
mv *LLK*.cxi all/
mv README_pynx.md all/
cd all/

# Apply a standard deviation filter
echo "Applying a standard deviation filter"
std_filter.py

# Running modes decomposition
echo "Running modes decomposition ..."
echo "/data/id01/inhouse/david/py38-env/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md"
/sware/exp/pynx/devel.p9/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md


# # Running strain analysis
# ssh simonne@rnice9 << EOF
# 	cd $1
# 	strain_ID01.py $2 $3
    
#     echo "Strain.py is running ..."
    
# 	exit

# EOF