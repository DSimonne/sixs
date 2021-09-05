#!/bin/bash 

# Moving to pynxraw directory
cd $1/$2S$3/pynxraw 

# Running phase retrieval
echo "Running phase retrieval with pynx ..."
echo "/sware/exp/pynx/devel.debian9/bin/pynx-id01cdi.py pynx_run.txt > README_pynx.md"
/sware/exp/pynx/devel.debian9/bin/pynx-id01cdi.py pynx_run.txt > README_pynx.md

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
cwd=$(pwd)

# Running modes decomposition
echo "Running modes decomposition ..."
echo "/sware/exp/pynx/devel.debian9/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md"
/sware/exp/pynx/devel.debian9/bin/pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_job.h5 > README_modes.md

# Running strain analysis
# ssh simonne@rnice9 << EOF
# 	cd $1
# 	strain_ID01.py $2 $3
#     
#     echo "Strain.py is running ..."
#     
# 	exit
# 
# EOF