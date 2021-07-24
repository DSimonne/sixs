#usr/bin/bash
cwd=$(pwd)
ssh simonne@slurm-access << EOF

	sbatch pynx_ID01.slurm $cwd/$1S$2/pynxraw
    
    echo "Phase retrieval is running ..."
    
	exit

EOF