#usr/bin/bash
cwd=$(pwd)
ssh simonne@slurm-access << EOF

	sbatch pynx_ID01.slurm $cwd $1 $2
    
    echo "Phase retrieval is running ..."
    
	exit

EOF

echo "Will also run strain analysis"
echo strain_ID01.py $1 $2

echo "If you have the conjugated object run:"
echo strain_ID01.py $1 $2 flip=True
