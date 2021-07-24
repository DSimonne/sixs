#usr/bin/bash
cwd=$(pwd)
ssh simonne@slurm-access << EOF

	sbatch modes_ID01.slurm $cwd/$1/pynxraw

	exit
EOF