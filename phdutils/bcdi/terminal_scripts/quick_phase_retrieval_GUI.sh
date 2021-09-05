#usr/bin/bash
cwd=$(pwd)
ssh simonne@slurm-nice-devel << EOF

	sbatch /home/esrf/simonne/Packages/phdutils/phdutils/bcdi/terminal_scripts/pynx_GUI.slurm $1
    
    echo "Phase retrieval is running ..."
    
	exit

EOF