#usr/bin/bash
cwd=$(pwd)
ssh $1@slurm-nice-devel << EOF

	sbatch /data/id01/inhouse/david/Packages/phdutils/phdutils/bcdi/terminal_scripts/pynx_GUI.slurm $2
    
    echo "Phase retrieval is running ..."
    
	exit

EOF