#usr/bin/bash
cwd=$(pwd)
ssh simonne@slurm-access << EOF

	sbatch pynx_ID01.slurm $cwd/$1/pynxraw

	exit

EOF

mkdir all
mv *LLK*.cxi all/
# mkdir all/pynx_images
# mv *Run*.png all/pynx_images
mv README_pynx.md all/
cd all/

echo "plotting slices ..."
slice_cxi_ID01.py ./ Module 2D mid

echo "Now run modes analysis or view the reconstruction before..."
echo "silx view *LLK*.cxi"
echo "pynx-cdi-analysis.py *LLK* modes=1 modes_output=modes_all.h5> README_modes_all.md"
