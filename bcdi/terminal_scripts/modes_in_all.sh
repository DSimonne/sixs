echo "running pynx reconstructions ..."
pynx-id01cdi.py pynx-run > README_pynx.md
mkdir all
mv *LLK* all/
cd all/
echo "running modes analysis ..."
pynx-cdi-analysis.py *LLK* modes=1 modes_crop=no modes_output=modes_all.h5> README_modes_all.md
echo "plotting slices ..."
slice_cxi.py ./ Module 2D mid
