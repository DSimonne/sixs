echo "sourcing environment"
source /sware/exp/pynx/devel.p9/bin/activate
echo "running pynx reconstructions ..."
pynx-id01cdi.py pynx-run-no-support.txt > README_pynx.md
mkdir all
mv *LLK* all/
cd all/
echo "running modes analysis ..."
pynx-cdi-analysis.py *LLK* modes=1 modes_crop=no modes_output=modes_all.h5> README_modes_all.md
echo "plotting slices ..."
source /data/id01/inhouse/david/Documents/Environments/p9.widgets/bin/activate
python /data/id01/inhouse/david/SIXS/Scripts/slice_cxi.py ./ Module 2D mid
