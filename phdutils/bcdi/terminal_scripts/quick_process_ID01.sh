#/usr/bin/bash
echo "Moving scan to new directory..."
echo movetodir_ID01.py $1 $2
movetodir_ID01.py $1 $2

echo "Correcting angles ..."
echo correct_angles_detector_ID01.py $1 $2
correct_angles_detector_ID01.py $1 $2

echo "Preprocessing scan..."
echo preprocess_bcdi_ID01.py $1 $2
echo "Or you can try:"
echo preprocess_bcdi_ID01.py $1 $2 "reload=True"

preprocess_bcdi_ID01.py $1 $2

echo "Ready to launch phase retrieval !"
echo cd $2S$3/pynxraw
echo quick_phase_retrieval.sh