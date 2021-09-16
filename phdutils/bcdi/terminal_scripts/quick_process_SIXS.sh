#/usr/bin/bash
echo "##################################################################"
echo "Moving scan to new directory..."
echo movetodir.py $1 $2 $3
echo "##################################################################"
movetodir.py $1 $2 $3

echo "##################################################################"
echo "Rotating scan..."
echo rotate.py $2 $3
echo "##################################################################"
rotate.py $2 $3

echo "##################################################################"
echo "Correcting angles ..."
echo correct_angles_detector.py $2 $3
echo "##################################################################"
correct_angles_detector.py $2 $3

echo "##################################################################"
echo "Preprocessing scan..."
echo preprocess_bcdi_merlin_ortho.py $2 $3
echo "Or you can try:"
echo preprocess_bcdi_merlin_ortho.py $2 $3 "reload=True"
echo "##################################################################"
preprocess_bcdi_merlin_ortho.py $2 $3

echo "##################################################################"
echo "Ready to launch phase retrieval !"
echo cd $2S$3/pynxraw
echo quick_phase_retrieval.sh
echo "##################################################################"
