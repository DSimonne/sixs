#/usr/bin/bash
echo "##################################################################"
echo "Moving scan to new directory..."
echo movetodir_ID01.py $1 $2
echo "##################################################################"
movetodir_ID01.py $1 $2

echo "##################################################################"
echo "Correcting angles ..."
echo correct_angles_detector_ID01.py $1 $2
echo "##################################################################"
correct_angles_detector_ID01.py $1 $2

echo "##################################################################"
echo "Preprocessing scan..."
echo preprocess_bcdi_ID01.py $1 $2
echo "Or you can try:"
echo preprocess_bcdi_ID01.py $1 $2 "reload=True"
echo "##################################################################"
preprocess_bcdi_ID01.py $1 $2

echo "##################################################################"
echo "Running phase retrieval and strain analysis!"
echo quick_phase_retrieval_ID01.sh $1 $2
echo "##################################################################"
quick_phase_retrieval_ID01.sh $1 $2