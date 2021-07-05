# Architecture of BCDI terminal scripts:

/data  # all the rocking curves collected
workflow_details.ipynb 	# Notebook that details how bcdi works
plots.ipynb 			# Quick plots to follow the evolution of the parameter for each scan
print_pos.py 			# Prints scan data
/Argon
	/Sxxxx
		/data
			# data file that is used during preprocessing
		/pynxraw
			# Output of pre processing scripts
			# Output of phase retrieval

		/result_crystal_frame
		/postprocessing
			# Output from correct_detector.py, such as the angles of COM and the lattice parameter 

/CondA
	/...
/CondB
	/...
/...