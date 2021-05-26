#!/home/david/anaconda3/envs/linux.BCDI_MI/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from phdutils.sixs import ReadNxs4 as rd
import ast
import sys
import os
import glob

def find_pos(filename):
	data = rd.DataSet(filename)

	# print("x:", np.round(data.x[0], 3))
	# print("y:", np.round(data.y[0], 3))
	# print("z:", np.round(data.z[0], 3))
	# print("mu:", np.round(data.mu[0], 3))
	# print("delta:", np.round(data.delta[0], 3))
	# print("omega:", np.round(data.omega[0], 3))
	# print("gamma:", np.round(data.gamma[0], 3))
	# print("gamma-mu:", np.round(data.gamma[0] - data.mu[0], 3))
	# print(f"rocking angle steps: {(data.mu[-1] - data.mu[-0]) / len(data.mu)}")
	    
	# print("gamma - mu:",data.gamma[0] - data.mu[0])
	# print("sum roi 1:",int(data.roi1_merlin.sum()))
	# print("sum roi 1:",int(data.roi4_merlin.sum()))
	# print("rocking curve step:",(data.mu[-1] - data.mu[-0]) / len(data.mu))
	# print("integration time:",data.integration_time[0])
	# print("nb of steps:",len(data.integration_time))

	# print("ssl3hg:", np.round(data.ssl3hg[0], 3))
	# print("ssl3vg:", np.round(data.ssl3vg[0], 3))
	# print("ssl1hg:", np.round(data.ssl1hg[0], 3))
	# print("ssl1vg:", np.round(data.ssl1vg[0], 3))

	metadata = [
	        data.x[0],
	        data.y[0],
	        data.z[0],
	        data.mu[0],
	        data.delta[0],
	        data.omega[0],
	        data.gamma[0],
	        data.gamma[0] - data.mu[0],
	        int(data.roi1_merlin.sum()),
	        int(data.roi4_merlin.sum()),
	        (data.mu[-1] - data.mu[-0]) / len(data.mu),
	        data.integration_time[0],
	        len(data.integration_time),
	        data.ssl3hg[0],
	        data.ssl3vg[0],
	        data.ssl1hg[0],
	        data.ssl1vg[0]
	            ]
	return metadata
    

# Part of script to allow systematic use

# Print total number of arguments
# print ('Total number of arguments:', format(len(sys.argv)))
 
# Print all arguments
# print ('Argument List:', str(sys.argv))
 
# Print arguments one by one
# print ('First argument:',  str(sys.argv[0]))
cd = os.getcwd()

print ('Scan (s):',  sys.argv[1])
scans = glob.glob(sys.argv[1])

# transform string of list into python list object
# if sys.argv[1].startswith("["):
#     scans = ast.literal_eval(sys.argv[1])
# else:
#     scans = [int(sys.argv[1])]

pos = np.array([find_pos(nb) for nb in scans])

data = pd.DataFrame(pos, columns = ["x", "y", "z", "mu", "delta", "omega", "gamma", 'gamma-mu',
                                    "roi1_sum", "roi4_sum", "step size", "integration time", "nb of steps",
                                    "ssl1vg", "ss1hg", "ssl3vg", "sslhg"], dtype=float)
data = data.round(3)
print(data.to_string())