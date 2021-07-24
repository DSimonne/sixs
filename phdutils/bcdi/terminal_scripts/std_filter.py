#!/usr/bin/python3

# use on p9
import numpy as np
import tables as tb
import glob
import os
import operator

# CXI files qlreqdy filtered by LLK
cxi_files = glob.glob("*LLK*.cxi")

# Keep standard deviation of reconstruction modules in dictionnary
std = {}

for filename in cxi_files:
    print("Computing standard deviation of object modulus for ", filename)
    with tb.open_file(filename, "r") as f:
        data = f.root.entry_1.image_1.data[:]
        std[filename] = np.std(np.abs(data))
        
n_keep = 5
n_llk = len(cxi_files)
sorted_dict = sorted(std.items(), key=operator.itemgetter(1))

for f, std in sorted_dict[n_keep:]:
    print(f"Removed scan {f}")
    os.remove(f)
    
print("Ready to run modes decomposition...")