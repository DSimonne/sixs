#!/home/david/anaconda3/envs/linux.BCDI_MI/bin/python
import os
import shutil
import sys
import ast
import glob

"""Python script to move file from datadir to folder where it will be used by preprocess.bcdi"""

print ('Total number of arguments:', format(len(sys.argv)))
 
# Print arguments one by one
print ('Data dir:',  sys.argv[1])
print ('Scan (s):',  sys.argv[2])

folder = sys.argv[1]

# transform string of list into python list object
if sys.argv[2].startswith("["):
    scans = ast.literal_eval(sys.argv[2])
    
elif sys.argv[2]=="all":
    subdirnames = [x[1] for x in os.walk(f"{folder}/")][0]
    scans = [s.replace("S", "") for s in sorted(subdirnames) if s.startswith("S")]
    print(scans, end="\n\n")
    
else:
    scans = [sys.argv[2]]

# Load data
for scan in scans:
    print(scan)
    try:
        os.mkdir(f"{folder}/S{scan}")
        print(f"Created {folder}/S{scan}")
    except FileExistsError:
        print(f"{folder}/S{scan} exists")
        pass

    try:
        os.mkdir(f"{folder}/S{scan}/data")
        print(f"Created {folder}/S{scan}/data")
    except FileExistsError:
        print(f"{folder}/S{scan}/data exists")
        pass

    try:
        os.mkdir(f"{folder}/S{scan}/pynxraw")
        print(f"Created {folder}/S{scan}/pynxraw")
    except FileExistsError:
        print(f"{folder}/S{scan}/pynxraw exists")
        pass

    try:
        shutil.copy("Scripts/pynx-run-no-support.txt", f"{folder}/S{scan}/pynxraw")
        print(f"Copied Scripts/pynx-run-no-support.txt to {folder}/S{scan}/pynxraw")
    except FileExistsError:
        print(f"{folder}/S{scan}/pynxraw/pynx-run-no-support.txt exists")
        pass

    try:
        filename = glob.glob(f"data/**/*mu*{scan}*", recursive=True)[0]
        shutil.copy2(filename, f"{folder}/S{scan}/data")
        print(f"Copied {filename} to {folder}/S{scan}/data")
    except FileExistsError:
        print(f"{folder}/S{scan}/data/{filename} exists")
        pass
    except IndexError:
        print("Not a mu scan")
        pass
    print("\n")