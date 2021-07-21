#!/users/simonne/anaconda3/envs/rnice.BCDI/bin/python3
import os
import shutil
import sys
import ast
import glob

"""Python script to move file from datadir to folder where it will be used by preprocess.bcdi
For ID01 data
    Arg 1: directory
    Arg 2: Scan(s) number, list or single value
"""
 
# Print help
try:
    print ('Target data dir:',  sys.argv[1])
    print ('Scan (s):',  sys.argv[2])
except IndexError:
    print("""
        Arg 1: Path of EXISTING target directory (e.g. Pt_Al2O3/) (subdirectories S{scan}/data & S{scan}/pynx_raw will be updated/created)
        Arg 2: Scan(s) number, list or single value

        Looks recursively for one mu or omega scan including the scan number (glob.glob).
        """)
    exit()

TG_folder = sys.argv[1]
scan_list = sys.argv[2]

# transform string of list into python list object
if scan_list.startswith("["):
    scans = ast.literal_eval(scan_list)
    
else:
    scans = [scan_list]

# Load data
for scan in scans:
    print(f"Moving scan {scan}...")
    try:
        os.mkdir(f"{TG_folder}S{scan}")
        print(f"Created {TG_folder}S{scan}")
    except FileExistsError:
        print(f"{TG_folder}S{scan} exists")
        pass

    try:
        os.mkdir(f"{TG_folder}S{scan}/data")
        print(f"Created {TG_folder}S{scan}/data")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/data exists")
        pass

    try:
        os.mkdir(f"{TG_folder}S{scan}/pynxraw")
        print(f"Created {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw exists")
        pass

    try:
        shutil.copy("/home/esrf/simonne/Packages/phdutils/phdutils/bcdi/pynx_run_ID01.txt", f"{TG_folder}S{scan}/pynxraw")
        print(f"Copied pynx_run_ID01.txt to {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw/pynx_run_ID01.txt exists")
        pass

    try:
        shutil.copy("/home/esrf/simonne/Packages/phdutils/phdutils/bcdi/PhasingNotebook.ipynb", f"{TG_folder}S{scan}/pynxraw")
        print(f"Copied PhasingNotebook.ipynb.txt to {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw/PhasingNotebook.ipynb exists")
        pass

    print("\n")