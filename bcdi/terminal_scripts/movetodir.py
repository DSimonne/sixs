#!/home/david/anaconda3/envs/linux.BCDI_MI/bin/python
import os
import shutil
import sys
import ast
import glob

"""Python script to move file from datadir to folder where it will be used by preprocess.bcdi
    Arg 1: directory
    Arg 2: Scan(s) number, list or single value
"""
 
# Print help
try:
    print ('OG data dir:',  sys.argv[1])
    print ('Target data dir:',  sys.argv[2])
    print ('Scan (s):',  sys.argv[3])
except IndexError:
    print("""
        Arg 1: Original data directory 
        Arg 2: Path of target directory (to be created, /S{scan} will be added and all the subfolders)
        Arg 3: Scan(s) number, list or single value
        """)
    exit()

OG_folder = sys.argv[1]
TG_folder = sys.argv[2]
scan_list = sys.argv[3]

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
        shutil.copy("/home/david/.local/lib/python3.9/site-packages/phdutils/bcdi/pynx-run-no-support.txt", f"{TG_folder}S{scan}/pynxraw")
        print(f"Copied pynx-run-no-support.txt to {TG_folder}S{scan}/pynxraw")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/pynxraw/pynx-run-no-support.txt exists")
        pass

    try:
        filename = glob.glob(f"{OG_folder}*mu*{scan}*", recursive=True)[0]
        shutil.copy2(filename, f"{TG_folder}S{scan}/data")
        print(f"Copied {filename} to {TG_folder}S{scan}/data")
    except FileExistsError:
        print(f"{TG_folder}S{scan}/data/{filename} exists")
        pass
    except IndexError:
        print("Not a mu scan")
        pass
    print("\n")