#!/home/david/anaconda3/envs/linux.BCDI_MI/bin/python
import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import ast
import shutil
import glob
import os

"""Python script to rotate the data for vertical configuration"""
 
# Print help
try:
    print ('Data dir:',  sys.argv[1])
    print ('Scan (s):',  sys.argv[2])
except IndexError:
    print("""
        Arg 1: Path of target directory (before /S{scan} ... )
        Arg 2: Scan(s) number, list or single value
        """)
    exit()

folder = sys.argv[1]
scan_list = sys.argv[2]
root_folder = os.getcwd() + "/" + folder  # folder of the experiment, where all scans are stored
sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).

# transform string of list into python list object
if scan_list.startswith("["):
    scans = ast.literal_eval(scan_list)
    
else:
    scans = [scan_list]


# Load data
for scan in scans:
    print(scan)
    filename = glob.glob(f"{root_folder}{sample_name}{scan}/data/*mu*{scan}*")[0]
    f_copy = filename.split(".nxs")[0] + "_R.nxs"
    
    shutil.copy2(filename, f_copy)
    print("Using a copy of the data: ", f_copy)

    print(f"Opening scan {scan} data ...")
    with tb.open_file(f_copy, "a") as f:
        # print(f)

        # Get data
        try:
            data_og = f.root.com.scan_data.data_02[:]
            print("Trying to climb eiger ...")
        except:
            try:
                data_og = f.root.com.scan_data.merlin_image[:]
                print("Calling merlin the enchanter ...")
            except:
                print("This data does not result from Eiger nor Merlin :/")

        # Just an index for plotting schemes
        half = int(data_og.shape[0]/2)

        # Rotate data
        data = np.transpose(data_og, axes=(0, 2, 1))
        for idx in range(data.shape[0]):
            tmp = data[idx, :, :]
            data[idx, :, :] = np.fliplr(tmp)
        print("Data well rotated by 90°.")  

        print("Saving example figures...", end="\n\n")
        plt.figure(figsize = (16, 9))
        plt.imshow(data_og[half, :, :], vmax = 10)
        plt.xlabel('Delta')
        plt.ylabel('Gamma')
        plt.tight_layout()
        plt.savefig(root_folder + sample_name + str(scan) + "/data/data_before_rotation.png")

        plt.figure(figsize = (16, 9))        
        plt.imshow(data[half, :, :], vmax = 10)
        plt.xlabel('Gamma')
        plt.ylabel('Delta')
        plt.tight_layout()
        plt.savefig(root_folder + sample_name + str(scan) + "/data/data_after_rotation.png")
        plt.close()

        # Overwrite data in copied file
        try:
            f.root.com.scan_data.data_02[:] = data
        except:
            try:
                f.root.com.scan_data.merlin_image[:] = data
            except:
                print("Could not overwrite data ><")