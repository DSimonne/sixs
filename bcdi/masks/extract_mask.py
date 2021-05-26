import glob
import numpy as np
import tables as tb

"""Extract mask as a 3D array of 0 et 1", from cxi file"""

Path = input("\nPlease write path of .cxi file: ")

with tb.open_file(Path, "r") as f:
    # print(f"\n{Path}")
    # Since .cxi files follow a specific architectture, we know where our data is.
    mask = f.root.entry_1.image_1.mask[:]

    ED = f.root.entry_1.data_1.data[:]
    print(f"Shape of real space complex electronic density array {np.shape(ED)}")
        
    # Check if the indices we have seem to be around the center of the array
    rocc = np.where(mask == 1)
    rnocc = np.where(mask == 0)

    print("Indices where result == 1:\n",rocc)
    print(f"Percentage of 3D array occupied by the mask :\n{np.shape(rocc)[1] / np.shape(rnocc)[1]}")
    
    np.savez("direct_mask.npz", mask = mask)
    
    print("Saved mask in local dir as direct_mask.npz")