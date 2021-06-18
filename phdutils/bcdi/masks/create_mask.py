import glob
import numpy as np
import tables as tb

"""Create a mask as a 3D array of 0 et 1"""

def CreateSupport(Path, Threshold):

    with tb.open_file(Path, "r") as f:
        print(f"\n{Path}")
        # Since .cxi files follow a specific architectture, we know where our data is.
        mask = f.root.entry_1.image_1.mask[:]
        print(f"All elements in mask array are equal to 1 : {np.all(mask == 1)}")
        print(f"Shape of support array {np.shape(mask)}")

        # support = f.root.entry_1.image_1.support[:]
        # print(f"All elements in support array are equal to 1 : {np.all(support == 1)}")

        ED = f.root.entry_1.data_1.data[:]
        print(f"Shape of real space complex electronic density array {np.shape(ED)}")

        # Find max value of real space image, we work with the module
        Amp = np.abs(ED)
        MaxRS = Amp.max()
        print(f"Maximum value in amplitude array: {MaxRS}")
        
        result = np.where(Amp < Threshold * MaxRS, 0, 1)
        # Check if the indices we have seem to be around the center of the array
        rocc = np.where(result == 1)
        rnocc = np.where(result == 0)
        
        print("Indices where result == 1:\n",rocc)
        print(f"Percentage of 3D array occupied by support:\n{np.shape(rocc)[1] / np.shape(rnocc)[1]}")

        return result

print("Example of used files : Pt_p2/pynxraw/S1398_pynx_norm_128_300_294_1_1_1-2020*cxi")

print("Example of folder : Pt_p2/pynxraw")
print("Example of file name : S1398_pynx_norm_128_300_294_1_1_1-2020")

folder = input("\nPlease write your folder like in the example: ")
filename = input("Please write your file name like in the example: ")

files = sorted(glob.glob(f"{folder}/{filename}*.cxi"))

print(f"\nUsed files: {files}")

T = input("\nThreshold value (int from 1 to 100): ")
t = int(T) /100

NS = int(input("\nAmount of .cxi files,  between the N files that were kept, for which the threshold must be respected: "))

# Take supports from cxi data
Supports = [CreateSupport(p, t) for p in files]


FirstSupport = Supports[0]

for S in Supports[1:]:
    FirstSupport = np.add(FirstSupport, S)
    
mask = np.where(FirstSupport >= NS, 1, 0)

print("\n\n")
np.savez(f"{folder}/mask_{t}.npz", mask = mask)
print(f"Mask saved in {folder} as mask_T{t}_N{NS}.npz")

rocc = np.where(mask == 1)
rnocc = np.where(mask == 0)

print("Indices where result == 1:\n", rocc)
print(f"Percentage of 3D array occupied by support:\n{np.shape(rocc)[1] / np.shape(rnocc)[1]}")