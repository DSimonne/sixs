#!/usr/bin/python3
import h5py
import ast
import sys
import os
import glob

def show_axes(file):

    with h5py.File(file) as f:
        axes = f["binoculars"]["axes"]
        print("File axes are:")
        for k in (axes.keys()):
            v = axes[k][...]
            print(f"\t {k}:[{v[0]:.3f}: {v[1]:.3f}: {v[2]:.3f}]")

# If used as script, iterate on glob string
if __name__ == "__main__":

    # Print help if error raised
    try:
        print(
            "#####################################################"
            f"\nGlob string, {sys.argv[1]}"
            "\n#####################################################\n"
        )
    except IndexError:
        print("""
            Arg 1: Glob string to seach for, try "path/to/data/*.nxs"
            """)
        exit()

    try:
        cd = os.getcwd()
        files = glob.glob(sys.argv[1])

        for f in files:
            print(f"#####################################################\n{f}")
            show_axes(f)
            print(f"#####################################################\n")

    except IndexError:
        print("""
            Arg 1: Glob string to seach for, try "path/to/data/*.nxs"
            """)
        exit()
