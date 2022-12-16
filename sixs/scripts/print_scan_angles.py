#!/usr/bin/python3
import h5py
import sys
import os
import glob
from sixs.experiments import coherence as coh
import tables as tb


def print_scan_angles(filename):
    """
    Print binoculars axes in the filename

    :param filename: string, absolute path to data
    """
    with tb.open_file(filename) as f:
        try:
            gamma = f.root.com.scan_data.gamma[...][0]
        except tb.NoSuchNodeError:
            pass
        print(
            "\n########################################"
            f"\nOpening {filename}"
            f"\n\tgamma = {gamma:.3f}"
            "\n########################################"
        )


# If used as script, iterate on glob string
if __name__ == "__main__":
    # Print help if error raised
    try:
        print(
            "#####################################################"
            f"\nDirectory, {sys.argv[1]}"
            f"\nStart number, {sys.argv[2]}"
            f"\nEnd number, {sys.argv[3]}"
            f"\nGlob Pattern, {sys.argv[4]}"
            "\n#####################################################\n"
        )

        files = coh.get_file_range(
            sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
            sys.argv[4],
        )

    except IndexError:
        print(
            """
            Arg 1: Directory
            Arg 2: Start number
            Arg 3: End number
            Arg 4: Glob Pattern

            e.g. print_scan_angles.py align/ 280 284 "*.nxs"
            """)
        exit()

    # Iterate on file list
    for f in files:
        print_scan_angles(sys.argv[1] + f)
