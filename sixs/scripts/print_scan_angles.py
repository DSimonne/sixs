#!/usr/bin/python3
import h5py
import sys
import os
import glob
from sixs.experiments import coherence as coh
import tables as tb
from prettytable import PrettyTable


def get_scan_angles(filename):
    """
    Print binoculars axes in the filename

    :param filename: string, absolute path to data
    """
    with tb.open_file(filename) as f:
        try:
            beta = f.root.com.scan_data.beta[...][0]
            mu = f.root.com.scan_data.mu[...][0]
            gamma = f.root.com.scan_data.gamma[...][0]
            delta = f.root.com.scan_data.delta[...][0]

        except tb.NoSuchNodeError:
            pass

        return os.path.basename(filename), beta, mu, gamma, delta


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
    table = PrettyTable(
        field_names=["Filename", "Beta", "Mu", "Gamma", "Delta"],
        header=True,
        float_format=".3"
    )
    for i, f in enumerate(files):
        table.add_row(get_scan_angles(sys.argv[1] + f))

    print(table)