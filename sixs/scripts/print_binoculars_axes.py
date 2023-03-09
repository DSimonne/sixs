#!/usr/bin/python3
import h5py
import sys
import os
import glob
from prettytable import PrettyTable


def print_binocular_axes(filename):
    """
    Print binoculars axes in the filename

    :param filename: string, absolute path to data
    """
    print(f"Opening {filename}")
    with h5py.File(filename) as f:
        axes = f["binoculars"]["axes"]

        table = PrettyTable(
            field_names=["Axe name", "Axe index", "Starting value",
                         "Final value", "Step", "Length"],
            header=True,
            float_format=".3"
        )
        for k in (axes.keys()):
            v = list(axes[k][:4])
            axe_length = int((axes[k][2] -  axes[k][1]) / axes[k][3])
            table.add_row([k] + v + [axe_length])

    print(table)

# If used as script, iterate on glob string
if __name__ == "__main__":
    # Print help if error raised
    try:
        print(
            "#####################################################"
            f"\nGlob string: {sys.argv[1]}"
            "\n#####################################################\n"
        )

        files = sorted(
            glob.glob(
                sys.argv[1],
                recursive=True
            ),
            key=os.path.getmtime
        )

    except IndexError:
        print(
            """
            Print the axes stored in the binoculars process output file and their respective range.

            Arg 1: Glob string, relative path to nexus file, e.g. "1335*.nxs"
            """)
        exit()

    # Iterate on file list
    for f in files:
        print_binocular_axes(f)
