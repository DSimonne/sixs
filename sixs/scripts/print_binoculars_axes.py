#!/usr/bin/python3
import h5py
import sys
import os
import glob


def print_binocular_axes(filename):
    """
    Print binoculars axes in the filename

    :param filename: string, absolute path to data
    """
    print(f"Opening {filename}")
    with h5py.File(filename) as f:
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
            Arg 1: Glob string, relative path to nexus file, e.g. "1335*.nxs"
            """)
        exit()

    # Iterate on file list
    for f in files:
        print_binocular_axes(f)
