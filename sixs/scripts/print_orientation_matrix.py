#!/usr/bin/python3
import tables as tb
import glob
import os
import pandas as pd
import sys


def extract_orientation_matrix(
    filename,
    ignore_string_list=["init", "pitch", "gap", "bragg", "ssl", "omega"],
):
    """
    Print the sample U matrix for the file.

    :param filename: string, absolute path to data
    :param ignore_string_list: ignore the scan if one of the string in the list
        is included

    return: orientation_matrix, list
    """
    if not any(s in filename for s in ignore_string_list):
        print(f"Opening {filename}")
        scan_index = filename.split("_")[-1].split(".nxs")[0]

        with tb.open_file(filename) as f:
            try:
                a = f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].A[0]
                ux = f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Ux[0]
                uy = f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Uy[0]
                uz = f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Uz[0]

            except (tb.exceptions.NoSuchNodeError, IndexError):
                a = f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].A[0]
                ux = f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Ux[0]
                uy = f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Uy[0]
                uz = f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Uz[0]

        scan_orientation_matrix = [
            scan_index,
            a,
            ux,
            uy,
            uz,
        ]

        return scan_orientation_matrix


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

    # Create data array
    orientation_matrix = np.array(
        [extract_orientation_matrix(f) for f in files]
    )

    df = pd.DataFrame(
        orientation_matrix,
        columns=[
            "Scan index",
            "a",
            "Ux",
            "Uy",
            "Uz",
        ],
    )

    # Print array, use pretty printer ?
    print(data.to_string())
