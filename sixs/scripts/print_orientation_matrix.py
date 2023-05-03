#!/usr/bin/python3
import tables as tb
import glob
import os
import pandas as pd
import sys
import numpy as np


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
                a = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].A[0], 4)
                b = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].B[0], 4)
                c = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].C[0], 4)
                alpha = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].alpha[0], 4)
                beta = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].beta[0], 4)
                gamma = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].gamma[0], 4)
                ux = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Ux[0], 4)
                uy = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Uy[0], 4)
                uz = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.h"].Uz[0], 4)

            except (tb.exceptions.NoSuchNodeError, IndexError):
                a = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].A[0], 4)
                b = f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].B[0]
                c = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].C[0], 4)
                alpha = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].alpha[0], 4)
                beta = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].beta[0], 4)
                gamma = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].gamma[0], 4)
                ux = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Ux[0], 4)
                uy = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Uy[0], 4)
                uz = np.round(f.root.com.SIXS["i14-c-cx1-ex-cm-med.v"].Uz[0], 4)

        scan_orientation_matrix = [
            scan_index,
            a, b, c,
            alpha, beta, gamma,
            ux, uy, uz,
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
            Print the orientation matrix used to compute hkl positions during the scan.

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
            "a", "b", "c",
            "alpha", "beta", "gamma",
            "Ux", "Uy", "Uz",
        ],
    )

    # Print array, use pretty printer ?
    print(df.to_string())
