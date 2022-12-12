#!/usr/bin/python3
import numpy as np
import pandas as pd
from sixs import ReadNxs4 as rd
import sys
import os
import glob


def find_pos(filename):
    """
    Print file metadata

    :param filename: string, absolute path to data

    return: metadata, list
    """
    print(f"Opening {filename}")
    try:
        scan_index = filename.split("_")[-1].split(".nxs")[0]

        data = rd.DataSet(filename)
        metadata = [
            scan_index,
            rotation,
            data.x[0],
            data.y[0],
            data.z[0],
            data.mu[0],
            data.delta[0],
            data.omega[0],
            data.gamma[0],
            data.gamma[0] - data.mu[0],
            int(data.roi1_merlin.sum()),
            int(data.roi4_merlin.sum()),
            (data.mu[-1] - data.mu[-0]) / len(data.mu),
            data.integration_time[0],
            len(data.integration_time),
            data.ssl3hg[0],
            data.ssl3vg[0],
            # data.ssl1hg[0],
            # data.ssl1vg[0]
        ]
        return metadata

    except:
        print("Could not load data, is the alias file up to date ?")
        return None


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
        print("""
            Arg 1: Glob string, relative path to nexus file, e.g. "1335*.nxs"
            """)
        exit()

    # Create data array
    pos = np.array(
        [find_pos(f) for f in files]
    )

    data = pd.DataFrame(
        pos,
        columns=[
            "scan_index",
            "rotation",
            "x",
            "y",
            "z",
            "mu",
            "delta",
            "omega",
            "gamma",
            "gamma-mu",
            "roi1_sum",
            "roi4_sum",
            "step size",
            "integration time",
            "nb of steps",
            "ssl3hg",
            "ssl3vg",
            # "ssl3vg",
            # "sslhg",
        ],
        dtype=float
    )
    data = data.round(3)

    # Print array, use pretty printer ?
    print(data.to_string())
