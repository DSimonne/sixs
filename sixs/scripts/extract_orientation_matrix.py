#!/usr/bin/python3
import tables as tb
import glob
import os
import pandas as pd


def extract_orientation_matrix(glob_string, save_as=False):
    """
    Ignores scans of type ("init", "pitch", "gap", "bragg", "ssl", "omega")

    :param glob_string: full path to nexus files, e.g. 
            "/nfs/ruche-sixs/sixs-soleil/com-sixs/2022/Run1/20211017_resta/**/*.nxs"
    :param save_as: path to final .csv file
    """
    files = sorted(
        glob.glob(
            glob_string,
            recursive=True
        ),
        key=os.path.getmtime
    )

    df = pd.DataFrame(
        columns=[
            "Index",
            "a",
            "Ux",
            "Uy",
            "Uz",
        ]
    )

    for file in files:
        if not any(s in file for s in ("init", "pitch", "gap", "bragg", "ssl", "omega")):
            scan_index = file.split("_")[-1].split(".nxs")[0]

            with tb.open_file(file) as f:
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
            row = pd.DataFrame({
                "Index": [scan_index],
                "a": [a],
                "Ux": [ux],
                "Uy": [uy],
                "Uz": [uz],
            }
            )

            df = df.append(row, ignore_index=True)

    # Save
    if isinstance(save_as, str):
        df.to_csv(save_as)

    else:
        print(df.to_string())


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
        print(
            "Arg 1: Glob string, full path to nexus file, e.g."
            "\t/nfs/ruche-sixs/sixs-soleil/com-sixs/2022/Run1/20211017_resta/**/*.nxs"
        )
        exit()

    try:
        extract_orientation_matrix(glob_string=sys.argv[1])

    except IndexError:
        print(
            "Arg 1: Glob string, relative path to nexus file, e.g."
            "\t/nfs/ruche-sixs/sixs-soleil/com-sixs/2022/Run1/20211017_resta/**/*.nxs"
        )
        exit()
