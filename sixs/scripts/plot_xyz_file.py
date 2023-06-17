#!/usr/bin/python3
import sys
import os

from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure
from plotly.offline import offline


def create_html_file(filepath):
    """Needs https://github.com/zotko/xyz2graph"""
    # Create the MolGraph object
    mg = MolGraph()

    # Read the data from the .xyz file
    mg.read_xyz(filepath)

    # Create the Plotly figure object
    fig = to_plotly_figure(mg)

    # Plot the figure
    offline.plot(fig)

    # Convert the molecular graph to the NetworkX graph
    G = to_networkx_graph(mg)


# If used as script
if __name__ == "__main__":
    # Print help if error raised
    try:
        print(
            "#####################################################"
            f"\nFilepath: {sys.argv[1]}"
            "\n#####################################################\n"
        )

        filepath = str(sys.argv[1])

        create_html_file(filepath)

    except IndexError:
        print(
            """
            Plots the xyz file as html.

            Arg 1: Relative or absolute path to `.xyz` file.
            """)
        exit()
