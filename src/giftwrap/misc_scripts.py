from pathlib import Path
import argparse
import inspect

import pandas as pd
from rich_argparse import RichHelpFormatter

from .utils import FlexFormatInfo


def print_R():
    parser = argparse.ArgumentParser(
        description="Print an R script to read a giftwrap HDF5 file.",
        formatter_class=RichHelpFormatter
    )
    args = parser.parse_args()  # No args
    with open(Path(__file__).parent / "read_gf_h5.R", "r") as f:
        print(f.read(), end="")
    exit(0)


def print_tech():
    parser = argparse.ArgumentParser(
        description="An example python file for defining a custom technology."
    )
    args = parser.parse_args()  # No args
    print("from giftwrap import FlexFormatInfo, PrefixTree")
    print(inspect.getsource(FlexFormatInfo), end="")
    exit(0)


def convert_probes():
    parser = argparse.ArgumentParser(
        description="Convert a 10X Genomics cellranger-based probe file to a giftwrap probe file. Prints to stdout.",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--input",
        default=None,
        required=False,
        type=str,
        help="The path to the input probe file. If not specified, will use the Human WTA 1.0.1 probes."
    )
    args = parser.parse_args()

    input = args.input
    if input is None:
        header = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }
        input = "https://cf.10xgenomics.com/supp/cell-exp/probeset/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2020-A.csv"
        probes = pd.read_csv(input, comment="#", storage_options=header)
    else:
        probes = pd.read_csv(input, comment="#")
    probes["lhs_probe"] = probes["probe_seq"].str.slice(0, 25)
    probes["rhs_probe"] = probes["probe_seq"].str.slice(25, 50)
    probes = probes.drop(columns=["probe_seq", "included", "gene_id"])
    probes = probes.rename(columns={"probe_id": "name"})
    probes['gene'] = probes['name'].str.split("|").str[1]
    probes['gap_probe_sequence'] = ""  # No gap expected
    probes['original_gap_probe_sequence'] = ""  # No gap expected

    if 'region' in probes.columns:
        probes = probes.drop(columns=['region'])

    # Print to stdout
    print(probes.to_csv(index=False, sep="\t"), end="")


if __name__ == "__main__":
    print_R()
