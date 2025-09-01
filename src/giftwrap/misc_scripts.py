import contextlib
import hashlib
import tempfile
import warnings, os
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # inherit to subprocesses

import sys
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
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
    )
    args = parser.parse_args()  # No args
    with open(Path(__file__).parent / "read_gf_h5.R", "r") as f:
        print(f.read(), end="")
    exit(0)


def print_tech():
    parser = argparse.ArgumentParser(
        description="An example python file for defining a custom technology."
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
    )
    args = parser.parse_args()  # No args
    print("from giftwrap import FlexFormatInfo, PrefixTree")
    print(inspect.getsource(FlexFormatInfo), end="")
    exit(0)


def int_to_hash(n, length=7):
    # Convert integer to bytes
    b = n.to_bytes((n.bit_length() + 7) // 8 or 1, byteorder="big")
    # Hash and get hex digest
    h = hashlib.sha1(b).hexdigest()
    # Return first `length` characters
    return h[:length]


def revert_probes():
    parser = argparse.ArgumentParser(
        description="Revert a GIFTwrap probe file to a 10X Genomics cellranger-based probe file. Prints to stdout.",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="The path to the input GIFTwrap probe file."
    )
    args = parser.parse_args()

    input = Path(args.input)
    if not input.exists():
        raise FileNotFoundError(f"Input file {input} does not exist.")

    from .step1_count_gapfills import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with warnings.catch_warnings(), open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            build_manifest(input, tmpdir, overwrite=True, allow_any_combination=False)
        probes = pd.read_csv(tmpdir / "manifest.tsv", sep="\t")

    # Next, we need to convert the GIFTwrap probe file back to the 10X Genomics format
    print("#probe_set_file_format=3.0")
    print("#panel_name=GIFTWRAP probe set")
    print("#panel_type=predesigned")
    print("#reference_genome=GRCh38")
    print("#reference_version=2024-A")
    print("gene_id,probe_seq,probe_id,included,region,gene_name")
    curr_hash = 0
    has_ref_and_alt = ~(probes['original_gap_probe_sequence'].isna() | probes['gap_probe_sequence'].isna()).all()
    for _, row in probes.iterrows():
        name = row['name']
        if 'gene' in row:
            gene = row['gene']
        else:
            gene = name.split(" ")[0]
        print(name, end="")
        if has_ref_and_alt:
            print(" REF", end=",")
            probe_seq = row['lhs_probe'] + (row['original_gap_probe_sequence']) + row['rhs_probe']
            probe_seq = probe_seq[:50]  # Ensure it is exactly 50 bases
            print(probe_seq, end=",")
            print(f"{name} REF|{gene}|{int_to_hash(curr_hash)},TRUE,unspliced,{gene} REF")
            curr_hash+=1
            print(name, " ALT", end=",")
            probe_seq = row['lhs_probe'] + row['gap_probe_sequence'] + row['rhs_probe']
            probe_seq = probe_seq[:50]  # Ensure it is exactly 50 bases
            print(probe_seq, end=",")
            print(f"{name} ALT|{gene}|{int_to_hash(curr_hash)},TRUE,unspliced,{gene} ALT")
            curr_hash+=1
        else:
            print(",", end="")
            probe_seq = row['lhs_probe'] + row['rhs_probe']
            probe_seq = probe_seq[:50]  # Ensure it is exactly 50 bases
            print(probe_seq, end=",")
            print(f"{name}|{gene}|{int_to_hash(curr_hash)},TRUE,unspliced,{gene}")
            curr_hash+=1


def convert_probes():
    parser = argparse.ArgumentParser(
        description="Convert a 10X Genomics cellranger-based probe file to a giftwrap probe file. Prints to stdout.",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
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
