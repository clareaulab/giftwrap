from pathlib import Path
import argparse


def print_R():
    parser = argparse.ArgumentParser(
        description="Print the R script to read a giftwrap HDF5 file."
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
    import inspect
    from utils import FlexFormatInfo
    print(inspect.getsource(FlexFormatInfo), end="")
    exit(0)


if __name__ == "__main__":
    print_R()
