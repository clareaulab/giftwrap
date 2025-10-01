import warnings, os
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # inherit to subprocesses

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from .utils import read_manifest, read_barcodes, maybe_multiprocess, maybe_gzip, write_sparse_matrix, \
    _tx_barcode_to_oligo, compile_flatfile


def collect_counts(input: Path, output: Path, manifest: pd.DataFrame, barcodes_df: pd.DataFrame, overwrite: bool, plex: int = 1, multiplex: bool = False, flatten: bool = False, all_pcr_thresholds: bool = False):
    """
    Generate an h5 file with counts for each barcode.
    :param input: The input file.
    :param output: The output directory.
    :param manifest: The manifest metadata.
    :param barcodes_df: The dataframe containing all barcodes and metadata.
    :param overwrite: Overwrite the output file if it exists.
    :param plex: The plex number.
    :param multiplex: Whether the run was multiplexed.
    :param flatten: Whether to also output a flattened tsv file.
    :param all_pcr_thresholds: Whether to include counts for all PCR duplicate thresholds.
    """
    # Check if the output file exists
    final_output = output / f"counts.{plex}.h5"
    if final_output.exists() and not overwrite:
        raise AssertionError(f"Output file already exists: {final_output}")
    elif final_output.exists():
        final_output.unlink()

    # Replace the barcode -plex with {probe bc}-1 to match cellranger output
    if multiplex:
        barcodes_df.barcode = barcodes_df.barcode.str.replace(f"-{plex}", f"{_tx_barcode_to_oligo[plex]}-1")

    probe_idx2name = {idx: name for idx, name in enumerate(manifest['name'])}

    # Get metadata cols
    barcode2h5_idx = {bc: idx for idx, bc in enumerate(barcodes_df.barcode.values)}

    if flatten:
        compile_flatfile(manifest, str(input), barcodes_df.barcode.values.tolist(), plex, str(output / f'flat_counts.{plex}.tsv.gz'))

    # Single pass through file - collect all data at once
    print("Reading and processing file...", end="")

    # Use defaultdicts for faster accumulation
    counts_data = defaultdict(int)
    dup_count_mapping = defaultdict(lambda: defaultdict(int))
    total_umi_data = defaultdict(int)
    percent_supporting_data = defaultdict(float)
    possible_probes = set()
    # Track original indices for cells and probes
    original_cell_indices = set()
    original_probe_indices = set()
    n_lines = 0

    # Pre-convert plex to int for faster comparison
    plex_int = int(plex)
    barcode_h5_indices = set(barcode2h5_idx.values())

    with maybe_gzip(input, 'r') as input_file:
        # Skip the header
        next(input_file)
        for line in input_file:
            cell_idx, probe_idx, probe_bc_idx, umi, gapfill, umi_dup_count, percent_supporting = line.strip().split("\t")
            cell_idx = int(cell_idx)
            probe_bc_idx = int(probe_bc_idx)

            # Fast filtering
            if cell_idx not in barcode_h5_indices or probe_bc_idx != plex_int:
                continue

            probe_idx = int(probe_idx)
            probe_name = probe_idx2name[probe_idx]
            probe_key = (probe_name, gapfill)
            possible_probes.add(probe_key)

            # Track original indices
            original_cell_indices.add(cell_idx)
            original_probe_indices.add(probe_idx)

            # Get barcode directly - avoid double lookup
            cell_barcode = barcodes_df.iloc[cell_idx].barcode
            cell_barcode_h5_idx = barcode2h5_idx[cell_barcode]

            # Store data for later matrix construction
            matrix_key = (cell_barcode_h5_idx, probe_key)
            counts_data[matrix_key] += 1
            umi_dup_count = int(umi_dup_count)
            total_umi_data[matrix_key] += umi_dup_count
            if all_pcr_thresholds:
                dup_count_mapping[umi_dup_count][matrix_key] += 1
            percent_supporting_data[matrix_key] += float(percent_supporting)
            n_lines += 1

    print(f"{len(possible_probes)} probe combinations found, {n_lines} valid lines processed.")

    # Create probe index mapping and track original probe indices for each probe_key
    probe2h5_idx = {probe_key: idx for idx, probe_key in enumerate(sorted(possible_probes))}
    probe_key_to_original_idx = {}
    for probe_key in possible_probes:
        probe_name = probe_key[0]
        # Find the original probe index from the manifest
        original_idx = manifest[manifest['name'] == probe_name]['index'].iloc[0]
        probe_key_to_original_idx[probe_key] = original_idx

    # Build sparse matrices efficiently using COO format
    print("Building sparse matrices...", end="")
    n_cells = len(barcode2h5_idx)
    n_probes = len(probe2h5_idx)

    # Pre-allocate arrays for COO matrix construction
    rows = []
    cols = []
    counts_vals = []
    umi_vals = []
    percent_vals = []
    max_dups = max(total_umi_data.values())
    for (cell_idx, probe_key), count in counts_data.items():
        probe_idx = probe2h5_idx[probe_key]
        rows.append(cell_idx)
        cols.append(probe_idx)
        counts_vals.append(count)
        umi_vals.append(total_umi_data[(cell_idx, probe_key)])
        # Normalize percent supporting by count
        percent_vals.append(percent_supporting_data[(cell_idx, probe_key)] / count)

    # Convert to numpy arrays for efficiency
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    counts_vals = np.array(counts_vals, dtype=np.uint32)
    umi_vals = np.array(umi_vals, dtype=np.uint32)
    percent_vals = np.array(percent_vals, dtype=np.float32)

    # Create COO matrices
    counts_matrix = scipy.sparse.coo_matrix((counts_vals, (rows, cols)), shape=(n_cells, n_probes), dtype=np.uint32)
    total_umi_dup_matrix = scipy.sparse.coo_matrix((umi_vals, (rows, cols)), shape=(n_cells, n_probes), dtype=np.uint32)
    percent_supporting_matrix = scipy.sparse.coo_matrix((percent_vals, (rows, cols)), shape=(n_cells, n_probes), dtype=np.float32)

    # Now, we can compute matrices for all PCR thresholds if needed
    # We will do this by progressively subtracing counts by dup count
    filtered_counts = None
    if all_pcr_thresholds:
        filtered_counts = dict()
        curr_counts_matrix = counts_matrix.copy().tolil()  # Start with full counts
        for i in range(0, max_dups):  # Duplicate number to remove
            if i in dup_count_mapping:
                for (cell_idx, probe_key), dup_count in dup_count_mapping[i].items():
                    probe_idx = probe2h5_idx[probe_key]
                    curr_counts_matrix[cell_idx, probe_idx] -= dup_count
            # Store a copy of the current counts matrix
            filtered_counts[i+1] = curr_counts_matrix.copy().tocoo()


    print("Done.")

    # Next we start writing the file
    with h5py.File(final_output, 'w') as output_file:
        # Prepare groups/datasets based on raw_probe_bc_matrix.h5 returned by cellranger
        matrix_grp = output_file.create_group("matrix")
        # List of barcodes
        matrix_grp.create_dataset("barcode",
                                  data=np.array(list(barcode2h5_idx.keys()), dtype='S'),
                                  compression='gzip')
        # Store original cell indices corresponding to each barcode
        original_cell_idx_array = np.array([barcodes_df.index[barcodes_df.barcode == bc].tolist()[0]
                                          for bc in barcode2h5_idx.keys()], dtype=np.uint32)
        matrix_grp.create_dataset("cell_index",
                                  data=original_cell_idx_array,
                                  compression='gzip')

        # List of probes
        matrix_grp.create_dataset("probe",
                                  data=np.array(list(probe2h5_idx.keys()), dtype='S'),
                                  compression='gzip')
        # Store original probe indices corresponding to each probe_key
        original_probe_idx_array = np.array([probe_key_to_original_idx[probe_key]
                                           for probe_key in sorted(probe2h5_idx.keys())], dtype=np.uint32)
        matrix_grp.create_dataset("probe_index",
                                  data=original_probe_idx_array,
                                  compression='gzip')
        output_file.flush()

        # Save cell metadata
        cell_metadata_grp = output_file.create_group("cell_metadata")
        cell_metadata_grp.create_dataset("columns", data=np.array(barcodes_df.columns.values.tolist(), dtype='S'), compression='gzip')
        for col in barcodes_df.columns:
            values = barcodes_df[col].values
            # If it is not an integer or float, then convert to string
            if not np.issubdtype(values.dtype, np.number):
                values = values.astype('S')
            cell_metadata_grp.create_dataset(col, data=values, compression='gzip')
        output_file.flush()

        print("Writing counts...", end="")

        write_sparse_matrix(matrix_grp, "data", counts_matrix)  # Shuffle for better compression
        output_file.flush()
        del counts_matrix  # Free up memory
        write_sparse_matrix(matrix_grp, "total_reads", total_umi_dup_matrix)
        output_file.flush()
        del total_umi_dup_matrix  # Free up memory
        write_sparse_matrix(matrix_grp, "percent_supporting", percent_supporting_matrix)
        output_file.flush()
        del percent_supporting_matrix  # Free up memory

        if all_pcr_thresholds:
            all_pcr_grp = output_file.create_group("pcr_thresholded_counts")
            all_pcr_grp.attrs['max_pcr_duplicates'] = max_dups
            for dup_threshold, matrix in filtered_counts.items():
                write_sparse_matrix(all_pcr_grp, f"pcr{dup_threshold}", matrix)
                output_file.flush()
                del matrix  # Free up memory

        print("Done.")

        print("Writing metadata...", end="")
        # Save the manifest data
        manifest_grp = output_file.create_group("probe_metadata")

        # Save the manifest dataframe in a separate dataset
        manifest_grp.create_dataset("name", data=np.array(manifest['name'], dtype='S'), compression='gzip')
        if 'gene' in manifest.columns:
            manifest_grp.create_dataset("gene", data=np.array(manifest['gene'], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("lhs_probe", data=np.array(manifest['lhs_probe'], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("rhs_probe", data=np.array(manifest['rhs_probe'], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("gap_probe_sequence", data=np.array(manifest['gap_probe_sequence'], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("original_sequence", data=np.array(manifest['original_gap_probe_sequence'], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("index", data=np.array(manifest['index'], dtype=np.uint32), compression='gzip')

        # Save some attributes
        output_file.attrs['plex'] = plex
        output_file.attrs['project'] = output.name
        output_file.attrs['created_date'] = str(pd.Timestamp.now())
        output_file.attrs['n_cells'] = len(barcode2h5_idx)
        output_file.attrs['n_probes'] = int(manifest.shape[0])
        output_file.attrs['n_probe_gapfill_combinations'] = len(probe2h5_idx)
        output_file.attrs['all_pcr_thresholds'] = all_pcr_thresholds
        output_file.attrs['max_pcr_duplicates'] = max_dups
        print("Done.")


def run(output: str, cores: int, overwrite: bool, was_multiplexed: bool, flatten: bool, all_pcr_thresholds: bool):
    if cores < 1:
        cores = os.cpu_count()

    output = Path(output)
    assert output.exists(), f"Output directory does not exist."
    input = output / "probe_reads.tsv.gz"
    if not input.exists():
        input = output / "probe_reads.tsv"
    assert input.exists(), f"Input file not found: {input}"

    print("Reading manifest and barcodes...", end="")
    manifest = read_manifest(output)
    barcodes_df = read_barcodes(output)
    print("Done.")

    # Multiplexed if there is more than one unique number plex indicated
    plexes = barcodes_df.plex.unique().tolist()
    multiplex = len(plexes) > 1
    # # Detect multiplexed experiment
    # demultiplexed_barcodes = dict()
    # for idx, bc in barcodes.items():
    #     probe_bc = bc.split("-")[1]
    #     if probe_bc not in demultiplexed_barcodes:
    #         demultiplexed_barcodes[probe_bc] = dict()
    #     demultiplexed_barcodes[probe_bc][idx] = bc

    if multiplex > 1:
        # Multiplexed run
        print(f"Detected {multiplex}-multiplexed run.")
        print("Collecting counts for each probe barcode...")
        mp = maybe_multiprocess(cores)
        with mp as pool:
            pool.starmap(
                collect_counts,
                [
                    (input, output, manifest, barcodes_df[barcodes_df.plex == plex].copy(), overwrite, plex, flatten, all_pcr_thresholds)
                    for plex in plexes
                ]
            )
        print(f"Counts data saved as counts.[{','.join(plexes)}].h5")
    else:
        if was_multiplexed or plexes[0] > 1:
            print(f"Detected multiplexed run using BC{plexes[0]}.")
        else:
            print("Detected single-plex run.")
        print("Collecting counts...")
        # No need to multithread
        collect_counts(input, output, manifest, barcodes_df, overwrite, int(plexes[0]), was_multiplexed, flatten, all_pcr_thresholds)
        print(f"Counts data saved as counts.1.h5.")

    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Collect counts into a single h5 file. Or multiple if the run was detected to be multiplexed.",
        formatter_class = RichHelpFormatter
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
    )

    parser.add_argument(
        "--output", '-o',
        required=True,
        type=str,
        help="Path to the output directory."
    )

    parser.add_argument(
        "--cores", '-c',
        required=False,
        type=int,
        default=1,
        help="The maximum number of cores to use."
    )

    parser.add_argument(
        "--multiplex", '-m',
        required=False,
        action="store_true",
        help="Hint to the program that the run should be expected to be multiplexed."
    )

    parser.add_argument(
        "--overwrite", '-f',
        required=False,
        action="store_true",
        help="Overwrite the output files if they exist."
    )

    parser.add_argument(
        "--flatten",
        required=False,
        action="store_true",
        help="Flatten the final output to a gzipped tsv file."
    )

    parser.add_argument(
        "--all_pcr_thresholds",
        required=False,
        action="store_true",
        help="If set, the resultant counts file will contain filtered counts for all possible minimum number of PCR duplicates. The parsed object will then have a new obsm field 'X_pcr{n}' for n in 1 to the maximum number of PCR duplicates observed in the data. This will increase the size of the output file, but allow for more flexible downstream filtering."
    )

    args = parser.parse_args()
    run(args.output, args.cores, args.overwrite, args.multiplex, args.flatten, args.all_pcr_thresholds)


if __name__ == "__main__":
    main()
