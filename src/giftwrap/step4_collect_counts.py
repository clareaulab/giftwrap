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


def collect_counts(input: Path, output: Path, manifest: pd.DataFrame, barcodes_df: pd.DataFrame, overwrite: bool, plex: int = 1, multiplex: bool = False, flatten: bool = False, max_pcr_thresholds: int = 10):
    """
    Generate an h5 file with counts for each barcode.
    :param input: The input file.
    :param output: The output directory.
    :param manifest: The manifest metadata.
    :param barcodes_df: The dataframe containing all barcodes and metadata for the current plex.
    :param overwrite: Overwrite the output file if it exists.
    :param plex: The plex number.
    :param multiplex: Whether the run was multiplexed.
    :param flatten: Whether to also output a flattened tsv file.
    :param max_pcr_thresholds: If greater than 1, generate counts matrices for PCR duplicate thresholds up to this number.
    """
    final_output = output / f"counts.{plex}.h5"
    if final_output.exists() and not overwrite:
        raise AssertionError(f"Output file already exists: {final_output}")
    elif final_output.exists():
        final_output.unlink()

    # Replace the barcode -plex with {probe bc}-1 to match cellranger output
    if multiplex:
        barcodes_df.barcode = barcodes_df.barcode.str.replace(f"-{plex}", f"{_tx_barcode_to_oligo.get(plex, '')}-1")

    # Pre-compute mappings to avoid extremely slow pandas lookups inside the main loop.
    # 1. Map original cell index (from file) to the new, dense HDF5 index (0, 1, 2...)
    original_idx_to_h5_idx = {orig_idx: h5_idx for h5_idx, orig_idx in enumerate(barcodes_df.index)}
    valid_original_indices = set(barcodes_df.index)
    # 2. Map probe name to its original index from the manifest file.
    name_to_original_idx = pd.Series(manifest['index'].values, index=manifest['name']).to_dict()
    # 3. Keep original mappings needed for file processing and writing.
    probe_idx2name = {idx: name for idx, name in enumerate(manifest['name'])}
    barcode2h5_idx = {bc: idx for idx, bc in enumerate(barcodes_df.barcode.values)}

    if flatten:
        compile_flatfile(manifest, str(input), barcodes_df.barcode.values.tolist(), plex,
                         str(output / f'flat_counts.{plex}.tsv.gz'))

    print("Reading and processing file...", end="")
    counts_data = defaultdict(int)
    dup_count_mapping = defaultdict(lambda: defaultdict(int))
    total_umi_data = defaultdict(int)
    percent_supporting_data = defaultdict(float)
    possible_probes = set()
    n_lines = 0
    plex_int = int(plex)

    with maybe_gzip(input, 'r') as input_file:
        # Skip the header
        next(input_file)
        for line in input_file:
            cell_idx_str, probe_idx_str, probe_bc_idx_str, _, gapfill, umi_dup_count_str, percent_supporting_str = line.strip().split("\t")

            # Fast filtering on probe barcode index
            if int(probe_bc_idx_str) != plex_int:
                continue

            # Fast filtering on original cell index
            cell_idx = int(cell_idx_str)
            if cell_idx not in valid_original_indices:
                continue

            # Process valid line
            probe_idx = int(probe_idx_str)
            probe_name = probe_idx2name[probe_idx]
            probe_key = (probe_name, gapfill)
            possible_probes.add(probe_key)

            cell_barcode_h5_idx = original_idx_to_h5_idx[cell_idx]

            matrix_key = (cell_barcode_h5_idx, probe_key)
            counts_data[matrix_key] += 1
            umi_dup_count = int(umi_dup_count_str)
            total_umi_data[matrix_key] += umi_dup_count
            percent_supporting_data[matrix_key] += float(percent_supporting_str)
            dup_count_mapping[umi_dup_count][matrix_key] += 1
            n_lines += 1

    print(f"{len(possible_probes)} probe combinations found, {n_lines} valid lines processed.")

    probe2h5_idx = {probe_key: idx for idx, probe_key in enumerate(sorted(possible_probes))}
    probe_key_to_original_idx = {}
    for probe_key in possible_probes:
        probe_name = probe_key[0]
        # OPTIMIZED: Use the pre-computed dictionary for a fast lookup.
        original_idx = name_to_original_idx[probe_name]
        probe_key_to_original_idx[probe_key] = original_idx

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
        percent_vals.append(percent_supporting_data[(cell_idx, probe_key)] / (count+1e-4))

    rows, cols = np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)
    counts_matrix = scipy.sparse.coo_matrix((np.array(counts_vals, dtype=np.uint32), (rows, cols)),
                                            shape=(n_cells, n_probes))
    total_umi_dup_matrix = scipy.sparse.coo_matrix((np.array(umi_vals, dtype=np.uint32), (rows, cols)),
                                                   shape=(n_cells, n_probes))
    percent_supporting_matrix = scipy.sparse.coo_matrix((np.array(percent_vals, dtype=np.float32), (rows, cols)),
                                                        shape=(n_cells, n_probes))
    # Now, we can compute matrices for all PCR thresholds if needed
    # We will do this by progressively subtracing counts by dup count
    filtered_counts = None
    if max_pcr_thresholds > 1:
        filtered_counts = dict()
        curr_counts_matrix = counts_matrix.copy().tolil()  # Start with full counts
        for i in range(0, min(max_pcr_thresholds, max_dups)):  # Duplicate number to remove
            if i in dup_count_mapping:
                for (cell_idx, probe_key), dup_count in dup_count_mapping[i].items():
                    probe_idx = probe2h5_idx[probe_key]
                    curr_counts_matrix[cell_idx, probe_idx] -= dup_count
            # Store a copy of the current counts matrix
            filtered_counts[i+1] = curr_counts_matrix.copy().tocoo()

    print("Done.")

    with h5py.File(final_output, 'w') as output_file:
        matrix_grp = output_file.create_group("matrix")
        matrix_grp.create_dataset("barcode", data=np.array(list(barcode2h5_idx.keys()), dtype='S'), compression='gzip')

        # The order of barcodes in barcode2h5_idx is the same as in barcodes_df.
        # We can directly use the index of the dataframe, which is much faster than a list comprehension with lookups.
        original_cell_idx_array = barcodes_df.index.values.astype(np.uint32)
        matrix_grp.create_dataset("cell_index", data=original_cell_idx_array, compression='gzip')

        sorted_probe_keys = sorted(probe2h5_idx.keys())
        matrix_grp.create_dataset("probe", data=np.array(sorted_probe_keys, dtype='S'), compression='gzip')

        original_probe_idx_array = np.array([probe_key_to_original_idx[pk] for pk in sorted_probe_keys],
                                            dtype=np.uint32)
        matrix_grp.create_dataset("probe_index", data=original_probe_idx_array, compression='gzip')
        output_file.flush()

        cell_metadata_grp = output_file.create_group("cell_metadata")
        cell_metadata_grp.create_dataset("columns", data=np.array(barcodes_df.columns.values.tolist(), dtype='S'),
                                         compression='gzip')
        for col in barcodes_df.columns:
            values = barcodes_df[col].values
            if not np.issubdtype(values.dtype, np.number):
                values = values.astype('S')
            cell_metadata_grp.create_dataset(col, data=values, compression='gzip')
        output_file.flush()

        print("Writing counts...", end="")
        write_sparse_matrix(matrix_grp, "data", counts_matrix)
        del counts_matrix
        write_sparse_matrix(matrix_grp, "total_reads", total_umi_dup_matrix)
        del total_umi_dup_matrix
        write_sparse_matrix(matrix_grp, "percent_supporting", percent_supporting_matrix)
        del percent_supporting_matrix
        output_file.flush()

        if max_pcr_thresholds > 1:
            all_pcr_grp = output_file.create_group("pcr_thresholded_counts")
            all_pcr_grp.attrs['max_pcr_duplicates'] = max_dups
            for dup_threshold, matrix in filtered_counts.items():
                write_sparse_matrix(all_pcr_grp, f"pcr{dup_threshold}", matrix)
                output_file.flush()
                del matrix  # Free up memory

        print("Done.")

        print("Writing metadata...", end="")
        manifest_grp = output_file.create_group("probe_metadata")
        for col in ['name', 'gene', 'lhs_probe', 'rhs_probe', 'gap_probe_sequence']:
            if col in manifest.columns:
                manifest_grp.create_dataset(col, data=np.array(manifest[col], dtype='S'), compression='gzip')
        manifest_grp.create_dataset("original_sequence",
                                    data=np.array(manifest['original_gap_probe_sequence'], dtype='S'),
                                    compression='gzip')
        manifest_grp.create_dataset("index", data=np.array(manifest['index'], dtype=np.uint32), compression='gzip')

        output_file.attrs['plex'] = plex
        output_file.attrs['project'] = output.name
        output_file.attrs['created_date'] = str(pd.Timestamp.now())
        output_file.attrs['n_cells'] = n_cells
        output_file.attrs['n_probes'] = int(manifest.shape[0])
        output_file.attrs['n_probe_gapfill_combinations'] = n_probes
        output_file.attrs['max_pcr_duplicates'] = max_dups
        print("Done.")


def run(output: str, cores: int, overwrite: bool, was_multiplexed: bool, flatten: bool, max_pcr_thresholds: int):
    if cores < 1:
        cores = os.cpu_count()

    output = Path(output)
    assert output.exists(), f"Output directory does not exist: {output}"
    input_gz = output / "probe_reads.tsv.gz"
    input_tsv = output / "probe_reads.tsv"
    input = input_gz if input_gz.exists() else input_tsv
    assert input.exists(), f"Input file not found in {output}"

    print("Reading manifest and barcodes...", end="")
    manifest = read_manifest(output)
    barcodes_df = read_barcodes(output)
    print("Done.")

    plexes = barcodes_df.plex.unique().tolist()
    multiplex = len(plexes) > 1

    if multiplex:
        print(f"Detected {len(plexes)}-plex run. Collecting counts for each probe barcode...")
        mp = maybe_multiprocess(cores)
        with mp as pool:
            pool.starmap(
                collect_counts,
                [
                    (input, output, manifest, barcodes_df[barcodes_df.plex == plex].copy(), overwrite, plex, multiplex,
                     flatten, max_pcr_thresholds)
                    for plex in plexes
                ]
            )
        print(f"Counts data saved as counts.[{','.join(map(str, sorted(plexes)))}].h5")
    else:
        plex = int(plexes[0])
        if was_multiplexed or plex > 1:
            print(f"Detected single-plex run using BC{plex}.")
        else:
            print("Detected single-plex run.")
        print("Collecting counts...")
        collect_counts(input, output, manifest, barcodes_df, overwrite, plex, was_multiplexed, flatten, max_pcr_thresholds)
        print(f"Counts data saved as counts.{plex}.h5.")

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
        "--max_pcr_thresholds",
        required=False,
        type=int,
        default=10,
        help="If greater than 1, the parsed object will then have a new layer field 'X_pcr{n}' for n in 1 to the maximum number of PCR duplicates observed in the data. This will increase the size of the output file, but allow for more flexible downstream filtering."
    )

    args = parser.parse_args()
    run(args.output, args.cores, args.overwrite, args.multiplex, args.flatten, args.max_pcr_thresholds)


if __name__ == "__main__":
    main()
