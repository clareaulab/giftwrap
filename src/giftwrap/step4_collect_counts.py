import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from .utils import read_manifest, read_barcodes, maybe_multiprocess, maybe_gzip, write_sparse_matrix, \
    _tx_barcode_to_oligo, compile_flatfile


def collect_counts(input: Path, output: Path, manifest: pd.DataFrame, barcodes_df: pd.DataFrame, overwrite: bool, plex: int = 1, multiplex: bool = False, flatten: bool = False):
    """
    Generate an h5 file with counts for each barcode.
    :param input: The input file.
    :param output: The output directory.
    :param manifest: The manifest metadata.
    :param barcodes_df: The dataframe containing all barcodes and metadata.
    :param overwrite: Overwrite the output file if it exists.
    :param plex: The plex number.
    :param multiplex: Whether the run was multiplexed.
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
        compile_flatfile(manifest, input, barcodes_df.barcode.values.tolist(), plex, output / f'flat_counts.{plex}.tsv.gz')

    # First we must scan for all possible probe id/gap fill combinations
    possible_probes = set()
    probe_bcs = set()
    n_lines = 0  # Track the number of lines for progress tracking later on
    print("Initial scan for possible probe gapfills...", end="")
    with maybe_gzip(input, 'r') as input_file:
        # Skip the header
        next(input_file)
        for line in input_file:
            cell_idx, probe_idx, probe_bc_idx, umi, gapfill, umi_count, percent_supporting = line.strip().split("\t")
            if int(cell_idx) not in barcode2h5_idx.values() or int(probe_bc_idx) != plex:
                continue
            probe_name = probe_idx2name[int(probe_idx)]
            possible_probes.add((probe_name, gapfill))
            probe_bcs.add(probe_bc_idx)
            n_lines += 1
    print(f"{len(possible_probes)} found.")

    # Now we will define probe/gapfill indices
    probe2h5_idx = dict()
    for idx, (probe_name, gapfill) in enumerate(sorted(possible_probes)):  # Sort for readability
        probe2h5_idx[(probe_name, gapfill)] = idx

    # Next we start reading the file
    with h5py.File(final_output, 'w') as output_file:
        # Prepare groups/datasets based on raw_probe_bc_matrix.h5 returned by cellranger
        matrix_grp = output_file.create_group("matrix")
        # List of barcodes
        matrix_grp.create_dataset("barcode",
                                  data=np.array(list(barcode2h5_idx.keys()), dtype='S'),
                                  compression='gzip')
        # List of probes
        matrix_grp.create_dataset("probe",
                                  data=np.array(list(probe2h5_idx.keys()), dtype='S'),
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

        with maybe_gzip(input, 'r') as input_file:
            # Skip the header
            next(input_file)

            # lil matrix for fast construction
            counts_matrix = scipy.sparse.lil_matrix((len(barcode2h5_idx), len(probe2h5_idx)), dtype=np.uint32)
            total_umi_dup_matrix = scipy.sparse.lil_matrix((len(barcode2h5_idx), len(probe2h5_idx)), dtype=np.uint32)
            percent_supporting_matrix = scipy.sparse.lil_matrix((len(barcode2h5_idx), len(probe2h5_idx)), dtype=np.float32)

            pbar = tqdm(total=n_lines, desc="Collecting counts", unit="umis")
            for line in input_file:
                # Split the line
                cell_idx, probe_idx, probe_bc_idx, umi, gapfill, umi_dup_count, percent_supporting = line.strip().split("\t")
                cell_idx = int(cell_idx)
                probe_idx = int(probe_idx)
                pbar.update(1)
                # Check whether we are dealing with the correct plexed umi for this invocation
                if cell_idx not in barcode2h5_idx.values() or int(probe_bc_idx) != plex:
                    continue  # Skip this line, should be handled elsewhere

                # Now we need to extract some data to add counts
                cell_barcode = barcodes_df.iloc[cell_idx].barcode
                cell_barcode_h5_idx = barcode2h5_idx[cell_barcode]
                probe_name = probe_idx2name[probe_idx]
                probe_h5_idx = probe2h5_idx[(probe_name, gapfill)]
                counts_matrix[cell_barcode_h5_idx, probe_h5_idx] += 1
                total_umi_dup_matrix[cell_barcode_h5_idx, probe_h5_idx] += int(umi_dup_count)
                percent_supporting_matrix[cell_barcode_h5_idx, probe_h5_idx] += float(percent_supporting)
            pbar.close()

        print("Writing counts...", end="")
        # Save the counts matrix
        # Normalize the percent supporting matrix
        # percent_supporting_matrix / counts_matrix will give us a non-sparse matrix, so we have to do some trickery
        # Iterate over the non-zero elements and divide them
        for (i, j) in zip(*percent_supporting_matrix.nonzero()):
            percent_supporting_matrix[i, j] /= counts_matrix[i, j]

        write_sparse_matrix(matrix_grp, "data", counts_matrix)  # Shuffle for better compression
        output_file.flush()
        del counts_matrix  # Free up memory
        write_sparse_matrix(matrix_grp, "total_reads", total_umi_dup_matrix)
        output_file.flush()
        del total_umi_dup_matrix  # Free up memory
        write_sparse_matrix(matrix_grp, "percent_supporting", percent_supporting_matrix)
        output_file.flush()
        del percent_supporting_matrix  # Free up memory
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
        output_file.attrs['n_probes'] = manifest.shape[0]
        output_file.attrs['n_probe_gapfill_combinations'] = len(probe2h5_idx)
        print("Done.")


def run(output: str, cores: int, overwrite: bool, was_multiplexed: bool, flatten: bool):
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
                    (input, output, manifest, barcodes_df[barcodes_df.plex == plex].copy(), overwrite, plex, flatten)
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
        collect_counts(input, output, manifest, barcodes_df, overwrite, int(plexes[0]), was_multiplexed, flatten)
        print(f"Counts data saved as counts.1.h5.")

    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Collect counts into a single h5 file. Or multiple if the run was detected to be multiplexed.",
        formatter_class = RichHelpFormatter
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

    args = parser.parse_args()
    run(args.output, args.cores, args.overwrite, args.multiplex, args.flatten)


if __name__ == "__main__":
    main()
