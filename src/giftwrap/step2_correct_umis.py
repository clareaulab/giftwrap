import argparse
import functools
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from .utils import maybe_multiprocess, batched, maybe_gzip, GzipNamedTemporaryFile, phred_string_to_probs, \
    permute_bases, generate_permuted_seqs, compute_max_distance


def process_lines(lines: list[str], threshold: int, allow_chimeras: bool) -> tuple[list[str], int, int]:
    probe_umi_to_lines = defaultdict(list)
    umi_len = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        split = line.split("\t")
        probe_id = split[1]
        probe_bc = split[2]
        umi = split[5]
        umi_len = max(umi_len, len(umi))
        probe_umi_to_lines[(probe_id, probe_bc, umi)].append(split)

    # Compute the threshold
    threshold = compute_max_distance(umi_len, threshold)

    all_valid_umis = defaultdict(set)
    final_lines = []
    corrected = 0

    # Nested structure probe_bc -> umi -> probe / count mappings
    tracked_umis = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Next we sort by count and iterate. Note that we remove the quality scores
    for (probe, probe_bc, umi), lines in sorted(probe_umi_to_lines.items(), key=lambda x: (int(x[0][0]), len(x[1])), reverse=True):
        if len(all_valid_umis[probe_bc]) == 0:  # We have no umis yet, so we have to assume the first one is a real umi
            all_valid_umis[probe_bc].add(umi)
            final_lines.extend([line[:6] for line in lines])
            tracked_umis[probe_bc][umi][probe] += len(lines)
            continue
        elif umi in all_valid_umis[probe_bc]:  # Exact match, so we don't need to do anything
            final_lines.extend([line[:6] for line in lines])
            tracked_umis[probe_bc][umi][probe] += len(lines)
            continue
        else:  # Now we need to see if there is a fuzzy match
            # match = rapidfuzz.process.extractOne(
            #     umi,
            #     all_valid_umis[probe_bc],
            #     scorer=rapidfuzz.distance.Levenshtein.distance,
            #     score_cutoff=threshold
            # )
            # if match is None:  # No match
            # Before promoting this to being a UMI, check the probability of each position being errant
            # We can do this by checking the umi quality scores
            qualities = np.array([phred_string_to_probs(line[6]) for line in lines]) # n_lines x n_bases (probability of error)
            # We can now calculate the probability of the base position being incorrect. Note that these are all
            # independent events
            probs = np.prod(qualities, axis=0)
            # Set all N positions to 100% error
            probs = np.where(np.array([base == 'N' for base in umi]), 1.0, probs)
            # Edit up to threshold positions to try finding an exact match
            found_existing = False
            for i in range(1, threshold + 1):
                for permuted_umi in generate_permuted_seqs(umi, probs, i):
                    if permuted_umi in all_valid_umis[probe_bc]:  # Found a match!
                        final_lines.extend([line[:5] + [permuted_umi] for line in lines])
                        corrected += len(lines)
                        found_existing = True
                        tracked_umis[probe_bc][permuted_umi][probe] += len(lines)
                        break

            if not found_existing:  # Still no match, so add to the list if there are no Ns
                all_valid_umis[probe_bc].add(umi)  # No match, so add to the list
                final_lines.extend([line[:6] for line in lines])
                tracked_umis[probe_bc][umi][probe] += len(lines)

            # else:  # There is a fuzzy match
            #     match_umi, score, match_umi_index = match
            #     for line in lines:
            #         line[5] = match_umi
            #     final_lines.extend(["\t".join(line[:6]) for line in lines])
            #     corrected += len(lines)

    # Now that we have all valid umis, create a list of umis to filter if chimeric
    dropped = 0
    if not allow_chimeras:
        to_drop = set()
        for probe_bc, umi_to_probes in tracked_umis.items():
            # If there are multiple probes for the same umi, we need to consider dropping
            for umi, probes in umi_to_probes.items():
                if len(probes) > 1:
                    # If there are multiple probes, we need to select the most common one
                    most_common_probe = max(probes.items(), key=lambda x: x[1])[0]
                    for probe, count in probes.items():
                        if probe != most_common_probe:
                            to_drop.add((
                                probe_bc, umi, probe
                            ))
                            dropped += count
        # Now create a filter for the final_lines list
        final_lines = filter(
            lambda x: (x[2], x[5], x[1]) not in to_drop,
            final_lines
        )

    # With UMIs corrected, we will now re-sort this chunk according to probe bc, cell bc, umi
    final_lines = sorted(final_lines, key=lambda x: (x[2], x[0], x[5]))

    return ["\t".join(line) for line in final_lines], corrected, dropped


def barcode_lines_generator(input_file_handle) -> tuple[list[str]]:
    # By assuming the lines are sorted by probe_bc/cell barcode/probe id, we can naively group them by just iterating through the file
    curr_barcode = None
    lines = []
    # Iterate cell-by-cell
    for line in input_file_handle:
        barcode = line.split("\t")[0]
        if barcode != curr_barcode:
            if len(lines) > 0:
                yield lines,
            lines = [line]
            curr_barcode = barcode
        else:
            lines.append(line)
    if len(lines) > 0:
        yield lines,


def run(output: str, threshold: int, cores: int, n_cells_per_batch: int, allow_chimeras: bool):
    if cores < 1:
        cores = os.cpu_count()

    output = Path(output)
    assert output.exists(), f"Output directory does not exist."
    input = output / "probe_reads.tsv.gz"
    if not input.exists():
        input = output / "probe_reads.tsv"
    assert input.exists(), f"Input file not found: {input}"

    # Note that this assumes that barcodes are sorted

    mp = maybe_multiprocess(cores)
    total_corrected = 0
    total_dropped = 0
    total = 0
    with GzipNamedTemporaryFile() as f, mp as pool:
        with maybe_gzip(input, 'r') as input_file:
            # Write the header
            header = next(input_file).rstrip()
            # Remove the umi quality column (last column)
            header = "\t".join(header.split("\t")[:-1])
            # Write the header back
            f.write(header)
            f.write("\n")
            # Process the reads
            lines_iterator = batched(barcode_lines_generator(input_file), n_cells_per_batch)  # Group by barcode

            job = None
            last_job = None

            def process_data(results):
                nonlocal total_corrected
                nonlocal total_dropped
                nonlocal total
                for final_lines, corrected, dropped in results:
                    f.write("\n".join(final_lines))
                    f.write("\n")
                    total_corrected += corrected
                    total_dropped += dropped
                    total += len(final_lines) + dropped

            # Note we parallelize the processing of reads
            # We first process a batch of reads while the next batch is being read
            for lines in (pbar := tqdm(lines_iterator, desc="Correcting UMIs", unit="barcode batches")):
                if job is not None:
                    last_job = job

                job = pool.starmap_async(
                    functools.partial(
                        process_lines,
                        threshold=threshold,
                        allow_chimeras=allow_chimeras
                    ),
                    lines
                )

                if last_job is not None:
                    process_data(last_job.get())
                pbar.set_postfix({"total": f"{total:,}", "corrected": f"{total_corrected:,}", "dropped": f"{total_dropped:,}"})

            # Process the last batch
            if job is not None:
                process_data(job.get())
                pbar.set_postfix({"total": f"{total:,}", "corrected": f"{total_corrected:,}", "dropped": f"{total_dropped:,}"})

    # Save a backup of the original file
    print("Backing up uncorrected file...", end="")
    if input.with_suffix('.bak.umi').exists():
        os.remove(input.with_suffix('.bak.umi'))
    input.rename(input.with_suffix(".bak.umi"))
    print("Done.")
    # os.rename(f.name, input)
    shutil.move(f.name, input)  # Fixes issues with moving across filesystems
    # Change the permissions to read/write enabled
    os.chmod(input, 0o766)
    print(f"{total_corrected} UMIs corrected.")
    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Correct cell UMIs and filter out identical values.", formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--output", '-o',
        required=True,
        type=str,
        help="The name of the output directory."
    )
    parser.add_argument(
        "--threshold", '-t',
        required=False,
        type=int,
        default=1,
        help="The edit distance threshold per 10bp for fuzzy matching. Default is 1."
    )
    parser.add_argument(
        '--cores', '-c',
        required=False,
        type=int,
        default=1,
        help="The number of cores to use. Less than 1 defaults to the number of available cores."
    )
    parser.add_argument(
        '--n_cells_per_batch', '-n',
        required=False,
        type=int,
        default=1_000,
        help="The number of cells to process in a given batch. Defaults to 1 thousand."
    )

    parser.add_argument(
        '--allow_chimeras', '-ac',
        required=False,
        action='store_true',
        help="Allow chimeric gapfills. If unset, umis that occur multiple times per cell will be collapsed to the most common probe. If set, there is no collapsing."
    )

    args = parser.parse_args()

    run(args.output, args.threshold, args.cores, args.n_cells_per_batch, args.allow_chimeras)


if __name__ == "__main__":
    main()
