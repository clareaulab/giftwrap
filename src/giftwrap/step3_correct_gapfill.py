import argparse
import functools
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from .utils import maybe_multiprocess, batched, maybe_gzip, GzipNamedTemporaryFile, phred_string_to_probs


def process_lines(lines: list[str]) -> tuple[str, bool]:
    split = lines[0]
    # Select the probe sequences (These should have been corrected in the count gapfills step and be the same for all lines)
    probe_idx = split[1]
    probe_bc_idx = split[2]
    cell_barcode_idx = split[0]
    umi = split[5]
    # Count the number to correct
    count = len(lines)
    # Get the gapfills
    gapfills = [tuple(line[3:5]) for line in lines]
    gapfill_seqs = [gapfill for gapfill, _ in gapfills]
    gapfill_quals = [gap_qual for _, gap_qual in gapfills]
    # only one sequence, don't correct
    # If there is complete consensus in the gap fill, don't correct
    if count < 2 or len(set(gapfill_seqs)) == 1:
        return f"{cell_barcode_idx}\t{probe_idx}\t{probe_bc_idx}\t{umi}\t{gapfills[0][0]}\t{count}\t1.0\n", False

    # Convert the quality scores to probabilities
    gapfill_probs = [phred_string_to_probs(q) for q in gapfill_quals]

    # Check if they are the same length
    if len(set(map(len, gapfill_seqs))) != 1:
        # There is a length mismatch.
        # Group by length and then compute the most likely length
        length2count = dict()
        for seq, prob in zip(gapfill_seqs, gapfill_probs):
            if len(seq) not in length2count:
                length2count[len(seq)] = 1
            else:
                length2count[len(seq)] += 1
        most_likely_length = max(length2count.keys(), key=lambda k: length2count[k])

        # FIXME: Re-incorporate errors?
        # length_counts = dict()
        # for seq, prob in zip(gapfill_seqs, gapfill_probs):
        #     # Compute the expected number of errors in the sequence
        #     errors = sum(prob)
        #     if len(seq) not in length_counts:
        #         length_counts[len(seq)] = []
        #     length_counts[len(seq)].append(errors)
        # # Compute the distribution of length counts
        # length_dist = {x: (len(length_counts[x]) / count) for x in length_counts.keys()}
        # # Compute the most likely length as having the least expected errors and most abundant length
        # most_likely_length = max(length_counts.keys(), key=lambda x: np.mean(length_counts[x]) * (1 - length_dist[x]))   # Lower the error rate, the better
        # Filter out the sequences that are not the most likely length
        gapfill_seqs = [seq for seq in gapfill_seqs if len(seq) == most_likely_length]
        gapfill_probs = [probs for probs in gapfill_probs if len(probs) == most_likely_length]

        # If there is only one unique sequence left, write it out
        if len(set(gapfill_seqs)) == 1:
            return f"{cell_barcode_idx}\t{probe_idx}\t{probe_bc_idx}\t{umi}\t{gapfill_seqs[0]}\t{count}\t{len(gapfill_seqs)/count}\n", True

    # Finally, compute the most likely sequence base-by-base
    seq_probs = [dict(A=list(), T=list(), C=list(), G=list()) for _ in range(len(gapfill_seqs[0]))]
    for seq, probs in zip(gapfill_seqs, gapfill_probs):
        for i, (base, prob) in enumerate(zip(seq, probs)):
            if base == 'N':
                seq_probs[i]['A'].append(prob)
                seq_probs[i]['T'].append(prob)
                seq_probs[i]['C'].append(prob)
                seq_probs[i]['G'].append(prob)
            else:
                seq_probs[i][base].append(prob)

    # Compute the most likely base for each position
    most_likely_seq = ""
    supporting = 0
    for probs in seq_probs:
        if len(probs['A']) > 0 and (len(probs['T']) == 0 and len(probs['C']) == 0 and len(probs['G']) == 0):
            most_likely_seq += 'A'
            supporting += 1 / len(seq_probs)
        elif len(probs['T']) > 0 and (len(probs['A']) == 0 and len(probs['C']) == 0 and len(probs['G']) == 0):
            most_likely_seq += 'T'
            supporting += 1 / len(seq_probs)
        elif len(probs['C']) > 0 and (len(probs['A']) == 0 and len(probs['T']) == 0 and len(probs['G']) == 0):
            most_likely_seq += 'C'
            supporting += 1 / len(seq_probs)
        elif len(probs['G']) > 0 and (len(probs['A']) == 0 and len(probs['T']) == 0 and len(probs['C']) == 0):
            most_likely_seq += 'G'
            supporting += 1 / len(seq_probs)
        elif len(probs['A']) == 0 and len(probs['T']) == 0 and len(probs['C']) == 0 and len(probs['G']) == 0:
            supporting += 1 / len(seq_probs)
            continue  # No data, this happens if there was no gapfill
        else:
            # No consensus, so we will use the most likely base
            # Remove keys with no values
            probs = {nuc: quals for nuc, quals in probs.items() if len(quals) > 0}
            n_total =  sum([len(quals) for quals in probs.values()])
            # most_likely_nuc = min(probs.keys(), key=lambda x: np.mean(probs[x]) * (1 - (len(probs[x])/n_total)))  # Lower the error rate, the better
            # FIXME: Include error probs?
            most_likely_nuc = max(probs.keys(), key=lambda k: len(probs[k]))
            most_likely_seq += most_likely_nuc
            supporting += (len(probs[most_likely_nuc]) / n_total) / len(seq_probs)

    # Compute the number of reads that contain the gapfill
    return f"{cell_barcode_idx}\t{probe_idx}\t{probe_bc_idx}\t{umi}\t{most_likely_seq}\t{count}\t{gapfill_seqs.count(most_likely_seq)/count}\n", True


def barcode_umi_name_lines_generator(input_file_handle) -> tuple[list[str]]:
    """
    We assume the lines are sorted by probe_barcode, cell_barcode, umi. Which should have happened in the previous steps.
    :param input_file_handle: The input file handle.
    :return: A tuple containing a list of lines for feeding into the processor.
    """
    curr_probe_barcode = None
    curr_cell_barcode = None
    curr_umi = None
    curr_probe = None
    lines = []
    for line in input_file_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        split = line.split("\t")
        cell_barcode_idx = split[0]
        probe_idx = split[1]
        probe_barcode_idx = split[2]
        umi = split[5]
        if curr_probe_barcode is None:
            curr_cell_barcode = cell_barcode_idx
            curr_probe_barcode = probe_barcode_idx
            curr_umi = umi
            curr_probe = probe_idx
        elif curr_cell_barcode != cell_barcode_idx or curr_probe_barcode != probe_barcode_idx or curr_umi != umi or curr_probe != probe_idx:
            yield lines,
            lines = []
            curr_cell_barcode = cell_barcode_idx
            curr_probe_barcode = probe_barcode_idx
            curr_umi = umi
            curr_probe = probe_idx
        lines.append(split)

    if len(lines) > 0:
        yield lines,


def run(output: str, cores: int, n_groups_per_batch: int):
    if cores < 1:
        cores = os.cpu_count()

    output = Path(output)
    assert output.exists(), f"Output directory does not exist."
    input = output / "probe_reads.tsv.gz"
    if not input.exists():
        input = output / "probe_reads.tsv"
    assert input.exists(), f"Input file not found: {input}"

    mp = maybe_multiprocess(cores)
    total_collapsed = 0
    total_corrected = 0

    # Note the input should already be sorted by barcode, umi, name
    with GzipNamedTemporaryFile() as f, mp as pool:
        with maybe_gzip(input, 'r') as input_file:
            # Skip the header
            next(input_file)
            # Write a new header. Since we have deduplicated, we will drop the gapfill quality and the umi columns
            f.write("cell_idx\tprobe_idx\tprobe_barcode\tumi\tgapfill\tpcr_duplicate_count\tpercent_supporting\n")
            # Process the data
            lines_iterator = batched(barcode_umi_name_lines_generator(input_file), n_groups_per_batch)

            job = None
            last_job = None

            def process_data(results):
                nonlocal total_corrected
                nonlocal total_collapsed
                for final_line, corrected in results:
                    f.write(final_line)
                    total_corrected += corrected
                    total_collapsed += 1

            # Note we parallelize the processing of the lines
            # We first process a batch of lines while the next batch is being read
            for lines in (pbar := tqdm(lines_iterator, desc="Correcting Gapfills...", unit="batches")):
                if job is not None:
                    last_job = job

                job = pool.starmap_async(
                    functools.partial(
                        process_lines,
                    ),
                    lines
                )

                if last_job is not None:
                    process_data(last_job.get())
                pbar.set_postfix(corrected=f"{total_corrected:,}", total=f"{total_collapsed:,}")

            # Process the last batch
            if job is not None:
                process_data(job.get())
                pbar.set_postfix(corrected=f"{total_corrected:,}", total=f"{total_collapsed:,}")

    print(f"{total_corrected} out of {total_collapsed} gapfills corrected.")
    # Save a backup of the original file
    print("Backing up uncorrected file...", end="")
    if input.with_suffix('.bak.gapfill').exists():
        os.remove(input.with_suffix('.bak.gapfill'))
    input.rename(input.with_suffix(".bak.gapfill"))
    print("Done.")
    # os.rename(f.name, input)
    shutil.move(f.name, input)  # Fixes issues with moving across filesystems
    # Change the permissions to read/write enabled
    os.chmod(input, 0o766)
    print(f"{total_collapsed} unique counts identified.")
    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Correct and deduplicate gap fills.", formatter_class=RichHelpFormatter
    )

    parser.add_argument(
        "--output", '-o',
        required=True,
        type=str,
        help="Path to the output directory."
    )

    parser.add_argument(
        '--cores', '-c',
        required=False,
        type=int,
        default=1,
        help="The number of cores to use. Less than 1 defaults to the number of available cores."
    )

    parser.add_argument(
        '--n_groups_per_batch', '-n',
        required=False,
        type=int,
        default=100_000,
        help="The number of (barcode, umi, name) groups to process in a given batch. Defaults to 100 thousand."
    )

    args = parser.parse_args()

    run(args.output, args.cores, args.n_groups_per_batch)


if __name__ == '__main__':
    main()
