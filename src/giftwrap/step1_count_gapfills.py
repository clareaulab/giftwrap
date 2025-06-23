import argparse
import functools
import os
import os.path as osp
import gzip
import shutil
from collections import namedtuple, Counter
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import fuzzysearch
import rapidfuzz
from rich_argparse import RichHelpFormatter

from .utils import maybe_multiprocess, batched, read_manifest, sort_tsv_file, FlexFormatInfo, VisiumHDFormatInfo, \
    VisiumFormatInfo, TechnologyFormatInfo, phred_string_to_probs, compute_max_distance, read_probes_input, read_fastqs


# Enum for possible states
class ReadProcessState(Enum):
    FILTERED_NO_CONSTANT = 0
    FILTERED_NO_RHS = 1
    FILTERED_NO_LHS = 2
    FILTERED_NO_PROBE_BARCODE = 3
    FILTERED_NO_CELL_BARCODE = 4
    CORRECTED_RHS = 5
    CORRECTED_LHS = 6
    CORRECTED_BARCODE = 7
    EXACT = 8
    TOTAL_READS = 9  # Placeholder so that we can count the total number of reads


ReadData = namedtuple("ReadData", ["probe_id", "probe_barcode", "gapfill", "gapfill_quality", "cell_barcode", "umi", "umi_quality", "coordinate_x", "coordinate_y"])


def process_reads(reads,
                 rhs_seqs, lhs_seqs,
                 lhs_seq2potential_rhs_seqs,
                 tech_info: TechnologyFormatInfo, names,
                 max_distance,
                 min_lhs_probe_size, min_rhs_probe_size,
                 max_lhs_probe_size, max_rhs_probe_size,
                 multiplex, barcode, allow_indels,
                 skip_constant_seq, unmapped_reads_prefix) -> list[tuple[list[ReadProcessState], Optional[ReadData]]]:
    if allow_indels:  # If allow indels, levenshtein distance is used
        def fuzzysearch_fn(subsequence, sequence, dist):
            return fuzzysearch.find_near_matches(subsequence, sequence, max_l_dist=dist)
    else:  # If not, we use the hamming distance
        def fuzzysearch_fn(subsequence, sequence, dist):
            return fuzzysearch.find_near_matches(subsequence, sequence, max_substitutions=dist, max_insertions=0, max_deletions=0)
    fuzzysearch_fn = functools.lru_cache(maxsize=len(reads) // 2)(fuzzysearch_fn)

    unmapped = []
    results = []
    for (r1, r2) in reads:
        res = process_read(r1, r2, rhs_seqs, lhs_seqs, lhs_seq2potential_rhs_seqs, tech_info, names, max_distance,
                         min_lhs_probe_size, min_rhs_probe_size, max_lhs_probe_size, max_rhs_probe_size, multiplex,
                         barcode, skip_constant_seq, unmapped_reads_prefix, fuzzysearch_fn)

        reasons = res[0]
        data = res[1]
        if data is None:  # Unmapped read
            unmapped.append((reasons[-1], r1, r2))

        results.append((reasons, data))

    save_unmapped_data(unmapped_reads_prefix, unmapped)

    return results


def save_unmapped_data(prefix, unmapped: list[tuple[ReadProcessState, tuple[tuple[str, str, str],tuple[str, str, str]]]]):
    """
    Add unmapped reads to fastq. And tag them with the reason. To deal with asynchronous writing, we will write to temp
    files and then concatenate them at the end.
    """
    if prefix is None:
        return

    prefix = Path(prefix+"_temp")
    r1_dir = prefix / "R1"
    r2_dir = prefix / "R2"
    if not prefix.exists():
        r1_dir.mkdir(parents=True, exist_ok=True)
        r2_dir.mkdir(parents=True, exist_ok=True)

    # Get arbitrary random name
    r1_file = None
    r2_file = None
    while r1_file is None or r2_file is None:
        hex_name = os.urandom(16).hex()
        r1_file = r1_dir / f"{hex_name}"
        r2_file = r2_dir / f"{hex_name}"
        if r1_file.exists() or r2_file.exists():
            r1_file = None
            r2_file = None
    r1_file.touch()
    r2_file.touch()
    with gzip.open(r1_file, 'wt') as f1, gzip.open(r2_file, 'wt') as f2:
        for reason, r1, r2 in unmapped:
            r1_title, r1_seq, r1_quality = r1
            r2_title, r2_seq, r2_quality = r2
            f1.write(f"@{r1_title} {reason}\n{r1_seq}\n+\n{r1_quality}\n")
            f2.write(f"@{r2_title} {reason}\n{r2_seq}\n+\n{r2_quality}\n")


def collect_unmapped_fastq(unmapped_reads_prefix):
    if unmapped_reads_prefix is None:
        return
    print("Collecting unmapped reads...", end="")
    temp_dir = Path(unmapped_reads_prefix+"_temp")
    assert temp_dir.exists()
    r1_dir = temp_dir / "R1"
    r2_dir = temp_dir / "R2"
    assert r1_dir.exists() and r2_dir.exists()
    out_R1_file = Path(unmapped_reads_prefix + "_R1.fastq.gz")
    out_R2_file = Path(unmapped_reads_prefix + "_R2.fastq.gz")

    with gzip.open(out_R1_file, 'at') as f1, gzip.open(out_R2_file, 'at') as f2:
        for r1_file in r1_dir.iterdir():
            # Get the corresponding r2 file
            r2_file = r2_dir / r1_file.name
            assert r2_file.exists()
            with gzip.open(r1_file, 'rt') as f1_temp, gzip.open(r2_file, 'rt') as f2_temp:
                shutil.copyfileobj(f1_temp, f1)
                shutil.copyfileobj(f2_temp, f2)
    # Remove the temp directory
    shutil.rmtree(temp_dir)
    print("Done.")


def process_read(r1, r2,
                 rhs_seqs, lhs_seqs,
                 lhs_seq2potential_rhs_seqs,
                 tech_info: TechnologyFormatInfo, names,
                 max_distance,
                 min_lhs_probe_size, min_rhs_probe_size,
                 max_lhs_probe_size, max_rhs_probe_size,
                 multiplex, barcode, skip_constant_seq,
                 unmapped_reads_prefix,
                 fuzzysearch_fn) -> tuple[list[ReadProcessState], Optional[ReadData]]:
    ((r1_title, r1_seq, r1_quality), (r2_title, r2_seq, r2_quality)) = r1, r2
    if tech_info.read1_length is None:
        r1_len = len(r1_seq)
    else:
        r1_len = tech_info.read1_length
    if tech_info.read2_length is None:
        r2_len = len(r2_seq)
    else:
        r2_len = tech_info.read2_length

    r1_seq = r1_seq[:r1_len]
    r1_quality = r1_quality[:r1_len]

    r2_seq = r2_seq[:r2_len]
    r2_quality = r2_quality[:r2_len]

    states = [ReadProcessState.TOTAL_READS]

    lhs_probes_sizes_differ = min_lhs_probe_size != max_lhs_probe_size
    rhs_probes_sizes_differ = min_rhs_probe_size != max_rhs_probe_size

    # First identify the LHS probe
    match = None
    if lhs_probes_sizes_differ:
        # We need to do a more expensive search since the probe size differences
        # will lead to inflated edit distances

        # Greedy search from longest possible probe size to shortest
        best_score = None
        for size in range(max_lhs_probe_size, min_lhs_probe_size-1, -1):
            potential_lhs_search_space = r2_seq[:size]
            potential_match = rapidfuzz.process.extractOne(
                potential_lhs_search_space,
                lhs_seqs,
                scorer=rapidfuzz.distance.Levenshtein.distance,
                score_cutoff=compute_max_distance(max_lhs_probe_size, max_distance)
            )
            if potential_match is not None:
                if best_score is None or potential_match[1] < best_score:
                    best_score = potential_match[1]
                    match = potential_match
                    match_query = potential_lhs_search_space
                if best_score == 0:  # Exact match
                    break
    else:
        match_query = r2_seq[:max_lhs_probe_size]
        match = rapidfuzz.process.extractOne(
            match_query,
            lhs_seqs,
            scorer=rapidfuzz.distance.Levenshtein.distance,
            score_cutoff=compute_max_distance(max_lhs_probe_size, max_distance)
        )

    if match is None:  # No match found
        return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_LHS], None

    match_size = len(match_query)
    lhs_probe_idx = match[2]

    if lhs_seqs[lhs_probe_idx] != match_query:
        states.append(ReadProcessState.CORRECTED_LHS)

    # prune out lhs sequence
    r2_seq = r2_seq[match_size:]
    r2_quality = r2_quality[match_size:]

    # Now, search for constant sequence to split the RHS and the probe barcode if it exists
    probe_barcode = ""
    if tech_info.has_constant_sequence:
        # Search for up to half of the constant sequence
        constant_seq_match = None
        for const_seq_len in range(len(tech_info.constant_sequence), len(tech_info.constant_sequence) // 2 - 1, -1):
            # The constant sequence may be truncated
            constant_seq_search = fuzzysearch_fn(tech_info.constant_sequence[:const_seq_len],
                                                 r2_seq, compute_max_distance(const_seq_len, max_distance))

            if len(constant_seq_search) > 0:
                potential_constant_seq_match = sorted(constant_seq_search, key=lambda x: (x.dist, -x.start))[0]
                if constant_seq_match is None or potential_constant_seq_match.dist < constant_seq_match.dist:
                    constant_seq_match = potential_constant_seq_match

        if constant_seq_match is None:  # No matches, short circuit
            if skip_constant_seq:
                # If we are skipping the constant seq, we will just assume the constant seq is not present
                rhs_end = len(r2_seq)
                constant_end = -1
                has_constant_seq = False
            else:
                return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_CONSTANT], None
        else:
            has_constant_seq = True
            # If the gap is too long, the constant seq may be cut off. So we will fuzzy match the first N bases of the
            # constant seq against the match to get the final distance
            rhs_end = constant_seq_match.start
            constant_end = constant_seq_match.end


        # Next, if the run is multiplexed or the probe barcode is specified
        # We will need to search for the probe barcode
        if has_constant_seq and tech_info.has_probe_barcode and (multiplex > 1 or barcode > 0):
            probe_bc_search_space = r2_seq[constant_end + tech_info.probe_barcode_start:constant_end + tech_info.probe_barcode_start + tech_info.probe_barcode_length]
            if len(probe_bc_search_space) == 0:  # Short circuit
                return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_PROBE_BARCODE], None

            if multiplex <= 1:  # Singleplex, so we search for the specified barcode
                probe_barcode = tech_info.probe_barcodes[barcode-1]
                if rapidfuzz.distance.Levenshtein.distance(probe_bc_search_space, probe_barcode) > compute_max_distance(len(probe_bc_search_space), max_distance):
                    return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_PROBE_BARCODE], None
            else:  # Multiplexed, so we need to search for the barcode
                probe_barcode_match = rapidfuzz.process.extractOne(
                    probe_bc_search_space,
                    tech_info.probe_barcodes[:multiplex],
                    scorer=rapidfuzz.distance.Levenshtein.distance,
                    score_cutoff=compute_max_distance(len(probe_bc_search_space), max_distance)
                )
                if probe_barcode_match is None:
                    return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_PROBE_BARCODE], None
                probe_barcode = probe_barcode_match[0]
        # Else don't verify since we don't need to demultiplex
        else:
            probe_barcode = tech_info.probe_barcodes[0]
        # If we got to here, there was a valid probe bc or no need to check
    else:
        # No constant
        rhs_end = len(r2_seq)

    # Filter out the constant seq + probe bc
    r2_seq = r2_seq[:rhs_end]
    r2_quality = r2_quality[:rhs_end]

    def find_rhs_match(sequence, rhs_sequence):
        # If there is a constant sequence, it was found and therefore we should have the full RHS length
        rhs_match = None
        if tech_info.has_constant_sequence:
            search = fuzzysearch_fn(rhs_sequence, sequence, compute_max_distance(len(rhs_sequence), max_distance))
            if len(search) > 0:
                rhs_match = sorted(search, key=lambda x: (x.dist, -x.start))[0]
        else:  # If these reads have no constant sequence, there is a chance that the RHS is cut off if it is at the end
            if len(rhs_sequence) < len(r2_seq):  # If the RHS is shorter than the read, we need to allow
                max_rhs_length = len(r2_seq)
                # Allow for up to 8mer RHS prefix (This matches our 10x flex constant sequence search)
                for rhs_len in range(max_rhs_length, 7, -1):
                    search = fuzzysearch_fn(rhs_sequence[:rhs_len], sequence,
                                            compute_max_distance(rhs_len, max_distance))
                    if len(search) == 0:
                        continue
                    search = sorted(search, key=lambda x: (x.dist, -x.start))[0]
                    if rhs_match is None or search.dist < rhs_match.dist:
                        rhs_match = search

            else:  # RHS is not shorter than the read so we should have the full sequence
                search = fuzzysearch_fn(rhs_sequence, sequence,
                                        compute_max_distance(len(rhs_sequence), max_distance))
                if len(search) > 0:
                    rhs_match = sorted(search, key=lambda x: (x.dist, -x.start))[0]
        return rhs_match

    # Now we have to map the RHS probe to the remaining R2 sequence
    # Since multiple RHS can be mapped to a single LHS, we must search for all possible RHS
    potential_rhs_seqs = lhs_seq2potential_rhs_seqs[lhs_seqs[lhs_probe_idx]]
    if len(potential_rhs_seqs) == 1:  # Only one possible RHS
        potential_rhs_probe_idx, potential_rhs_probe_seq = potential_rhs_seqs[0]
        # Since there is only one possible RHS, if this doesn't match then this is invalid
        # Note that if the RHS may be cut off, we need to adjust the search space by allowing for deletions
        rhs_match = find_rhs_match(r2_seq, potential_rhs_probe_seq)

        if rhs_match is None:  # Short circuit
            return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_RHS], None

        rhs_idx = potential_rhs_probe_idx
    else:  # Multiple possible RHS
        # Find the best match
        rhs_match = None
        rhs_idx = None
        # Repeat the same logic as above
        for potential_rhs_probe_idx, potential_rhs_probe_seq in potential_rhs_seqs:
            potential_rhs_match = find_rhs_match(r2_seq, potential_rhs_probe_seq)

            if potential_rhs_match is None:
                continue

            if rhs_match is None or rhs_match.dist > potential_rhs_match.dist:
                rhs_match = potential_rhs_match
                rhs_idx = potential_rhs_probe_idx
        if rhs_match is None:  # Short circuit
            return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_RHS], None

    if rhs_seqs[rhs_idx] != r2_seq[rhs_match.start:rhs_match.end]:
        states.append(ReadProcessState.CORRECTED_RHS)

    # Probe index would be the index of the RHS probe since we eliminated all potential redundancy
    probe_idx = rhs_idx
    # Prune out the RHS probe
    r2_seq = r2_seq[:rhs_match.start]
    r2_quality = r2_quality[:rhs_match.start]

    # What remains should be the gapfill sequence
    gapfill_seq = r2_seq
    gapfill_quality = r2_quality

    # Now that we have parsed R2, we need to parse R1
    umi = r1_seq[tech_info.umi_start:tech_info.umi_start + tech_info.umi_length]
    umi_quality = r1_quality[tech_info.umi_start:tech_info.umi_start + tech_info.umi_length]
    # Prune out the umi
    if tech_info.umi_start < tech_info.cell_barcode_start:
        # Cell barcode is after the UMI, so we can ignore the umi
        start_idx = tech_info.cell_barcode_start
        end_idx = tech_info.cell_barcode_start + tech_info.max_cell_barcode_length
        # cell_barcode = r1_seq[tech_info.cell_barcode_start:tech_info.cell_barcode_start+tech_info.max_cell_barcode_length]
        # cell_barcode_quality = r1_quality[tech_info.cell_barcode_start:tech_info.cell_barcode_start+tech_info.max_cell_barcode_length]
    else:
        # Cell barcode is before the UMI, so we need to extract from before the umi
        start_idx = tech_info.cell_barcode_start
        end_idx = tech_info.cell_barcode_start + tech_info.max_cell_barcode_length
        # cell_barcode = r1_seq[tech_info.cell_barcode_start:tech_info.umi_start]
        # cell_barcode_quality = r1_quality[tech_info.cell_barcode_start:tech_info.umi_start]

    # We have a likely probe. Now we need to correct the barcode
    # was_corrected, cell_barcode = correct_barcode(cell_barcode,
    #                                               np.array(phred_string_to_probs(cell_barcode_quality)),
    #                                               tech_info,
    #                                               )
    cell_barcode, was_corrected = tech_info.correct_barcode(r1_seq,
                                                            compute_max_distance(end_idx - start_idx, max_distance),
                                                            start_idx,
                                                            end_idx,
                                                            )

    if cell_barcode is None:
        return [ReadProcessState.TOTAL_READS, ReadProcessState.FILTERED_NO_CELL_BARCODE], None

    if was_corrected:
        states.append(ReadProcessState.CORRECTED_BARCODE)

    if len(states) == 1:  # Should be equal to one because of the TOTAL_READS state
        states.append(ReadProcessState.EXACT)

    coordinate_x = None
    coordinate_y = None
    if tech_info.is_spatial:
        coordinate_x, coordinate_y = tech_info.barcode2coordinates(cell_barcode)

    return states, ReadData(probe_idx, probe_barcode, gapfill_seq, gapfill_quality, cell_barcode, umi, umi_quality, coordinate_x, coordinate_y)


def search_files(read1s, read2s, output_dir, tech_info,
                 cores=1, n_reads_per_batch=1_000_000, max_distance=2,
                 multiplex=1, barcode=1, allow_indels=False,
                 skip_constant_seq=False, unmapped_reads_prefix=None):
    probes = read_manifest(output_dir)

    if unmapped_reads_prefix:
        unmapped_reads_prefix = os.path.join(output_dir, unmapped_reads_prefix)

    lhs_seqs = probes['lhs_probe'].tolist()
    rhs_seqs = probes['rhs_probe'].tolist()
    names = probes['name'].tolist()

    # Some lhs can map to multiple possible RHS (or vice versa)
    lhs_seq2potential_rhs_seqs = dict()
    for lhs_seq, (rhs_i, rhs_seq) in zip(lhs_seqs, enumerate(rhs_seqs)):
        if lhs_seq not in lhs_seq2potential_rhs_seqs:
            lhs_seq2potential_rhs_seqs[lhs_seq] = []
        lhs_seq2potential_rhs_seqs[lhs_seq].append((rhs_i, rhs_seq))
    # Sort each of the potential RHS by length (Largest first) to make searching attempt to match more difficult sequences first
    for lhs_seq in lhs_seq2potential_rhs_seqs:
        lhs_seq2potential_rhs_seqs[lhs_seq] = list(
            sorted(lhs_seq2potential_rhs_seqs[lhs_seq], key=lambda x: len(x[1]), reverse=True)
        )

    # For selecting search spaces
    max_lhs_probe_size = max(len(lhs) for lhs in lhs_seqs)
    max_rhs_probe_size = max(len(rhs) for rhs in rhs_seqs)
    # If probe sizes are all the same we can short-circuit the search
    min_lhs_probe_size = min(len(lhs) for lhs in lhs_seqs)
    min_rhs_probe_size = min(len(rhs) for rhs in rhs_seqs)

    read1_iterator, read2_iterator = read_fastqs(read1s, read2s)

    # Note we have to map to tuple because starmap expects tuple inputs
    n_jobs = max(cores, 1)
    batched_reads = batched(map(lambda x: (x,), batched(zip(read1_iterator, read2_iterator), n_reads_per_batch // n_jobs)), n_jobs)

    mp = maybe_multiprocess(cores)

    result_reason_counter = Counter()

    barcodes_encountered = dict()

    # Metrics
    total = 0  # Total number of probes
    probe_ids_encountered = set()
    with mp as pool:
        with gzip.open(output_dir / "probe_reads.tsv.gz", 'wt') as f, gzip.open(output_dir / "barcodes.tsv.gz", 'wt') as f2:
            f.write(f"cell_idx\tprobe_idx\tprobe_barcode\tgapfill\tgapfill_quality\tumi\tumi_quality\n")
            f2.write("barcode\tplex")
            if tech_info.is_spatial:
                f2.write("\tin_tissue\tarray_col\tarray_row")
            f2.write("\n")

            job = None
            last_job = None

            def process_data(results):
                nonlocal total
                # Returns a tuple of outputs to write
                for results_batch in results:
                    for (states, data) in results_batch:
                        result_reason_counter.update(states)
                        if data is None:
                            continue
                        total += 1
                        probe_ids_encountered.add(data.probe_id)
                        if tech_info.has_probe_barcode:
                            probe_bc = tech_info.probe_barcode_index(data.probe_barcode)
                        else:
                            probe_bc = 1
                        complete_cell_barcode = tech_info.make_barcode_string(data.cell_barcode, probe_bc, data.coordinate_x, data.coordinate_y, tech_info.has_probe_barcode and (multiplex > 1 or barcode > 0))
                        if complete_cell_barcode not in barcodes_encountered:
                            barcode_id = len(barcodes_encountered)
                            barcodes_encountered[complete_cell_barcode] = barcode_id
                            f2.write(f"{complete_cell_barcode}\t{probe_bc}")  # Record the barcode
                            if tech_info.is_spatial:
                                f2.write(f"\t1\t{data.coordinate_x}\t{data.coordinate_y}")
                            f2.write("\n")
                        cell_id = barcodes_encountered[complete_cell_barcode]
                        # Record the read
                        f.write(f"{cell_id}\t{data.probe_id}\t{probe_bc}\t{data.gapfill}\t{data.gapfill_quality}\t{data.umi}\t{data.umi_quality}\n")

            # Note we parallelize the processing of reads
            # We first process a batch of reads while the next batch is being read
            for i, batch in (pbar := tqdm(enumerate(batched_reads), desc="Processing reads", unit="batches")):
                if job is not None:
                    last_job = job
                job = pool.starmap_async(
                    functools.partial(process_reads,
                                      rhs_seqs=rhs_seqs,
                                      lhs_seqs=lhs_seqs,
                                      lhs_seq2potential_rhs_seqs=lhs_seq2potential_rhs_seqs,
                                      tech_info=tech_info,
                                      names=names,
                                      max_distance=max_distance,
                                      min_lhs_probe_size=min_lhs_probe_size,
                                      min_rhs_probe_size=min_rhs_probe_size,
                                      max_lhs_probe_size=max_lhs_probe_size,
                                      max_rhs_probe_size=max_rhs_probe_size,
                                      multiplex=multiplex,
                                      barcode=barcode,
                                      allow_indels=allow_indels,
                                      skip_constant_seq=skip_constant_seq,
                                      unmapped_reads_prefix=unmapped_reads_prefix),
                    batch
                )
                if last_job is not None:  # Output the previous run, then continue reading the file while the next batch is being processed
                    process_data(last_job.get())
                pbar.set_postfix({name.name: f"{count:,}" for name, count in result_reason_counter.items()})

            if job is not None:  # Process the final batch
                process_data(job.get())
                pbar.set_postfix({name.name: f"{count:,}" for name, count in result_reason_counter.items()})

    # If we were writing unmapped reads, we need to collect them
    collect_unmapped_fastq(unmapped_reads_prefix)

    print("Reads processed.")

    print("Writing statistics...", end="")
    with open(output_dir / "fastq_metrics.tsv", 'w') as f:
        f.write("metric\tvalue\n")
        f.write(f"PROBE_CONTAINING_READS\t{total}\n")
        f.write(f"POSSIBLE_PROBES\t{probes.shape[0]}\n")
        f.write(f"PROBES_ENCOUNTERED\t{len(probe_ids_encountered)}\n")
        for state, count in result_reason_counter.items():
            f.write(f"{state.name}\t{count}\n")

    print("Done")

    print("Sorting reads by cell...", end="")
    sort_tsv_file(output_dir / "probe_reads.tsv.gz", [2, 0, 1], cores=cores)  # Sort by probe bc, cell idx, probe idx
    print("Done!")
    print(f"{total} reads extracted.")


# @functools.lru_cache(maxsize=1_000)
# def find_exact_match(barcode: str, tech_info: TechnologyFormatInfo) -> Optional[str]:
#     for length, bcs in tech_info.length2barcodes().items():  # We assume this is sorted from largest -> smallest
#         barcode_sub = barcode[:length]
#         if barcode_sub in bcs:
#             return barcode_sub
#     return None


# def correct_barcode(read: str, start_idx: int, end_idx: int, tech_info: TechnologyFormatInfo, max_dist: int) -> tuple[bool, Optional[str]]:
#     """
#     Correct a barcode by permuting the barcode by max_dist and checking if it is in the set of barcodes.
#     :param read: The barcode-containing read string.
#     :param barcode_quality: The barcode position qualities (Probability of error).
#     :param tech_info: Information about the technology used.
#     :param max_dist: The maximum edit distance to search for.
#     :return: The corrected barcode, or None if no barcode was found.
#     """
#     match, corrected = tech_info.correct_barcode(barcode, max_dist)  # TODO: Replace rapidfuzz entirely with the trie?
#     return corrected, match

    # # based on: https://github.com/caleblareau/errorcorrect_10xatac_barcodes/blob/main/process_10x_barcodes.py#L115
    # match = find_exact_match(barcode, tech_info)
    # if match:
    #     return False, match
    # elif max_dist <= 0:
    #     return False, None  # The max_dist requires an exact match
    #
    # # If more Ns than max_dist, return None
    # N_indices = [i for i, base in enumerate(barcode) if base == "N"]
    # if len(N_indices) > max_dist:
    #     return False, None
    #
    # # Set the N positions to high error to prioritize their correction
    # barcode_quality[N_indices] = 1.0
    #
    # # Try to correct the barcode
    # for edits in range(1, max_dist + 1):
    #     # Generate all possible edits
    #     # First try to edit just positions with N
    #     if len(N_indices) > edits:  # Not possible to correct fully
    #         continue
    #     elif len(N_indices) > 0:
    #         # all_edits_N = len(N_indices) == edits
    #         # Replace the Ns with all possible bases
    #         for possible_seq in permute_bases(barcode, N_indices):
    #             # match = find_exact_match(possible_seq, tech_info)
    #             # if match:  # Found a base
    #             #     return True, match
    #             # elif not all_edits_N:  # Additional edits must be made
    #             is_corrected, corrected_barcode = correct_barcode(possible_seq, barcode_quality, tech_info, max_dist - edits)
    #             if corrected_barcode is not None:
    #                 return True, corrected_barcode
    #         # If we are here, we failed to correct the Ns so iterate over the loop
    #         continue
    #
    #     # At this point, there are no Ns so we have to do naive correction
    #     for possible_seq in generate_permuted_seqs(barcode, barcode_quality, edits):
    #         match = find_exact_match(possible_seq, tech_info)
    #         if match:
    #             return True, match
    #
    # # If we are here, nothing worked
    # return False, None


def build_manifest(probes, output: Path, overwrite, allow_any_combination):
    print("Indexing probes...", end="")
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        else:
            raise AssertionError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=overwrite)

    df = read_probes_input(probes)

    print(f"{df.shape[0]} unique probes found.")

    if allow_any_combination:
        df['was_defined'] = True

        additional_columns = [c for c in df.columns if c not in {'lhs_probe', 'rhs_probe', 'name', 'was_defined'}]

        # Map all possible LHS to all possible RHS if not already defined
        lhs_name_tuples = df[['lhs_probe', 'name']].drop_duplicates('lhs_probe').itertuples(index=False, name=None)
        rhs_name_tuples = df[['rhs_probe', 'name']].drop_duplicates('rhs_probe').itertuples(index=False, name=None)
        to_add = {
            'lhs_probe': [],
            'rhs_probe': [],
            'name': [],
            'was_defined': [],
        }
        for c in additional_columns:
            to_add[c] = []
        for lhs_probe, lhs_name in lhs_name_tuples:
            for rhs_probe, rhs_name in rhs_name_tuples:
                if df[(df['lhs_probe'] == lhs_probe) & (df['rhs_probe'] == rhs_probe)].shape[0] == 0:
                    to_add['lhs_probe'].append(lhs_probe)
                    to_add['rhs_probe'].append(rhs_probe)
                    to_add['name'].append(f"{lhs_name}/{rhs_name}")
                    to_add['was_defined'].append(False)
                    for c in additional_columns:
                        to_add[c].append(None)
        to_add = pd.DataFrame(to_add)
        df = pd.concat([df, to_add], ignore_index=True)

        print(f"{(~df['was_defined']).sum()} decoy pairings added.")

    # Create an index column
    df.reset_index(drop=True, inplace=True)
    df["index"] = df.index

    # Write the manifest to the output directory
    df.to_csv(output / "manifest.tsv", index=False, sep="\t")


def run(probes,
        read1,
        read2,
        project,
        output,
        cores,
        n_reads_per_batch,
        max_distance,
        technology,
        tech_df,
        overwrite,
        multiplex,
        barcode,
        r1_len,
        r2_len,
        allow_indels,
        skip_constant_seq,
        allow_any_combination,
        unmapped_reads_prefix,
        cellranger_output):
    if (read1 == read2 == project) and project is None:
        raise AssertionError("At least one of the read1, read2, or project arguments must be provided.")
    assert not (multiplex > 1 and barcode > 0), "Multiplex and barcode arguments are mutually exclusive."
    assert (not skip_constant_seq) or (multiplex < 2 and barcode < 2), "Skipping the constant sequence is only valid for singleplex sequencing."

    if isinstance(cellranger_output, str):
        cellranger_output = [cellranger_output]
    has_cellranger = cellranger_output is not None and len(cellranger_output) > 0
    if has_cellranger:
        cellranger_output = [Path(x) for x in cellranger_output]
        print("WTA CellRanger output provided.")
    else:
        cellranger_output = None

    print("Searching for fastq files...", end="")
    if project is not None:
        read1s = []
        read2s = []
        for r1 in sorted(Path(project).parent.glob(Path(project).name + "*_R1*")):
            if r1.suffix not in {".fastq", ".gz", '.fq',}: # Skip non-fastq files
                continue
            read1s.append(str(r1))
            possible_r2 = Path(str(r1).replace("R1", "R2"))
            if not possible_r2.exists():
                raise FileNotFoundError(f"Matching R2 file not found: {possible_r2}")
            read2s.append(str(possible_r2))
    else:
        if '.' in read1 or '.' in read2:  # Assuming these are file names
            assert osp.exists(read1), f"Read1 file not found: {read1}"
            assert osp.exists(read2), f"Read2 file not found: {read2}"
            assert (".gz" in read1) == (".gz" in read2), "Read1 and Read2 must either both be gzipped or not gzipped."
            read1s = [read1]
            read2s = [read2]
        else:  # Assume these are patterns
            read1s = []
            read2s = []
            for r1 in sorted(Path(read1).parent.glob(Path(read1).name + "*")):
                read1s.append(str(r1))
                possible_r2 = Path(read2).parent / r1.name.replace(read1, read2)
                if not possible_r2.exists():
                    raise FileNotFoundError(f"Matching R2 file not found: {possible_r2}")
                read2s.append(str(possible_r2))
    print(f"Found {len(read1s)} pairs of fastq files:")
    for r1, r2 in zip(read1s, read2s):
        print(f"{r1} | {r2}")

    assert osp.exists(probes), f"Probes file not found: {probes}"
    if cores < 1:
        cores = os.cpu_count()
    output = Path(output)

    print("Searching for cell barcodes...", end="")

    cellranger = shutil.which("cellranger")
    if cellranger is None:
        cellranger = shutil.which("spaceranger")
    if cellranger is None:
        barcode_dir = None
    else:
        barcode_dir = Path(cellranger).parent / "lib" / "python" / "cellranger" / "barcodes"
        if not barcode_dir.exists():
            print(f"Warning: Cellranger barcodes directory not found: {barcode_dir}")
            print("Falling back to default barcodes.")
            barcode_dir = None
    print("Done!")

    # Get the technology information
    print("Extracting sequencing technology information...", end="")
    tech_info: TechnologyFormatInfo
    if technology == "Custom":
        print(f"Loading custom technology definition from {tech_df}...", end="")
        tech_module = Path(tech_df).stem
        tech_module = __import__(tech_module)
        # Get the classes and select the first one.
        clazz = [getattr(tech_module, a) for a in dir(tech_module) if isinstance(getattr(tech_module, a), type)][0]
        tech_info = clazz(
            barcode_dir,
            r1_len,
            r2_len
        )
        print("Loaded", clazz.__name__)
    elif technology == "Flex":
        tech_info = FlexFormatInfo(
            barcode_dir,
            r1_len,
            r2_len,
            cellranger_output,
        )
    elif technology == "VisiumHD":
        tech_info = VisiumHDFormatInfo(
            None,
            barcode_dir,
            r1_len,
            r2_len,
            cellranger_output
        )
    else:  # Visium-vVERSION
        version = int(technology.split("-v")[1])
        tech_info = VisiumFormatInfo(
            version,
            barcode_dir,
            r1_len,
            r2_len,
            cellranger_output
        )
    print(f"{tech_info.n_barcodes} cell barcodes found.")

    build_manifest(probes, output, overwrite, allow_any_combination)

    search_files(read1s, read2s, output, tech_info,
                 cores=cores, n_reads_per_batch=n_reads_per_batch, max_distance=max_distance,
                 multiplex=multiplex, barcode=barcode, allow_indels=allow_indels,
                 skip_constant_seq=skip_constant_seq, unmapped_reads_prefix=unmapped_reads_prefix)
    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Quantify the genotypes of Gap-filling probes.", formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--probes", '-p',
        required=True,
        type=str,
        help="Path to the generated gap-filling probe set file."
    )
    parser.add_argument(
        "-r1", "--read1",
        required=False,
        type=str,
        help="Path to the R1 file. Either the fastq/fastq.gz file, or a file prefix to find a set of files."
    )
    parser.add_argument(
        "-r2", "--read2",
        required=False,
        type=str,
        help="Path to the R2 file. Either the fastq/fastq.gz file, or a file prefix to find a set of files."
    )
    parser.add_argument(
        "--unmapped_reads",
        required=False,
        type=str,
        default=None,
        help="If provided, unmapped reads are written to the file prefix given."
    )
    parser.add_argument(
        '--project',
        required=False,
        type=str,
        default=None,
        help="The generic name for the project. Used to automatically find R1 and R2 fastq files. Mutually exclusive with -r1 and -r2 arguments."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Name of the output directory."
    )
    parser.add_argument(
        '-c', '--cores',
        type=int,
        default=1,
        help="The number of cores to use. Less than 1 defaults to the number of available cores."
    )
    parser.add_argument(
        '-n', '--n_reads_per_batch',
        type=int,
        default=10_000_000,
        help="The number of reads to process in a batch. Defaults to 10 million"
    )
    parser.add_argument(
        "-t", '--threshold',
        type=int,
        default=1,
        help="The maximum edit distance for fuzzy matching probes and cell barcodes per 10bp."
    )
    # Enumerate the technologies supported (Flex or Visium)
    parser.add_argument(
        "--technology", '-e',
        required=False,
        type=str,
        default="Flex",
        choices=["Flex", "Visium-v1", "Visium-v2", 'Visium-v3', 'Visium-v4', 'Visium-v5', 'VisiumHD', "Custom"],
        help="The technology used to generate the gap-filling probes. Default is Flex. If 'Custom', you must provide the --tech_def argument."
    )
    parser.add_argument(
        "--tech_def",
        required=False,
        type=str,
        default=None,
        help="The path to the technology definition python file to import. Must include a single class definition that inherits from TechnologyFormatInfo."
    )
    # Overwrite the output directory
    parser.add_argument(
        "--overwrite", '-f',
        required=False,
        action="store_true",
        help="Overwrite the output directory if it exists."
    )
    # Multiplex?
    parser.add_argument(
        '--multiplex', '-m',
        required=False,
        type=int,
        default=1,
        help="The number of probes to multiplexed in the Flex run. Defaults to single plex."
    )
    parser.add_argument(
        '--barcode', '-b',
        required=False,
        type=int,
        default=0,
        help="The barcode number to use for the Flex run. Defaults to no barcode."
    )
    # Sequencing info
    parser.add_argument(
        "--r1_length",
        type=int,
        default=None,
        help="The length of the R1 read. Can optimize the probe mapping speed and accuracy."
    )
    parser.add_argument(
        "--r2_length",
        type=int,
        default=None,
        help="The length of the R2 read. Can optimize the probe mapping speed and accuracy."
    )
    parser.add_argument(
        "--allow_indels",
        required=False,
        action="store_true",
        help="Allow indels in the probe error correction. Note that cell barcode correction is based on the technology used."
    )
    parser.add_argument(
        "--skip_constant_seq",
        required=False,
        action="store_true",
        help="If the technology (i.e. Flex) has a constant sequence in the probe design, do not filter reads for missing it. This is useful for reads that are too short to capture the full probes."
    )
    parser.add_argument('--allow_any_combination',
                        action='store_true',
                        help='Allow any combination of probes to be used for gapfill')
    parser.add_argument(
        "--cellranger_output", '-wta',
        action="append",
        required=False,
        default=None,
        help="Path to either the filtered_feature_bc_matrix.h5 or the sample_filtered_feature_bc_matrix folder from CellRanger. "
             "Can be specified multiple times to merge multiple samples if multiplex (in order of provided barcodes)."
    )

    args = parser.parse_args()

    run(
        args.probes,
        args.read1,
        args.read2,
        args.project,
        args.output,
        args.cores,
        args.n_reads_per_batch,
        args.threshold,
        args.technology,
        args.tech_def,
        args.overwrite,
        args.multiplex,
        args.barcode,
        args.r1_length,
        args.r2_length,
        args.allow_indels,
        args.skip_constant_seq,
        args.allow_any_combination,
        args.unmapped_reads,
        args.cellranger_output
    )


if __name__ == '__main__':
    main()
