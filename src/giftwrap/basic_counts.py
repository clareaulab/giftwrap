import argparse
import gzip
from pathlib import Path
import functools

from rich_argparse import RichHelpFormatter
import pandas as pd
from tqdm import tqdm
import fuzzysearch

from .utils import read_probes_input, FlexFormatInfo, read_fastqs


@functools.lru_cache(maxsize=1024)
def lhs_probe_search(lhs_universe: tuple[tuple[tuple[str, str]]], r2_seq: str, fuzzy_search: bool) -> tuple[list[tuple[str, str]], str, int]:
    lhs_hit = None
    if fuzzy_search:
        start_i = len(r2_seq)
        best_dist = 10000
        for lhs_probe_group in lhs_universe:
            if lhs_hit is None:
                for lhs_probe, lhs_probe_name in lhs_probe_group:
                    match = fuzzysearch.find_near_matches(lhs_probe, r2_seq, max_l_dist=2)
                    if len(match) > 0 and (match[0].end <= start_i or match[0].dist < best_dist):
                        lhs_hit = (lhs_probe, lhs_probe_name)
                        start_i = match[0].start
                        best_dist = match[0].dist
                        if best_dist == 0:
                            break
    else:
        start_i = 0
        for lhs_probe_group in lhs_universe:
            if lhs_hit is None:  # Only move to the next group if no hits are found with the longest LHS
                for lhs_probe, lhs_probe_name in lhs_probe_group:
                    if r2_seq.startswith(lhs_probe):
                        lhs_hit = (lhs_probe, lhs_probe_name)
                        break

    if lhs_hit is None:  # No hits found
        return lhs_hit, r2_seq, -1
    return lhs_hit, r2_seq[start_i + len(lhs_hit[0]):], start_i


@functools.lru_cache(maxsize=1024)
def rhs_probe_search(rhs_universe: tuple[tuple[tuple[str, str]]], r2_seq: str, fuzzy_search: bool) -> tuple[list[tuple[str, str]], str, int]:
    rhs_hit = None
    start_i = len(r2_seq)
    if fuzzy_search:
        best_dist = 10000
        for rhs_probe_group in rhs_universe:
            if rhs_hit is None:
                for rhs_probe, rhs_probe_name in rhs_probe_group:
                    match = fuzzysearch.find_near_matches(rhs_probe, r2_seq, max_l_dist=2)
                    if len(match) > 0 and (match[0].start <= start_i or match[0].dist < best_dist):
                        rhs_hit = (rhs_probe, rhs_probe_name)
                        start_i = match[0].start
                        best_dist = match[0].dist
                        if best_dist == 0:
                            break
    else:
        for rhs_probe_group in rhs_universe:
            if rhs_hit is None:
                for rhs_probe, rhs_probe_name in rhs_probe_group:
                    if rhs_probe in r2_seq:
                        index = r2_seq.index(rhs_probe)
                        if index < start_i:
                            rhs_hit = (rhs_probe, rhs_probe_name)
                            start_i = index

    if rhs_hit is None:  # No hits found
        return rhs_hit, r2_seq, -1
    return rhs_hit, r2_seq[:start_i], start_i


def maybe_write(f_r1, f_r2, r1_name, r1_seq, r1_qual, r2_name, r2_seq, r2_qual):
    if f_r1 is not None:
        f_r1.write(f"@{r1_name}\n{r1_seq}\n+\n{r1_qual}\n")
    if f_r2 is not None:
        f_r2.write(f"@{r2_name}\n{r2_seq}\n+\n{r2_qual}\n")


def run(probes, project, output, make_unparsed_fastq, correct_barcodes, fuzzy_search):
    probes = read_probes_input(probes)
    # Sort probes by the the length of the lhs and rhs probe
    probes["lhs_len"] = probes.lhs_probe.str.len()
    probes["rhs_len"] = probes.rhs_probe.str.len()
    # Split probes by length (check for longest length first)
    lhs_probes = []
    rhs_probes = []
    curr_len = None
    for i, row in probes.sort_values(["lhs_len"], ascending=False).iterrows():
        if curr_len is None or curr_len != row["lhs_len"]:
            curr_len = row["lhs_len"]
            lhs_probes.append(list())
        lhs_probes[-1].append((row["lhs_probe"], row["name"]))
    curr_len = None
    for i, row in probes.sort_values(["rhs_len"], ascending=False).iterrows():
        if curr_len is None or curr_len != row["rhs_len"]:
            curr_len = row["rhs_len"]
            rhs_probes.append(list())
        rhs_probes[-1].append((row["rhs_probe"], row["name"]))
    # Convert lists to tuples so that they are immutable
    lhs_probes = tuple(tuple(group) for group in lhs_probes)
    rhs_probes = tuple(tuple(group) for group in rhs_probes)
    matching_probe_pairs = {(row["lhs_probe"], row["rhs_probe"]): row["name"] for i, row in probes.iterrows()}
    name2gapfill = {row["name"]: row["gap_probe_sequence"] for i, row in probes.iterrows()}

    base_output = output + "_parsed_reads.tsv"
    agg_output = output + "_parsed_counts.tsv"
    if make_unparsed_fastq:
        unparsed_r1 = output  + "_unparsed_R1.fastq.gz"
        unparsed_r2 = output + "_unparsed_R2.fastq.gz"

        unparsed_r1 = gzip.open(unparsed_r1, "wt")
        unparsed_r2 = gzip.open(unparsed_r2, "wt")
    else:
        unparsed_r1 = None
        unparsed_r2 = None

    # Find the R1 and R2 files
    read1s = []
    read2s = []
    for r1 in sorted(Path(project).parent.glob(Path(project).name + "*_R1*")):
        if r1.suffix not in {".fastq", ".gz", '.fq'}:  # Skip non-fastq files
            continue
        read1s.append(str(r1))
        possible_r2 = Path(str(r1).replace("R1", "R2"))
        if not possible_r2.exists():
            raise FileNotFoundError(f"Matching R2 file not found: {possible_r2}")
        read2s.append(str(possible_r2))

    flex_format = FlexFormatInfo()
    
    read1_iterator, read2_iterator = read_fastqs(read1s, read2s)

    tracker = dict(
        lhs_probe=[],
        lhs_sequence=[],
        lhs_probe_start=[],
        rhs_probe=[],
        rhs_sequence=[],
        rhs_probe_start=[],
        cell_barcode=[],
        umi=[],
        gapfill=[],
        matched_pairing=[],  # Whether the correct probes were called
        matched_gapfill=[]  # Whether the correct probes were paired + gapfill
    )

    for r1, r2 in tqdm(zip(read1_iterator, read2_iterator), desc="Processing reads", unit="reads"):
        try:
            (r1_name, r1_seq, r1_qual), (r2_name, r2_seq, r2_qual) = r1, r2

            # LHS probe search first (longest to shortest)
            lhs_hit, adjusted_r2_seq, lhs_start = lhs_probe_search(lhs_probes, r2_seq, fuzzy_search)
            if lhs_hit is None:  # Reject
                maybe_write(unparsed_r1, unparsed_r2, r1_name, r1_seq, r1_qual, r2_name, r2_seq, r2_qual)
                continue

            # RHS probe search next (longest to shortest)
            rhs_hit, adjusted_r2_seq, rhs_start = rhs_probe_search(rhs_probes, adjusted_r2_seq, fuzzy_search)
            if rhs_hit is None:  # Reject
                maybe_write(unparsed_r1, unparsed_r2, r1_name, r1_seq, r1_qual, r2_name, r2_seq, r2_qual)
                continue

            # rhs_start is wrt R2 without the LHS
            rhs_start += lhs_start + len(lhs_hit[0])

            # Remaining R2 should be gapfill
            gapfill = adjusted_r2_seq

            # Now extract the cell barcode and UMI
            umi = r1_seq[flex_format.umi_start:flex_format.umi_start + flex_format.umi_length]

            if correct_barcodes:
                # Correct the cell barcode
                cell_barcode = flex_format.correct_barcode(r1_seq, max_mismatches=2,
                                                           start_idx=flex_format.cell_barcode_start, end_idx=flex_format.umi_start)
                if cell_barcode is None:
                    maybe_write(unparsed_r1, unparsed_r2, r1_name, r1_seq, r1_qual, r2_name, r2_seq, r2_qual)
                    continue
                else:
                    cell_barcode = cell_barcode[0]
            else:
                cell_barcode = r1_seq[flex_format.cell_barcode_start:flex_format.umi_start]

            tracker["lhs_probe"].append(lhs_hit[1])
            tracker["lhs_sequence"].append(lhs_hit[0])
            tracker["lhs_probe_start"].append(lhs_start)
            tracker["rhs_probe"].append(rhs_hit[1])
            tracker["rhs_sequence"].append(rhs_hit[0])
            tracker["rhs_probe_start"].append(rhs_start)
            tracker["cell_barcode"].append(cell_barcode)
            tracker["umi"].append(umi)
            tracker["gapfill"].append(gapfill)
            probe_name = matching_probe_pairs.get((lhs_hit[0], rhs_hit[0]), None)
            expected_gapfill = name2gapfill.get(probe_name, None)
            tracker["matched_pairing"].append(probe_name is not None)
            tracker["matched_gapfill"].append(gapfill == expected_gapfill)
        # Catch interrupt
        except KeyboardInterrupt:
            print("Interrupted. Exiting early.")
            break

    if make_unparsed_fastq:
        unparsed_r1.close()
        unparsed_r2.close()

    tracker = pd.DataFrame(tracker)

    if tracker.empty:
        print("No reads parsed. Exiting.")
        return

    # Count the occurrences of all the unique combinations and add as new column
    tracker = tracker.groupby(["lhs_probe", "lhs_sequence", "lhs_probe_start", "rhs_probe", "rhs_sequence", "rhs_probe_start", "cell_barcode", "umi", "gapfill", "matched_pairing", "matched_gapfill"]).size().reset_index()
    tracker.columns = ["lhs_probe", "lhs_sequence", "lhs_probe_start", "rhs_probe", "rhs_sequence", "rhs_probe_start", "cell_barcode", "umi", "gapfill", "matched_pairing", "matched_gapfill", 'count']
    tracker.to_csv(base_output, sep="\t", index=False)

    # Aggregate to psuedobulk
    tracker = tracker.groupby(["lhs_probe", "lhs_sequence", "lhs_probe_start", "rhs_probe", "rhs_sequence", "rhs_probe_start", "gapfill", "matched_pairing", "matched_gapfill"]).agg({'count': 'sum'}).reset_index()
    tracker.columns = ["lhs_probe", "lhs_sequence", "lhs_probe_start", "rhs_probe", "rhs_sequence", "rhs_probe_start", "gapfill", "matched_pairing", "matched_gapfill", 'count']
    tracker.to_csv(agg_output, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Quick and basic script to quantify *reads* without error correction for diagnosing potential issues. This is currently only applicable for Flex.",
        formatter_class=RichHelpFormatter
    )

    # No bells and whistles

    # Input arguments
    parser.add_argument(
        "--probes", '-p',
        required=True,
        type=str,
        help="Path to the generated gap-filling probe set file."
    )

    parser.add_argument(
        "--project",
        required=True,
        type=str,
        help="The name of the project for finding the R1 and R2 files."
    )

    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help="The prefix for output files."
    )

    parser.add_argument(
        '--unparsed-fastq',
        action='store_true',
        help="If set, a new fastq will be created containing the unparsed reads."
    )

    parser.add_argument(
        '--correct-barcodes',
        action='store_true',
        help="If set, will attempt to correct cell barcodes. Uncorrectable barcodes will be considered unparsed."
    )

    parser.add_argument(
        '--fuzzy-search',
        action='store_true',
        help="If set, will use fuzzy search to find probe sequences."
    )

    args = parser.parse_args()

    run(args.probes, args.project, args.output, args.unparsed_fastq, args.correct_barcodes, args.fuzzy_search)


if __name__ == "__main__":
    main()
    # run("../../trna_comb_g3_AAV_SPLIT.csv", "../../G3", "test", False, True, True)
