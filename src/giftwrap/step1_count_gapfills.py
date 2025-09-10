import warnings, os
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # inherit to subprocesses

import argparse
import functools
import os.path as osp
import gzip
import shutil
import sys
from collections import namedtuple, Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

from .utils import maybe_multiprocess, batched, read_manifest, sort_tsv_file, FlexFormatInfo, VisiumHDFormatInfo, \
    VisiumFormatInfo, TechnologyFormatInfo, compute_max_distance, read_probes_input, read_fastqs, \
    ReadProcessState, ProbeParser

ReadData = namedtuple("ReadData", ["probe_id", "probe_barcode", "gapfill", "gapfill_quality", "cell_barcode", "umi", "umi_quality", "coordinate_x", "coordinate_y"])


def process_reads(reads: list[tuple[tuple[str, str, str], tuple[str, str, str]]],
                  tech_info: TechnologyFormatInfo,
                  probe_parser: ProbeParser,
                  max_distance: int,
                  skip_constant_seq: bool,
                  unmapped_reads_prefix: Optional[str],
                  flexible_start: bool) -> list[tuple[list[ReadProcessState], Optional[ReadData]]]:
    unmapped = []
    results = []
    for (r1, r2) in reads:
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

        probe_idx, gap_seq, gap_start, gap_end, probe_bc, states = probe_parser.parse_probe(
            r2_seq, max_distance, skip_constant_seq, flexible_start
        )

        states = states.copy()

        if probe_idx is None:  # Failed to parse R2
            results.append(
                (states, None)
            )
            unmapped.append((states[-1], r1, r2))
            continue

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

        cell_barcode, corrections = tech_info.correct_barcode(r1_seq,
                                                                compute_max_distance(end_idx - start_idx, max_distance),
                                                                start_idx,
                                                                end_idx,
                                                                )

        if cell_barcode is None:
            results.append(
                (states + [ReadProcessState.FILTERED_NO_CELL_BARCODE], None)
            )
            unmapped.append((ReadProcessState.FILTERED_NO_CELL_BARCODE, r1, r2))
            continue

        if corrections > 0:
            states.append(ReadProcessState.CORRECTED_BARCODE)

        if len(states) == 1:  # Should be equal to one because of the TOTAL_READS state
            states.append(ReadProcessState.EXACT)

        coordinate_x = None
        coordinate_y = None
        if tech_info.is_spatial:
            coordinate_x, coordinate_y = tech_info.barcode2coordinates(cell_barcode)

        results.append((
            states,
            ReadData(
                probe_idx, probe_bc, gap_seq, r2_quality[gap_start:gap_end], cell_barcode, umi, umi_quality, coordinate_x, coordinate_y
            )
        ))
    #
    # unmapped = []
    # results = []
    # for (r1, r2) in reads:
    #     res = process_read(r1, r2, rhs_seqs, lhs_seqs, lhs_seq2potential_rhs_seqs, tech_info, names, max_distance,
    #                      min_lhs_probe_size, min_rhs_probe_size, max_lhs_probe_size, max_rhs_probe_size, multiplex,
    #                      barcode, skip_constant_seq, unmapped_reads_prefix, flexible_start_mapping, fuzzysearch_fn)
    #
    #     reasons = res[0]
    #     data = res[1]
    #     if data is None:  # Unmapped read
    #         unmapped.append((reasons[-1], r1, r2))
    #
    #     results.append((reasons, data))

    save_unmapped_data(unmapped_reads_prefix, unmapped)

    return results


def save_unmapped_data(prefix, unmapped: list[tuple[
    ReadProcessState, tuple[tuple[str, str, str],tuple[str, str, str]]]]):
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


def search_files(read1s, read2s, output_dir, tech_info,
                 cores=1, n_reads_per_batch=1_000_000, max_distance=2,
                 multiplex=1, barcode=1, allow_indels=False,
                 skip_constant_seq=False, unmapped_reads_prefix=None,
                 flexible_start=False):
    probes = read_manifest(output_dir)

    lhs_seqs = probes['lhs_probe'].tolist()
    rhs_seqs = probes['rhs_probe'].tolist()
    names = probes['name'].tolist()
    if multiplex > 1:
        probe_bcs = list(range(1, multiplex+1))
    elif barcode > 0:
        probe_bcs = [barcode]
    else:
        probe_bcs = None
    probe_parser = ProbeParser(
        lhs_seqs, rhs_seqs, names, tech_info, probe_bcs, allow_indels
    )

    if unmapped_reads_prefix:
        unmapped_reads_prefix = os.path.join(output_dir, unmapped_reads_prefix)

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
                                      tech_info=tech_info,
                                      probe_parser=probe_parser,
                                      max_distance=max_distance,
                                      skip_constant_seq=skip_constant_seq,
                                      unmapped_reads_prefix=unmapped_reads_prefix,
                                      flexible_start=flexible_start),
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


def build_manifest(probes, output: Path, overwrite, allow_any_combination, trim_probes):
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

    if trim_probes > 0:
        print("Trimming probes to an expected length of", trim_probes)
        for i, row in list(df.iterrows()):
            original_gap_probe_sequence = row['original_gap_probe_sequence']
            gap_probe_sequence = row['gap_probe_sequence']
            orig_gap_is_na = (original_gap_probe_sequence == "NA" or pd.isna(original_gap_probe_sequence))
            gap_is_na = (gap_probe_sequence == "NA" or pd.isna(gap_probe_sequence))
            if orig_gap_is_na and gap_is_na:
                gap_length = 0
            elif not orig_gap_is_na and gap_is_na:
                gap_length = len(original_gap_probe_sequence)
            elif orig_gap_is_na and not gap_is_na:
                gap_length = len(gap_probe_sequence)
            else:
                gap_length = max(len(original_gap_probe_sequence), len(gap_probe_sequence))

            to_trim_from_rhs = trim_probes - gap_length - len(row['lhs_probe'])
            if to_trim_from_rhs < 0:
                print(f"Warning: Probe {row['name']} is shorter than the trim length by {-to_trim_from_rhs}bp. Not trimming.")
                continue
            new_rhs = row['rhs_probe'][:to_trim_from_rhs]
            if len(new_rhs) < 5:
                print(f"Warning: Probe {row['name']} is trimmed to less than 5bp. This may cause issues!")
            df.at[i, 'rhs_probe'] = new_rhs
        print("Probes trimmed.")

    # Create an index column
    df.reset_index(drop=True, inplace=True)
    df["index"] = df.index

    # Write the manifest to the output directory
    df.to_csv(output / "manifest.tsv", index=False, sep="\t")


def run(probes,
        trim_probes,
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
        cellranger_output,
        flexible_start):
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

    build_manifest(probes, output, overwrite, allow_any_combination, trim_probes)

    search_files(read1s, read2s, output, tech_info,
                 cores=cores, n_reads_per_batch=n_reads_per_batch, max_distance=max_distance,
                 multiplex=multiplex, barcode=barcode, allow_indels=allow_indels,
                 skip_constant_seq=skip_constant_seq, unmapped_reads_prefix=unmapped_reads_prefix,
                 flexible_start=flexible_start)
    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Quantify the genotypes of Gap-filling probes.", formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {sys.modules['giftwrap'].__version__}",
        help="Show the version of the GIFTwrap pipeline."
    )
    parser.add_argument(
        "--probes", '-p',
        required=True,
        type=str,
        help="Path to the generated gap-filling probe set file."
    )
    parser.add_argument(
        "--trim_probes",
        type=int,
        default=-1,
        required=False,
        help="If > 0, trim the probes to the given length before mapping. This can be useful if the probes have a common sequence at the end that is not expected to be sequenced. If the probe file contains expected gapfills, this will be used to inform the trimming."
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
        "--flexible_start_mapping",
        required=False,
        default=False,
        action="store_true",
        help="If set with, we no longer assume that the R2 read starts with the LHS probe and that there may be an insertion that would need to be trimmed."
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
        choices=["Flex", 'VisiumHD', "Visium-v1", "Visium-v2", 'Visium-v3', 'Visium-v4', 'Visium-v5', "Custom"],
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
        args.trim_probes,
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
        args.cellranger_output,
        args.flexible_start_mapping
    )


if __name__ == '__main__':
    main()
