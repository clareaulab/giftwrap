import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich_argparse import RichHelpFormatter


def streaming_subprocess_run(args: list, **kwargs):
    """
    Similar to subprocess.run but live streams outputs to the console.
    https://stackoverflow.com/a/62233864
    """
    print(">", " ".join(args), flush=True)
    stream = True
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, **kwargs)
    while stream:
        stream = process.poll() is None
        for line in process.stdout:
            print(line.rstrip("\n"), flush=True)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete pipeline to generate gapfill counts data. Note that this is slightly opinionated "
                    "for simplicity. For more control, run the individual scripts one by one (giftwrap-count, "
                    "giftwrap-correct-umis, giftwrap-correct-gapfill, giftwrap-collect, giftwrap-summarize).",
        formatter_class=RichHelpFormatter
    )

    # Input arguments
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
        '--project',
        required=False,
        type=str,
        default=None,
        help="The generic name for the project. Used to automatically find R1 and R2 fastq files. Mutually exclusive with -r1 and -r2 arguments."
    )
    parser.add_argument(
        '--multiplex', '-m',
        required=False,
        type=int,
        default=0,
        help="The number of probes to multiplexed in the Flex run with the same probe set. Mutually exclusive with --barcode. Defaults to single plex."
    )
    parser.add_argument(
        '--barcode', '-b',
        required=False,
        type=int,
        default=1,
        help="The barcode number to use for the Flex run. Mutually exclusive with --multiplex. Defaults BC01"
    )
    parser.add_argument(
        "--unmapped_reads",
        required=False,
        type=str,
        default=None,
        help="If provided, unmapped reads are written to the file prefix given."
    )
    # Pipeline arguments
    parser.add_argument(
        "--output", '-o',
        required=True,
        type=str,
        help="The name of the output directory."
    )
    parser.add_argument(
        "-c", "--cores",
        required=False,
        type=int,
        default=1,
        help="The number of cores to use. Less than 1 defaults to the number of available cores."
    )
    # Enumerate the technologies supported (Flex or Visium)
    parser.add_argument(
        "--technology", '-e',
        required=False,
        type=str,
        default="Flex",
        choices=["Flex", "Visium-v1", "Visium-v2", 'Visium-v3', 'Visium-v4', 'Visium-v5', 'VisiumHD', "Custom"],
        help="The technology used to generate the gap-filling probes. Default is Flex."
    )
    parser.add_argument(
        "--tech_def",
        required=False,
        type=str,
        default=None,
        help="The path to the technology definition python file to import. Must include a single class definition that inherits from TechnologyFormatInfo."
    )
    # Matched WTA for additional analyses
    parser.add_argument(
        "--cellranger_output", '-wta',
        action="append",
        required=False,
        default=None,
        help="Path to either the filtered_feature_bc_matrix.h5 or the sample_filtered_feature_bc_matrix folder from CellRanger. "
             "Can be specified multiple times to merge multiple samples if multiplex (in order of the counts.N.h5 files is sorted by N)."
    )
    parser.add_argument(
        "--overwrite", '-f',
        action="store_true",
        required=False,
        help="Overwrite the output directory if it already exists."
    )
    parser.add_argument(
        "--skip_constant_seq",
        required=False,
        action="store_true",
        help="If the technology (i.e. Flex) has a constant sequence in the probe design, do not filter reads for missing it. This is useful for reads that are too short to capture the full probes."
    )
    parser.add_argument(
        '--allow_any_combination',
        required=False,
        action="store_true",
        help="Allow any combination of probes to be counted. By default, only the probes that are in the gapfill set are counted."
    )
    parser.add_argument(
        '--flatten',
        required=False,
        action="store_true",
        help="After processing, save a processed, flattened version of the data as a gzipped tsv file. Note that this is an inefficient storage format for large datasets."
    )
    parser.add_argument(
        '--allow_chimeras', '-ac',
        required=False,
        action='store_true',
        help="Allow chimeric gapfills. If unset, umis that occur multiple times per cell will be dropped except for the most common probe. If set, there is no collapsing."
    )
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

    args = parser.parse_args()

    probes = args.probes
    read1 = args.read1
    read2 = args.read2
    project = args.project
    multiplex = args.multiplex
    barcode = args.barcode
    output = args.output
    cores = args.cores
    technology = args.technology
    tech_def = args.tech_def
    cellranger_output = args.cellranger_output
    overwrite = args.overwrite
    skip_constant_seq = args.skip_constant_seq


    print("Gapfill counts pipeline started.")
    print("================================", flush=True)

    wta_args = []
    if cellranger_output is not None and len(cellranger_output) > 0:
        if isinstance(cellranger_output, str):
            cellranger_output = [cellranger_output]
        for wta in cellranger_output:
            wta_args += ["-wta", wta]

    # Check if step 1 is already done
    output = Path(output)
    if not (output / "steps" / "COUNT_GAPFILLS").exists() or overwrite:
        # Step 1: Count gapfills
        print("Step 1: Counting gapfills.", file=sys.stderr)
        start = datetime.now()
        returncode = streaming_subprocess_run([
            # sys.executable,
            sys.path[0] + "/giftwrap-count",
            "-p", probes,
            "-o", str(output),
            "-c", str(cores),
            "-e", technology
            ]
            + (['--tech_def', tech_def] if tech_def is not None else [])
            + (['--overwrite'] if overwrite else [])
            + (["-r1", read1, "-r2", read2] if project is None else ["--project", project])
            + (['--r1_length', str(args.r1_length)] if args.r1_length is not None else [])
            + (['--r2_length', str(args.r2_length)] if args.r2_length is not None else [])
            + (['-m', str(multiplex)] if multiplex > 0 else [])
            + (['-b', str(barcode)] if barcode > 0 else [])
            + (['--skip_constant_seq'] if skip_constant_seq else [])
            + (['--allow_any_combination'] if args.allow_any_combination else [])
            + (['--unmapped_reads', args.unmapped_reads] if args.unmapped_reads is not None else [])
            + wta_args
        )
        if returncode != 0:
            print("Error: Failed to count gapfills.", file=sys.stderr)
            sys.exit(1)
            return
        (output / "steps").mkdir()
        (output / "steps" / "COUNT_GAPFILLS").touch()
        print("Step 1: Counting gapfills took",  (datetime.now() - start).total_seconds(), "seconds.", file=sys.stderr, flush=True)
    else:
        print("Step 1: Gapfill counting already done.", file=sys.stderr, flush=True)

    # Step 2: Correct UMIs
    if not (output / "steps" / "CORRECT_UMIS").exists() or overwrite:
        print("Step 2: Correcting UMIs.", file=sys.stderr, flush=True)
        start = datetime.now()
        returncode = streaming_subprocess_run([
            # sys.executable,
            sys.path[0] + "/giftwrap-correct-umis",
            "-o", str(output),
            "-c", str(cores)
            ] + (['--allow_chimeras'] if args.allow_chimeras else [])
        )
        if returncode != 0:
            print("Error: Failed to correct UMIs.", file=sys.stderr, flush=True)
            sys.exit(1)
            return
        (output / "steps" / "CORRECT_UMIS").touch()
        print("Step 2: Correcting UMIs took",  (datetime.now() - start).total_seconds(), "seconds.", file=sys.stderr, flush=True)
    else:
        print("Step 2: UMI correction already done.", file=sys.stderr, flush=True)

    # Step 3: Correct gapfills
    if not (output / "steps" / "CORRECT_GAPFILLS").exists() or overwrite:
        print("Step 3: Correcting gapfills.", file=sys.stderr, flush=True)
        start = datetime.now()
        returncode = streaming_subprocess_run([
            # sys.executable,
            sys.path[0] + "/giftwrap-correct-gapfill",
            "-o", str(output),
            "-c", str(cores)
        ]
        )
        if returncode != 0:
            print("Error: Failed to correct gapfills.", file=sys.stderr, flush=True)
            sys.exit(1)
            return
        (output / "steps" / "CORRECT_GAPFILLS").touch()
        print("Step 3: Correcting gapfills took",  (datetime.now() - start).total_seconds(), "seconds.", file=sys.stderr, flush=True)
    else:
        print("Step 3: Gapfill correction already done.", file=sys.stderr, flush=True)

    # Step 4: Collect counts
    if not (output / "steps" / "COLLECT_COUNTS").exists() or overwrite:
        print("Step 4: Collecting counts.", file=sys.stderr, flush=True)
        start = datetime.now()
        returncode = streaming_subprocess_run([
            # sys.executable,
            sys.path[0] + "/giftwrap-collect",
            "-o", str(output),
            "-c", str(cores)
        ] + (['--multiplex'] if multiplex > 0 else [])
        + (['--overwrite'] if overwrite else [])
        + (['--flatten'] if args.flatten else []))
        if returncode != 0:
            print("Error: Failed to collect counts.", file=sys.stderr, flush=True)
            sys.exit(1)
            return
        (output / "steps" / "COLLECT_COUNTS").touch()
        print("Step 4: Collecting counts took", (datetime.now() - start).total_seconds(), "seconds.", file=sys.stderr, flush=True)
    else:
        print("Step 4: Count collection already done.", file=sys.stderr, flush=True)

    # Step 5: Summary statistics and plots
    if not (output / "steps" / "ANALYSIS").exists() or overwrite:
        print("Step 5: Generating summary statistics and plots.", file=sys.stderr, flush=True)
        start = datetime.now()
        returncode = streaming_subprocess_run([
            # sys.executable,
            sys.path[0] + "/giftwrap-summarize",
            "-o", str(output),
        ] + (['--overwrite'] if overwrite else [])
          + (['--flatten'] if args.flatten else [])
          + wta_args)
        if returncode != 0:
            print("Error: Failed to generate summary statistics and plots.", file=sys.stderr, flush=True)
            sys.exit(1)
            return
        (output / "steps" / "ANALYSIS").touch()
        print("Step 5: Generating summary statistics and plots took", (datetime.now() - start).total_seconds(), "seconds.", file=sys.stderr, flush=True)
    else:
        print("Step 5: Summary statistics and plots already done.", file=sys.stderr, flush=True)

    print("Gapfill counts pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
