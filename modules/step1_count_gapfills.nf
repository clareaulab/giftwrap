/*
 * COUNT_GAPFILLS  (Step 1)
 *
 * Maps paired FASTQ reads to probes, extracts gapfill sequences,
 * cell barcodes, and UMIs. Writes probe_reads.tsv.gz, barcodes.tsv.gz,
 * manifest.tsv, and fastq_metrics.tsv into the output directory.
 *
 * Input tuple: (sample_id, r1_fastq, r2_fastq, barcode, probes, wta)
 *   sample_id – unique sample name; used as the output directory name
 *   r1_fastq  – per-sample R1 FASTQ produced by CUTADAPT_PREPROCESS
 *   r2_fastq  – per-sample R2 FASTQ produced by CUTADAPT_PREPROCESS
 *   barcode   – plex barcode number (e.g. "1")
 *   probes    – staged probe TSV file
 *   wta       – path to Cell Ranger WTA h5, or "" if not available
 */

process COUNT_GAPFILLS {

    input:
    tuple val(sample_id), path(r1_fastq), path(r2_fastq), val(barcode), path(probes), val(wta)

    output:
    tuple val(sample_id), val(wta), path("${sample_id}")

    script:
    def bc_arg  = barcode ? "-b '${barcode}'" : ""
    def wta_arg = wta     ? "-wta '${wta}'"   : ""
    """
    python -m giftwrap.step1_count_gapfills \\
        --probes     '${probes}' \\
        --technology '${params.technology}' \\
        --output     '${sample_id}' \\
        --cores      ${task.cpus} \\
        --overwrite \\
        --read1 '${r1_fastq}' \\
        --read2 '${r2_fastq}' \\
        ${bc_arg} \\
        ${wta_arg}
    """
}
