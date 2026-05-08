/*
 * CUTADAPT_PREPROCESS
 *
 * Filters a paired-end FASTQ (R1 + R2) by a sample's Flex barcode sequence.
 * The Flex plate barcode appears in R2 (the probe read); reads where R2 does not
 * contain the barcode are discarded (--discard-untrimmed). -A (uppercase) targets
 * R2 in paired-end mode; -a (lowercase) would target R1 which is wrong here.
 * Produces per-sample R1 and R2 files that giftwrap step 1 consumes via
 * --read1 / --read2.
 *
 * The barcode sequence is resolved in main.nf from the FLEX_BARCODES constant
 * and passed into this process as barcode_seq.
 *
 * Input tuple: (sample_id, barcode_seq, r1_fastq, r2_fastq, barcode, probes, wta)
 *   sample_id   – unique sample name
 *   barcode_seq – Flex barcode DNA sequence (e.g. "ACTTTAGG") for cutadapt -A
 *   r1_fastq    – raw R1 FASTQ file (contains cell barcode + UMI)
 *   r2_fastq    – raw R2 FASTQ file (contains biological sequence)
 *   barcode     – plex barcode number; passed through to step 1
 *   probes      – probe TSV file; passed through to step 1
 *   wta         – WTA h5 path or ""; passed through to step 1
 */

process CUTADAPT_PREPROCESS {

    input:
    tuple val(sample_id), val(barcode_seq), path(r1_fastq), path(r2_fastq), val(barcode), path(probes), val(wta)

    output:
    tuple val(sample_id), path("${sample_id}_R1.fastq.gz"), path("${sample_id}_R2.fastq.gz"), val(barcode), path(probes), val(wta)

    script:
    """
    cutadapt \\
        -A '${barcode_seq}' \\
        -O 8 \\
        --discard-untrimmed \\
        --action=none \\
        -j ${task.cpus} \\
        -o '${sample_id}_R1.fastq.gz' \\
        -p '${sample_id}_R2.fastq.gz' \\
        '${r1_fastq}' \\
        '${r2_fastq}'
    """
}
