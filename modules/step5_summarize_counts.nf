/*
 * SUMMARIZE_COUNTS  (Step 5)
 *
 * Reads counts.{plex}.h5 files from the output directory and produces:
 *   - counts.{plex}.summary.tsv  (QC metrics table)
 *   - counts.{plex}.summary.pdf  (multi-page visualization report)
 *   - counts.{plex}.filtered.h5  (if barcode/PCR filtering applied)
 *
 * Publishes final outputs to <params.output>/<sample_id>/outputs/
 */

process SUMMARIZE_COUNTS {

    publishDir { "${params.output}/${sample_id}" }, mode: 'copy', overwrite: true, saveAs: { fn ->
        "outputs/${file(fn).name}"
    }

    input:
    tuple val(sample_id), val(wta), path(output_dir)

    output:
    path(output_dir)

    script:
    def wta_arg      = wta ? "-wta '${wta}'" : ""
    def flatten_flag = params.flatten ? '--flatten' : ''
    """
    python -m giftwrap.step5_summarize_counts \\
        --output            '${output_dir}' \\
        --overwrite \\
        --reads_per_gapfill ${params.reads_per_gapfill} \\
        ${wta_arg} \\
        ${flatten_flag}
    """
}