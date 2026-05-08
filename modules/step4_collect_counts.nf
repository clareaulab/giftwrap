/*
 * COLLECT_COUNTS  (Step 4)
 *
 * Aggregates deduplicated reads into sparse HDF5 count matrices
 * (one per plex). Reads probe_reads.tsv.gz, manifest.tsv, and
 * barcodes.tsv.gz from the output directory; writes counts.{plex}.h5.
 *
 * Publishes stable intermediate outputs to <params.output>/<sample_id>/outputs/
 * so they remain accessible even if Step 5 is re-run.
 */

process COLLECT_COUNTS {

    publishDir { "${params.output}/${sample_id}" }, mode: 'copy', overwrite: true, saveAs: { fn ->
        def name = file(fn).name
        if (name ==~ /fastq_metrics\.tsv|manifest\.tsv|barcodes\.tsv\.gz|probe_reads\.tsv\.gz|counts\.\d+\.h5/) "outputs/${name}"
        else null
    }

    input:
    tuple val(sample_id), val(wta), path(output_dir)

    output:
    tuple val(sample_id), val(wta), path(output_dir)

    script:
    def multiplex_flag = params.multiplex > 1 ? '--multiplex' : ''
    def flatten_flag   = params.flatten ? '--flatten' : ''
    """
    python -m giftwrap.step4_collect_counts \\
        --output             '${output_dir}' \\
        --cores              ${task.cpus} \\
        --max_pcr_thresholds ${params.max_pcr_thresholds} \\
        --overwrite \\
        ${multiplex_flag} \\
        ${flatten_flag}
    """
}
