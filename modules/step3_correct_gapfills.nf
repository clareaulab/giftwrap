/*
 * CORRECT_GAPFILLS  (Step 3)
 *
 * Deduplicates PCR copies into one row per unique molecule using
 * quality-weighted base-by-base consensus calling. Operates in-place
 * on the output directory from Step 2:
 *   - updates probe_reads.tsv.gz (deduplicated, consensus-corrected)
 *   - creates probe_reads.tsv.bak.gapfill backup
 */

process CORRECT_GAPFILLS {

    input:
    tuple val(sample_id), val(wta), path(output_dir)

    output:
    tuple val(sample_id), val(wta), path(output_dir)

    script:
    """
    python -m giftwrap.step3_correct_gapfill \\
        --output '${output_dir}' \\
        --cores  ${task.cpus}
    """
}
