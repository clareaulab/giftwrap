/*
 * CORRECT_UMIS  (Step 2)
 *
 * Deduplicates reads, error-corrects UMI sequences, and removes chimeric
 * reads. Operates in-place on the output directory from Step 1:
 *   - updates probe_reads.tsv.gz (UMI-corrected)
 *   - creates probe_reads.tsv.bak.umi backup
 */

process CORRECT_UMIS {

    input:
    tuple val(sample_id), val(wta), path(output_dir)

    output:
    tuple val(sample_id), val(wta), path(output_dir)

    script:
    def chimera_flag = params.allow_chimeras ? '--allow_chimeras' : ''
    """
    python -m giftwrap.step2_correct_umis \\
        --output    '${output_dir}' \\
        --threshold ${params.umi_edit_distance} \\
        --cores     ${task.cpus} \\
        ${chimera_flag}
    """
}
