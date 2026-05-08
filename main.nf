nextflow.enable.dsl=2

/*
 * GIFTwrap Nextflow Pipeline
 *
 * Cutadapt always runs to demultiplex each sample from the multiplexed FASTQ
 * by its Flex barcode. The fastq_dir must contain exactly one R1 and one R2
 * file (matched by *_R1_*.fastq.gz / *_R2_*.fastq.gz patterns).
 *
 * Samplesheet format (CSV with header row):
 *   sample,barcode,probes,wta
 *   SAMPLE1,1,/path/to/probes.tsv,/path/to/sample_filtered_feature_bc_matrix.h5
 *   SAMPLE2,2,/path/to/probes2.tsv,
 *
 * Usage:
 *   nextflow run main.nf \
 *       --samplesheet samplesheet.csv \
 *       --fastq_dir   /path/to/raw_fastqs/ \
 *       --output      results/
 *
 * Add -profile slurm (single dash) to submit each task as a SLURM job.
 * All samples run in parallel; each publishes to: <output>/<sample>/outputs/
 */

include { CUTADAPT_PREPROCESS } from './modules/cutadapt_preprocess'
include { COUNT_GAPFILLS      } from './modules/step1_count_gapfills'
include { CORRECT_UMIS        } from './modules/step2_correct_umis'
include { CORRECT_GAPFILLS    } from './modules/step3_correct_gapfills'
include { COLLECT_COUNTS      } from './modules/step4_collect_counts'
include { SUMMARIZE_COUNTS    } from './modules/step5_summarize_counts'

// ── Parameters ─────────────────────────────────────────────────────────────
// All step-tuning defaults are in nextflow.config.

params.samplesheet = null   // required: CSV with sample,barcode,probes,wta
params.output      = "results"
params.fastq_dir   = null   // required: dir containing R1/R2 fastq.gz files

// ── Workflow ────────────────────────────────────────────────────────────────

workflow {

    def FLEX_BARCODES = [
        '1':'ACTTTAGG', '2':'AACGGGAA', '3':'AGTAGGCT', '4':'ATGTTGAC',
        '5':'ACAGACCT', '6':'ATCCCAAC', '7':'AAGTAGAG', '8':'AGCTGTGA',
        '9':'ACAGTCTG', '10':'AGTGAGTG', '11':'AGAGGCAA', '12':'ACTACTCA',
        '13':'ATACGTCA', '14':'ATCATGTG', '15':'AACGCCGA', '16':'ATTCGGTT'
    ]

    if (!params.samplesheet) error "Please provide --samplesheet <path/to/samplesheet.csv>"
    if (!params.fastq_dir)   error "Please provide --fastq_dir <dir>"

    // Find the single R1 and R2 file from fastq_dir.
    r1_ch = Channel.fromPath("${params.fastq_dir}/*_R1_*.fastq.gz").first()
    r2_ch = Channel.fromPath("${params.fastq_dir}/*_R2_*.fastq.gz").first()

    // Parse samplesheet into [sample_id, barcode, probes_file, wta]
    samples_ch = Channel
        .fromPath(params.samplesheet)
        .splitCsv(header: true, strip: true)
        .map { row ->
            def barcode   = row.barcode?.trim()
            if (!barcode) error "Samplesheet must include a 'barcode' column for sample '${row.sample}'"
            def probePath = row.probes?.trim()
            if (!probePath) error "Missing probes path for sample '${row.sample}'"
            def wta = row.wta?.trim() ?: ""
            tuple(row.sample, barcode, file(probePath), wta)
        }

    // Attach the shared R1 and R2 to every sample, then look up the Flex sequence.
    cutadapt_in = samples_ch
        .combine(r1_ch)
        .combine(r2_ch)
        .map { sample, barcode, probes, wta, r1, r2 ->
            def barcodeSeq = FLEX_BARCODES[barcode]
            if (!barcodeSeq) error "Barcode '${barcode}' not in Flex barcode map (valid: 1-16)"
            tuple(sample, barcodeSeq, r1, r2, barcode, probes, wta)
        }

    // cutadapt produces per-sample R1 and R2; pass both to step 1.
    step1_in_ch = CUTADAPT_PREPROCESS(cutadapt_in).map { sample, r1, r2, barcode, probes, wta ->
        tuple(sample, r1, r2, barcode, probes, wta)
    }

    // ── Steps 1–5 (all samples run in parallel) ───────────────────────────
    step1_out_ch = COUNT_GAPFILLS(step1_in_ch)
    step2_out_ch = CORRECT_UMIS(step1_out_ch)
    step3_out_ch = CORRECT_GAPFILLS(step2_out_ch)
    step4_out_ch = COLLECT_COUNTS(step3_out_ch)
    SUMMARIZE_COUNTS(step4_out_ch)
}
