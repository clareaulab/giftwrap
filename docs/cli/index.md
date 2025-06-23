---
name: CLI Reference
description: A reference guide for the `giftwrap` command line interface (CLI) for processing GIFT-seq data.
---

# GIFTwrap Commands:
1. [giftwrap](giftwrap.md): The main command for processing GIFT-seq data that runs the full pipeline end-to-end.
2. [giftwrap-count](giftwrap_count.md): Command for manually counting GIFT-seq data from FastQ files.
3. [giftwrap-correct-umis](giftwrap_correct_umis.md): Command for manually correcting UMIs in GIFT-seq data.
4. [giftwrap-correct-gapfill](giftwrap_correct_gapfill.md): Command for manually correcting gapfill sequences in GIFT-seq data.
5. [giftwrap-collect](giftwrap_collect.md): Command for manually collecting GIFT-seq data after processing.
6. [giftwrap-summarize](giftwrap_summarize.md): Command for manually performing basic QC on GIFT-seq data after collection.
7. [giftwrap-generate-r](giftwrap_generate_r.md): Command for generating an R script backbone for importing GIFT-seq data into Seurat. See the [Seurat integration tutorial](../tutorials/seurat_integration.md) for more information on how to use this command.
8. [giftwrap-generate-tech](giftwrap_generate_tech.md): Command for generating a scaffold for a new GIFT-seq technology design. See the [Extending GIFTwrap tutorial](../tutorials/extending_giftwrap.md) for more information on how to use this command.
9. [giftwrap-convert-probes](giftwrap_convert_probes.md): Converts a 10X Cell Ranger probe set to a GIFTwrap-compatible probe set file.
10. [giftwrap-diagnostic](giftwrap_diagnostic.md): Command to quickly parse reads without any corrections/processing to check for issues with the sequencing data.
