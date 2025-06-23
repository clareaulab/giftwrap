---
title: Getting Started with GIFTwrap
summary: A guide to help you get started with the GIFTwrap package for GIFT-seq data analysis.
---
# Getting Started
GIFTwrap has two main components:

1. The `giftwrap` command line interface (CLI) for processing GIFT-seq sequencing data.

2. The `giftwrap` python package for downstream analysis of GIFT-seq data.

It is important to note that the `giftwrap` python API will often require external dependencies such as scanpy that are not installed by default. Refer to the [installation guide](../installation.md) for more information on how to install these dependencies.

## General Workflow
Below we outline the general workflow for using GIFTwrap to analyze GIFT-seq data.

1. **Transcriptome Processing**: While not required, it is recommended to first process the transcriptome data through `cellranger` to obtain robust cell calls. This will help in the quality control of the GIFT-seq data. If GIFT-seq is in a pooled library, you should expect that many reads may not map to the transcriptome panel given by `cellranger`. This is normal and expected, as the GIFT-seq data is designed to capture genotypes rather than transcriptomes. However, we recommend splitting GIFT-seq into a separate library to improve sensitivity of genotyping.
2. **GIFT-seq Data Processing**: Use the `giftwrap` CLI to process the GIFT-seq FastQ files into counts matrices. This step will generate a finalized counts matrix that can be used for downstream analysis. While the `giftwrap` command bundles all step of processing, you can also run each step separately if needed. More information on the CLI can be found in the [processing GIFT-seq data tutorial](./processing_giftseq_data.md).
3. **Quality Control and Analysis**: Perform quality control and basic analysis on the GIFT-seq data using the `giftwrap` Python API. The API is designed to be easily incorporated with scverse-based workflows, see the [analyzing GIFT-seq data tutorial](./analyzing_giftseq_data.md) for more information. If you have a preference for Seurat, we provide limited Seurat integration, see the [Seurat integration tutorial](./seurat_integration.md) for more information. Additionally, if the GIFT-seq library is spatial, you can use an extended set of tools for spatial analysis, see the [spatial analysis tutorial](./spatial_giftseq.md) for more information.
4. **Imputation**: Since GIFT-seq genotyping is highly sensitive to expression levels of transcripts, we provide basic tools for imputing genotypes based on transcriptomic data in addition to partial genotyping data. Note that these tools do not infer genotypes *de novo*, but rather take advantage of non-randomly distributed genotypes in the population to impute missing genotypes. See the [imputation tutorial](./imputation.md) for more information.

## Additional Resources
- [Installation Guide](../installation.md): Detailed instructions on how to install the `giftwrap` package and its dependencies.
- [API Documentation](../api/index.md): Comprehensive documentation of the `giftwrap` Python API, including functions and classes available for GIFT-seq data analysis.
- [Extending GIFTwrap Tutorial](./extending_giftwrap.md): Information on how to extend the `giftwrap` package for custom library designs.
