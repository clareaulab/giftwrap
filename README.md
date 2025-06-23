GIFT-wrap
=========
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![PyPI](https://img.shields.io/pypi/v/giftwrap-sc.svg)](https://pypi.org/project/giftwrap-sc/) 

This package provides tools for dealing with GIFT-seq data. The typical workflow is to use the all-in-one command 
`giftwrap`. Ideally, the WTA panel should be generated beforehand to allow for basic analysis and QC. Additionally, 
this package provides programmatic utilities for dealing with this data.

For detailed documentation and tutorials, please refer to the [documentation website](https://clareaulab.github.io/giftwrap).

# Installation
Since the package provides a CLI, it is recommended to install it with `pipx` or `uvx` if you do not plan to use the
python API, so that the CLI is available on your PATH and does not interfere with your current environment. In this case
you can replace `pip` with `pipx` or `uvx` in the following commands.

Recommended installation:
```bash
pip install giftwrap[all]
```
This installs the requirements for spatial data processing and basic summary analyses of paired WTA data.

Alternatively, you can install with fewer dependencies as follows:
```bash
pip install giftwrap[analysis]  # For additional basic summary analysis
pip install giftwrap[spatial]  # For spatial counts processing
pip install giftwrap  # Minimal installation
```

# Command Line Usage
Most basic command
`giftwrap --probes path/to/probes.xlsx --project fastqs/fastq_prefix -o output_dir/`

This will process the fastq files in `fastqs/` with the prefix `fastq_prefix` and the probes in `path/to/probes.xlsx`. The outputs
will be saved in `output_dir/`.

Note that the probes file should be either in `.tsv`, `.csv`, or `.xlsx` format. The following columns can be included:

- `name` (required): By convention: "gene_name HGSVc", example: "TP53 c.215G>A".
- `lhs_probe` (required): The left-hand side probe sequence.
- `rhs_probe` (required): The right-hand side probe sequence.
- `gap_probe_sequence` (optional): The expected gapfill sequence.
- `original_gap_probe_sequence` (optional): If targeting an SNV, the wild-type expected gapfill sequence.
- `gene` (optional): The gene name the probe targets. If not specified, we assume the first word in `name` is the gene name.

## Advanced options
The pipeline can also be run with the following arguments:
- `--cores`: The number of cores to use for processing. Note: VisiumHD has a high complexity cell barcoding scheme, making its counts more memory-bottlenecked than CPU bottlenecked.
- `-wta`: The path to either the filtered_feature_bc_matrix.h5 or the sample_filtered_feature_bc_matrix folder from CellRanger for the WTA panel.
- `-f`: If passed, overwrite files instead of exiting if they already exist.
- `--technology`: The technology. I.e. Flex/VisiumHD. Default is Flex.
- `--tech_def`: Custom technology definition path. See below for more details.
- `-r1` / `-r2`: Instead of specifying a fastq prefix to search with `--project`, provide the specific R1 and R2 files.
- `--multiplex`: If the fastq files are multiplexed, this flag should be set with the number of expected samples.
- `--barcode`: Similar to `--multiplex`, but only the provided barcode is processed (remaining ignored).

Additionally each step can be run individually with the following commands:
- `giftwrap-count`
- `giftwrap-correct-umis`
- `giftwrap-correct-gapfill`
- `giftwrap-collect`
- `giftwrap-summarize`

## Technology Definition
To add a custom technology definition, you must create a python file describing its features by subclassing 
`giftwrap.utils.TechnologyFormatInfo`. To get a template, you can output the Flex technology difference with:

```bash
giftwrap-generate-tech > tech_def.py
```

You can then modify the python file as needed. To use the custom technology definition you must pass the following 
arguments to the `giftwrap` or `giftwrap-count` commands:
- `--technology Custom`: Specifies a custom technology defintiion
- `--tech_def path/to/tech_def.py`: The path to your custom technology definition file.

# Analyzing processed GIFT-seq data
## Files for analysis
The final output file of the pipeline that should be used for analysis is the `counts.N.h5` file in the output directory,
where `N` is the plex number (1 by default when the data is not multiplexed).
Since gapfills are lower fidelity compared to standard WTA panels, it is highly recommended to run cellranger/spaceranger
on the WTA panel, and provide its output to giftwrap with the `--wta` argument. This will allow giftwrap to filter the
`counts.N.h5` file to only include valid cells based on the WTA panel called by cellranger/spaceranger. If provided,
users should make use of the `counts.N.filtered.h5` file instead of the unfiltered version.

Additionally, there will be several generated files in the output directory with statistics which may be used for 
quality control:

* `fastq_metrics.tsv`: Summary statistics of the parsing of the given fastq files
    - `TOTAL_READS`: Total number of reads processed by the count step.
    - `PROBE_CONTAINING_READS`: Total number of reads that contained a valid probe (including umi/cell barcode).
    - `POSSIBLE_PROBES`: The total number of probes defined.
    - `PROBES_ENCOUNTERED`: The number of probes that were encountered in the fastq files.
    - `EXACT`: The number of reads that contained probes and required no error correction.
    - `CORRECTED_BARCODE`: The number of reads that required cell barcode correction.
    - `CORRECTED_LHS`: The number of reads that required left-hand side probe correction.
    - `CORRECTED_RHS`: The number of reads that required right-hand side probe correction.
    - `FILTERED_NO_CELL_BARCODE`: The number of reads that were filtered out due to no valid cell barcode.
    - `FILTERED_NO_PROBE_BARCODE`: The number of reads that were filtered out due to no valid probe barcode. Only applicable for multiplex runs.
    - `FILTERED_NO_LHS`: The number of reads that were filtered out due to no valid left-hand side probe.
    - `FILTERED_NO_RHS`: The number of reads that were filtered out due to no valid right-hand side probe.
    - `FILTERED_NO_CONSTANT`: The number of reads that were filtered out due to no valid constant sequence region. Only applicable for Flex.

* `counts.N.summary.csv`: Summary statistics about the final (filtered if available) output of the pipeline.
    - `TOTAL_CELLS`: The total number of cells in the output.
    - `GAPFILL_CONTAINING_CELLS`: The number of cells that contained at least one gapfill read.
    - `UMIS_PER_CELL_MEAN`: The mean number of UMIs per cell.
    - `UMIS_PER_CELL_MEDIAN`: The median number of UMIs per cell.
    - `UMIS_PER_CELL_STD`: The standard deviation of the number of UMIs per cell.
    - `UMIS_PER_CELL_MIN`: The minimum number of UMIs per cell.
    - `UMIS_PER_CELL_MIN_EXCLUDING_ZERO`: The minimum number of UMIs per cell excluding cells with zero UMIs.
    - `UMIS_PER_CELL_MAX`: The maximum number of UMIs per cell.
    - `CELLS_PER_GAPFILL_MEAN`: The mean number of cells with gapfills per probe.
    - `CELLS_PER_GAPFILL_MEDIAN`: The median number of cells with gapfills per probe.
    - `CELLS_PER_GAPFILL_STD`: The standard deviation of the number of cells with gapfills per probe.
    - `CELLS_PER_GAPFILL_MIN`: The minimum number of cells with gapfills per probe.
    - `CELLS_PER_GAPFILL_MAX`: The maximum number of cells with gapfills per probe.

* `counts.N.summary.pdf`: Basic analysis of the final (filtered if available) output of the pipeline. The report includes various figures with descriptions inline.

## Single-Cell Analysis
Single-cell resolution gapfill data are provided in the `counts.N.h5`/`counts.N.filtered.h5` file. This file is related
to the .h5 files typically produced by cellranger. The following concepts are important to understand, however.

* Counts data are stored in a CellxFeature sparse matrix format.
* Features are NOT single genes/probes. But rather, they represent unique combinations of probes and gapfill product. Information about these are included as metadata.
* In addition to pure counts, there are also the `total_reads` matrix representing the number of reads (including PCR duplicates) supporting a given gapfill and a `percent_supporting` matrix representing the average % of PCR duplicates that support a gapfill call exactly.

These data can be loaded into standard single-cell analysis tools such as scanpy and Seurat. Since this package is written 
in python, we provide utilities to deal with these files. But we do provide sample R code to read in the data as well.

### Python
The giftwrap module may be directly imported into python to read the data into a scanpy/AnnData object:
```python
import giftwrap as gw

gapfill_adata = gw.read_h5_file("counts.N.h5")
```
Standard preprocessing and analysis can be done with scanpy as usual. But we additionally provide some specialized 
functions to make dealing with the data GIFT-seq data easier. Below highlights the main features:
```python
import giftwrap as gw
import scanpy as sc

adata = ...  # Load and pre-process your WTA data here
gapfill_adata = gw.read_h5_file("counts.1.filtered.h5")

# Basic filtering of low-quality genotypes
gw.pp.filter_gapfills(gapfill_adata)

# Ensure the WTA and gapfill data only have the same cells (in the same order)
adata, gapfill_adata = gw.tl.intersect_wta(adata, gapfill_adata)

# Call genotypes from the gapfill data
gw.tl.call_genotypes(gapfill_adata)

# Potentially useful plots (reuqires scanpy)
sc.tl.dendrogram(gapfill_adata, "sample", n_pcs=0)
gw.pl.dendrogram(gapfill_adata, "sample")  # Basic linkage plot across all genotypes per sample

gw.pl.matrixplot(gapfill_adata, "TP53 c.215G>A", "celltype")  # Plot a matrixplot of the genotypes of TP53 c.215G>A

# Plot UMAPs of the genotypes
# Note this assumes that the WTA data has been pre-processed and is ready for sc.pl.umap

gw.tl.transfer_genotypes(adata, gapfill_adata)  # We will be plotting this on the WTA UMAP basis, so transfer genotype info
gw.pl.umap(adata, "TP53 c.215G>A")  # Plot the UMAP colored by the genotype of TP53 c.215G>A

# Note: You can plot a UMAP or t-SNE of the gapfill data directly, but it may be much more noisy.

# Collapse all genotypes into a single probe feature, useful if you don't care about the specific genotypes
gw.tl.collapse_gapfills(gapfill_adata)
```

### R
We package an R script that you may use to read in data. To easily retrieve it, you can run the following command:
```bash
giftwrap-generate-r > read_giftwrap.R
```
This R script requires the `Matrix` and `rhdf5` packages to be installed. All the individual components of the h5 file
are read with the `read()` function. If you have `Seurat` installed, you can directly read the data into a Seurat object
with the `read_seurat()` function.

### Spatial Analysis example
Note: This expects the giftwrap-sc[spatial] extras to be installed. 
Additionally, we recommend installing spatialdata-io and spatialdata-plot as well.
```python
import giftwrap as gw
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot

wta = sdio.visium_hd(...)  # Load the WTA data
gf = gw.read_h5_file("counts.1.filtered.h5")  # Load the gapfill data

# Join the WTA and gapfill data
wta = gw.sp.join_with_wta(wta, gf)

# Plot the data
wta.pl.
```

# Building

First, install uv: https://github.com/astral-sh/uv

Then set up the environment with: `uv sync --all-extras`

Build the package with: `uv build`

Publish with uv `uv publish --token <your_token>`
