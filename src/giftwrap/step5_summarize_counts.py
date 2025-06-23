import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, gaussian_kde
from sankeyflow import Sankey
from rich_argparse import RichHelpFormatter

from .utils import filter_h5_file, read_h5_file, read_wta, sequencing_saturation, sequence_saturation_curve, maybe_gzip
from .analysis.tools import collapse_gapfills


def density(x, y):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    return kde(xy)

def best_fit(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p(np.unique(x))


def make_sankey(fastq_stats: dict, counts_stats: dict) -> Sankey:
    nodes = [
        [("Reads", fastq_stats.get("TOTAL_READS", 0))],

        [("Exact Matches", fastq_stats.get("EXACT", 0)), ("Corrected LHS", fastq_stats.get("CORRECTED_LHS", 0)),
         ("Corrected RHS", fastq_stats.get("CORRECTED_RHS", 0)),
         ("Corrected Cell Barcode", fastq_stats.get("CORRECTED_BARCODE", 0)),
         ("Filtered", fastq_stats.get("TOTAL_READS", 0) - fastq_stats.get("PROBE_CONTAINING_READS", 0))],

        [("Mapped Reads", fastq_stats.get("PROBE_CONTAINING_READS", 0)),
         ("No LHS", fastq_stats.get("FILTERED_NO_LHS", 0)), ("No RHS", fastq_stats.get("FILTERED_NO_RHS", 0)),
         ("No Cell Barcode", fastq_stats.get("FILTERED_NO_CELL_BARCODE", 0)),
         ("No Constant Sequence", fastq_stats.get('FILTERED_NO_CONSTANT', 0)),
         ("No Probe Barcode", fastq_stats.get("FILTERED_NO_PROBE_BARCODE", 0))],

        [("Total UMIs", counts_stats.get("TOTAL_UMIS", 0)), ("Total Cells", counts_stats.get("TOTAL_CELLS", 0))]
    ]

    flows = [
        ("Reads", "Exact Matches", fastq_stats.get("EXACT", 0)),
        ("Reads", "Corrected LHS", fastq_stats.get("CORRECTED_LHS", 0)),
        ("Reads", "Corrected RHS", fastq_stats.get("CORRECTED_RHS", 0)),
        ("Reads", "Corrected Cell Barcode", fastq_stats.get("CORRECTED_BARCODE", 0)),
        ("Reads", "Filtered", fastq_stats.get("TOTAL_READS", 0) - fastq_stats.get("PROBE_CONTAINING_READS", 0)),
        ("Filtered", "No LHS", fastq_stats.get("FILTERED_NO_LHS", 0)),
        ("Filtered", "No RHS", fastq_stats.get("FILTERED_NO_RHS", 0)),
        ("Filtered", "No Cell Barcode", fastq_stats.get("FILTERED_NO_CELL_BARCODE", 0)),
        ("Filtered", "No Constant Sequence", fastq_stats.get('FILTERED_NO_CONSTANT', 0)),
        ("Filtered", "No Probe Barcode", fastq_stats.get("FILTERED_NO_PROBE_BARCODE", 0)),
        ("Exact Matches", "Mapped Reads", fastq_stats.get("EXACT", 0)),
        ("Corrected LHS", "Mapped Reads", fastq_stats.get("CORRECTED_LHS", 0)),
        ("Corrected RHS", "Mapped Reads", fastq_stats.get("CORRECTED_RHS", 0)),
        ("Corrected Cell Barcode", "Mapped Reads", fastq_stats.get("CORRECTED_BARCODE", 0)),
        ("Mapped Reads", "Total UMIs", counts_stats.get("TOTAL_UMIS", 0)),
        ("Mapped Reads", "Total Cells", counts_stats.get("TOTAL_CELLS", 0))
    ]

    # # Prune any zeros
    # for node in nodes:
    #     to_remove = []
    #     for i, (name, value) in enumerate(node):
    #         if value == 0:
    #             to_remove.append(i)
    #     for i in to_remove[::-1]:
    #         node.pop(i)
    #
    # to_remove = []
    # for i, flow in enumerate(flows):
    #     if flow[1] not in [n[0] for n in nodes[-1]] or flow[2] == 0:
    #         to_remove.append(i)
    # for i in to_remove[::-1]:
    #     flows.pop(i)

    return Sankey(flows=flows, nodes=nodes)


def make_pdf_report(output_file, gapfill_adata, adata):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except:
        print("Matplotlib not found. Please install it to generate the PDF report.")
        return

    print("Generating PDF report, this may take a few minutes...")
    with PdfPages(output_file) as pdf:
        # Compute a sankey diagram describing the overall flow of data processing
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.set_dpi(600)
        fig.suptitle("Data processing flow")
        # Read the counts metrics file (long format table) into a dictionary
        fastq_metrics = pd.read_table(output_file.parent / "fastq_metrics.tsv").set_index("metric").to_dict()["value"]
        # Read the counts metrics file (long format table) into a dictionary
        counts_metrics = pd.read_table(output_file.with_suffix(".tsv")).set_index("statistic").to_dict()["value"]
        # Create the sankey diagram
        sankey = make_sankey(fastq_metrics, counts_metrics)
        sankey.draw(ax)
        fig.text(0.5, 0.005,
                    "This figure shows the flow of data processing from the raw fastq files to the final gapfill counts. "
                    "The width of the lines represents the number of reads or cells at each step.",
                    ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Barcode rank plot (log-umis vs log-cell rank)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.set_dpi(600)
        fig.suptitle("Barcode rank plot")
        umis_per_cell = np.asarray(gapfill_adata.X.sum(axis=1)).flatten()
        gapfill_adata.obs['umis_per_cell'] = umis_per_cell
        cells_per_gapfill = np.asarray((gapfill_adata.X > 0).sum(axis=0)).flatten()
        cell_rank = np.argsort(umis_per_cell)[::-1]
        ax.scatter(
            x=np.arange(1, gapfill_adata.shape[0] + 1),
            y=umis_per_cell[cell_rank]
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Cell rank")
        ax.set_ylabel("UMIs")
        fig.text(0.5, 0.005,
                 "Cells are ranked by the number of UMIs they contain. "
                 "The x-axis is the log-rank of the cell, and the y-axis is the log-number of UMIs. "
                 "This is a similar plot as generated by CellRanger standard outputs.",
                 ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Plot the sequencing saturation curve
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.set_dpi(600)
        fig.suptitle("Sequencing saturation curve")
        saturations = sequence_saturation_curve(collapse_gapfills(gapfill_adata).layers['total_reads'], n_points=1000)
        ax.plot(saturations[:,0], saturations[:,1])
        # Use log10 scale
        # ax.set_xscale("log")
        ax.set_xlabel("Mean reads per cell")
        ax.set_ylabel("Sequencing saturation")
        fig.text(0.5, 0.005,
                    "This figure shows the sequencing saturation curve. "
                    "The x-axis is the mean number of reads per cell after down-sampling, and the y-axis is the sequencing saturation. "
                 "This is a similar plot as generated by CellRanger standard ouputs.",
                    ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Plot the distributions of umis per cell and cells per gapfill
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        fig.set_dpi(600)
        fig.suptitle("UMIs distributions")
        axs[0].hist(umis_per_cell, bins=100)
        # Add horizontal line to show the # of probes
        axs[0].axvline(x=gapfill_adata.var['probe'].nunique(), color='r', linestyle='--')
        axs[0].set_xlabel("UMIs per cell")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("UMIs per cell distribution")

        axs[1].hist(cells_per_gapfill, bins=100)
        axs[1].set_xlabel("Cells per gapfill")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Cells per gapfill distribution")

        fig.text(0.5, 0.005,
                    "The top plot shows the distribution of UMIs per cell, including a line depicting the number of probes. "
                    "Good quality data should have # UMIs > # probes per cell. "
                    "The bottom plot shows the distribution of cells per containing each possible probe/gapfill combination.",
                 ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Plot the number of unique gapfills per gene and the number of cells containing a gapfill for a gene
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        fig.set_dpi(600)
        fig.suptitle("Unique gapfills per gene")
        unique_genes = gapfill_adata.var['probe'].unique().tolist()
        gapfills_per_gene = np.zeros((gapfill_adata.shape[0],len(unique_genes)), dtype=float)
        for i, gene in enumerate(unique_genes):
            gapfills_per_gene[:,i] = gapfill_adata[:, gapfill_adata.var['probe'] == gene].X.sum(axis=1).flatten()
        axs[0].boxplot(gapfills_per_gene, tick_labels=unique_genes, sym='+')
        # Rotate the x-axis labels
        for tick in axs[0].get_xticklabels():
            tick.set_rotation(90)
        axs[0].set_ylabel("UMIs")
        axs[0].set_xlabel("Probe")

        axs[1].hist((gapfills_per_gene > 0).sum(axis=0), bins=25)
        axs[1].set_xlabel("Cells with a gapfill.")
        axs[1].set_ylabel("Cells")
        axs[1].set_title("Genes containing gapfill distribution")

        fig.text(0.5, 0.005,
                 "The top plot shows the distribution of the number of UMIs for each probe to evaluate the variance of the gapfill across cells. "
                 "The bottom plot shows the distribution of the number of probes each cell contains to evaluate probe coverage.",
                 ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Number of supporting reads per gene gapfill
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.set_dpi(600)
        fig.suptitle("Supporting reads per gapfill")
        total_reads_per_probe = gapfill_adata.layers['total_reads'].todense().__array__().sum(0).flatten()
        # Sort
        sorted_probes = np.argsort(total_reads_per_probe)[::-1]
        ax.bar(np.arange(len(sorted_probes)), total_reads_per_probe[sorted_probes])
        # Get the sum and divide by the number of possible probe/gapfills to get the expected uniform distribution
        ax.axhline(y=gapfill_adata.layers['total_reads'].sum() / gapfill_adata.shape[1], color='r', linestyle='--')
        ax.set_yscale("log")
        ax.set_ylabel("Reads")
        # Remove xticks
        ax.set_xticks([])
        ax.set_xlabel("Probe/Gapfill Pair")
        fig.text(0.5, 0.005,
                 "This figure shows the distributions of reads supporting each unique gapfill. "
                 "The red dashed line shows the expected number of reads per gapfill if they were uniformly captured in PCR. "
                 "Note that these are not UMIs. "
                 "The plot can be used to evaluate the quality of gapfill calls as more reads per gapfill can reduce the likelihood of technical error.",
                 ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Perform basic clustering/projection on just the gapfill data
        try:
            import scanpy as sc
        except:
            print("Scanpy not found. Completing the PDF report without analysis")
            return
        # Save the raw data
        gapfill_adata_old = gapfill_adata.copy()
        # Remove cells with no gapfill counts (for figure purposes)
        gapfill_adata = gapfill_adata[gapfill_adata.X.sum(axis=1) > 0, :]
        # Normalize the data
        sc.pp.normalize_total(gapfill_adata, target_sum=1e2)  # Note lower target sum because of typically lower UMI counts
        sc.pp.log1p(gapfill_adata)
        # Perform PCA
        sc.tl.pca(gapfill_adata)
        # Perform clustering
        sc.pp.neighbors(gapfill_adata)
        try:
            sc.tl.leiden(gapfill_adata, key_added='cluster')
        except:
            try:
                sc.tl.louvain(gapfill_adata, key_added='cluster')
            except:
                print("Leiden and louvain clustering packages not found. Skipping clustering.")
                gapfill_adata.obs['cluster'] = 0
        # Compute t-sne and umap
        sc.tl.tsne(gapfill_adata)
        sc.tl.umap(gapfill_adata)
        # Plot the results
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        fig.set_dpi(600)
        fig.suptitle("Clustering of gapfill data")
        sc.pl.tsne(gapfill_adata, color='cluster', ax=axs[0], show=False)
        sc.pl.umap(gapfill_adata, color='cluster', ax=axs[1], show=False)
        fig.text(0.5, 0.005,
                    "This figure shows the results of clustering purely on the gapfill data on both a t-SNE and UMAP projection.",
                    ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


        # If we have cellranger data, compare psuedobulk counts of gapfill vs WTA.
        if adata is not None:
            # Get the original gapfill adata back
            gapfill_adata = gapfill_adata_old
            # Re-order cells in adata to match gapfill adata
            adata = adata[gapfill_adata.obs.index, :]
            # Get the gene names
            if 'gene' not in gapfill_adata.var.columns:  # No gene name included in manifest, we will try to guess the gene name
                print("Warning: Gene names not present in metadata, guessing gene names...")
                gapfill_adata.var['gene'] = [p.replace(" ", "_").split('_')[0] for p in gapfill_adata.var['probe']]
            # Check if our gene names match any in the WTA
            adata.var['gene'] = adata.var_names.values
            if len(set(gapfill_adata.var['gene']) & set(adata.var['gene'])) == 0:
                print("Warning: No gene names match between WTA and gapfill data. Skipping comparison to WTA.")
                return
            # Compute correlation between gapfill and WTA
            wta_expression = np.zeros((gapfill_adata.shape[0], gapfill_adata.var['probe'].nunique()))
            gapfill_expression = np.zeros((gapfill_adata.shape[0], gapfill_adata.var['probe'].nunique()))
            for i, gap_probe in enumerate(gapfill_adata.var['probe'].unique()):
                gap_gene = gapfill_adata.var['gene'][gapfill_adata.var['probe'] == gap_probe].values[0]
                if gap_gene in adata.var_names:
                    wta_expression[:, i] = adata[:, adata.var_names == gap_gene].X.toarray().flatten()
                # Aggregate to probe level
                gapfill_expression[:, i] = gapfill_adata[:, gapfill_adata.var['probe'] == gap_probe].X.toarray().sum(axis=1).flatten()

            fig, axs = plt.subplots(2, 1, figsize=(8, 12))
            fig.set_dpi(600)
            fig.suptitle("Gapfill vs WTA")
            # Compute spearman's rank correlation for single cells
            sc_corr_results = spearmanr(gapfill_expression.flatten(), wta_expression.flatten())
            # Plot a density plot of the correlation
            axs[0].scatter(gapfill_expression.flatten(), wta_expression.flatten())
                           # c=density(gapfill_expression.flatten(), wta_expression.flatten()), zorder=2)
            # Add a line of best fit
            axs[0].plot(np.unique(gapfill_expression.flatten()), best_fit(gapfill_expression.flatten(), wta_expression.flatten()),
                        zorder=1, linestyle='--', color='k')
            axs[0].set_xlabel("Gapfill expression")
            axs[0].set_ylabel("WTA expression")
            # axs[0].set_xscale("log")
            # axs[0].set_yscale("log")
            axs[0].set_title(
                f"Paired single-cell expression, Spearman's rho={sc_corr_results.correlation:.2f} (p={sc_corr_results.pvalue:.2e})")
            # Compute the same for psuedobulk
            gapfill_expression = gapfill_expression.sum(axis=0)  # Sum across cells
            wta_expression = wta_expression.sum(axis=0)
            pb_corr_results = spearmanr(gapfill_expression, wta_expression)
            # Plot a density plot of the correlation
            axs[1].scatter(gapfill_expression, wta_expression)
                           #c=density(gapfill_expression, wta_expression))
            # Add a line of best fit
            axs[1].plot(np.unique(gapfill_expression.flatten()), best_fit(gapfill_expression.flatten(), wta_expression.flatten()),
                        zorder=1, linestyle='--', color='k')
            axs[1].set_xlabel("Gapfill expression")
            axs[1].set_ylabel("WTA expression")
            # axs[1].set_xscale("log")
            # axs[1].set_yscale("log")
            axs[1].set_title(
                f"Paired psuedobulk expression, Spearman's rho={pb_corr_results.correlation:.2f} (p={pb_corr_results.pvalue:.2e})")
            fig.text(0.5, 0.005, "This figure shows the correlation between the expression of genes in the gapfill and WTA data. "
                                 "The top plot shows the correlation for paired cells, while the bottom plot shows the correlation for psuedobulk expression. "
                                "Note that genes present in the gapfill panel that are not present in the WTA will have zero expression on the y-axis.",
                        ha='center', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

            # Finally, compute WTA UMAP, which we will project gapfill data onto
            sc.pp.normalize_total(adata, target_sum=1e4)  # Note higher target sum because of typically higher UMI counts in WTA
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata)
            sc.tl.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            # Project gapfill data onto WTA UMAP
            gapfill_adata.obsm['X_umap'] = adata[gapfill_adata.obs.index.values].obsm['X_umap']

            # Plot summary statistics
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            fig.set_dpi(600)
            fig.suptitle("Gapfill on WTA UMAP")
            sc.pl.umap(gapfill_adata, color='umis_per_cell', ax=axs[0], title="UMIs per cell", show=False)

            # Compute the majority gapfill for the probe with the most UMIs and the fewest UMIs
            gapfill_adata.obs['majority_gapfill'] = 'N/A'
            probe_X = np.zeros((gapfill_adata.var['probe'].nunique(),))
            probes = gapfill_adata.var['probe'].unique()
            for i, probe in enumerate(gapfill_adata.var['probe'].unique()):
                probe_X[i] = gapfill_adata[:, gapfill_adata.var['probe'] == probe].X.mean()
            max_probe = probes[np.argmax(probe_X)]
            min_probe = probes[np.argmin(probe_X)]
            for i, probe in enumerate([max_probe, min_probe]):
                ax = axs[i+1]
                majority_gapfills = list()
                for cell in gapfill_adata.obs.index:
                    cell_data = gapfill_adata[cell, gapfill_adata.var['probe'] == probe].X
                    if cell_data.sum() > 0:
                        majority_gapfills.append(gapfill_adata.var['gapfill'][cell_data.argmax()])
                    else:
                        majority_gapfills.append("N/A")
                gapfill_adata.obs['majority_gapfill'] = majority_gapfills
                sc.pl.umap(gapfill_adata, color='majority_gapfill', ax=ax, title=f"Majority gapfill for {probe}", show=False)

            fig.text(0.5, 0.005,
                    "The top figure shows the distribution of UMIs per cell on the WTA UMAP. "
            "The middle and bottom figures show the majority gapfill for the probe with the most and fewest average UMIs, respectively.",
                    ha='center', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)


def summarize_counts(input: Path, summary_output: Path, summary_pdf_output: Path, counts_output: Path, flattened_counts_output: Path, cellranger_output: Optional[Path], flatten: bool):
    print("Summarizing counts file ", input, " to ", summary_output, ", ", summary_pdf_output, ", and ", counts_output, " (This will take awhile)...")

    # Read cellranger to an anndata file if provided
    if cellranger_output is not None:
        obj = read_wta(
            cellranger_output,
            fallback_to_barcodes=True
        )
        # If an array was returned, we only have barcodes, else we have an andata object
        if isinstance(obj, np.ndarray):
            adata = None
            barcodes = obj
            print("Cellranger output provided, AnnData only partially parsed. Using barcodes only.")
        else:
            adata = obj
            barcodes = adata.obs_names.values
    else:
        adata = None
        barcodes = None
        print("No cellranger output provided, unable to filter counts...")

    if barcodes is not None:
        print(f"Cellranger identified {barcodes.shape[0]} cells. Filtering counts...", end="")
        # Filter the counts file to only include cells in the cellranger output
        filter_h5_file(input, counts_output, barcodes)
        if flatten:  # We need to filter the flattened counts as well
            with maybe_gzip(input.parent / input.name.replace(".h5", ".tsv.gz").replace('counts.', 'flat_counts.'), 'r') as f_in:
                with maybe_gzip(flattened_counts_output, 'w') as f_out:
                    first = True
                    for line in f_in:
                        if line.split("\t")[0] in barcodes or first:
                            f_out.write(line)
                            first = False
        gapfill_adata = read_h5_file(counts_output)
        print(f"Filtered to {gapfill_adata.shape[0]} gapfill cells.")
    else:
        gapfill_adata = read_h5_file(input)

    # Compute umi statistics
    stats = dict(
        statistic=[],
        value=[]
    )
    # Note that X is a csr_matrix so we must make sure it's dense
    umis_per_cell = np.asarray(gapfill_adata.X.sum(axis=1)).flatten()
    cells_per_gapfill = np.asarray((gapfill_adata.X > 0).sum(axis=0)).flatten()
    # Number of cells
    stats["statistic"].append("TOTAL_CELLS")
    stats["value"].append(gapfill_adata.shape[0])
    stats["statistic"].append("TOTAL_UMIS")
    stats["value"].append(gapfill_adata.X.sum())
    # Number of non-zero cells
    stats["statistic"].append("GAPFILL_CONTAINING_CELLS")
    stats["value"].append((umis_per_cell > 0).sum())
    # Summary stats for umis per cell
    stats["statistic"].append("UMIS_PER_CELL_MEAN")
    stats["value"].append(umis_per_cell.mean())
    stats["statistic"].append("UMIS_PER_CELL_MEDIAN")
    stats["value"].append(np.median(umis_per_cell))
    stats["statistic"].append("UMIS_PER_CELL_STD")
    stats["value"].append(umis_per_cell.std())
    stats["statistic"].append("UMIS_PER_CELL_MIN")
    stats["value"].append(umis_per_cell.min())
    stats["statistic"].append("UMIS_PER_CELL_MIN_EXCLUDING_ZERO")
    stats["value"].append(umis_per_cell[umis_per_cell > 0].min())
    stats["statistic"].append("UMIS_PER_CELL_MAX")
    stats["value"].append(umis_per_cell.max())
    # Summary stats for cells per gapfill
    stats["statistic"].append("CELLS_PER_GAPFILL_MEAN")
    stats["value"].append(cells_per_gapfill.mean())
    stats["statistic"].append("CELLS_PER_GAPFILL_MEDIAN")
    stats["value"].append(np.median(cells_per_gapfill))
    stats["statistic"].append("CELLS_PER_GAPFILL_STD")
    stats["value"].append(cells_per_gapfill.std())
    stats["statistic"].append("CELLS_PER_GAPFILL_MIN")
    stats["value"].append(cells_per_gapfill.min())
    stats["statistic"].append("CELLS_PER_GAPFILL_MAX")
    stats["value"].append(cells_per_gapfill.max())
    stats["statistic"].append("SEQUENCING_SATURATION")
    stats["value"].append(sequencing_saturation(collapse_gapfills(gapfill_adata).layers['total_reads']))

    pd.DataFrame(stats).to_csv(summary_output, index=False, sep="\t")

    # Generate a PDF report
    make_pdf_report(summary_pdf_output, gapfill_adata, adata)


def run(output, overwrite, cellranger_output, flatten):
    if isinstance(cellranger_output, str):
        cellranger_output = [cellranger_output]
    has_cellranger = cellranger_output is not None and len(cellranger_output) > 0
    if has_cellranger:
        cellranger_output = [Path(x) for x in cellranger_output]
        print("WTA CellRanger output provided.")

    output = Path(output)
    assert output.exists(), f"Output directory does not exist."
    # Search for counts.*.h5 files in the output dir
    print("Searching for counts files...", end="")
    counts_files = []
    for file in output.iterdir():
        if file.is_file() and file.name.startswith("counts.") and file.name.endswith(".h5") and not file.name.endswith(".filtered.h5"):
            counts_files.append(file)
    # Sort by the number in the filename
    counts_files.sort(key=lambda x: int(x.name.split(".")[1]))
    print(f"Found {len(counts_files)} files.")

    if has_cellranger and len(counts_files) != len(cellranger_output):
        print("WARNING: Number of CellRanger outputs does not match the number of counts files. This can lead to unexpected results.")

    for i, counts_file in enumerate(counts_files):
        if has_cellranger and i < len(cellranger_output):
            counts_cellranger = cellranger_output[i]
            print("Matching counts file", counts_file, "with CellRanger output", counts_cellranger)
        else:
            counts_cellranger = None

        summary_output_file = output / counts_file.name.replace(".h5", ".summary.tsv")
        summary_pdf_output_file = output / counts_file.name.replace(".h5", ".summary.pdf")
        h5_output_file = output / counts_file.name.replace(".h5", ".filtered.h5")
        flattened_output_file = output / counts_file.name.replace(".h5", ".filtered.tsv.gz").replace('counts.', 'flat_counts.')

        if summary_output_file.exists() or h5_output_file.exists():
            if overwrite:
                print("Overwriting existing files.")
            else:
                print("Skipping existing files. Use --overwrite to overwrite.")
                continue

        summarize_counts(counts_file, summary_output_file, summary_pdf_output_file, h5_output_file, flattened_output_file, counts_cellranger, flatten)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics for counts.", formatter_class=RichHelpFormatter
    )

    parser.add_argument(
        "--output", '-o',
        required=True,
        type=str,
        help="Path to the output directory."
    )

    parser.add_argument(
        "--overwrite", '-f',
        required=False,
        action="store_true",
        help="Overwrite the output files if they exist."
    )

    parser.add_argument(
        "--cellranger_output", '-wta',
        action="append",
        required=False,
        default=None,
        help="Path to either the filtered_feature_bc_matrix.h5 or the sample_filtered_feature_bc_matrix folder from CellRanger. "
             "Can be specified multiple times to merge multiple samples if multiplex (in order of the counts.N.h5 files is sorted by N)."
    )

    parser.add_argument(
        "--flatten",
        required=False,
        action="store_true",
        help="Flatten the final output to a gzipped tsv file."
    )

    args = parser.parse_args()
    run(args.output, args.overwrite, args.cellranger_output, args.flatten)


if __name__ == "__main__":
    main()
