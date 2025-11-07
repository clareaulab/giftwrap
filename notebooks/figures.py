import gc
from collections import defaultdict

import giftwrap as gw
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.stats import gaussian_kde
import adjustText
import seaborn as sns
import spatialdata as sd
import scanpy as sc
mpl.rcParams['figure.dpi'] = 300


def plot_HE(sdata):
    return (sdata.pl.render_images(f"_hires_image")
                .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title="Hematoxylin & Eosin Stain")
            )


def plot_library_size(sdata, table, resolution: int = 2, include_0bp: bool = False):
    assert table in ('gf', '')
    if table == 'gf':
        table = "gf_"

    if not include_0bp and table == 'gf_':
        zero_bp_probes = get_all_0bp_probes(sdata.tables[f'gf_square_{resolution:03d}um'])
        library = sdata.tables[f'gf_square_{resolution:03d}um'][:, ~sdata.tables[f'gf_square_{resolution:03d}um'].var.probe.isin(zero_bp_probes)].X.sum(1)
    else:
        library = sdata.tables[f'{table}square_{resolution:03d}um'].X.sum(1)
    sdata[f'square_{resolution:03d}um'].obs['library_size'] = library
    return (sdata.pl.render_images(f"_hires_image", alpha=0.8)
                .pl.render_shapes(element=f'_square_{resolution:03d}um', color='library_size', method='matplotlib', v='p95')
                .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title="Library Size")
            )

def plot_sites_genotyped(sdata, resolution: int = 2, at_least_one: bool = False):
    genotype_df = sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'].copy()
    # Remove 0bp
    zero_bp_probes = get_all_0bp_probes(sdata.tables[f'gf_square_{resolution:03d}um'])
    genotype_df = genotype_df.loc[:, ~genotype_df.columns.isin(zero_bp_probes)]
    number_genotyped = (~(genotype_df.isna() | (genotype_df == "N/A"))).sum(1).values
    if at_least_one:
        sdata[f'square_{resolution:03d}um'].obs['any_genotyped'] = number_genotyped >= 1
        return (sdata.pl.render_images(f"_hires_image", alpha=0.8)
                    .pl.render_shapes(element=f'_square_{resolution:03d}um', color='any_genotyped', method='matplotlib', cmap='bwr', cmin=0, cmax=1)
                    .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title="Bins with Any Genotype Called")
                )
    else:
        total_genotypes = genotype_df.shape[1]
        sdata[f'square_{resolution:03d}um'].obs['percent_genotyped'] = number_genotyped / total_genotypes * 100
        return (sdata.pl.render_images(f"_hires_image", alpha=0.8)
                    .pl.render_shapes(element=f'_square_{resolution:03d}um', color='percent_genotyped', method='matplotlib', cmap='bwr', cmin=0, cmax=100)
                    .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title="Percent of Sites Genotyped")
                )

def plot_library_specific_probe(sdata, probe: str, gapfill: str, resolution: int = 2):
    table = "gf_"
    sdata[f'square_{resolution:03d}um'].obs['library_size'] = sdata.tables[f'{table}square_{resolution:03d}um'][:, (sdata.tables[f'{table}square_{resolution:03d}um'].var.probe == probe) & (sdata.tables[f'{table}square_{resolution:03d}um'].var.gapfill == gapfill)].X.sum(1)
    return (sdata.pl.render_images(f"_hires_image", alpha=0.8)
                .pl.render_shapes(element=f'_square_{resolution:03d}um', color='library_size', method='matplotlib', v='p98')
                .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title=f"Library Size for {probe} {gapfill}")
            )


def compare_library_size_per_bin(sdata, resolution: int = 2, include_0bp: bool = False):
    # Compare library size per bin between WTA and GIFT-seq
    wta_lib = sdata.tables[f'square_{resolution:03d}um'].X.sum(1).__array__().flatten()
    if not include_0bp:
        zero_bp_probes = get_all_0bp_probes(sdata.tables[f'gf_square_{resolution:03d}um'])
        gf_lib = sdata.tables[f'gf_square_{resolution:03d}um'][:, ~sdata.tables[f'gf_square_{resolution:03d}um'].var.probe.isin(zero_bp_probes)].X.sum(1).flatten()
    else:
        gf_lib = sdata.tables[f'gf_square_{resolution:03d}um'].X.sum(1).flatten()
    xy = np.vstack([wta_lib, gf_lib])
    density = gaussian_kde(xy, bw_method="silverman")(xy)
    plt.figure(figsize=(5, 5))
    plt.scatter(wta_lib, gf_lib, c=density, cmap='viridis', alpha=0.5)
    plt.xlabel("WTA Library Size")
    plt.ylabel("GIFT-seq Library Size")
    plt.title(f"Library Size Comparison per {resolution}um Bin")
    plt.show()

def get_0bp_probe(adata, probe_name: str):
    curr_gene = adata.var[adata.var.probe == probe_name].gene.values[0]
    zero_bp_probe = adata.var[(adata.var.gene == curr_gene) & (adata.var.probe.str.contains("0bp") | (adata.var.probe == adata.var.gene))].probe.values
    if len(zero_bp_probe) < 1 or zero_bp_probe[0] == probe_name:
        return None
    return zero_bp_probe[0]

def get_all_0bp_probes(adata):
    zero_bp_probes = []
    for probe in adata.var.probe.unique():
        zero_bp_probe = get_0bp_probe(adata, probe)
        if zero_bp_probe is not None and zero_bp_probe not in zero_bp_probes:
            zero_bp_probes.append(zero_bp_probe)
    return zero_bp_probes

def plot_relative_efficiency(sdata, resolution: int = 2, min_gf_count: int = 0, min_0bp_count: int = 0):
    # gf_data = sdata
    if isinstance(sdata, ad.AnnData):
        gf_data = sdata
    else:
        gf_data = sdata.tables[f'gf_square_{resolution:03d}um']
    to_plot = {
        'probe': [],
        'gene': [],
        '0bp': [],
        'gf': []
    }
    for probe in gf_data.var.probe.unique():
        zero_bp_probe = get_0bp_probe(gf_data, probe)
        if zero_bp_probe is None:
            print(f"Can't find 0bp for: {probe}")
            continue
        gf_counts = gf_data[:, gf_data.var.probe == probe].X
        zero_bp_counts = gf_data[:, gf_data.var.probe == zero_bp_probe].X
        to_plot['probe'].append(probe)
        to_plot['gene'].append(gf_data.var[gf_data.var.probe == probe].gene.values[0])
        to_plot['gf'].append(gf_counts.sum())
        to_plot['0bp'].append(zero_bp_counts.sum())

    fig, ax = plt.subplots()
    ax.scatter(
        to_plot['0bp'],
        to_plot['gf'],
        alpha=0.7
    )

    df = pd.DataFrame(to_plot)
    median_ratio = ((df['gf'] + 1) / (df['0bp'] + 1)).median()
    texts = []
    for x, y, probe_name in zip(to_plot['0bp'], to_plot['gf'], to_plot['probe']):
        if x > min_0bp_count or y > min_gf_count:
            texts.append(ax.text(x, y, probe_name, fontsize=8))
    adjustText.adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', lw=0.5))

    ax.plot([1, max(to_plot['0bp'])], [median_ratio, median_ratio * max(to_plot['0bp'])], color='red', linestyle='--', label=f'Median Ratio: {median_ratio:.2f}')

    ax.set_xlabel("0bp Control Probe Counts")
    ax.set_ylabel("GIFT-seq Probe Counts")
    ax.set_title("GIFT-seq Probe Counts vs 0bp Control Probe Counts")

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    return fig, ax

def plot_genotypes(sdata, probe, resolution: int = 2, imputed: bool = False, use_anndata: bool = False):
    # gf_adata = sdata.tables[f'gf_square_{resolution:03d}um']
    if imputed:
        orig = sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'].copy()
        sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'] = sdata.tables[f'gf_square_{resolution:03d}um'].obsm['imputed_genotype']
    res = gw.sp.plot_genotypes(
        sdata.tables[f'gf_square_{resolution:03d}um'] if use_anndata else sdata, probe, image_name="hires_image", resolution=resolution
    )
    if imputed:
        sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'] = orig
    return res

def print_summary_stats(sdata, resolution: int = 2, include_0bp: bool = False):
    gf_adata = sdata.tables[f'gf_square_{resolution:03d}um']
    wta_adata = sdata.tables[f'square_{resolution:03d}um']

    print("Ignoring 0bp probes for summary stats...")

    if not include_0bp:
        zero_bp_probes = get_all_0bp_probes(gf_adata)
        gf_adata = gf_adata[:, ~gf_adata.var.probe.isin(zero_bp_probes)].copy()

    # Aggregate to probe level
    adata = gw.tl.collapse_gapfills(gf_adata)

    # N probes targeted and N with at least one count
    n_probes = adata.shape[1]
    n_at_least_one = (adata.X.sum(0) > 0).sum()
    print(f"Number of probes targeted: {n_probes}")
    print(f"Number of probes with at least one count: {n_at_least_one} ({n_at_least_one / n_probes * 100:.2f}%)")

    # Median counts per bin per probe (i.e. the median of matrix)
    median_counts_per_bin_per_probe = np.median(adata.X.flatten())
    print(f"Median counts per bin per probe: {median_counts_per_bin_per_probe:.2f}")
    # Mean counts per bin per probe (i.e. the mean of matrix)
    mean_counts_per_bin_per_probe = np.mean(adata.X.flatten())
    print(f"Mean counts per bin per probe: {mean_counts_per_bin_per_probe:.2f}")

    # Median counts per bin per gene for wta
    median_counts_per_bin_per_gene_wta = np.median(wta_adata.X.toarray().flatten())
    print(f"Median counts per bin per gene (WTA): {median_counts_per_bin_per_gene_wta:.2f}")
    mean_counts_per_bin_per_gene_wta = np.mean(wta_adata.X.toarray().flatten())
    print(f"Mean counts per bin per gene (WTA): {mean_counts_per_bin_per_gene_wta:.2f}")

def _assign_genotype_calls(table, probe, wt_gf, alt_gf):
    table.obs[probe] = 'N/A'
    for i, row in table.var.iterrows():
        gapfill = row['gapfill']
        call = 'WT' if gapfill == wt_gf else 'ALT' if gapfill == alt_gf else "Other"
        table.obs.loc[table.X[:, table.var.index.get_loc(i)].flatten() > 0, probe] = call
    # Call heterozygous if both WT and ALT are present
    wt_mask = (table.var.probe == probe) & (table.var.gapfill == wt_gf)
    alt_mask = (table.var.probe == probe) & (table.var.gapfill == alt_gf)
    wt_present = table.X[:, np.where(wt_mask)[0]].flatten() > 0
    alt_present = table.X[:, np.where(alt_mask)[0]].flatten() > 0
    if wt_present.shape != alt_present.shape:
        table.obs['both_present'] = False
        return table
    both_present = wt_present & alt_present
    table.obs['both_present'] = both_present
    table.obs.loc[table.obs['both_present'], probe] = 'HET'
    return table

def genotype_cell_line_barplots(sdata, probe: str, wt_gf: str, alt_gf: str, resolution: int = 2, ax=None):
    table = sdata.tables[f'gf_square_{resolution:03d}um'].copy()
    # Add the cell_line annotation to the obs if it doesn't exist
    if 'cell_line' not in table.obs.columns:
        wta = sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Add cell_line to gf table from wta by matching indices
            table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either gf or wta data.")
    table = table[:, table.var.probe == probe].copy()
    if table.shape[1] == 0:
        raise ValueError(f"Probe {probe} not found in data.")
    table = _assign_genotype_calls(table, probe, wt_gf, alt_gf)
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    prop_data = (
        table.obs.groupby(['cell_line', probe])
        .size()
        .reset_index(name='count')
        .groupby('cell_line')
        .apply(lambda x: x.assign(proportion=x['count'] / x['count'].sum()))
        .reset_index(drop=True)
    )
    # Filter N/A cell line
    prop_data = prop_data[prop_data['cell_line'] != 'N/A']
    sns.barplot(
        data=prop_data,
        x='cell_line',
        y='proportion',
        hue=probe,
        ax=ax,
        palette={'N/A': 'orange', 'WT': 'blue', 'ALT': 'red', 'HET': 'green', 'Other': 'lightgray'}
    )
    ax.set_title(f"Genotype Call Proportions for {probe} by Cell Line")
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Proportion of Bins")
    ax.legend(title="Genotype Call")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return ax

def genotype_accuracy_barplot(sdata, probe: str, wt_gf: str, alt_gf: str, celltype2genotype_acc: dict, resolution: int = 2, filter_na: bool = True, ax=None):
    table = sdata.tables[f'gf_square_{resolution:03d}um'].copy()
    if 'cell_line' not in table.obs.columns:
        wta = sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Add cell_line to gf table from wta by matching indices
            table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either wta data.")
    table = table[:, table.var.probe == probe].copy()
    if table.shape[1] == 0:
        raise ValueError(f"Probe {probe} not found in data.")
    table = _assign_genotype_calls(table, probe, wt_gf, alt_gf)

    accuracy_data = {
        'cell_line': [],
        'accuracy': []
    }
    for cell_type, expected_genotype in celltype2genotype_acc.items():
        subset = table.obs[table.obs['cell_line'] == cell_type]
        if len(subset) == 0:
            continue
        correct_calls = subset[subset[probe] == expected_genotype]
        if filter_na:
            accuracy = len(correct_calls) / (len(subset[subset[probe] != 'N/A'])+1e-4)
        else:
            accuracy = len(correct_calls) / (len(subset)+1e-4)
        accuracy_data['cell_line'].append(cell_type)
        accuracy_data['accuracy'].append(accuracy)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=pd.DataFrame(accuracy_data), x='cell_line', y='accuracy')
    else:
        sns.barplot(data=pd.DataFrame(accuracy_data), x='cell_line', y='accuracy', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Genotype Call Accuracy for {probe} by Cell Line")
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return ax


def genotype_psuedobulk_accuracy_by_pcr(
    sdata, probe: str, wt_gf: str, alt_gf: str, celltype2genotype_acc: dict, celltypes: list[str],
    resolution: int = 2, max_threshold: int = None
):
    # Get the table once and keep it in memory
    table = sdata.tables[f'gf_square_{resolution:03d}um']
    assert table.uns['all_pcr_thresholds']
    if max_threshold is None:
        max_threshold = table.uns['max_pcr_duplicates']

    # Prepare cell line annotations once
    if 'cell_line' not in table.obs.columns:
        wta = sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Create a copy only if we need to add cell_line annotations
            table = table.copy()
            table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either gf or wta data.")

    # Filter to probe once
    probe_mask = table.var.probe == probe
    if not probe_mask.any():
        raise ValueError(f"Probe {probe} not found in data.")

    # Get the probe-specific data once
    probe_table = table[:, probe_mask.values]
    gapfills = probe_table.var['gapfill'].values
    cell_lines = probe_table.obs['cell_line'].values

    # Create gapfill mapping once
    gapfill_to_idx = {gf: i for i, gf in enumerate(gapfills)}
    wt_idx = gapfill_to_idx.get(wt_gf, -1)
    alt_idx = gapfill_to_idx.get(alt_gf, -1)

    data = defaultdict(list)

    # Precompute celltype masks to avoid repeated computation
    unique_celltypes = set(cell_lines)
    celltype_masks = {ct: (cell_lines == ct) for ct in unique_celltypes if ct != 'N/A'}

    for threshold in range(1, max_threshold):
        # Get the appropriate data matrix for this threshold
        if threshold == 1:
            X_data = probe_table.X
        else:
            X_data = probe_table.layers[f'X_pcr_threshold_{threshold}']

        if hasattr(X_data, 'toarray'):
            X_data = X_data.toarray()

        pseudobulk_counts = {}

        for celltype, mask in celltype_masks.items():
            if not mask.any():
                pseudobulk_counts[celltype] = np.zeros(X_data.shape[1], dtype=X_data.dtype)
                continue
            counts = X_data[mask, :].sum(axis=0)
            if hasattr(counts, 'A1'):
                counts = counts.A1
            pseudobulk_counts[celltype] = counts

        for celltype in celltypes:
            counts = pseudobulk_counts.get(celltype, None)
            if counts is None or counts.sum() == 0:
                data[celltype].append(0.0)
                continue

            wt_count = counts[wt_idx] if wt_idx >= 0 else 0
            alt_count = counts[alt_idx] if alt_idx >= 0 else 0
            other_count = counts.sum() - wt_count - alt_count
            total = wt_count + alt_count + other_count

            if total == 0:
                accuracy = 0.0
            else:
                expected_genotype = celltype2genotype_acc.get(celltype, None)
                if expected_genotype == 'WT':
                    correct = wt_count
                elif expected_genotype == 'ALT':
                    correct = alt_count
                else:
                    correct = 0
                accuracy = correct / total

            data[celltype].append(accuracy)

        # Explicitly delete large arrays to help GC
        del X_data
        del pseudobulk_counts

    plt.figure(figsize=(8, 6))
    for celltype, accuracies in data.items():
        plt.plot(range(1, max_threshold), accuracies, label=celltype)
    plt.xlabel("PCR Duplicate Threshold")
    plt.ylabel("Genotype Call Accuracy")
    plt.title(f"Genotype Call Accuracy for {probe} by Cell Line vs PCR Duplicate Threshold")
    plt.ylim(0, 1)
    plt.legend(title="Cell Line")
    plt.show()
    plt.clf()


def psuedobulk_labels(sdata, probe: str, resolution: int = 2) -> pd.DataFrame:
    """
    Create a dataframe that simply contains the counts for each gapfill grouped by annotated cell lines in space.
    Optimized to avoid unnecessary copies and dense conversions.
    """
    table = sdata.tables[f'gf_square_{resolution:03d}um']
    if 'cell_line' not in table.obs.columns:
        wta = sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Add cell_line to gf table from wta by matching indices
            table = table.copy()
            table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either gf or wta data.")
    probe_mask = table.var.probe == probe
    if not probe_mask.any():
        raise ValueError(f"Probe {probe} not found in data.")
    # Avoid .copy() and .toarray() if possible
    X = table.X[:, probe_mask.values]
    if hasattr(X, "toarray"):
        # Only convert to dense if needed for DataFrame construction
        X = X.toarray()
    df = pd.DataFrame(X, columns=table.var['gapfill'][probe_mask], index=table.obs_names)
    df = df.join(table.obs[['cell_line']])
    df = df.groupby('cell_line').sum(numeric_only=True).reset_index()
    return df

def pseudobulk_genotype_table(sdata, probe: str, wt_gf: str, alt_gf: str, celltype2genotype_acc: dict, resolution: int = 2):
    labels = psuedobulk_labels(sdata, probe, resolution)
    # First, remove unlabelled cell lines
    labels = labels[labels['cell_line'] != 'N/A']
    # Next rename gapfills to WT, ALT, Other
    gapfill_to_genotype = {wt_gf: 'WT', alt_gf: 'ALT'}
    labels = labels.rename(columns=gapfill_to_genotype)
    # Identify columns that are not cell_line, WT, or ALT
    other_cols = [col for col in labels.columns if col not in ['cell_line', 'WT', 'ALT']]
    # Ensure WT and ALT columns exist, create with zeros if missing
    if 'WT' not in labels.columns:
        labels['WT'] = 0
    if 'ALT' not in labels.columns:
        labels['ALT'] = 0
    # Sum all 'Other' columns into a single 'Other' column
    if other_cols:
        labels['Other'] = labels[other_cols].sum(axis=1)
        labels = labels[['cell_line', 'WT', 'ALT', 'Other']]
    else:
        labels['Other'] = 0
        labels = labels[['cell_line', 'WT', 'ALT', 'Other']]
    # Next, add a column for expected genotype
    labels['expected_genotype'] = labels['cell_line'].map(celltype2genotype_acc)
    return labels


def _collect_dual_vs_gapfill(dual_sdata, gap_sdata, probe_dual, probe_gf, wt_gfs, alt_gfs, celltype2genotype_acc, resolution=2):
    if 'cell_line' not in dual_sdata.tables[f'gf_square_{resolution:03d}um'].obs.columns:
        wta = dual_sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Add cell_line to gf table from wta by matching indices
            dual_sdata.tables[f'gf_square_{resolution:03d}um'].obs['cell_line'] = wta.obs.loc[dual_sdata.tables[f'gf_square_{resolution:03d}um'].obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either dual or wta data.")
    if 'cell_line' not in gap_sdata.tables[f'gf_square_{resolution:03d}um'].obs.columns:
        wta = gap_sdata.tables[f'square_{resolution:03d}um']
        if 'cell_line' in wta.obs.columns:
            # Add cell_line to gf table from wta by matching indices
            gap_sdata.tables[f'gf_square_{resolution:03d}um'].obs['cell_line'] = wta.obs.loc[gap_sdata.tables[f'gf_square_{resolution:03d}um'].obs_names, 'cell_line'].values
        else:
            raise ValueError("cell_line annotation not found in either gapfill or wta data.")

    # Ignoring het cell lines:
    celltype2genotype_acc = celltype2genotype_acc.copy()
    celltype2genotype_acc = {k: v for k, v in celltype2genotype_acc.items() if v in ('WT', 'ALT')}

    df = {
        'true_call': [],
        'predicted_call': [],
        'method': []
    }

    for ct, true_call in celltype2genotype_acc.items():
        dual_table = dual_sdata.tables[f'gf_square_{resolution:03d}um'].copy()
        dual_table = dual_table[dual_table.obs['cell_line'] == ct, dual_table.var.probe == probe_dual]
        gap_table = gap_sdata.tables[f'gf_square_{resolution:03d}um'].copy()
        gap_table = gap_table[gap_table.obs['cell_line'] == ct, gap_table.var.probe == probe_gf]
        if dual_table.shape[0] == 0 or gap_table.shape[0] == 0:
            continue
        # Select cells that have at least one count for the probe
        dual_table.obs['library_size'] = dual_table.X.sum(1)
        dual_table = dual_table[dual_table.obs['library_size'] > 0, :]
        gap_table.obs['library_size'] = gap_table.X.sum(1)
        gap_table = gap_table[gap_table.obs['library_size'] > 0, :]
        if dual_table.shape[0] == 0 or gap_table.shape[0] == 0:
            print(f"Warning: No cells with counts for probe {probe_dual} in dual or {probe_gf} in gapfill for cell type {ct}")
            continue
        # For each cell, get the gapfill with the only count. If multiple, then set to Het
        dual_calls = []
        for i in range(dual_table.shape[0]):
            counts = dual_table.X[i, :].toarray().flatten() if hasattr(dual_table.X, 'toarray') else dual_table.X[i, :].flatten()
            if counts.sum() == 0:
                dual_calls.append('N/A')
            elif (counts > 0).sum() > 1:
                dual_calls.append('HET')
            else:
                gf = dual_table.var['gapfill'][counts > 0].values[0]
                call = 'WT' if gf in wt_gfs else 'ALT' if gf in alt_gfs else 'Other'
                dual_calls.append(call)
        gap_calls = []
        for i in range(gap_table.shape[0]):
            counts = gap_table.X[i, :].toarray().flatten() if hasattr(gap_table.X, 'toarray') else gap_table.X[i, :].flatten()
            if counts.sum() == 0:
                gap_calls.append('N/A')
            elif (counts > 0).sum() > 1:
                gap_calls.append('HET')
            else:
                gf = gap_table.var['gapfill'][counts > 0].values[0]
                call = 'WT' if gf in wt_gfs else 'ALT' if gf in alt_gfs else 'Other'
                gap_calls.append(call)
        df['true_call'].extend([true_call] * len(dual_calls))
        df['predicted_call'].extend(dual_calls)
        df['method'].extend(['Dual'] * len(dual_calls))
        df['true_call'].extend([true_call] * len(gap_calls))
        df['predicted_call'].extend(gap_calls)
        df['method'].extend(['Gapfill'] * len(gap_calls))

    df = pd.DataFrame(df)
    df['probe'] = probe_dual
    return df

def boxplot_of_dualprobe_vs_gapfill(
    dual_sdata, gap_sdata, annotated_genotypes, celltype_genotypes, wt_alleles, alt_alleles, resolution=2
):
    # Suppress ImplicitModificationWarning
    import warnings
    from anndata import ImplicitModificationWarning
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
    dfs = []
    for probe in gap_sdata.tables[f'gf_square_{resolution:03d}um'].var.probe.unique():
        probe_norm = probe.split("|")
        if len(probe_norm) > 1:
            probe_norm = " ".join(probe_norm[1:3])
        else:
            probe_norm = probe
        if probe_norm not in annotated_genotypes:
            continue

        # Now find the corresponding dual probe
        dual_probe = None
        for dp in dual_sdata.tables[f'gf_square_{resolution:03d}um'].var.probe.unique():
            if dp.split(">")[0] == probe_norm.split(">")[0]:
                dual_probe = dp
                break
        if dual_probe is None:
            print(f"Could not find dual probe for {probe}")
            continue
        print(f"Comparing dual probe {dual_probe} to gapfill probe {probe}")
        gf_wt_allele = wt_alleles[probe_norm]
        gf_alt_allele = alt_alleles[probe_norm]
        dp_wt_allele = "" if ">" not in dual_probe else dual_probe.split(">")[0][-1]
        dp_alt_allele = "" if ">" not in dual_probe else dual_probe.split(">")[1]
        ct_dict = {
            "HEL": "HET" if len(celltype_genotypes["HEL"][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes["HEL"][probe_norm] else "ALT"),
            "K562": "HET" if len(celltype_genotypes["K562"][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes["K562"][probe_norm] else "ALT"),
            "SET2": "HET" if len(celltype_genotypes["SET2"][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes["SET2"][probe_norm] else "ALT"),
        }
        df = _collect_dual_vs_gapfill(
            dual_sdata, gap_sdata, dual_probe, probe,
            wt_gfs=[dp_wt_allele, gf_wt_allele], alt_gfs=[dp_alt_allele, gf_alt_allele],
            celltype2genotype_acc=ct_dict, resolution=resolution
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    # Compute the proportion of correct calls for each method and probe split by WT and ALT
    summary = (
        df.groupby(['probe', 'method', 'true_call'])
        .apply(lambda x: pd.Series({
            'count_correct': (x['predicted_call'] == x['true_call']).sum(),
            'count_incorrect': (x['predicted_call'] != x['true_call']).sum(),
            'proportion_correct': (x['predicted_call'] == x['true_call']).mean()
        }))
        .reset_index()
    )

    # Now make a box plot for WT and one for ALT
    # Each will plot the distribution of correct calls for dual probe and gapfill probe side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.boxplot(
        data=summary[summary['true_call'] == 'WT'],
        x='method',
        y='proportion_correct',
        ax=axes[0],
        palette={'Dual': 'blue', 'Gapfill': 'orange'}
    )
    axes[0].set_title("WT Genotype Call Accuracy")
    axes[0].set_ylabel("Proportion of Correct Calls")
    axes[0].set_xlabel("Method")
    axes[0].set_ylim(0, 1)

    sns.boxplot(
        data=summary[summary['true_call'] == 'ALT'],
        x='method',
        y='proportion_correct',
        ax=axes[1],
        palette={'Dual': 'blue', 'Gapfill': 'orange'}
    )
    axes[1].set_title("ALT Genotype Call Accuracy")
    axes[1].set_ylabel("Proportion of Correct Calls")
    axes[1].set_xlabel("Method")
    axes[1].set_ylim(0, 1)
    plt.suptitle("Genotype Call Accuracy: Dual Probe vs Gapfill Probe")
    plt.tight_layout()
    plt.show()
    return summary, (fig, axes)

def plot_genotype_umi_comparison(sdata, cell_line1: str, cell_line2: str, annotated_genotypes, celltype_genotypes, wt_alleles, alt_alleles, resolution: int = 2, min_umi_threshold: int = 0):
    """
    Compare UMI counts between two cell lines for probes with different genotypes.

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill data
    cell_line1 : str
        Name of first cell line
    cell_line2 : str
        Name of second cell line
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    min_umi_threshold : int
        Minimum UMI count threshold to include a probe (default: 0)

    Returns:
    --------
    fig, ax, df : matplotlib figure and axis objects, and the underlying dataframe
    """
    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(sdata, ad.AnnData):
            wta = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # Prepare data for plotting
    plot_data = {
        'probe': [],
        'genotype': [],
        f'{cell_line1}_umi': [],
        f'{cell_line2}_umi': [],
        'label': [],
        f'{cell_line1}_genotype': [],
        f'{cell_line2}_genotype': []
    }

    for probe in non_zero_probes:
        # Normalize probe name (same logic as boxplot_of_dualprobe_vs_gapfill)
        probe_norm = probe.split("|")
        if len(probe_norm) > 1:
            probe_norm = " ".join(probe_norm[1:3])
        else:
            probe_norm = probe

        # Check if probe has genotype information
        if probe_norm not in annotated_genotypes:
            continue

        # Check if both cell lines have genotype info
        if cell_line1 not in celltype_genotypes or cell_line2 not in celltype_genotypes:
            continue

        if probe_norm not in celltype_genotypes[cell_line1] or probe_norm not in celltype_genotypes[cell_line2]:
            continue

        # Determine genotype call (WT, ALT, or HET)
        gt1 = "HET" if len(celltype_genotypes[cell_line1][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes[cell_line1][probe_norm] else "ALT")
        gt2 = "HET" if len(celltype_genotypes[cell_line2][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes[cell_line2][probe_norm] else "ALT")

        # Skip if either is HET
        if gt1 == 'HET' or gt2 == 'HET':
            continue

        # Skip if both have the same genotype (WT/WT or ALT/ALT)
        if gt1 == gt2:
            continue

        # Get probe-specific data
        probe_mask = table.var.probe == probe
        probe_table = table[:, probe_mask]

        # Detect if this is dual probe or gapfill probe data
        available_gapfills = probe_table.var.gapfill.unique().tolist()
        is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)  # Single nucleotide gapfills

        # Get WT and ALT alleles for this probe
        if is_dual_probe:
            # For dual probes, extract from probe name (e.g., "AKAP9 c.1389G>T" -> WT='G', ALT='T')
            if ">" in probe_norm:
                variant_part = probe_norm.split()[-1]  # Get "c.1389G>T"
                if ">" in variant_part:
                    bases = variant_part.split(">")
                    wt_allele = bases[0][-1]  # Last character before '>'
                    alt_allele = bases[1]     # Everything after '>'
                else:
                    continue
            else:
                continue
        else:
            # For gapfill probes, use the provided dictionaries
            wt_allele = wt_alleles[probe_norm]
            alt_allele = alt_alleles[probe_norm]

        # Calculate UMI sums for WT and ALT genotypes in each cell line
        for genotype, allele in [('WT', wt_allele), ('ALT', alt_allele)]:
            # Filter to gapfills matching this genotype
            gf_mask = probe_table.var.gapfill == allele

            if not gf_mask.any():
                continue

            gf_data = probe_table[:, gf_mask]

            # Sum UMIs for cell line 1
            cl1_mask = gf_data.obs['cell_line'] == cell_line1
            cl1_sum = gf_data.X[cl1_mask.values, :].sum() if cl1_mask.any() else 0

            # Sum UMIs for cell line 2
            cl2_mask = gf_data.obs['cell_line'] == cell_line2
            cl2_sum = gf_data.X[cl2_mask.values, :].sum() if cl2_mask.any() else 0

            # Apply threshold filter
            if cl1_sum < min_umi_threshold and cl2_sum < min_umi_threshold:
                continue

            plot_data['probe'].append(probe)
            plot_data['genotype'].append(genotype)
            plot_data[f'{cell_line1}_umi'].append(cl1_sum)
            plot_data[f'{cell_line2}_umi'].append(cl2_sum)
            plot_data['label'].append(f"{probe}|{genotype}")
            plot_data[f'{cell_line1}_genotype'].append(gt1)
            plot_data[f'{cell_line2}_genotype'].append(gt2)

    df = pd.DataFrame(plot_data)

    if len(df) == 0:
        raise ValueError(f"No probes found with different non-HET genotypes between {cell_line1} and {cell_line2}")

    # Create scatterplot with color coding by genotype
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by genotype
    colors = {'WT': 'blue', 'ALT': 'red'}
    for genotype in ['WT', 'ALT']:
        mask = df['genotype'] == genotype
        if mask.any():
            ax.scatter(
                df.loc[mask, f'{cell_line1}_umi'],
                df.loc[mask, f'{cell_line2}_umi'],
                alpha=0.6,
                s=50,
                c=colors[genotype],
                label=genotype
            )

    # Add diagonal line for reference
    max_val = max(df[f'{cell_line1}_umi'].max(), df[f'{cell_line2}_umi'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')

    ax.set_xlabel(f'{cell_line1} UMI Sum')
    ax.set_ylabel(f'{cell_line2} UMI Sum')
    ax.set_title(f'UMI Comparison: {cell_line1} vs {cell_line2}\n(Probes with Different Non-HET Genotypes)')
    ax.legend()

    # Optionally add labels for high-count probes
    if len(df) > 0:
        high_count_threshold = df[[f'{cell_line1}_umi', f'{cell_line2}_umi']].max().max() * 0.5
        texts = []
        for idx, row in df.iterrows():
            if row[f'{cell_line1}_umi'] > high_count_threshold or row[f'{cell_line2}_umi'] > high_count_threshold:
                texts.append(ax.text(row[f'{cell_line1}_umi'], row[f'{cell_line2}_umi'],
                                   row['label'], fontsize=8, alpha=0.7))

        if texts:
            adjustText.adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.5))

    plt.tight_layout()

    return fig, ax, df


def plot_genotype_precision_by_cell(sdata, cell_line1: str, cell_line2: str, annotated_genotypes, celltype_genotypes, wt_alleles, alt_alleles, resolution: int = 2, min_umi_threshold: int = 0, normalize_by_0bp: bool = False, verbose: bool = False, return_detailed_info: bool = False, use_density: bool = False):
    """
    Evaluate genotyping precision by plotting cell-level UMI counts supporting each cell line's genotype.

    For each cell/bin, counts UMIs that support Cell Line 1's genotype vs Cell Line 2's genotype
    across all probes where the two cell lines have different genotypes (one WT, one ALT).
    Points are colored by the actual cell_line annotation to visualize genotyping precision.

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill data
    cell_line1 : str
        Name of first cell line
    cell_line2 : str
        Name of second cell line
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    min_umi_threshold : int
        Minimum total UMI count threshold to include a cell (default: 0)
    normalize_by_0bp : bool
        If True, normalize each probe's UMI counts by dividing by its corresponding
        0bp probe's UMI counts for each cell (default: False)
    verbose : bool
        If True, print detailed information about probes and calculations (default: False)
    return_detailed_info : bool
        If True, return detailed per-probe breakdown as fourth return value (default: False)
    use_density : bool
        If True, plot KDE density contours instead of scatter plots (default: False)

    Returns:
    --------
    fig, ax, df : matplotlib figure and axis objects, and the underlying dataframe
    detailed_info : dict (only if return_detailed_info=True)
        Dictionary containing per-probe breakdown with keys:
        - 'probe_info': DataFrame with probe-level information
        - 'per_probe_contributions': DataFrame with per-cell, per-probe UMI contributions
    """
    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(sdata, ad.AnnData):
            wta = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # First pass: identify differentially-genotyped probes and their allele mapping
    diff_genotyped_probes = {}  # {probe: {'cl1_genotype': 'WT'/'ALT', 'cl2_genotype': 'WT'/'ALT', 'wt_allele': str, 'alt_allele': str}}

    for probe in non_zero_probes:
        # Normalize probe name
        probe_norm = probe.split("|")
        if len(probe_norm) > 1:
            probe_norm = " ".join(probe_norm[1:3])
        else:
            probe_norm = probe

        # Check if probe has genotype information
        if probe_norm not in annotated_genotypes:
            continue

        # Check if both cell lines have genotype info
        if cell_line1 not in celltype_genotypes or cell_line2 not in celltype_genotypes:
            continue

        if probe_norm not in celltype_genotypes[cell_line1] or probe_norm not in celltype_genotypes[cell_line2]:
            continue

        # Determine genotype call (WT, ALT, or HET)
        gt1 = "HET" if len(celltype_genotypes[cell_line1][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes[cell_line1][probe_norm] else "ALT")
        gt2 = "HET" if len(celltype_genotypes[cell_line2][probe_norm]) > 1 else ("WT" if wt_alleles[probe_norm] in celltype_genotypes[cell_line2][probe_norm] else "ALT")

        # Skip if either is HET
        if gt1 == 'HET' or gt2 == 'HET':
            continue

        # Skip if both have the same genotype (WT/WT or ALT/ALT)
        if gt1 == gt2:
            continue

        # Get probe-specific data to detect dual vs gapfill probe
        probe_mask = table.var.probe == probe
        probe_table = table[:, probe_mask]

        # Detect if this is dual probe or gapfill probe data
        available_gapfills = probe_table.var.gapfill.unique().tolist()
        is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)  # Single nucleotide gapfills

        # Get WT and ALT alleles for this probe
        if is_dual_probe:
            # For dual probes, extract from probe name
            if ">" in probe_norm:
                variant_part = probe_norm.split()[-1]
                if ">" in variant_part:
                    bases = variant_part.split(">")
                    wt_allele = bases[0][-1]
                    alt_allele = bases[1]
                else:
                    continue
            else:
                continue
        else:
            # For gapfill probes, use the provided dictionaries
            wt_allele = wt_alleles[probe_norm]
            alt_allele = alt_alleles[probe_norm]

        # Store this probe's info
        diff_genotyped_probes[probe] = {
            'cl1_genotype': gt1,
            'cl2_genotype': gt2,
            'wt_allele': wt_allele,
            'alt_allele': alt_allele
        }

    if len(diff_genotyped_probes) == 0:
        raise ValueError(f"No probes found with different non-HET genotypes between {cell_line1} and {cell_line2}")

    if verbose:
        print(f"\n=== Genotype Precision Analysis ===")
        print(f"Comparing: {cell_line1} vs {cell_line2}")
        print(f"Found {len(diff_genotyped_probes)} differentially-genotyped probes")
        print(f"Normalization: {'0bp' if normalize_by_0bp else 'None'}")
        print(f"\nProbe Details:")
        for probe, info in diff_genotyped_probes.items():
            print(f"  {probe}:")
            print(f"    {cell_line1}: {info['cl1_genotype']} (allele: {info['wt_allele'] if info['cl1_genotype'] == 'WT' else info['alt_allele']})")
            print(f"    {cell_line2}: {info['cl2_genotype']} (allele: {info['wt_allele'] if info['cl2_genotype'] == 'WT' else info['alt_allele']})")

    # If normalizing by 0bp, create mapping from probe to 0bp probe
    probe_to_0bp = {}
    if normalize_by_0bp:
        for probe in diff_genotyped_probes.keys():
            zero_bp_probe = get_0bp_probe(table, probe)
            if zero_bp_probe is None:
                print(f"Warning: No 0bp probe found for {probe}, skipping normalization for this probe")
            probe_to_0bp[probe] = zero_bp_probe

        if verbose:
            print(f"\n0bp Probe Mapping:")
            for probe, zero_bp in probe_to_0bp.items():
                print(f"  {probe} -> {zero_bp}")

    # Second pass: for each cell, count UMIs supporting each cell line's genotype
    n_cells = table.shape[0]
    cl1_support_umis = np.zeros(n_cells)
    cl2_support_umis = np.zeros(n_cells)

    # For detailed tracking
    if return_detailed_info:
        per_probe_data = []

    for probe, probe_info in diff_genotyped_probes.items():
        probe_mask = table.var.probe == probe

        cl1_gt = probe_info['cl1_genotype']
        cl2_gt = probe_info['cl2_genotype']
        wt_allele = probe_info['wt_allele']
        alt_allele = probe_info['alt_allele']

        # Determine which allele supports which cell line
        cl1_allele = wt_allele if cl1_gt == 'WT' else alt_allele
        cl2_allele = wt_allele if cl2_gt == 'WT' else alt_allele

        # Get UMI counts for CL1-supporting allele
        cl1_allele_mask = (table.var.probe == probe) & (table.var.gapfill == cl1_allele)
        if cl1_allele_mask.any():
            cl1_counts = table[:, cl1_allele_mask].X
            if hasattr(cl1_counts, 'toarray'):
                cl1_counts = cl1_counts.toarray()
            cl1_counts = cl1_counts.sum(axis=1).flatten()
        else:
            cl1_counts = np.zeros(n_cells)

        # Get UMI counts for CL2-supporting allele
        cl2_allele_mask = (table.var.probe == probe) & (table.var.gapfill == cl2_allele)
        if cl2_allele_mask.any():
            cl2_counts = table[:, cl2_allele_mask].X
            if hasattr(cl2_counts, 'toarray'):
                cl2_counts = cl2_counts.toarray()
            cl2_counts = cl2_counts.sum(axis=1).flatten()
        else:
            cl2_counts = np.zeros(n_cells)

        # Normalize by 0bp probe if requested
        zero_bp_counts_array = None
        if normalize_by_0bp and probe in probe_to_0bp:
            zero_bp_probe = probe_to_0bp[probe]

            # Get 0bp probe UMI counts for each cell
            zero_bp_mask = table.var.probe == zero_bp_probe
            if zero_bp_mask.any():
                zero_bp_counts = table[:, zero_bp_mask].X.sum(axis=1)
                if hasattr(zero_bp_counts, 'A1'):
                    zero_bp_counts = zero_bp_counts.A1
                zero_bp_counts = zero_bp_counts.flatten()
                zero_bp_counts_array = zero_bp_counts.copy()  # Save for logging

                # Normalize: divide by 0bp counts (avoid division by zero)
                # Where 0bp count is 0, set normalized value to 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    cl1_counts = np.where(zero_bp_counts > 0, cl1_counts / (zero_bp_counts+1), cl1_counts)
                    cl2_counts = np.where(zero_bp_counts > 0, cl2_counts / (zero_bp_counts+1), cl2_counts)
            else:
                print(f"Warning: No 0bp probe counts found for {zero_bp_probe}, skipping normalization for this probe")

        # Log per-probe statistics
        if verbose:
            print(f"\n  Processing {probe}:")
            print(f"    Total {cell_line1} UMIs: {cl1_counts.sum():.2f}")
            print(f"    Total {cell_line2} UMIs: {cl2_counts.sum():.2f}")
            if normalize_by_0bp and zero_bp_counts_array is not None:
                print(f"    Mean 0bp counts: {zero_bp_counts_array.mean():.2f}")

        # Collect detailed per-probe data
        if return_detailed_info:
            per_probe_data.append({
                'probe': probe,
                'cl1_genotype': cl1_gt,
                'cl2_genotype': cl2_gt,
                'cl1_allele': cl1_allele,
                'cl2_allele': cl2_allele,
                'total_cl1_umis': cl1_counts.sum(),
                'total_cl2_umis': cl2_counts.sum(),
                'mean_cl1_umis_per_cell': cl1_counts.mean(),
                'mean_cl2_umis_per_cell': cl2_counts.mean(),
                'median_cl1_umis_per_cell': np.median(cl1_counts),
                'median_cl2_umis_per_cell': np.median(cl2_counts),
                'n_cells_with_cl1_umis': (cl1_counts > 0).sum(),
                'n_cells_with_cl2_umis': (cl2_counts > 0).sum(),
            })

        # Add to cumulative counts
        cl1_support_umis += cl1_counts
        cl2_support_umis += cl2_counts

    # Create dataframe with cell-level data
    plot_data = pd.DataFrame({
        'cell': table.obs_names,
        'cell_line': table.obs['cell_line'].values,
        f'{cell_line1}_support_umis': cl1_support_umis,
        f'{cell_line2}_support_umis': cl2_support_umis
    })

    # Filter to only the two cell lines being compared
    plot_data = plot_data[plot_data['cell_line'].isin([cell_line1, cell_line2])]

    # Filter out cells with 0 UMIs on both axes
    plot_data = plot_data[
        (plot_data[f'{cell_line1}_support_umis'] > min_umi_threshold) |
        (plot_data[f'{cell_line2}_support_umis'] > min_umi_threshold)
    ]

    if len(plot_data) == 0:
        raise ValueError(f"No cells found with UMIs above threshold {min_umi_threshold} for {cell_line1} or {cell_line2}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Use a color palette for the two cell lines
    colors = plt.cm.tab10([0, 1])  # Use first two colors
    color_map = {cell_line1: colors[0], cell_line2: colors[1]}

    if use_density:
        # Create a common grid for both cell lines
        x_all = plot_data[f'{cell_line1}_support_umis'].values
        y_all = plot_data[f'{cell_line2}_support_umis'].values

        # Determine grid bounds
        x_min_global, x_max_global = x_all.min(), x_all.max()
        y_min_global, y_max_global = y_all.min(), y_all.max()

        # Add padding
        x_range = x_max_global - x_min_global
        y_range = y_max_global - y_min_global
        x_min_global -= 0.05 * x_range
        x_max_global += 0.05 * x_range
        y_min_global -= 0.05 * y_range
        y_max_global += 0.05 * y_range

        # Create mesh grid (higher resolution for smoother heatmap)
        xx, yy = np.mgrid[x_min_global:x_max_global:200j, y_min_global:y_max_global:200j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Plot density heatmap for each cell line
        for i, cell_line in enumerate([cell_line1, cell_line2]):
            mask = plot_data['cell_line'] == cell_line
            if mask.any():
                x = plot_data.loc[mask, f'{cell_line1}_support_umis'].values
                y = plot_data.loc[mask, f'{cell_line2}_support_umis'].values

                # Only compute KDE if we have enough points
                if len(x) > 10:
                    # Create KDE
                    xy = np.vstack([x, y])
                    try:
                        kde = gaussian_kde(xy)
                        density = kde(positions).reshape(xx.shape)

                        # Normalize density to 0-1 range for better visualization
                        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)

                        # Create density heatmap with alpha blending
                        cmap = plt.cm.Blues if cell_line == cell_line1 else plt.cm.Oranges
                        cmap = cmap.copy()
                        cmap.set_bad(alpha=0)  # Make zero density transparent

                        # Mask out very low density values to avoid background noise
                        density_masked = np.ma.masked_where(density_norm < 0.05, density_norm)

                        ax.imshow(
                            density_masked.T,
                            origin='lower',
                            extent=[x_min_global, x_max_global, y_min_global, y_max_global],
                            cmap=cmap,
                            alpha=0.5,
                            aspect='auto',
                            interpolation='gaussian'
                        )

                        # Add a dummy point for legend
                        ax.plot([], [], color=color_map[cell_line], linewidth=4, label=cell_line)
                    except Exception as e:
                        print(f"Warning: Could not compute KDE for {cell_line}: {e}")
                        # Fallback to scatter if KDE fails
                        ax.scatter(x, y, alpha=0.6, s=50, c=[color_map[cell_line]], label=cell_line)
                else:
                    # Too few points for KDE, use scatter
                    ax.scatter(x, y, alpha=0.6, s=50, c=[color_map[cell_line]], label=cell_line)
    else:
        # Plot scatter points for each cell line
        for cell_line in [cell_line1, cell_line2]:
            mask = plot_data['cell_line'] == cell_line
            if mask.any():
                ax.scatter(
                    plot_data.loc[mask, f'{cell_line1}_support_umis'],
                    plot_data.loc[mask, f'{cell_line2}_support_umis'],
                    alpha=0.6,
                    s=50,
                    c=[color_map[cell_line]],
                    label=cell_line
                )

    # Add diagonal line for reference
    max_val = max(
        plot_data[f'{cell_line1}_support_umis'].max(),
        plot_data[f'{cell_line2}_support_umis'].max()
    )
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='y=x')

    # Set labels based on normalization
    if normalize_by_0bp:
        x_label = f'{cell_line1} Genotype Support (Normalized by 0bp)'
        y_label = f'{cell_line2} Genotype Support (Normalized by 0bp)'
        title_suffix = '(0bp normalized)'
    else:
        x_label = f'{cell_line1} Genotype Support (UMI Count)'
        y_label = f'{cell_line2} Genotype Support (UMI Count)'
        title_suffix = ''

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f'Genotyping Precision: {cell_line1} vs {cell_line2} {title_suffix}\n'
        f'({len(diff_genotyped_probes)} differentially-genotyped probes, {len(plot_data)} cells)',
        fontsize=14
    )
    ax.legend(title='Actual Cell Line', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Prepare detailed info if requested
    if return_detailed_info:
        probe_info_df = pd.DataFrame(per_probe_data)

        if verbose:
            print(f"\n=== Per-Probe Summary ===")
            print(probe_info_df.to_string())

        detailed_info = {
            'probe_info': probe_info_df,
            'n_probes': len(diff_genotyped_probes),
            'n_cells_plotted': len(plot_data),
            'normalization_used': normalize_by_0bp
        }

        return fig, ax, plot_data, detailed_info

    return fig, ax, plot_data


def plot_celltype_specific_probes_spatial(
    sdata,
    cell_line: str,
    annotated_genotypes,
    celltype_genotypes,
    wt_alleles,
    alt_alleles,
    resolution: int = 2,
    include_het: bool = False,
    color_by_celline: bool = False,
    log_scale_histograms: bool = True,
    figsize: tuple = (15, 15)
):
    """
    Plot spatial distribution of UMI counts for cell type-specific probes with marginal histograms.

    This function identifies probes where the specified cell line has a different genotype compared
    to other cell lines (e.g., cell line is ALT while others are WT, or vice versa). It then
    visualizes the spatial distribution of these cell-type specific probes with marginal histograms
    showing aggregate UMI counts along x and y axes.

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    cell_line : str
        Name of the cell line to analyze (e.g., 'HEL', 'K562', 'SET2')
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    include_het : bool
        If True, allow probes where other (non-target) cell lines have HET genotypes
        to be considered cell-type specific. When counting UMIs for such probes,
        spatial bins belonging to HET cell lines are excluded from the visualization
        The target cell line is never allowed to be HET
        (default: False)
    color_by_cellline : bool
        If True, color the spatial plot by cell_line annotation instead of UMI counts
        (default: False)
    log_scale_histograms : bool
        If True, apply log scaling to the y-axis of the marginal histograms
        (default: True)
    figsize : tuple
        Figure size for the plot (default: (15, 15))

    Returns:
    --------
    fig, axes, df : matplotlib figure, axes objects (main, top, right), and the underlying dataframe
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(sdata, ad.AnnData):
            wta = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    table.obs['x_coord'] = [c[1] for c in coords]
    table.obs['y_coord'] = [c[0] for c in coords]

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # Prepare data for identifying cell-type specific probes
    probe_info = {
        'probe': [],
        'probe_norm': [],
        'target_genotype': [],
        'is_specific': [],
        'het_cell_lines': [],  # Track which cell lines have HET for this probe
        'same_genotype_cell_lines': []  # Track which cell lines have same genotype as target
    }

    for probe in non_zero_probes:
        probe_norm = probe.split("|")
        if len(probe_norm) > 1:
            probe_norm = " ".join(probe_norm[1:3])
        else:
            probe_norm = probe

        if probe_norm not in annotated_genotypes:
            continue
        if cell_line not in celltype_genotypes:
            continue
        if probe_norm not in celltype_genotypes[cell_line]:
            continue

        target_alleles = set(celltype_genotypes[cell_line][probe_norm])

        if len(target_alleles) > 1:
            target_genotype = "HET"
        elif wt_alleles[probe_norm] in target_alleles:
            target_genotype = "WT"
        elif alt_alleles[probe_norm] in target_alleles:
            target_genotype = "ALT"
        else:
            target_genotype = "Unknown"

        # ALWAYS skip HET target genotypes (target cell line is never allowed to be HET)
        if target_genotype == 'HET':
            continue

        # Skip Unknown genotypes (alleles not matching WT or ALT)
        if target_genotype == 'Unknown':
            continue

        has_different_genotype = False  # Track if at least one cell line has different genotype
        het_cell_lines = []  # Track which other cell lines have HET for this probe
        same_genotype_cell_lines = []  # Track which cell lines have same genotype as target

        for other_cell_line, genotypes in celltype_genotypes.items():
            if other_cell_line == cell_line:
                continue
            if probe_norm in genotypes:
                other_alleles = set(genotypes[probe_norm])

                # Determine other cell line's genotype
                if len(other_alleles) > 1:
                    other_genotype = "HET"
                elif wt_alleles[probe_norm] in other_alleles:
                    other_genotype = "WT"
                elif alt_alleles[probe_norm] in other_alleles:
                    other_genotype = "ALT"
                else:
                    other_genotype = "Unknown"

                # Track HET cell lines
                if other_genotype == 'HET':
                    het_cell_lines.append(other_cell_line)
                    if not include_het:
                        # When include_het=False, any HET in other cell lines disqualifies this probe
                        break
                    # When include_het=True, HET is considered different from WT/ALT
                    elif target_genotype in ('WT', 'ALT'):
                        has_different_genotype = True

                # Check if other cell line has same genotype as target
                elif target_genotype in ('WT', 'ALT') and other_genotype == target_genotype:
                    same_genotype_cell_lines.append(other_cell_line)

                # Check if other cell line has different non-HET genotype
                elif target_genotype in ('WT', 'ALT') and other_genotype in ('WT', 'ALT') and target_genotype != other_genotype:
                    has_different_genotype = True

        # Only mark as specific if at least one cell line has a different genotype
        is_specific = has_different_genotype

        # When include_het=False, don't consider specific if any other cell line is HET
        if not include_het and len(het_cell_lines) > 0:
            is_specific = False

        probe_info['probe'].append(probe)
        probe_info['probe_norm'].append(probe_norm)
        probe_info['target_genotype'].append(target_genotype)
        probe_info['is_specific'].append(is_specific)
        probe_info['het_cell_lines'].append(het_cell_lines)
        probe_info['same_genotype_cell_lines'].append(same_genotype_cell_lines)

    # Filter to only cell-type specific probes
    df_probes = pd.DataFrame(probe_info)
    df_probes = df_probes[df_probes['is_specific']]

    if len(df_probes) == 0:
        raise ValueError(f"No cell-type specific probes found for {cell_line}")

    # Calculate UMI counts per spatial location
    # Filter by the specific genotype (WT or ALT allele) for each probe
    umi_counts = np.zeros(table.shape[0])

    for _, row in df_probes.iterrows():
        probe = row['probe']
        probe_norm = row['probe_norm']
        target_genotype = row['target_genotype']
        het_cell_lines = row['het_cell_lines']
        same_genotype_cell_lines = row['same_genotype_cell_lines']

        # Get probe-specific data to detect dual vs gapfill probe
        probe_mask = table.var.probe == probe
        probe_table = table[:, probe_mask]

        # Detect if this is dual probe or gapfill probe data
        available_gapfills = probe_table.var.gapfill.unique().tolist()
        is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)  # Single nucleotide gapfills

        # Get WT and ALT alleles for this probe
        if is_dual_probe:
            # For dual probes, extract from probe name (e.g., "AKAP9 c.1389G>T" -> WT='G', ALT='T')
            if ">" in probe_norm:
                variant_part = probe_norm.split()[-1]  # Get "c.1389G>T"
                if ">" in variant_part:
                    bases = variant_part.split(">")
                    wt_allele = bases[0][-1]  # Last character before '>'
                    alt_allele = bases[1]     # Everything after '>'
                else:
                    continue
            else:
                continue
        else:
            # For gapfill probes, use the provided dictionaries
            wt_allele = wt_alleles[probe_norm]
            alt_allele = alt_alleles[probe_norm]

        if target_genotype == 'WT':
            valid_alleles = [wt_allele]
        elif target_genotype == 'ALT':
            valid_alleles = [alt_allele]
        elif target_genotype == 'HET':
            valid_alleles = [wt_allele, alt_allele]
        else:
            continue

        # Filter to specific gapfill alleles
        gapfill_mask = table.var.gapfill.isin(valid_alleles)
        combined_mask = probe_mask & gapfill_mask

        if combined_mask.any():
            probe_counts = table[:, combined_mask].X.sum(axis=1)
            if hasattr(probe_counts, 'A1'):
                probe_counts = probe_counts.A1
            probe_counts = probe_counts.flatten()

            # Build list of cell lines to exclude from UMI counts
            exclude_cell_lines = []

            # When include_het=True, exclude UMI counts from HET cell lines for this probe
            if include_het and len(het_cell_lines) > 0:
                exclude_cell_lines.extend(het_cell_lines)

            # Always exclude cell lines with the same genotype as target
            if len(same_genotype_cell_lines) > 0:
                exclude_cell_lines.extend(same_genotype_cell_lines)

            # Apply exclusion mask if there are cell lines to exclude
            if len(exclude_cell_lines) > 0:
                # Create a mask for bins NOT belonging to excluded cell lines
                cell_line_mask = ~table.obs['cell_line'].isin(exclude_cell_lines)
                # Zero out counts from excluded cell lines
                probe_counts = probe_counts * cell_line_mask.values

            umi_counts += probe_counts

    # Get all spatial coordinates
    x_coords = table.obs['x_coord'].values
    y_coords = table.obs['y_coord'].values

    # Remove NaN coordinates
    valid_mask = ~(np.isnan(x_coords) & np.isnan(y_coords))
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    umi_counts_valid = umi_counts[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create 2D matrix for heatmap
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

    if color_by_celline:
        # Get cell_line values for each coordinate
        cell_line_values = table.obs['cell_line'].values[valid_mask]

        # Create a categorical mapping for cell lines
        unique_cell_lines = sorted(table.obs['cell_line'].unique())
        cell_line_to_num = {cl: i for i, cl in enumerate(unique_cell_lines)}

        # Fill in the matrix with cell_line indices
        for x, y, cl in zip(x_coords, y_coords, cell_line_values):
            heatmap_matrix[y - y_min, x - x_min] = cell_line_to_num.get(cl, np.nan)
    else:
        # Fill in the matrix with UMI counts
        for x, y, count in zip(x_coords, y_coords, umi_counts_valid):
            heatmap_matrix[y - y_min, x - x_min] = count

    # Compute marginal histograms (aggregate UMI counts)
    if color_by_celline:
        # When coloring by cell line, marginals still show UMI counts
        x_marginal = np.zeros(x_max - x_min + 1)
        y_marginal = np.zeros(y_max - y_min + 1)
        for x, y, count in zip(x_coords, y_coords, umi_counts_valid):
            x_marginal[x - x_min] += count
            y_marginal[y - y_min] += count
    else:
        x_marginal = np.nansum(heatmap_matrix, axis=0)  # Sum along y-axis for each x
        y_marginal = np.nansum(heatmap_matrix, axis=1)  # Sum along x-axis for each y

    # Create figure with GridSpec for main plot + marginals + colorbar
    fig = plt.figure(figsize=figsize)
    if color_by_celline:
        # No colorbar needed when using legend
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.05,
                      width_ratios=[1, 1, 1, 1])
    else:
        gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.05,
                      width_ratios=[1, 1, 1, 1, 0.15])  # Last column for colorbar

    # Create axes
    if color_by_celline:
        ax_main = fig.add_subplot(gs[1:, :-1])  # Main heatmap
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)  # Top marginal
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)  # Right marginal
    else:
        ax_main = fig.add_subplot(gs[1:, :-2])  # Main heatmap (exclude rightmost 2 columns)
        ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)  # Top marginal
        ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)  # Right marginal
        ax_cbar = fig.add_subplot(gs[1:, -1])  # Colorbar axes

    # Plot main heatmap
    if color_by_celline:
        # Use a categorical colormap for cell lines
        n_cell_lines = len(unique_cell_lines)
        cmap = plt.cm.get_cmap('tab10', n_cell_lines)
        cmap_copy = cmap.copy()
        cmap_copy.set_bad(color='lightgray')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='auto',
            origin='upper',
            cmap=cmap_copy,
            interpolation='nearest',
            extent=[x_min, x_max + 1, y_max + 1, y_min],
            vmin=0,
            vmax=n_cell_lines - 1
        )

        # Create legend patches for each cell line
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap(i), label=cl)
                          for i, cl in enumerate(unique_cell_lines)]
        ax_main.legend(handles=legend_elements, loc='upper right',
                      title='Cell Line', framealpha=0.9)
    else:
        # Use a colormap that handles NaN values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='auto',
            origin='upper',  # Invert y-axis so 0 is at the top
            cmap=cmap,
            interpolation='nearest',
            extent=[x_min, x_max + 1, y_max + 1, y_min]  # Flipped y extent for origin='upper'
        )

        # Add colorbar to dedicated axes
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('UMI Count', rotation=270, labelpad=20)

    ax_main.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (μm)', fontsize=12)
    ax_main.set_title(f'{cell_line}-Specific Probes\n({len(df_probes)} probes)', fontsize=14, fontweight='bold')

    # Plot top marginal (x-axis aggregate)
    x_positions = np.arange(x_min, x_max + 1)
    ax_top.bar(x_positions, x_marginal, width=1.0, color='steelblue', alpha=0.7, edgecolor='none')
    ax_top.set_ylabel('Total UMIs', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # Apply log scaling to top histogram if requested
    if log_scale_histograms:
        ax_top.set_yscale('log')

    # Plot right marginal (y-axis aggregate)
    # Reverse y_marginal to match the flipped y-axis from origin='upper'
    y_positions = np.arange(y_min, y_max + 1)
    ax_right.barh(y_positions, y_marginal, height=1.0, color='steelblue', alpha=0.7, edgecolor='none')
    ax_right.set_xlabel('Total UMIs', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    # Apply log scaling to right histogram if requested
    if log_scale_histograms:
        ax_right.set_xscale('log')

    # Create summary dataframe
    summary_df = df_probes.copy()
    summary_df['total_umis'] = [
        table[:, table.var.probe == probe].X.sum() for probe in df_probes['probe']
    ]

    # Return figure, axes, and summary data
    axes = {
        'main': ax_main,
        'top': ax_top,
        'right': ax_right
    }

    return fig, axes, summary_df


def plot_celltype_specific_probes_spatial_multi_cellline(
    sdata,
    annotated_genotypes,
    celltype_genotypes,
    wt_alleles,
    alt_alleles,
    resolution: int = 2,
    include_het: bool = False,
    color_by_celline: bool = False,
    log_scale_marginals: bool = True,
    figsize: tuple = (15, 15)
):
    """
    Plot spatial distribution of UMI counts for cell type-specific probes with line plot marginals for all cell lines.

    This function loops through all cell lines, identifies cell-line-specific probes for each
    (using the same logic as plot_celltype_specific_probes_spatial), and displays line plot
    marginals showing the spatial distribution of each cell line's specific probes with different colors.

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    include_het : bool
        If True, allow probes where other (non-target) cell lines have HET genotypes
        to be considered cell-type specific. When counting UMIs for such probes,
        spatial bins belonging to HET cell lines are excluded from the visualization
        The target cell line is never allowed to be HET
        (default: False)
    color_by_celline : bool
        If True, color the spatial plot by cell_line annotation instead of UMI counts
        (default: False)
    log_scale_marginals : bool
        If True, apply log scaling to the marginal line plots
        (default: True)
    figsize : tuple
        Figure size for the plot (default: (15, 15))

    Returns:
    --------
    fig, axes, summary_data : matplotlib figure, axes objects (main, top, right), and summary dictionary
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(sdata, ad.AnnData):
            wta = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    table.obs['x_coord'] = [c[1] for c in coords]
    table.obs['y_coord'] = [c[0] for c in coords]

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # Get all unique cell lines and assign colors
    unique_cell_lines = sorted(table.obs['cell_line'].unique())
    n_cell_lines = len(unique_cell_lines)

    # Use tab10 colormap for up to 10 cell lines, otherwise use a larger palette
    if n_cell_lines <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_cell_lines]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_cell_lines]

    cell_line_colors = {cl: colors[i] for i, cl in enumerate(unique_cell_lines)}

    # Initialize UMI count tracking per cell line
    umi_counts_by_cellline = {cl: np.zeros(table.shape[0]) for cl in unique_cell_lines}
    umi_counts_total = np.zeros(table.shape[0])

    # Track probe info across all cell lines
    all_probes_info = []

    # Loop through each cell line to find its specific probes
    for target_cell_line in unique_cell_lines:
        if target_cell_line not in celltype_genotypes:
            continue

        # Prepare data for identifying cell-type specific probes for this cell line
        probe_info = {
            'cell_line': [],
            'probe': [],
            'probe_norm': [],
            'target_genotype': [],
            'is_specific': [],
            'het_cell_lines': [],
            'same_genotype_cell_lines': []
        }

        for probe in non_zero_probes:
            probe_norm = probe.split("|")
            if len(probe_norm) > 1:
                probe_norm = " ".join(probe_norm[1:3])
            else:
                probe_norm = probe

            if probe_norm not in annotated_genotypes:
                continue
            if probe_norm not in celltype_genotypes[target_cell_line]:
                continue

            target_alleles = set(celltype_genotypes[target_cell_line][probe_norm])

            if len(target_alleles) > 1:
                target_genotype = "HET"
            elif wt_alleles[probe_norm] in target_alleles:
                target_genotype = "WT"
            elif alt_alleles[probe_norm] in target_alleles:
                target_genotype = "ALT"
            else:
                target_genotype = "Unknown"

            # ALWAYS skip HET target genotypes
            if target_genotype == 'HET':
                continue

            # Skip Unknown genotypes
            if target_genotype == 'Unknown':
                continue

            has_different_genotype = False
            het_cell_lines = []
            same_genotype_cell_lines = []

            for other_cell_line, genotypes in celltype_genotypes.items():
                if other_cell_line == target_cell_line:
                    continue
                if probe_norm in genotypes:
                    other_alleles = set(genotypes[probe_norm])

                    if len(other_alleles) > 1:
                        other_genotype = "HET"
                    elif wt_alleles[probe_norm] in other_alleles:
                        other_genotype = "WT"
                    elif alt_alleles[probe_norm] in other_alleles:
                        other_genotype = "ALT"
                    else:
                        other_genotype = "Unknown"

                    if other_genotype == 'HET':
                        het_cell_lines.append(other_cell_line)
                        if not include_het:
                            break
                        elif target_genotype in ('WT', 'ALT'):
                            has_different_genotype = True

                    elif target_genotype in ('WT', 'ALT') and other_genotype == target_genotype:
                        same_genotype_cell_lines.append(other_cell_line)

                    elif target_genotype in ('WT', 'ALT') and other_genotype in ('WT', 'ALT') and target_genotype != other_genotype:
                        has_different_genotype = True

            is_specific = has_different_genotype

            if not include_het and len(het_cell_lines) > 0:
                is_specific = False

            probe_info['cell_line'].append(target_cell_line)
            probe_info['probe'].append(probe)
            probe_info['probe_norm'].append(probe_norm)
            probe_info['target_genotype'].append(target_genotype)
            probe_info['is_specific'].append(is_specific)
            probe_info['het_cell_lines'].append(het_cell_lines)
            probe_info['same_genotype_cell_lines'].append(same_genotype_cell_lines)

        # Filter to only cell-type specific probes for this cell line
        df_probes = pd.DataFrame(probe_info)
        df_probes = df_probes[df_probes['is_specific']]

        # Calculate UMI counts for this cell line's specific probes
        for _, row in df_probes.iterrows():
            probe = row['probe']
            probe_norm = row['probe_norm']
            target_genotype = row['target_genotype']
            het_cell_lines = row['het_cell_lines']
            same_genotype_cell_lines = row['same_genotype_cell_lines']

            # Get probe-specific data
            probe_mask = table.var.probe == probe
            probe_table = table[:, probe_mask]

            # Detect if dual probe or gapfill probe
            available_gapfills = probe_table.var.gapfill.unique().tolist()
            is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)

            # Get WT and ALT alleles
            if is_dual_probe:
                if ">" in probe_norm:
                    variant_part = probe_norm.split()[-1]
                    if ">" in variant_part:
                        bases = variant_part.split(">")
                        wt_allele = bases[0][-1]
                        alt_allele = bases[1]
                    else:
                        continue
                else:
                    continue
            else:
                wt_allele = wt_alleles[probe_norm]
                alt_allele = alt_alleles[probe_norm]

            if target_genotype == 'WT':
                valid_alleles = [wt_allele]
            elif target_genotype == 'ALT':
                valid_alleles = [alt_allele]
            elif target_genotype == 'HET':
                valid_alleles = [wt_allele, alt_allele]
            else:
                continue

            # Filter to specific gapfill alleles
            gapfill_mask = table.var.gapfill.isin(valid_alleles)
            combined_mask = probe_mask & gapfill_mask

            if combined_mask.any():
                probe_counts = table[:, combined_mask].X.sum(axis=1)
                if hasattr(probe_counts, 'A1'):
                    probe_counts = probe_counts.A1
                probe_counts = probe_counts.flatten()

                # Build list of cell lines to exclude from UMI counts
                exclude_cell_lines = []

                # When include_het=True, exclude UMI counts from HET cell lines for this probe
                if include_het and len(het_cell_lines) > 0:
                    exclude_cell_lines.extend(het_cell_lines)

                # Always exclude cell lines with the same genotype as target
                if len(same_genotype_cell_lines) > 0:
                    exclude_cell_lines.extend(same_genotype_cell_lines)

                # Apply exclusion mask (keep all non-excluded cell lines, just like the original function)
                if len(exclude_cell_lines) > 0:
                    # Create a mask for bins NOT belonging to excluded cell lines
                    non_excluded_mask = ~table.obs['cell_line'].isin(exclude_cell_lines)
                    # Zero out counts from excluded cell lines
                    probe_counts_filtered = probe_counts * non_excluded_mask.values
                else:
                    probe_counts_filtered = probe_counts.copy()

                # Add to this cell line's total (this tracks which probes were specific to this cell line)
                umi_counts_by_cellline[target_cell_line] += probe_counts_filtered
                umi_counts_total += probe_counts_filtered

        # Store probe info for this cell line
        all_probes_info.append(df_probes)

    # Combine all probe info across cell lines
    if len(all_probes_info) > 0:
        summary_df = pd.concat(all_probes_info, ignore_index=True)
    else:
        summary_df = pd.DataFrame()

    # Get all spatial coordinates
    x_coords = table.obs['x_coord'].values
    y_coords = table.obs['y_coord'].values

    # Remove NaN coordinates
    valid_mask = ~(np.isnan(x_coords) & np.isnan(y_coords))
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    umi_counts_valid = umi_counts_total[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create 2D matrix for heatmap
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

    if color_by_celline:
        # Get cell_line values for each coordinate
        cell_line_values = table.obs['cell_line'].values[valid_mask]

        # Create a categorical mapping for cell lines
        cell_line_to_num = {cl: i for i, cl in enumerate(unique_cell_lines)}

        # Fill in the matrix with cell_line indices
        for x, y, cl in zip(x_coords, y_coords, cell_line_values):
            heatmap_matrix[y - y_min, x - x_min] = cell_line_to_num.get(cl, np.nan)
    else:
        # Fill in the matrix with UMI counts
        for x, y, count in zip(x_coords, y_coords, umi_counts_valid):
            heatmap_matrix[y - y_min, x - x_min] = count

    # Compute marginal line plots per cell line
    x_marginals_by_cellline = {cl: np.zeros(x_max - x_min + 1) for cl in unique_cell_lines}
    y_marginals_by_cellline = {cl: np.zeros(y_max - y_min + 1) for cl in unique_cell_lines}

    for cl in unique_cell_lines:
        cl_counts_valid = umi_counts_by_cellline[cl][valid_mask]
        for x, y, count in zip(x_coords, y_coords, cl_counts_valid):
            x_marginals_by_cellline[cl][x - x_min] += count
            y_marginals_by_cellline[cl][y - y_min] += count

    # Create figure with GridSpec for main plot + marginals + colorbar
    fig = plt.figure(figsize=figsize)
    if color_by_celline:
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.05,
                      width_ratios=[1, 1, 1, 1])
    else:
        gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.05,
                      width_ratios=[1, 1, 1, 1, 0.15])

    # Create axes
    if color_by_celline:
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    else:
        ax_main = fig.add_subplot(gs[1:, :-2])
        ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)
        ax_cbar = fig.add_subplot(gs[1:, -1])

    # Plot main heatmap
    if color_by_celline:
        # Create discrete colormap using the same colors as the line plots
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([cell_line_colors[cl] for cl in unique_cell_lines])
        cmap_copy = cmap.copy()
        cmap_copy.set_bad(color='lightgray')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='auto',
            origin='upper',
            cmap=cmap_copy,
            interpolation='nearest',
            extent=[x_min, x_max + 1, y_max + 1, y_min],
            vmin=0,
            vmax=n_cell_lines - 1
        )

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cell_line_colors[cl], label=cl)
                          for cl in unique_cell_lines]
        ax_main.legend(handles=legend_elements, loc='upper right',
                      title='Cell Line', framealpha=0.9)
    else:
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='auto',
            origin='upper',
            cmap=cmap,
            interpolation='nearest',
            extent=[x_min, x_max + 1, y_max + 1, y_min]
        )

        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('UMI Count', rotation=270, labelpad=20)

    ax_main.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (μm)', fontsize=12)
    ax_main.set_title(f'Cell Line-Specific Probes', fontsize=14, fontweight='bold')

    # Plot top marginal (x-axis aggregate) as line plots
    x_positions = np.arange(x_min, x_max + 1)
    for cl in unique_cell_lines:
        ax_top.plot(x_positions, x_marginals_by_cellline[cl],
                   color=cell_line_colors[cl], label=cl, linewidth=2, alpha=0.8)

    ax_top.set_ylabel('Total UMIs', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Apply log scaling to top marginal if requested
    if log_scale_marginals:
        ax_top.set_yscale('log')

    # Plot right marginal (y-axis aggregate) as line plots
    y_positions = np.arange(y_min, y_max + 1)
    for cl in unique_cell_lines:
        ax_right.plot(y_marginals_by_cellline[cl], y_positions,
                     color=cell_line_colors[cl], label=cl, linewidth=2, alpha=0.8)

    ax_right.set_xlabel('Total UMIs', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    # Apply log scaling to right marginal if requested
    if log_scale_marginals:
        ax_right.set_xscale('log')

    # Return figure, axes, and summary data
    axes = {
        'main': ax_main,
        'top': ax_top,
        'right': ax_right
    }

    return fig, axes, summary_df


def cache_to_zarr(sdata: sd.SpatialData, zarr_path: str, reload: bool = True):
    """
    Cache SpatialData to ZARR format for faster subsequent loading.

    Best Practice:
    - ZARR format enables lazy loading and faster I/O
    - Essential for large VisiumHD datasets (>1GB)
    - Allows memory-efficient operations

    Parameters:
    -----------
    sdata : SpatialData
        SpatialData object to cache
    zarr_path : str
        Path where ZARR store will be saved
    reload : bool
        If True, reload from ZARR and return the reloaded object

    Returns:
    --------
    sdata : SpatialData or None
        Reloaded SpatialData if reload=True, else None
    """
    print(f"\nCaching to ZARR: {zarr_path}")
    sdata.write(zarr_path, overwrite=True)

    if reload:
        # Clear memory before reloading
        del sdata
        gc.collect()

        print(f"Reloading from ZARR...")
        sdata = sd.read_zarr(zarr_path)
        print(f"✓ Reloaded from ZARR")
        return sdata

    return None


def qc_spatialdata(sdata: sd.SpatialData, gf_adata: ad.AnnData, resolutions: list = [2, 8, 16], mt_pattern=r"^MT-"):
    """
    Perform QC on SpatialData object and print summary statistics.

    Parameters:
    -----------
    sdata : SpatialData
        SpatialData object to QC
    resolutions : list
        List of resolutions (in microns) to check for gapfill tables (default: [2, 8, 16])
    mt_pattern : str
        Regex pattern to identify mitochondrial genes (default: r"^MT-")

    Returns:
    --------
    None
    """
    sdata = gw.sp.join_with_wta(sdata, gf_adata)
    for res in resolutions:
        table_name = f'square_{res:03d}um'
        gf_table_name = f'gf_square_{res:03d}um'
        wta_table = sdata.tables[table_name].copy()
        gf_table = sdata.tables[gf_table_name].copy()

        # WTA metrics
        # First, create boolean column marking mitochondrial genes
        wta_table.var['mt'] = wta_table.var_names.str.match(mt_pattern)

        # Then calculate QC metrics (this will create pct_counts_mt automatically)
        sc.pp.calculate_qc_metrics(
            wta_table,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        sc.pp.filter_cells(wta_table, min_counts=100)
        sc.pp.filter_genes(wta_table, min_cells=10)

        # Get good cells
        good_cells = wta_table.obs[
            (wta_table.obs['pct_counts_mt'] < 30)
        ].index.tolist()

        # Gapfill filtering
        gf_table = gw.pp.filter_gapfills(gf_table, min_cells=10)
        gf_table = gw.tl.call_genotypes(gf_table)

        wta_table = wta_table[good_cells, :]
        gf_table = gf_table[good_cells, :]

        # Replace tables in sdata
        sdata.tables[table_name] = wta_table
        sdata.tables[gf_table_name] = gf_table

    return sdata


def adjust_spatialdata(sdata: sd.SpatialData, table: ad.AnnData, resolution: int, sample_name: str):
    # Subset the AnnData object, not just the obs DataFrame
    subdata = table[table.obs['sample'] == sample_name, :].copy()
    # Re-integrate into spatialdata
    # Need to strip the sample name from obs names
    subdata.obs_names = [name.split('-')[0] + '-' + name.split('-')[1] for name in subdata.obs_names]

    # Get the corresponding square table to copy region and instance_id
    square_table = sdata.tables[f'square_{resolution:03d}um']
    subdata.uns['spatialdata_attrs'] = square_table.uns['spatialdata_attrs']
    sdata.tables[f'filtered_square_{resolution:03d}um'] = subdata

    return sdata


def combine_spatial_tables(sdata1: sd.SpatialData, sdata2: sd.SpatialData,
                           name1: str, name2: str,
                           resolution: int) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Combine spatial and gapfill tables at specified resolution into single AnnData objects.

    Parameters:
    -----------
    sdata : SpatialData
        SpatialData object containing spatial and gapfill tables
    resolution : int
        Resolution in microns (e.g., 2, 8, 16)

    Returns:
    --------
    combined_wta : AnnData
        Combined WTA AnnData with spatial coordinates
    combined_gf : AnnData
        Combined gapfill AnnData with spatial coordinates
    """
    table_name = f'square_{resolution:03d}um'
    gf_table_name = f'gf_square_{resolution:03d}um'

    wta1 = sdata1.tables[table_name].copy()
    wta1.obs['sample'] = name1
    gf1 = sdata1.tables[gf_table_name].copy()
    gf1.obs['sample'] = name1
    wta2 = sdata2.tables[table_name].copy()
    wta2.obs['sample'] = name2
    gf2 = sdata2.tables[gf_table_name].copy()
    gf2.obs['sample'] = name2

    combined_wta = ad.concat([wta1, wta2], axis=0, join='outer', label='sample', keys=[name1, name2], index_unique='-')
    combined_gf = ad.concat([gf1, gf2], axis=0, join='outer', label='sample', keys=[name1, name2], index_unique='-')

    combined_wta = gw.tl.transfer_genotypes(combined_wta, combined_gf)

    return combined_wta, combined_gf


def plot_gapfill_saturation_cdf(sdata, cell_line: str, resolution: int = 2, figsize: tuple = (10, 6)):
    """
    Plot cumulative distribution function (CDF) of gapfill UMI counts per bin for a specific cell line.

    This function helps identify the saturation point by showing what percentage of bins
    have captured at least a given number of gapfill UMIs (excluding 0bp probes).

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill data
    cell_line : str
        Name of the cell line to analyze (e.g., 'HEL', 'K562', 'SET2')
    resolution : int
        Resolution in microns (default: 2)
    figsize : tuple
        Figure size for the plot (default: (10, 6))

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(sdata, ad.AnnData):
            wta = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Filter to the specified cell line
    cell_line_table = table[table.obs['cell_line'] == cell_line, :].copy()

    if cell_line_table.shape[0] == 0:
        raise ValueError(f"No bins found for cell line: {cell_line}")

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(cell_line_table)
    non_zero_mask = ~cell_line_table.var.probe.isin(zero_bp_probes)

    # Calculate UMI counts per bin (excluding 0bp probes)
    if non_zero_mask.any():
        umi_counts_per_bin = cell_line_table[:, non_zero_mask].X.sum(axis=1)
        if hasattr(umi_counts_per_bin, 'A1'):
            umi_counts_per_bin = umi_counts_per_bin.A1
        umi_counts_per_bin = umi_counts_per_bin.flatten()
    else:
        raise ValueError("No non-0bp probes found in data")

    # Sort UMI counts
    sorted_counts = np.sort(umi_counts_per_bin)

    # Calculate CDF: cumulative percent of bins for each UMI count
    n_bins = len(umi_counts_per_bin)
    unique_counts = np.unique(sorted_counts)
    cdf_values = []

    for threshold in unique_counts:
        cumulative_percent = (umi_counts_per_bin <= threshold).sum() / n_bins * 100
        cdf_values.append(cumulative_percent)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(unique_counts, cdf_values, linewidth=2, color='steelblue')
    ax.set_xlabel('Gapfill UMI Count', fontsize=12)
    ax.set_ylabel('Cumulative Percent of Total Bins (%)', fontsize=12)
    ax.set_title(f'Gapfill UMI Saturation: {cell_line}\n({cell_line_table.shape[0]} bins, {non_zero_mask.sum()} non-0bp features)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    return fig, ax


def identify_distinct_allele_probes(
    sdata,
    cell_lines: list,
    celltype_genotypes: dict,
    wt_alleles: dict,
    alt_alleles: dict,
    annotated_genotypes,
    resolution: int = 2,
    top_n: int = None,
    min_umis: int = 0
):
    """
    Identify probes with distinct alleles (WT/ALT/HET) across the specified cell lines,
    ranked by total UMI count.

    This function finds probes where each of the 3 cell lines has a different genotype
    (e.g., one is WT, one is ALT, and one is HET), which are useful for distinguishing
    between cell lines based on genetic variants.

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    cell_lines : list
        List of cell line names to compare (e.g., ['HEL', 'K562', 'SET2'])
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    resolution : int
        Resolution in microns (default: 2)
    top_n : int, optional
        If specified, return only the top N probes by UMI count (default: None, returns all)
    min_umis : int
        Minimum total UMI count threshold for a probe to be included (default: 0)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - probe: Full probe name
        - probe_norm: Normalized probe name
        - total_umis: Total UMI count across all cell lines
        - {cell_line}_genotype: Genotype for each cell line (WT/ALT/HET)
        - genotype_pattern: String showing pattern like "WT|ALT|HET"
    """
    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # Validate cell_lines parameter
    if len(cell_lines) < 2:
        raise ValueError("At least 2 cell lines must be provided")

    # Prepare data for identifying distinct-allele probes
    probe_results = []

    for probe in non_zero_probes:
        probe_norm = probe.split("|")
        if len(probe_norm) > 1:
            probe_norm = " ".join(probe_norm[1:3])
        else:
            probe_norm = probe

        if probe_norm not in annotated_genotypes:
            continue

        # Check that all cell lines have genotype data for this probe
        has_all_genotypes = all(
            cell_line in celltype_genotypes and probe_norm in celltype_genotypes[cell_line]
            for cell_line in cell_lines
        )

        if not has_all_genotypes:
            continue

        # Determine genotype for each cell line
        genotypes = {}
        for cell_line in cell_lines:
            alleles = set(celltype_genotypes[cell_line][probe_norm])

            if len(alleles) > 1:
                genotypes[cell_line] = "HET"
            elif wt_alleles[probe_norm] in alleles:
                genotypes[cell_line] = "WT"
            elif alt_alleles[probe_norm] in alleles:
                genotypes[cell_line] = "ALT"
            else:
                genotypes[cell_line] = "Unknown"

        # Skip if any cell line has Unknown genotype
        if any(gt == "Unknown" for gt in genotypes.values()):
            continue

        # Check if all genotypes are distinct OR at least 2 are different
        unique_genotypes = set(genotypes.values())
        if len(unique_genotypes) < 2:
            # All cell lines have the same genotype - skip
            continue

        # Calculate total UMI count for this probe
        probe_mask = table.var.probe == probe
        total_umis = table[:, probe_mask].X.sum()

        if total_umis < min_umis:
            continue

        # Store result
        result = {
            'probe': probe,
            'probe_norm': probe_norm,
            'total_umis': int(total_umis)
        }

        # Add genotype for each cell line
        for cell_line in cell_lines:
            result[f'{cell_line}_genotype'] = genotypes[cell_line]

        # Create genotype pattern string
        result['genotype_pattern'] = '|'.join([genotypes[cl] for cl in cell_lines])

        probe_results.append(result)

    # Convert to DataFrame and sort by UMI count
    df = pd.DataFrame(probe_results)

    if len(df) == 0:
        print(f"Warning: No probes found with distinct alleles across cell lines {cell_lines}")
        return df

    df = df.sort_values('total_umis', ascending=False).reset_index(drop=True)

    # Filter to top N if specified
    if top_n is not None:
        df = df.head(top_n)

    return df


def plot_wt_alt_alleles_spatial(
    sdata,
    probe_name: str,
    wt_alleles: dict,
    alt_alleles: dict,
    resolution: int = 2,
    figsize: tuple = (15, 15)
):
    """
    Plot spatial distribution of WT and ALT allele genotypes for a specific probe.

    This function colors each spatial bin by its genotype for the given probe:
    - Blue for WT (wild-type)
    - Red for ALT (alternate)
    - Purple/magenta for HET (heterozygous, interpolated between WT and ALT)

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    probe_name : str
        Name of the probe to visualize. Can be either the full probe name from the data
        (e.g., "SET2_homozygous|SPN|c.879C>T") or a substring that matches
        (e.g., "SPN c.879C>T" or just "SPN")
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    figsize : tuple
        Figure size for the plot (default: (15, 15))

    Returns:
    --------
    fig, ax, genotype_data
        - fig: matplotlib figure object
        - ax: matplotlib axes object
        - genotype_data: DataFrame with spatial coordinates and genotype calls
    """
    import matplotlib.colors as mcolors

    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Find the full probe name in the data
    matching_probes = [p for p in table.var.probe.unique() if probe_name in p]

    if len(matching_probes) == 0:
        raise ValueError(f"Probe '{probe_name}' not found in data")
    elif len(matching_probes) > 1:
        print(f"Warning: Multiple probes match '{probe_name}'. Using first match: {matching_probes[0]}")

    probe = matching_probes[0]

    # Normalize the probe name to match the format in wt_alleles/alt_alleles dictionaries
    probe_norm = probe.split("|")
    if len(probe_norm) > 1:
        probe_norm = " ".join(probe_norm[1:3])
    else:
        probe_norm = probe

    # Get available gapfills for this probe to determine if it's a dual probe
    probe_mask = table.var.probe == probe
    probe_table = table[:, probe_mask]
    available_gapfills = probe_table.var.gapfill.unique().tolist()
    is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)  # Single nucleotide gapfills

    # Get WT and ALT alleles for this probe
    if is_dual_probe:
        # For dual probes, extract from probe name (e.g., "AKAP9 c.1389G>T" -> WT='G', ALT='T')
        if ">" in probe_norm:
            variant_part = probe_norm.split()[-1]  # Get "c.1389G>T"
            if ">" in variant_part:
                bases = variant_part.split(">")
                wt_allele = bases[0][-1]  # Last character before '>'
                alt_allele = bases[1]     # Everything after '>'
            else:
                raise ValueError(f"Cannot parse variant notation from probe name: '{probe_norm}'")
        else:
            raise ValueError(f"Dual probe '{probe_norm}' does not contain '>' notation for WT/ALT extraction")
    else:
        # For gapfill probes, use the provided dictionaries
        if probe_norm not in wt_alleles or probe_norm not in alt_alleles:
            raise ValueError(f"WT/ALT alleles not defined for gapfill probe '{probe_norm}' (original: '{probe}')")
        wt_allele = wt_alleles[probe_norm]
        alt_allele = alt_alleles[probe_norm]

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    table.obs['x_coord'] = [c[1] for c in coords]
    table.obs['y_coord'] = [c[0] for c in coords]

    # Get UMI counts for WT and ALT alleles per bin
    probe_mask = table.var.probe == probe
    wt_mask = table.var.gapfill == wt_allele
    alt_mask = table.var.gapfill == alt_allele

    wt_combined_mask = probe_mask & wt_mask
    alt_combined_mask = probe_mask & alt_mask

    if wt_combined_mask.any():
        wt_counts = table[:, wt_combined_mask].X.sum(axis=1)
        if hasattr(wt_counts, 'A1'):
            wt_counts = wt_counts.A1
        wt_counts = wt_counts.flatten()
    else:
        wt_counts = np.zeros(table.shape[0])

    if alt_combined_mask.any():
        alt_counts = table[:, alt_combined_mask].X.sum(axis=1)
        if hasattr(alt_counts, 'A1'):
            alt_counts = alt_counts.A1
        alt_counts = alt_counts.flatten()
    else:
        alt_counts = np.zeros(table.shape[0])

    # Determine genotype for each bin
    # WT = 0, HET = 0.5, ALT = 1, N/A (no coverage) = NaN
    genotype_values = np.full(table.shape[0], np.nan)

    for i in range(table.shape[0]):
        wt_count = wt_counts[i]
        alt_count = alt_counts[i]
        total_count = wt_count + alt_count

        if total_count > 0:
            # Calculate ALT allele fraction
            alt_fraction = alt_count / total_count
            genotype_values[i] = alt_fraction

    # Get spatial coordinates
    x_coords = table.obs['x_coord'].values
    y_coords = table.obs['y_coord'].values

    # Remove NaN coordinates
    valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    genotype_values_valid = genotype_values[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create 2D matrix for heatmap
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

    for x, y, genotype in zip(x_coords, y_coords, genotype_values_valid):
        heatmap_matrix[y - y_min, x - x_min] = genotype

    # Create custom colormap: Blue (WT=0) -> Purple (HET=0.5) -> Red (ALT=1)
    colors = ['blue', 'purple', 'red']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('wt_alt', colors, N=n_bins)
    cmap.set_bad(color='lightgray')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        heatmap_matrix,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        interpolation='nearest',
        extent=[x_min, x_max + 1, y_max + 1, y_min],
        vmin=0,
        vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Genotype', rotation=270, labelpad=20, fontsize=12)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'WT ({wt_allele})', 'HET', f'ALT ({alt_allele})'])

    ax.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax.set_ylabel('Y Coordinate (μm)', fontsize=12)

    # Count genotypes for title
    n_wt = np.sum(genotype_values_valid == 0)
    n_het = np.sum((genotype_values_valid > 0) & (genotype_values_valid < 1))
    n_alt = np.sum(genotype_values_valid == 1)
    n_bins_with_data = np.sum(~np.isnan(genotype_values_valid))

    ax.set_title(
        f'{probe_name}\nWT: {n_wt} bins | HET: {n_het} bins | ALT: {n_alt} bins | Total: {n_bins_with_data} bins',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # Create summary dataframe
    genotype_data = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'genotype_value': genotype_values_valid,
        'wt_count': wt_counts[valid_mask],
        'alt_count': alt_counts[valid_mask]
    })

    return fig, ax, genotype_data