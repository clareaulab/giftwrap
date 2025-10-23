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
                .pl.render_shapes(element=f'_square_{resolution:03d}um', color='library_size', method='matplotlib', v='p98')
                .pl.show(coordinate_systems="", figsize=(25, 25), na_in_legend=False, title="Library Size")
            )

def plot_sites_genotyped(sdata, resolution: int = 2):
    number_genotyped = (~(sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'].isna() | (sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'].isna() != "N/A"))).sum(1).values
    total_genotypes = sdata.tables[f'gf_square_{resolution:03d}um'].obsm['genotype'].shape[1]
    sdata[f'square_{resolution:03d}um'].obs['percent_genotyped'] = number_genotyped / total_genotypes * 100
    return (sdata.pl.render_images(f"_hires_image", alpha=0.8)
                .pl.render_shapes(element=f'_square_{resolution:03d}um', color='percent_genotyped', method='matplotlib')
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
            raise ValueError("cell_line annotation not found in either gf or wta data.")
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
