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
            accuracy = len(correct_calls) / len(subset[subset[probe] != 'N/A'])
        else:
            accuracy = len(correct_calls) / len(subset)
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