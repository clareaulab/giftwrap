"""
This module provides plotting functions for visualizing gapfill and genotype data in AnnData objects.
"""

from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from .tools import _compute_alignments, _generate_genotype_frequencies

try:
    import scanpy as sc
except ImportError:
    sc = None


def _check_genotypes(adata: ad.AnnData):
    if 'genotype' not in adata.obsm:
        raise ValueError("Genotypes not found in adata. Please run call_genotypes first.")


# Gapfill-adata plots


def plot_logo(gapfill_adata: ad.AnnData,
              probe: str,
              groupby: str = None,
              group: str = None,
              compare_to: str = None,
              genotype_mode: Literal['genotype', 'raw'] = None,
              align: bool = True,
              reverse_complement_gapfill: bool = False,
              threads: int = 1) -> tuple['logomaker.Logo', plt.Axes]:
    """
    Generate a logo plot for a single probe in the gapfill adata object.

    The final plot depends on the combination of input parameters given.

    - When alignments are disabled, sequences are padded to the same length with gaps naively.
    - When genotype_mode is set to 'genotype', the logo frequencies are plotted by the number of cells for each genotype.
    - When genotype_mode is set to 'raw', the logo frequencies are plotted by the total number of UMIs for each genotype.
    - If a compared group is provided, the logo frequencies are first normalized to the total frequencies per position,
        then the alt frequencies are subtracted from the ref frequencies.

    :param gapfill_adata: The gapfill adata object. If there is an obsm['genotype'] slot, called genotypes will be used.
        If not, raw probe gapfill frequencies will be used. Note that this can generate some unexpected results in this
        case due to rare genotypes captured with varying gap lengths.
    :param probe: The probe name.
    :param groupby: The groupby column to use for subsetting the adata object. If None, the entire adata object is used.
    :param group: A specific group to plot. If provided, this will subset the adata object to only include cells
    :param compare_to: If provided, a second group to compare the logo frequencies against.
    :param genotype_mode: The mode for genotypes. If 'genotype', requires that genotypes were previously called and
        the logo frequency will be plotted by the number of cells for each genotype. When 'raw', the logo frequency
        will be plotted by the total number of UMIs for each genotype. By default, we use 'genotype' if genotypes
        have been called, otherwise 'raw'.
    :param align: Whether to align the logos using pyFAMSA.
    :param reverse_complement_gapfill: If True, the gapfill sequences will be reverse complemented before plotting.
        I.e. we plot the mRNA sequence itself, rather than the cDNA sequence.
    :param threads: The number of threads to use for alignment. Default is 1.
    :return: The logo object and its matplotlib axes.
    """
    # Subset the adata to the specified group
    if groupby is not None and group is not None:
        if groupby not in gapfill_adata.obs:
            raise ValueError(f"Groupby {groupby} not found in gapfill_adata.obs.")
        group_mask = gapfill_adata.obs[groupby] == group
        if compare_to is not None:
            compared_mask = gapfill_adata.obs[groupby] == compare_to
            compared_adata = gapfill_adata[compared_mask, :]
        else:
            compared_adata = None
        gapfill_adata = gapfill_adata[group_mask, :]
    else:
        compared_adata = None

    import logomaker

    ref_group_frequency_name, ref_genotype2count = _generate_genotype_frequencies(
        gapfill_adata,
        probe,
        genotype_mode
    )

    if compared_adata is not None:
        ref_group_frequency_name += " (Normalized)"

        _, alt_genotype2count = _generate_genotype_frequencies(
            compared_adata,
            probe,
            genotype_mode
        )
        ref_genotype2count, alt_genotype2count = _compute_alignments(
            ref_frequencies=ref_genotype2count,
            alt_frequencies=alt_genotype2count,
            align=align,
            threads=threads
        )
        # Prepare the data for logomaker (index = Position, columns = Nucleotides, Values = Frequencies)
        seq_length = max(len(logo) for logo in ref_genotype2count.keys())
        ref_data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        for logo, freq in ref_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in ref_data:
                    ref_data[nucleotide][i] += freq
        ref_data = pd.DataFrame(ref_data).set_index('Relative Position (bp)')
        alt_data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        # Subtract from the alt frequencies
        for logo, freq in alt_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in alt_data:
                    alt_data[nucleotide][i] += freq
        alt_data = pd.DataFrame(alt_data).set_index('Relative Position (bp)')

        # Normalize by total frequencies per position then subtract alt data from ref data
        ref_data = ref_data.div(ref_data.sum(axis=1), axis=0)
        alt_data = alt_data.div(alt_data.sum(axis=1), axis=0)
        data = ref_data - alt_data
    else:
        ref_genotype2count, _ = _compute_alignments(
            ref_frequencies=ref_genotype2count,
            alt_frequencies=None,
            align=align,
            threads=threads
        )
        # Prepare the data for logomaker (index = Position, columns = Nucleotides, Values = Frequencies)
        seq_length = max(len(logo) for logo in ref_genotype2count.keys())
        data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        for logo, freq in ref_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in data:
                    data[nucleotide][i] += freq
        data = pd.DataFrame(data).set_index('Relative Position (bp)')

    if reverse_complement_gapfill:  # Reverse the positions and then reverse complement the columns
        # First, reverse the positions and relabel the positions
        data = data.iloc[::-1]
        data.index = pd.Index(list(range(len(data.index))), name='Relative Position (bp)')
        # Then complement the columns
        data = data.rename(columns={
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        })

    # Now we need to plot the logo with logomaker
    logo = logomaker.Logo(
        data,
        shade_below=0.25,
        fade_below=0.25,
        stack_order='small_on_top',
    )

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(fmt='%d', anchor=0, rotation=45)

    logo.ax.set_xlabel('Relative Position (bp)')
    logo.ax.xaxis.set_ticks_position('none')
    logo.ax.xaxis.set_tick_params(pad=-1)

    logo.ax.set_ylabel(ref_group_frequency_name)

    return logo, logo.ax


def dotplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a dotplot of the gapfills for a single probe. Similar to dotplots in sc.pl.dotplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.dotplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.dotplot(gapfill_adata, var_names, groupby, **kwargs)


def tracksplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a tracksplot of the gapfills for a single probe. Similar to tracksplots in sc.pl.tracksplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.tracksplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.tracksplot(gapfill_adata, var_names, groupby, **kwargs)


def matrixplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a matrixplot of the gapfills for a single probe. Similar to matrixplots in sc.pl.matrixplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.matrixplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.matrixplot(gapfill_adata, var_names, groupby, **kwargs)


def violin(gapfill_adata: ad.AnnData, probe: str, **kwargs):
    """
    Generate a violin plot of the gapfills for a single probe. Similar to violin plots in sc.pl.violin.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param kwargs: Arguments passed to sc.pl.violin.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.violin(gapfill_adata, var_names, **kwargs)

# Genotyped-adata plots

def clustermap(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a clustermap of the genotypes. Similar to clustermaps in sc.pl.clustermap.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.clustermap.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    # Move the genotype to obs
    if 'genotype' in adata.obs:
        print("Warning: Overwriting existing genotype column in adata.obs.")

    adata.obs['genotype'] = adata.obsm['genotype'][genotype]

    return_val = sc.pl.clustermap(adata, **kwargs)

    # Drop the fake column
    adata.obs.drop(columns=['genotype'], inplace=True)

    return return_val


def tsne(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a t-SNE plot colored by the specified genotype.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.tsne.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    if 'genotype' in adata.obs or 'genotype_proportion' in adata.obs:
        print("Warning: Overwriting existing genotype and genotype_proportion columns in adata.obs.")

    # Add fake obs columns so that we may plot the genotype and its probability
    adata.obs['genotype'] = adata.obsm['genotype'][genotype]
    adata.obs['genotype_proportion'] = adata.obsm['genotype_proportion'][genotype]

    return_val = sc.pl.tsne(adata, color=['genotype', 'genotype_proportion'], **kwargs)

    # Drop the fake columns
    adata.obs.drop(columns=['genotype', 'genotype_proportion'], inplace=True)

    return return_val


def umap(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a UMAP plot colored by the specified genotype.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.umap.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    if 'genotype' in adata.obs or 'genotype_proportion' in adata.obs:
        print("Warning: Overwriting existing genotype and genotype_proportion columns in adata.obs.")

    # Add fake obs columns so that we may plot the genotype and its probability
    adata.obs['genotype'] = adata.obsm['genotype'][genotype]
    adata.obs['genotype_proportion'] = adata.obsm['genotype_proportion'][genotype]

    return_val = sc.pl.umap(adata, color=['genotype', 'genotype_proportion'], **kwargs)

    # Drop the fake columns
    adata.obs.drop(columns=['genotype', 'genotype_proportion'], inplace=True)

    return return_val
