import anndata as ad
import matplotlib.pyplot as plt
try:
    import scanpy as sc
except ImportError:
    sc = None


def _check_genotypes(adata: ad.AnnData):
    if 'genotype' not in adata.obsm:
        raise ValueError("Genotypes not found in adata. Please run call_genotypes first.")


# Gapfill-adata plots

# def motif_plot()

def dendrogram(gapfill_adata: ad.AnnData, groupby: str, **kwargs):
    """
    Generate a dendrogram of the gapfills. Similar to dendrograms in sc.pl.dendrogram.
        Note, this requires sc.tl.dendrogram to be run first.
    :param gapfill_adata: The gapfill adata object.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.dendrogram.
    :return: The figure/axes.
    """
    return sc.pl.dendrogram(gapfill_adata, groupby, **kwargs)


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
