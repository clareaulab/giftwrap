import os

import pandas as pd

import anndata as ad
import numpy as np
from giftwrap.utils import maybe_multiprocess
from tqdm.auto import tqdm


def collapse_gapfills(adata: ad.AnnData) -> ad.AnnData:
    """
    Collapse various gapfills into a single feature per probe. This yields an AnnData object much more similar to a
    typical scRNA-seq dataset.
    :param adata: The AnnData object containing the gapfills.
    :return: A stripped-down copy of the AnnData object with the gapfills collapsed.
    """
    # Collapse the gapfills that have the same probe value
    new_X = np.zeros((adata.shape[0], adata.var["probe"].nunique()))
    new_obs = adata.obs.copy()
    new_var = adata.var.groupby("probe").first().reset_index().drop(columns=["gapfill"]).set_index("probe")
    for i, probe in enumerate(new_var.index.values):
        new_X[:, i] = adata.X[:, adata.var["probe"] == probe].sum(axis=1).flatten()
    # Do the same for layers
    new_layers = dict()
    for layer in adata.layers.keys():
        new_X_layer = np.zeros((adata.shape[0], adata.var["probe"].nunique()))
        for i, probe in enumerate(new_var.index.values):
            if layer == 'percent_supporting':
                new_X_layer[:, i] = adata.layers[layer][:, adata.var["probe"] == probe].mean(axis=1).flatten()
            else:
                new_X_layer[:, i] = adata.layers[layer][:, adata.var["probe"] == probe].sum(axis=1).flatten()
        new_layers[layer] = new_X_layer
    return ad.AnnData(X=new_X, obs=new_obs, var=new_var, layers=new_layers)


def intersect_wta(wta_adata: ad.AnnData, gapfill_adata: ad.AnnData) -> (ad.AnnData, ad.AnnData):
    """
    Intersect two AnnData objects, keeping only the cells that are in both datasets.
    :param wta_adata: The AnnData object containing the WTA data.
    :param gapfill_adata: The AnnData object containing the gapfill data.
    :return: Returns a tuple of the two AnnData objects with the cells that are not in both datasets removed.
    """
    x = [x for x in wta_adata.obs.index.values if x in gapfill_adata.obs.index.values]
    return wta_adata[x, :], gapfill_adata[x, :]


def call_genotypes(adata: ad.AnnData,
                   threshold: float = 0.66,
                   cores: int = 1) -> ad.AnnData:
    """
    Adds a "genotype" obsm to the AnnData object that contains the genotype calls for each cell and a "genotype_p" obsm
    that contains the cumulative fraction of UMIs for the called genotype.
    The algorithm is extremely basic, and computes genotypes as follows:

    For each cell:
        For each probe:
            Collect all gapfills with >0 UMIs
            If there are no gapfills, return NAN
            If there is a single gapfill, return that gapfill.
            Else
                Sort gapfills by UMI count, select the combination of gapfills that lead to UMIs cumulative
                    proportion > threshold.
                Return this combination as the genotype sorted and joined by "/".

    :param adata: The AnnData object containing the gapfills.
    :param threshold: The minimum cumulative fraction of UMIs to call a genotype.
    :param cores: The number of cores to use for parallel processing. If <1, uses all available cores.
    :return: The same, AnnData object with the genotype calls added.
    """
    if cores < 1:
        cores = os.cpu_count()
    elif cores == 1:
        print("Info: if genotyping takes too long, consider setting cores > 1.")

    probes = adata.var["probe"].unique().tolist()

    mp = maybe_multiprocess(cores)
    genotypes = dict()
    genotypes_p = dict()
    with mp as pool:
        for probe in (pbar := tqdm(probes, desc="Genotyping ")):
            pbar.set_postfix_str(f"Probe {probe}...")
            probe_genotypes = adata.var["gapfill"][adata.var["probe"] == probe].values
            gapfill_counts = adata.X[:, adata.var["probe"] == probe].toarray()
            results = pool.starmap(
                _genotype_call_job,
                [(probe_genotypes, counts, threshold) for counts in gapfill_counts]
            )
            genotypes[probe] = [x[0] for x in results]
            genotypes_p[probe] = [x[1] for x in results]

    adata.obsm["genotype"] = pd.DataFrame(genotypes, index=adata.obs.index)
    adata.obsm["genotype_p"] = pd.DataFrame(genotypes_p, index=adata.obs.index)
    return adata


def _genotype_call_job(genotypes: np.array, counts: np.array, threshold: float) -> (str, float):
    """
    Call the genotype for a single cell and probe.
    :param genotypes: The list of genotypes for the probe.
    :param counts: The counts for the gapfills.
    :param threshold: The cumulative fraction of UMIs to call a genotype.
    :return: Returns a tuple of the genotype call and the cumulative fraction of UMIs for the called genotype.
    """
    # First check for short circuit cases
    if counts.sum() == 0:  # No UMIs
        return np.nan, 0.0
    if counts.shape[0] == 1:  # Only one gapfill
        return genotypes[0], 1.0
    if (counts > 0).sum() == 1:  # All UMIs in a single gapfill
        return genotypes[counts.argmax()], 1.0

    # Now we have to do a more expensive computation
    sorted_indices = np.argsort(counts)[::-1]
    counts = counts[sorted_indices]
    genotypes = genotypes[sorted_indices]

    cumulative = np.cumsum(counts) / counts.sum()
    idx = np.argmax(cumulative >= threshold)
    return "/".join(genotypes[:idx + 1]), cumulative[idx]


def transfer_genotypes(wta_adata: ad.AnnData, gapfill_adata: ad.AnnData) -> ad.AnnData:
    """
    Transfer the genotypes from the gapfill data to the WTA data. This is useful for visualizing the genotypes on the
        WTA UMAP. This simply copies the genotype and genotype_p obsm from the gapfill data to the WTA data.
    :param wta_adata: The AnnData object containing the WTA data.
    :param gapfill_adata: The AnnData object containing the gapfill data.
    :return: The WTA data with the genotypes transferred.
    """
    assert "genotype" in gapfill_adata.obsm, "Gapfill data does not contain genotypes. Please run call_genotypes first."

    cell_ids_wta = wta_adata.obs.index.values
    cell_ids_gapfill = gapfill_adata.obs.index.values
    intersected_cell_ids = np.array(set(cell_ids_wta).intersection(set(cell_ids_gapfill)))

    if intersected_cell_ids.shape[0] == cell_ids_wta.shape[0]:
        # All WTA cells are in the gapfill data
        wta_adata.obsm["genotype"] = gapfill_adata[cell_ids_wta].obsm["genotype"]
        wta_adata.obsm["genotype_p"] = gapfill_adata[cell_ids_wta].obsm["genotype_p"]
    elif intersected_cell_ids.shape[0] < cell_ids_wta.shape[0]:
        # Not all WTA cells have gapfill. Need to pad with NaNs.
        genotype = gapfill_adata[intersected_cell_ids].obsm["genotype"]
        genotype_p = gapfill_adata[intersected_cell_ids].obsm["genotype_p"]
        # Append missing ids with NaNs
        missing_ids = np.setdiff1d(cell_ids_wta, intersected_cell_ids)
        intersected_cell_ids = np.concatenate([intersected_cell_ids, missing_ids])
        genotype = pd.concat([genotype, pd.DataFrame(index=missing_ids, columns=genotype.columns)], axis=0)
        genotype_p = pd.concat([genotype_p, pd.DataFrame(index=missing_ids, columns=genotype_p.columns)], axis=0)
        # Re-order the WTA
        wta_adata = wta_adata[intersected_cell_ids]
        wta_adata.obsm["genotype"] = genotype
        wta_adata.obsm["genotype_p"] = genotype_p
    else:
        raise ValueError("This should never happen.")

    return wta_adata
