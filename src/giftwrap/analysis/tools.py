import os

import pandas as pd

import anndata as ad
import numpy as np
import scipy

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
                   flavor: str = "basic",
                   threshold: float = 0.66,
                   cores: int = 1) -> ad.AnnData:
    """
    Adds a "genotype" obsm to the AnnData object that contains the genotype calls for each cell, a "genotype_counts"
    obsm that contains the number of UMIs supporting the called genotype, and a "genotype_p" obsm
    that contains the cumulative fraction of UMIs for the called genotype.

    The 'basic' flavor of the algorithm simply accumulates variants until a certain umi cumulative
    proportion is reached. This is useful for calling genotypes in a simple and fast manner and is defined as follows:

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
    :param flavor: The flavor of genotyping to use.
    :param threshold: The minimum cumulative fraction of UMIs to call a genotype.
    :param cores: The number of cores to use for parallel processing. If <1, uses all available cores.
    :return: The same, AnnData object with the genotype calls added.
    """
    available_flavors = ("basic",)
    assert flavor in available_flavors, f"Flavor {flavor} not recognized. Available flavors: {available_flavors}."

    if cores < 1:
        cores = os.cpu_count()
    elif cores == 1:
        print("Info: if genotyping takes too long, consider setting cores > 1.")

    probes = adata.var["probe"].unique().tolist()

    mp = maybe_multiprocess(cores)
    genotypes = dict()
    genotypes_p = dict()
    genotypes_counts = dict()
    N_cells = adata.shape[0]
    with mp as pool:
        for probe in (pbar := tqdm(probes, desc="Genotyping ")):
            pbar.set_postfix_str(f"Probe {probe}...")
            probe_genotypes = adata.var["gapfill"][adata.var["probe"] == probe].values
            if scipy.sparse.issparse(adata.X):
                gapfill_counts = adata.X[:, (adata.var["probe"] == probe).values].toarray()
            else:
                gapfill_counts = adata.X[:, adata.var["probe"] == probe]

            # Chunk the gapfill counts into the number of cores
            gapfill_counts = np.array_split(gapfill_counts, cores)
            # Call the genotypes in parallel
            results = pool.starmap(
                _genotype_call_job,
                [(probe_genotypes, counts, threshold) for counts in gapfill_counts]
            )

            # Collect the results
            start_idx = 0
            for genotype, count, p in results:
                if probe not in genotypes:
                    genotypes[probe] = np.full(N_cells, np.nan, dtype=object)
                    genotypes_counts[probe] = np.zeros(N_cells, dtype=int)
                    genotypes_p[probe] = np.zeros(N_cells, dtype=float)

                # Fill in the results
                end_idx = start_idx + count.shape[0]
                genotypes[probe][start_idx:end_idx] = genotype
                genotypes_counts[probe][start_idx:end_idx] = count
                genotypes_p[probe][start_idx:end_idx] = p

                start_idx = end_idx

    adata.uns['genotype_call_args'] = {
        "flavor": flavor,
        "threshold": threshold,
        "cores": cores
    }
    adata.obsm["genotype"] = pd.DataFrame(genotypes, index=adata.obs.index)
    adata.obsm["genotype_counts"] = pd.DataFrame(genotypes_counts, index=adata.obs.index)
    adata.obsm["genotype_proportion"] = pd.DataFrame(genotypes_p, index=adata.obs.index)
    return adata


def _genotype_call_job(genotypes: np.array, counts: np.array, threshold: float) -> (np.array, np.array, np.array):
    """
    Call the genotype for a single cell and probe.
    :param genotypes: The string list of genotypes for the probe (N_genotypes_for_probe)
    :param counts: The counts for the gapfills for the probe (N_cells x N_gapfills_for_probe)
    :param threshold: The cumulative fraction of UMIs to call a genotype.
    :return: Returns a tuple of the genotype call, number of umis supporting the genotype, and the cumulative fraction of UMIs for the called genotype.
    """
    N_cells, N_genotypes = counts.shape
    calls = np.full(N_cells, np.nan, dtype=object)
    n_umis = np.zeros(N_cells, dtype=int)
    p_umis = np.zeros(N_cells, dtype=float)

    library = counts.sum(-1)

    # Case 1: No UMIs, should be NaN, 0, 0.0
    all_zero_mask = (library == 0)

    # Case 2: Only one possible detected genotype option, no need to do compute
    if N_genotypes == 1:
        calls[~all_zero_mask] = genotypes[0]
        n_umis[~all_zero_mask] = counts.sum(-1)[~all_zero_mask]
        p_umis[~all_zero_mask] = 1.0
        return calls, n_umis, p_umis

    # Case 3: All umis in a single gapfill
    single_gapfill_mask = ((counts > 0).sum(-1) == 1) & (~all_zero_mask)
    if np.any(single_gapfill_mask):
        # Find the correct genotypes
        gapfill_indices = np.argmax(counts[single_gapfill_mask], -1)
        calls[single_gapfill_mask] = genotypes[gapfill_indices]
        n_umis[single_gapfill_mask] = counts[single_gapfill_mask].sum(-1)
        p_umis[single_gapfill_mask] = 1.0

    # Case 4: All other cases, requiring more expensive computation
    remaining_mask = ~all_zero_mask & ~single_gapfill_mask
    if np.any(remaining_mask):
        # Get the counts and genotypes for the remaining cells
        remaining_counts = counts[remaining_mask]

        # Compute sorted indices (descending order)
        sorted_indices = np.argsort(remaining_counts, axis=-1)[:, ::-1]
        sorted_counts = np.take_along_axis(remaining_counts, sorted_indices, axis=-1)
        sorted_genotypes = np.take_along_axis(genotypes[np.newaxis, :], sorted_indices, axis=-1)

        # Compute cumulative proportions
        cumulative = np.cumsum(sorted_counts, axis=-1) / sorted_counts.sum(axis=-1, keepdims=True)

        # Find the index where cumulative proportion exceeds the threshold
        idx = np.argmax(cumulative >= threshold, axis=-1)
        # Finally, compute the genotype calls, counts, and proportions
        for subset_i, orig_i in enumerate(np.where(remaining_mask)[0]):
            # Get the index of the first genotype that exceeds the threshold
            if idx[subset_i] == 0:
                calls[orig_i] = sorted_genotypes[subset_i, 0]
                n_umis[orig_i] = sorted_counts[subset_i, 0]
                p_umis[orig_i] = 1.0
            else:
                calls[orig_i] = "/".join(sorted_genotypes[subset_i, :idx[subset_i] + 1])
                n_umis[orig_i] = sorted_counts[subset_i, :idx[subset_i] + 1].sum()
                p_umis[orig_i] = cumulative[subset_i, idx[subset_i]]

    return calls, n_umis, p_umis


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
    intersected_cell_ids = np.intersect1d(cell_ids_wta, cell_ids_gapfill)

    if intersected_cell_ids.shape[0] == cell_ids_wta.shape[0]:
        # All WTA cells are in the gapfill data
        wta_adata.obsm["genotype"] = gapfill_adata[cell_ids_wta].obsm["genotype"]
        wta_adata.obsm["genotype_proportion"] = gapfill_adata[cell_ids_wta].obsm["genotype_proportion"]
        wta_adata.obsm["genotype_counts"] = gapfill_adata[cell_ids_wta].obsm["genotype_counts"]
    elif intersected_cell_ids.shape[0] < cell_ids_wta.shape[0]:
        # Not all WTA cells have gapfill. Need to pad with NaNs.
        genotype = gapfill_adata[intersected_cell_ids].obsm["genotype"]
        genotype_p = gapfill_adata[intersected_cell_ids].obsm["genotype_proportion"]
        genotype_counts = gapfill_adata[intersected_cell_ids].obsm["genotype_counts"]
        # Append missing ids with NaNs
        missing_ids = np.setdiff1d(cell_ids_wta, intersected_cell_ids)
        intersected_cell_ids = np.concatenate([intersected_cell_ids, missing_ids])
        genotype = pd.concat([genotype, pd.DataFrame(index=missing_ids, columns=genotype.columns)], axis=0)
        genotype_p = pd.concat([genotype_p, pd.DataFrame(index=missing_ids, columns=genotype_p.columns)], axis=0)
        genotype_counts = pd.concat([genotype_counts, pd.DataFrame(index=missing_ids, columns=genotype_counts.columns)], axis=0)
        # Re-order the WTA
        wta_adata = wta_adata[intersected_cell_ids]
        wta_adata.obsm["genotype"] = genotype
        wta_adata.obsm["genotype_proportion"] = genotype_p
        wta_adata.obsm["genotype_counts"] = genotype_counts
    else:
        raise ValueError("This should never happen.")

    return wta_adata
