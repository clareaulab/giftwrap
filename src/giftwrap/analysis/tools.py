import os
import itertools

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
        new_X[:, i] = adata[:, adata.var["probe"] == probe].X.sum(axis=1).flatten()
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
                   threshold: float = 0.5,
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
            pbar.set_postfix_str(f"Probe {probe}")
            probe_genotypes = adata.var["gapfill"][adata.var["probe"] == probe].values
            if scipy.sparse.issparse(adata.X):
                gapfill_counts = adata[:, adata.var["probe"] == probe].X.toarray()
            else:
                gapfill_counts = adata[:, adata.var["probe"] == probe].X

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


def impute_genotypes(adata: ad.AnnData,
                     cluster_key: str,
                     k: int = 100,
                     threshold: float = 0.66,
                     impute_all: bool = False,
                     hold_out: float = 0.05,
                     cores: int = 1) -> ad.AnnData:
    """
    Imputes the genotypes with the following procedure:
        - For each cluster, independently:
        - Compute a neighbors graph on the genotyped cells in the cluster.
        - For each cell, select the closest k neighbors. If there are less than k neighbors, select all neighbors.
        - For each probe, compute the distribution of the genotypes in the selected neighbors.
        - Perform a test to determine whether the genotype is heterozygous.
        - Finally, impute the genotype in cells with missing calls.
    :param adata: The anndata object containing the genotypes. This object should have a "genotype" obsm.
    :param cluster_key: The key in adata.obs that contains the cluster labels.
    :param k: The number of neighbors to use for imputation.
    :param threshold: The threshold for determining whether a genotype is heterozygous.
    :param impute_all: Whether to impute all genotypes, or only missing genotypes.
    :param hold_out: The fraction of cells to hold out for testing. This is used to compute the imputation accuracy.
    :param cores: The number of cores to use for parallel processing. If <1, uses all available cores.
    :return: The anndata object with the genotypes imputed.

    The AnnData object will now have a "genotype_imputed" obsm containing the new genotypes and
    a "genotype_imputed_certainty" obsm containing the likelihood of the genotype being correct according to the
    neighborhood graph.
    """
    if "genotype" not in adata.obsm:
        raise ValueError("The AnnData object does not contain genotypes. Please run call_genotypes first.")

    hold_out = max(min(hold_out, 1.0), 0.0)
    if hold_out == 0.0:
        print("Info: Imputation accuracy will not be computed, as no cells are held out for testing.")
        mask = None
    else:
        # Generate a random mask to hold out cells for testing
        mask = np.random.rand(*adata.obsm["genotype"].shape) < hold_out
        # Only hold out not currently NaN genotypes
        mask = mask & (~adata.obsm["genotype"].isna().values)

    if cores < 1:
        cores = os.cpu_count()
    elif cores == 1:
        print("Info: if imputation takes too long, consider setting cores > 1.")

    mp = maybe_multiprocess(cores)

    # Split the anndata according to cluster identity
    clusters = adata.obs[cluster_key].unique()
    with mp as pool:
        results = pool.starmap(
            _impute_within_cluster,
            tqdm([(adata, cluster_key, cluster, k, threshold, impute_all, mask) for cluster in clusters], desc="Imputing cluster genotypes...", total=len(clusters), unit="cluster"),
        )

    # Combine results
    imputed_genotypes = pd.concat([r[0] for r in results])
    imputed_certainty = pd.concat([r[1] for r in results])

    if mask is not None:
        correct_imputation_counts = [r[2] for r in results]
        accuracy = sum(correct_imputation_counts) / mask.sum()
        adata.uns['imputation_accuracy'] = accuracy
        print(f"Imputation accuracy: {accuracy:.2f} ({mask.sum():,} out of {mask.size:,} cell/allele pairs held out)")

    # Add to AnnData object
    adata.obsm['genotype_imputed'] = imputed_genotypes.loc[adata.obs_names]
    adata.obsm['genotype_imputed_certainty'] = imputed_certainty.loc[adata.obs_names]

    return adata


def _nan_aware_distance(x: np.array, y: np.array, distance_func = scipy.spatial.distance.euclidean) -> float:
    """
    Compute a distance between two arrays
    :param x: Feature array 1.
    :param y: Feature array 2.
    :param distance_func: The distance function to use. Defaults to scipy.spatial.eucldean.
    :return: The distance normalized to the number of features that are not present.
        (i.e. the expected number of mismatches in unobserved genotypes).
    """
    nan_features1 = np.isnan(x)
    nan_features2 = np.isnan(y)
    nan_features = nan_features1 | nan_features2
    if np.all(nan_features):
        return np.nan  # Undefined if non-complementary
    else:
        return (distance_func(x[~nan_features], y[~nan_features]) / (~nan_features).sum()) * nan_features.sum()  # Normalize by the number of features that are not NaN then penalize by nan features


def _impute_within_cluster(adata: ad.AnnData,
                           cluster_key: str,
                           cluster: str,
                           k: int = 25,
                           threshold: float = .66,
                           impute_all: bool = False,
                           mask: np.ndarray = None,
                           ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Helper function to impute genotypes within a single cluster.
    :param adata: The AnnData object containing cells from a single cluster.
    :param cluster_key: The key in adata.obs that contains the cluster labels.
    :param cluster: The cluster to impute genotypes for.
    :param k: The number of neighbors to use for imputation.
    :param threshold: The threshold for determining whether a genotype is heterozygous.
    :param impute_all: Whether to impute all cells, or only missing genotypes.
    :param mask: A matrix identifying cells x genotypes to hold out for testing.
    :return: A tuple of (imputed genotypes DataFrame, imputation certainty DataFrame, number of correct imputed genotypes).
    """
    cluster_filter = adata.obs[cluster_key] == cluster
    adata = adata[cluster_filter, :].copy()
    orig_index = adata.obs_names
    if mask is not None:
        mask = mask[cluster_filter, :]
        if mask.shape[0] == 0:
            mask = None
        else:
            original_genotypes = adata.obsm["genotype"].copy()
            adata.obsm["genotype"] = adata.obsm["genotype"].mask(mask, other=np.nan)

    # Get cells with genotypes
    has_genotype = ~adata.obsm['genotype'].isna().all(axis=1)
    adata.obsm['genotype_certainty'] = pd.DataFrame(index=adata.obsm['genotype'].index, columns=adata.obsm['genotype'].columns, data=np.full((adata.shape[0], adata.obsm['genotype'].shape[1]), 0.))
    unable_to_impute = ~has_genotype  # We need to add these cells back later with empty genotypes
    to_genotype = adata[has_genotype].copy()
    if to_genotype.shape[0] == 0:
        # Can't genotype
        return pd.DataFrame(index=adata.obs_names, columns=adata.obsm['genotype'].columns), \
            pd.DataFrame(index=adata.obs_names, columns=adata.obsm['genotype'].columns)

    # Expand the genotypes to one-hot and concatenate to one big matrix
    genotypes_matrix = []  # Will be n_cells x n_genotypes
    genotype_allele = []  # Will be n_genotypes x n_genotypes
    genotype_labels = []  # Will be n_genotypes
    for geno in to_genotype.obsm['genotype'].columns.values:
        # De-concatenate genotypes (i.e. split CAG/GAG to two genotypes: CAG and GAG)
        raw_genotypes = to_genotype.obsm['genotype'][geno].str.split("/").explode().unique()
        # Convert nan strings back to NaN
        raw_genotypes = np.where(raw_genotypes == "nan", np.nan, raw_genotypes)
        allele_indicators = []
        allele = []
        for raw_genotype in raw_genotypes:
            if raw_genotype == "nan" or pd.isna(raw_genotype):
                continue
            # Check if cells contain any of the raw genotypes
            allele_indicators.append(
                (to_genotype.obsm['genotype'][geno].str.split("/").apply(
                    lambda x: raw_genotype in x if isinstance(x, list) else (np.nan if pd.isna(x) else x == raw_genotype)
                )).astype(float)
            )
            allele.append(np.nan if raw_genotype == "nan" else raw_genotype)  # FIXME: Fix the string coercsion in the .str.split call chains
        if len(allele_indicators) == 0:  # No genotypes found for this probe
            genotypes_matrix.append(np.full((0, to_genotype.shape[0]), np.nan))
            genotype_allele.append(np.full((0, to_genotype.shape[0]), np.nan))
            genotype_labels.append(geno)
        else:
            genotypes_matrix.append(np.stack(allele_indicators))
            genotype_allele.append(np.stack(allele))
            genotype_labels.append(geno)

    # To fill
    distance_matrix = np.full((to_genotype.shape[0], to_genotype.shape[0]), np.nan)
    # Set diagonal to 0
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0
    # Compute all possible combinations
    for i,j in itertools.combinations(range(to_genotype.shape[0]), 2):
        if i == j:
            continue
        # Extract all genotypes from the nested data
        cell_i_vector = np.concatenate([g[:, i] for g in genotypes_matrix], axis=0)
        cell_j_vector = np.concatenate([g[:, j] for g in genotypes_matrix], axis=0)
        distance_matrix[i,j] = _nan_aware_distance(cell_i_vector, cell_j_vector)
        distance_matrix[j,i] = distance_matrix[i,j]

    # Finally, for each cell compute the nearest neighbors to impute missing genotypes
    # Get all indices with any NA (so that we can ignore cells with no missing genotypes
    to_fill_mask = np.full((to_genotype.shape[0], to_genotype.obsm['genotype'].shape[1]), True) if impute_all else to_genotype.obsm['genotype'].isna().values
    # Get indices with at least one genotype to impute
    for idx in np.argwhere(np.any(to_fill_mask, axis=1)):
        idx = idx.item()
        genotypes_to_fill = to_fill_mask[idx, :]
        # Get the nearest neighbors
        neighbor_indices = np.argsort(distance_matrix[idx, :], axis=0)
        for genotype_idx in np.argwhere(genotypes_to_fill):
            genotype_idx = genotype_idx.item()
            genotype_vector = genotypes_matrix[genotype_idx][:, neighbor_indices]  # Get the genotype vector for this genotype
            if np.all(np.isnan(genotype_vector)):
                # If all neighbors have NaN genotype, skip this genotype
                continue
            # Get the nearest neighbors that have a genotype in this position
            nearest = genotype_vector[:, ~np.isnan(genotype_vector).all(0)][:, 1:k+1]  # Skip the first one, which is the cell itself
            if nearest.size == 0:  # No neighbors to impute
                continue
            # Select the genotypes from the neighbors
            possible_genotypes = genotype_allele[genotype_idx]
            if len(possible_genotypes) == 0:  # No genotypes to impute
                continue
            genotype_counts = np.nansum(nearest, axis=1)  # Get the counts of the genotypes in the neighbors
            # correct_genotype = original_genotypes[has_genotype].iloc[idx, genotype_idx]
            # Explode the genotype strings if heterozygous (via /) and duplicate the counts accordingly
            if any("/" in g for g in possible_genotypes):
                new_genotypes = {}
                for g, c in zip(possible_genotypes, genotype_counts):
                    if pd.isna(g):
                        continue
                    # Split the genotype by / and duplicate the counts accordingly
                    alleles = g.split("/")
                    for allele in alleles:
                        if allele not in new_genotypes:
                            new_genotypes[allele] = 0
                        new_genotypes[allele] += c
                possible_genotypes = np.array(new_genotypes.keys())
                genotype_counts = np.array(new_genotypes.values())
            # Collect the cumulative proportion of counts >=50%
            total_count = np.sum(genotype_counts)
            if total_count == 0:
                continue
            sorted_indices = np.argsort(genotype_counts)[::-1]
            cumulative_proportion = np.cumsum(genotype_counts[sorted_indices] / total_count)
            best_idx = np.argmax(cumulative_proportion >= threshold)
            if best_idx == 0:  # Scalar
                new_genotype = possible_genotypes[sorted_indices[0]]
            else:
                new_genotype = "/".join(possible_genotypes[sorted_indices[:best_idx + 1]])
            to_genotype.obsm['genotype'].iloc[idx, to_genotype.obsm['genotype'].columns.get_loc(genotype_labels[genotype_idx])] = str(new_genotype)
            to_genotype.obsm['genotype_certainty'].iloc[idx, to_genotype.obsm['genotype_certainty'].columns.get_loc(genotype_labels[genotype_idx])] = float(cumulative_proportion[best_idx])

    # Validate the imputed genotypes
    correct_genotypes = 0.
    if mask is not None:
        # Check the imputed genotypes against the original genotypes
        new_masked_genotypes = to_genotype.obsm['genotype'].values[mask]
        old_masked_genotypes = original_genotypes.values[mask]

        correct_genotypes += np.sum(
            new_masked_genotypes == old_masked_genotypes
        )

        if not impute_all:  # Replace the imputed genotypes where we know the truth with the original genotypes
            to_genotype.obsm['genotype'].loc[mask] = original_genotypes.loc[mask]

    # Recompile the adata by adding the non-imputed cells back
    adata = ad.concat([to_genotype, adata[unable_to_impute]], axis=0)
    # Reorder the adata to match the original order
    adata = adata[orig_index, :]

    # Return the genotype and genotype_certainty dataframes
    return (adata.obsm['genotype'].loc[adata.obs_names],
            adata.obsm['genotype_certainty'].loc[adata.obs_names],
            correct_genotypes)
