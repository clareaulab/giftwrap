"""
This module provides various tools for analyzing and manipulating a GIFT-seq dataset. Most true analysis tools live
here.
"""
import functools
import os
import itertools
from collections import defaultdict

import pandas as pd

import anndata as ad
import numpy as np
import scipy

from giftwrap.utils import maybe_multiprocess
from tqdm.auto import tqdm

# Numba-compiled metric for genotype connectivity (must be at module level)
try:
    import numba

    @numba.njit()
    def _nan_aware_metric_encoded(x, y):
        """Numba-compiled metric that decodes NaN mask from concatenated representation"""
        # The array is [genotype_values, validity_mask] concatenated
        # Split point is at half the array length
        total_len = len(x)
        split = total_len // 2

        # Split into genotype values and mask
        x_values = x[:split]
        x_mask = x[split:] > 0.5  # Convert float to bool
        y_values = y[:split]
        y_mask = y[split:] > 0.5

        # Compute NaN-aware Euclidean distance
        sum_sq = 0.0
        count = 0
        for i in range(split):
            if x_mask[i] and y_mask[i]:
                diff = x_values[i] - y_values[i]
                sum_sq += diff * diff
                count += 1

        if count == 0:
            return np.nan
        return np.sqrt(sum_sq)

except ImportError:
    # Numba not available, will use fallback version
    _nan_aware_metric_encoded = None


def collapse_gapfills(adata: ad.AnnData) -> ad.AnnData:
    """
    Collapse various gapfills into a single feature per probe. This yields an AnnData object much more similar to a
    typical scRNA-seq dataset.
    :param adata: The AnnData object containing the gapfills.
    :return: A stripped-down copy of the AnnData object with the gapfills collapsed.
    """
    # Collapse the gapfills that have the same probe value
    new_obs = adata.obs.copy()
    new_var = adata.var.groupby("probe").first().reset_index().drop(columns=["gapfill"]).set_index("probe")

    # Create probe to column index mapping for fast lookup
    probe_to_cols = adata.var.groupby("probe").indices

    # Vectorized X aggregation
    n_cells = adata.shape[0]
    n_probes = len(new_var)

    if scipy.sparse.issparse(adata.X):
        # For sparse matrices, convert to CSC for efficient column operations
        X_csc = adata.X.tocsc()
        new_X = np.zeros((n_cells, n_probes))
        for i, probe in enumerate(new_var.index.values):
            cols = probe_to_cols[probe]
            new_X[:, i] = X_csc[:, cols].sum(axis=1).A1
    else:
        new_X = np.zeros((n_cells, n_probes))
        for i, probe in enumerate(new_var.index.values):
            cols = probe_to_cols[probe]
            new_X[:, i] = adata.X[:, cols].sum(axis=1)

    # Do the same for layers using vectorized operations
    new_layers = dict()
    for layer_name, layer_data in adata.layers.items():
        new_layer = np.zeros((n_cells, n_probes))
        if scipy.sparse.issparse(layer_data):
            layer_csc = layer_data.tocsc()
            for i, probe in enumerate(new_var.index.values):
                cols = probe_to_cols[probe]
                if layer_name == 'percent_supporting':
                    new_layer[:, i] = layer_csc[:, cols].mean(axis=1).A1
                else:
                    new_layer[:, i] = layer_csc[:, cols].sum(axis=1).A1
        else:
            for i, probe in enumerate(new_var.index.values):
                cols = probe_to_cols[probe]
                if layer_name == 'percent_supporting':
                    new_layer[:, i] = layer_data[:, cols].mean(axis=1)
                else:
                    new_layer[:, i] = layer_data[:, cols].sum(axis=1)
        new_layers[layer_name] = new_layer

    return ad.AnnData(X=new_X, obs=new_obs, var=new_var, layers=new_layers)


def intersect_wta(wta_adata: ad.AnnData, gapfill_adata: ad.AnnData) -> (ad.AnnData, ad.AnnData):
    """
    Intersect two AnnData objects, keeping only the cells that are in both datasets.
    :param wta_adata: The AnnData object containing the WTA data.
    :param gapfill_adata: The AnnData object containing the gapfill data.
    :return: Returns a tuple of the two AnnData objects with the cells that are not in both datasets removed.
    """
    # Use numpy intersect1d for O(n log n) complexity instead of O(n²)
    intersected_cells = np.intersect1d(wta_adata.obs.index.values, gapfill_adata.obs.index.values)
    return wta_adata[intersected_cells, :], gapfill_adata[intersected_cells, :]


def annotate_alleles(adata: ad.AnnData, probe_df: pd.DataFrame = None, annotate_unrecognized_as_alt: bool = True, call_het: bool = True) -> ad.AnnData:
    """
    Adds "genotype_allele" to adata.obsm. This will contain the annotations: "REF", "ALT", "HET" (or "REF/ALT"), or "UNKNOWN".
    The adata object provided must have a "genotype" obsm containing the genotype calls. Use gw.tl.call_genotypes first.

    If a probe_df is provided, it should contain columns "probe", "reference_gapfill", and "expected_gapfill" to use for annotation.

    :param adata: The AnnData object containing the genotypes.
    :param probe_df: A DataFrame containing the probe annotations. If not provided, will attempt to be inferred.
    :param annotate_unrecognized_as_alt: If True, any unrecognized alleles will be annotated as "ALT". If False, they will be "UNKNOWN".
        Regardless, missing genotypes will be annotated as "UNKNOWN".
    :param call_het: If True, heterozygous genotypes (mixed REF/ALT) will be annotated as "HET". If False, they will be "REF/ALT".
    :return: The same AnnData object with the genotype alleles added.
    """
    assert "genotype" in adata.obsm, "AnnData object does not contain genotypes. Please run call_genotypes first."

    # Extract probe metadata
    if probe_df is None:
        # Infer from adata
        if "probe_metadata" in adata.uns:
            probe_df = adata.uns["probe_metadata"].copy()
        else:
            # Fall back to extracting from var
            probe_df = adata.var.groupby("probe").first().reset_index()[["probe", "reference_gapfill", "expected_gapfill"]]

    # Validate probe_df has required columns
    required_cols = ["probe", "reference_gapfill", "expected_gapfill"]
    missing_cols = [col for col in required_cols if col not in probe_df.columns]
    if missing_cols:
        raise ValueError(f"probe_df is missing required columns: {missing_cols}")

    # Create a mapping from probe to reference and expected gapfills
    probe_ref_map = probe_df.set_index("probe")["reference_gapfill"].to_dict()
    probe_exp_map = probe_df.set_index("probe")["expected_gapfill"].to_dict()

    # Get the genotype DataFrame
    genotype_df = adata.obsm["genotype"]

    # Initialize allele annotation DataFrame with same shape as genotype_df
    allele_df = pd.DataFrame(index=genotype_df.index, columns=genotype_df.columns, dtype="string")

    # Iterate over each probe (column in genotype_df)
    for probe in genotype_df.columns:
        if probe not in probe_ref_map or probe not in probe_exp_map:
            # Probe not in mapping, annotate all as UNKNOWN
            allele_df[probe] = "UNKNOWN"
            continue

        ref_gapfill = probe_ref_map[probe]
        exp_gapfill = probe_exp_map[probe]

        # Get genotype calls for this probe
        genotypes = genotype_df[probe]

        # Annotate each cell's genotype for this probe
        annotations = []
        for genotype in genotypes:
            if pd.isna(genotype) or genotype == "" or genotype is np.nan:
                # Missing genotype
                annotations.append("UNKNOWN")
            else:
                # Split multi-allelic genotypes (e.g., "ALT1/ALT2")
                alleles = str(genotype).split("/")
                allele_types = set()

                for allele in alleles:
                    if allele == ref_gapfill:
                        allele_types.add("REF")
                    elif allele == exp_gapfill:
                        allele_types.add("ALT")
                    else:
                        # Unrecognized allele
                        if annotate_unrecognized_as_alt:
                            allele_types.add("ALT")
                        else:
                            allele_types.add("UNKNOWN")

                # Combine allele types
                if "UNKNOWN" in allele_types:
                    annotations.append("UNKNOWN")
                elif len(allele_types) == 0:
                    annotations.append("UNKNOWN")
                elif len(allele_types) == 1:
                    annotations.append(list(allele_types)[0])
                else:
                    # Mixed REF/ALT -> heterozygous
                    if call_het:
                        annotations.append("HET")
                    else:
                        sorted_types = sorted(list(allele_types))
                        annotations.append("/".join(sorted_types))

        allele_df[probe] = annotations

    # Convert all columns to category for memory efficiency
    allele_df = allele_df.astype('category')

    # Add to obsm
    adata.obsm["genotype_allele"] = allele_df
    return adata


def calculate_mutational_burden(adata: ad.AnnData, normalize: bool = True) -> ad.AnnData:
    """
    Calculates the mutational burden for each cell based on allele annotations. The mutational burden is defined as
    the ratio of mutated sites to total valid sites, where:
    - REF = 0 (no mutation)
    - HET or REF/ALT = 0.5 (heterozygous, one mutated allele)
    - ALT = 1.0 (homozygous mutation)
    - UNKNOWN sites are excluded from calculation

    The adata object must have a "genotype_allele" obsm containing allele annotations. Use gw.tl.annotate_alleles first.

    The function adds "mutational_burden" to adata.obs with the calculated burden for each cell.

    :param adata: The AnnData object containing the allele annotations.
    :param normalize: If True, normalize the burden values to range from 0-1 across the dataset. If False, return raw burden scores.
    :return: The same AnnData object with mutational burden added to obs.
    """
    assert "genotype_allele" in adata.obsm, "AnnData object does not contain allele annotations. Please run annotate_alleles first."

    # Get the allele annotation DataFrame
    allele_df = adata.obsm["genotype_allele"]

    # Initialize arrays for burden score and total counts
    n_cells = adata.shape[0]
    burden_scores = np.zeros(n_cells, dtype=float)
    total_counts = np.zeros(n_cells, dtype=float)

    # Iterate over each cell
    for cell_idx in range(n_cells):
        cell_alleles = allele_df.iloc[cell_idx]

        for allele in cell_alleles:
            # Skip UNKNOWN genotypes - these are not valid sites
            if pd.isna(allele) or allele == "UNKNOWN":
                continue

            # Count as valid site
            total_counts[cell_idx] += 1

            # Add mutation score based on allele type
            allele_str = str(allele)
            if allele_str == "REF":
                # Reference allele: no mutation
                burden_scores[cell_idx] += 0.0
            elif allele_str == "HET" or allele_str == "REF/ALT" or allele_str == "ALT/REF":
                # Heterozygous: half mutated
                burden_scores[cell_idx] += 0.5
            elif allele_str == "ALT":
                # Homozygous alternate: fully mutated
                burden_scores[cell_idx] += 1.0
            elif "ALT" in allele_str and "REF" in allele_str:
                # Any other REF/ALT combination
                burden_scores[cell_idx] += 0.5
            elif "ALT" in allele_str:
                # Pure ALT without REF (e.g., "ALT/ALT" or just contains ALT)
                burden_scores[cell_idx] += 1.0

    # Calculate burden as average mutation score across valid sites
    # Avoid division by zero
    burden = np.zeros(n_cells, dtype=float)
    valid_mask = total_counts > 0
    burden[valid_mask] = burden_scores[valid_mask] / total_counts[valid_mask]

    # Normalize if requested
    if normalize:
        # Normalize to 0-1 range across the dataset
        min_burden = burden.min()
        max_burden = burden.max()

        if max_burden > min_burden:
            burden = (burden - min_burden) / (max_burden - min_burden)
        # If all values are the same, burden remains unchanged (all zeros or all same value)

    # Add to obs
    adata.obs["mutational_burden"] = burden

    return adata


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

    # Create DataFrames - optimize categorical conversion by doing it during DataFrame creation
    # This is much faster than converting each column individually
    genotype_df = pd.DataFrame(genotypes, index=adata.obs.index, dtype="string")
    # Convert all columns to category at once using apply - faster than column-by-column
    genotype_df = genotype_df.astype('category')

    adata.obsm["genotype"] = genotype_df
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

        # Vectorize the genotype call computation
        orig_indices = np.where(remaining_mask)[0]

        # Handle single genotype case (idx == 0 or threshold met on first genotype)
        single_geno_mask = (idx == 0) | (cumulative[:, 0] >= threshold)
        if np.any(single_geno_mask):
            single_orig_indices = orig_indices[single_geno_mask]
            calls[single_orig_indices] = sorted_genotypes[single_geno_mask, 0]
            n_umis[single_orig_indices] = sorted_counts[single_geno_mask, 0]
            p_umis[single_orig_indices] = cumulative[single_geno_mask, 0]

        # Handle multi-genotype case (idx > 0 and threshold not met on first)
        multi_geno_mask = ~single_geno_mask
        if np.any(multi_geno_mask):
            for i, orig_i in enumerate(orig_indices[multi_geno_mask]):
                subset_i = np.where(multi_geno_mask)[0][i]
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
        genotype_df = gapfill_adata[cell_ids_wta].obsm["genotype"].copy()
        # Optimize: convert all columns at once instead of one by one
        genotype_df = genotype_df.astype("string").astype('category')

        wta_adata.obsm["genotype"] = genotype_df
        wta_adata.obsm["genotype_proportion"] = gapfill_adata[cell_ids_wta].obsm["genotype_proportion"]
        wta_adata.obsm["genotype_counts"] = gapfill_adata[cell_ids_wta].obsm["genotype_counts"]
    elif intersected_cell_ids.shape[0] < cell_ids_wta.shape[0]:
        # Not all WTA cells have gapfill. Need to pad with NaNs.
        genotype = gapfill_adata[intersected_cell_ids].obsm["genotype"].copy()
        genotype_p = gapfill_adata[intersected_cell_ids].obsm["genotype_proportion"]
        genotype_counts = gapfill_adata[intersected_cell_ids].obsm["genotype_counts"]
        # Append missing ids with NaNs
        missing_ids = np.setdiff1d(cell_ids_wta, intersected_cell_ids)
        intersected_cell_ids = np.concatenate([intersected_cell_ids, missing_ids])
        genotype = pd.concat([genotype, pd.DataFrame(index=missing_ids, columns=genotype.columns)], axis=0)
        genotype_p = pd.concat([genotype_p, pd.DataFrame(index=missing_ids, columns=genotype_p.columns)], axis=0)
        genotype_counts = pd.concat([genotype_counts, pd.DataFrame(index=missing_ids, columns=genotype_counts.columns)], axis=0)

        # Convert genotypes to nullable string categorical dtype
        for col in genotype.columns:
            genotype[col] = genotype[col].astype("string").astype('category')

        # Re-order the WTA
        wta_adata = wta_adata[intersected_cell_ids]
        wta_adata.obsm["genotype"] = genotype
        wta_adata.obsm["genotype_proportion"] = genotype_p
        wta_adata.obsm["genotype_counts"] = genotype_counts
    else:
        raise ValueError("This should never happen.")

    return wta_adata


def genotype_connectivity(adata: ad.AnnData,
                          key_added: str = 'genotype_connectivity',
                          distance_func = scipy.spatial.distance.euclidean,
                          n_neighbors: int = 15
                          ) -> ad.AnnData:
    """
    Compute a connectivity matrix based on genotype similarity using a NaN-aware distance metric.

    :param adata: AnnData containing genotype calls in `adata.obsm['genotype']`.
    :param key_added: Key in `adata.obsp` where the connectivity will be stored.
    :param distance_func: Callable to compute distance between two vectors (default: `scipy.spatial.distance.euclidean`).
    :param n_neighbors: Number of neighbors to use for the graph (default: 15).
    :return: The same AnnData object with the connectivity matrix stored at `adata.obsp[key_added]`.

    Example:
        gw.tl.genotype_connectivity(adata)
        sc.tl.umap(adata, neighbors_key='genotype_connectivity')
        sc.pl.umap(adata)
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for genotype_connectivity. Please install scanpy to use this function.")
    assert "genotype" in adata.obsm, "Gapfill data does not contain genotypes. Please run call_genotypes first."

    # Compute genotype matrix
    all_genotype_vectors, _, _ = _encoded_genotype_matrix(adata)

    # Encode NaN mask into the representation by concatenating data + mask
    # First half: genotype values (NaNs replaced with 0)
    # Second half: binary mask (1 = valid, 0 = NaN)
    n_features = all_genotype_vectors.shape[1]
    nan_mask = ~np.isnan(all_genotype_vectors)  # True where valid
    genotype_filled = np.where(np.isnan(all_genotype_vectors), 0.0, all_genotype_vectors)

    # Concatenate: [genotype_values, validity_mask]
    encoded_with_mask = np.concatenate([genotype_filled, nan_mask.astype(float)], axis=1)

    # Store encoded representation
    adata.obsm['_genotype_encoded'] = encoded_with_mask

    # Choose metric function (prefer numba-compiled if available)
    if _nan_aware_metric_encoded is not None:
        metric_func = _nan_aware_metric_encoded
    else:
        # Fallback to non-compiled version if Numba not available
        def metric_func(x, y):
            """Metric that decodes NaN mask from concatenated representation"""
            total_len = len(x)
            split = total_len // 2

            x_values = x[:split]
            x_mask = x[split:] > 0.5
            y_values = y[:split]
            y_mask = y[split:] > 0.5

            # Compute NaN-aware Euclidean distance
            valid = x_mask & y_mask
            if not np.any(valid):
                return np.nan
            diff = x_values[valid] - y_values[valid]
            return np.sqrt(np.sum(diff * diff))

    # Use scanpy's neighbor computation with custom metric
    sc.pp.neighbors(adata, use_rep='_genotype_encoded',
                   metric=metric_func,
                   n_neighbors=n_neighbors,
                   key_added=key_added)

    # Note: We keep _genotype_encoded in obsm because scanpy's downstream tools
    # (like sc.tl.umap) may need to access it via the neighbors_key parameter

    return adata


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

    # Convert imputed genotypes to nullable string categorical dtype - optimize by doing all at once
    imputed_genotypes = imputed_genotypes.astype("string").astype('category')

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
    # Find non-NaN features
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if not valid_mask.any():
        return np.nan  # Undefined

    # Compute distance only on valid features
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    dist = distance_func(valid_x, valid_y)

    # Normalize and penalize for NaN features
    n_valid = valid_mask.sum()
    n_nan = (~valid_mask).sum()
    normalized_dist = (dist / n_valid) * n_nan if n_valid > 0 else np.nan

    return normalized_dist


def _compute_nan_aware_dist_matrix(all_genotype_vectors, distance_func = scipy.spatial.distance.euclidean) -> np.ndarray:
    """
    Vectorized computation of NaN-aware pairwise distance matrix with chunked processing
    to avoid memory overflow on large datasets.
    """
    n_cells, n_features = all_genotype_vectors.shape

    # For euclidean distance, we can use chunked computation
    if distance_func == scipy.spatial.distance.euclidean:
        # Determine chunk size based on available memory (aim for ~1GB chunks)
        # Each chunk processes chunk_size x n_cells x n_features booleans (1 byte each)
        # We need 2 copies (for X_i and X_j masks) plus computation space
        target_memory_gb = 1.0
        bytes_per_element = 8  # float64
        # Estimate: chunk_size * n_cells * n_features * bytes_per_element * 4 (for intermediate arrays) < target_memory
        chunk_size = max(1, int((target_memory_gb * 1e9) / (n_cells * n_features * bytes_per_element * 4)))
        chunk_size = min(chunk_size, n_cells)  # Don't exceed total cells

        print(f"Computing distance matrix in chunks of {chunk_size} cells to avoid memory overflow...")

        # Pre-allocate the distance matrix
        distance_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)

        # Process in chunks
        for i in tqdm(range(0, n_cells, chunk_size), desc="Computing distances"):
            end_i = min(i + chunk_size, n_cells)
            chunk_i = all_genotype_vectors[i:end_i]  # (chunk_size, n_features)

            # Expand for broadcasting: (chunk_size, 1, n_features)
            X_i = chunk_i[:, np.newaxis, :]

            # Process inner loop in chunks too
            for j in range(0, n_cells, chunk_size):
                end_j = min(j + chunk_size, n_cells)
                chunk_j = all_genotype_vectors[j:end_j]  # (chunk_size, n_features)

                # Expand for broadcasting: (1, chunk_size, n_features)
                X_j = chunk_j[np.newaxis, :, :]

                # Compute valid mask for this block
                valid_mask = ~(np.isnan(X_i) | np.isnan(X_j))  # (chunk_i_size, chunk_j_size, n_features)

                # Replace NaNs with 0
                X_i_clean = np.where(np.isnan(X_i), 0.0, X_i)
                X_j_clean = np.where(np.isnan(X_j), 0.0, X_j)

                # Compute squared differences
                diff_sq = (X_i_clean - X_j_clean) ** 2
                diff_sq_masked = diff_sq * valid_mask

                # Count valid and NaN features per pair
                n_valid = valid_mask.sum(axis=2)  # (chunk_i_size, chunk_j_size)
                n_nan = n_features - n_valid

                # Sum and take square root for euclidean distance
                dist_sum = diff_sq_masked.sum(axis=2)

                # Compute distance with normalization and NaN penalty
                with np.errstate(divide='ignore', invalid='ignore'):
                    distance = np.sqrt(dist_sum)
                    normalized_dist = np.where(n_valid > 0, (distance / n_valid) * n_nan, np.nan)

                # Store in the full matrix
                distance_matrix[i:end_i, j:end_j] = normalized_dist

        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0.0)

        return distance_matrix
    else:
        # Fall back to loop-based approach for custom distance functions
        # Use sklearn's pairwise_distances for parallelization
        from sklearn.metrics import pairwise_distances

        def nan_aware_metric(x, y):
            return _nan_aware_distance(x, y, distance_func)

        return pairwise_distances(all_genotype_vectors, metric=nan_aware_metric, n_jobs=-1)


def _encoded_genotype_matrix(adata: ad.AnnData):
    genotypes_matrix = []
    genotype_allele = []
    genotype_labels = []

    genotype_data = adata.obsm['genotype']
    for geno in genotype_data.columns.values:
        # Use vectorized operations for splitting genotypes
        split_series = genotype_data[geno].str.split("/")

        # Get all unique alleles more efficiently
        all_alleles = set()
        for cell_alleles in split_series.dropna():
            if isinstance(cell_alleles, list):
                all_alleles.update(cell_alleles)
            else:
                all_alleles.add(cell_alleles)

        # Remove nan strings
        all_alleles.discard("nan")
        all_alleles = [a for a in all_alleles if pd.notna(a)]

        if len(all_alleles) == 0:
            genotypes_matrix.append(np.full((0, adata.shape[0]), np.nan))
            genotype_allele.append(np.full((0,), np.nan))
        else:
            # Vectorized indicator computation
            indicators = np.zeros((len(all_alleles), adata.shape[0]))
            for i, allele in enumerate(all_alleles):
                # Vectorized check for allele presence
                has_allele = split_series.apply(
                    lambda x: allele in x if isinstance(x, list) else (
                        np.nan if pd.isna(x) else x == allele
                    )
                ).astype(float)
                indicators[i, :] = has_allele.values

            genotypes_matrix.append(indicators)
            genotype_allele.append(np.array(all_alleles))

        genotype_labels.append(geno)

    # Concatenate all genotype vectors for efficient distance computation
    all_genotype_vectors = np.concatenate([g for g in genotypes_matrix], axis=0)
    all_genotype_vectors = all_genotype_vectors.T  # Shape: (n_cells, n_features)

    return all_genotype_vectors, genotype_labels, genotype_allele


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
            pd.DataFrame(index=adata.obs_names, columns=adata.obsm['genotype'].columns), 0.0

    all_genotype_vectors, genotype_labels, genotype_allele = _encoded_genotype_matrix(to_genotype)
    
    # Compute pairwise distances using vectorized operations
    distance_matrix = _compute_nan_aware_dist_matrix(all_genotype_vectors)

    # Finally, for each cell compute the nearest neighbors to impute missing genotypes
    # Get all indices with any NA (so that we can ignore cells with no missing genotypes
    to_fill_mask = np.full((to_genotype.shape[0], to_genotype.obsm['genotype'].shape[1]), True) if impute_all else to_genotype.obsm['genotype'].isna().values

    # Pre-compute neighbor indices for all cells to avoid repeated sorting
    neighbor_indices_all = np.argsort(distance_matrix, axis=1)
    
    # Vectorize where possible
    cells_to_process = np.where(np.any(to_fill_mask, axis=1))[0]
    
    for cell_idx in cells_to_process:
        genotypes_to_fill = to_fill_mask[cell_idx, :]
        neighbor_indices = neighbor_indices_all[cell_idx]
        
        for genotype_idx in np.where(genotypes_to_fill)[0]:
            genotype_vector = all_genotype_vectors[neighbor_indices, :][:, genotype_idx]
            
            if np.all(np.isnan(genotype_vector)):
                # If all neighbors have NaN genotype, skip this genotype
                continue
            
            # Get valid neighbors (skip self at index 0)
            valid_neighbors = ~np.isnan(genotype_vector).all(0)
            if not valid_neighbors[1:k+1].any():
                continue
            
            nearest = genotype_vector[:, valid_neighbors][:, 1:k+1]
            if nearest.size == 0:
                continue
            
            possible_genotypes = genotype_allele[genotype_idx]
            if len(possible_genotypes) == 0:
                continue
            
            genotype_counts = np.nansum(nearest, axis=1)
            
            # Optimize heterozygous genotype handling
            if any("/" in str(g) for g in possible_genotypes):
                genotype_count_dict = {}
                for g, c in zip(possible_genotypes, genotype_counts):
                    if pd.isna(g):
                        continue
                    alleles = str(g).split("/")
                    for allele in alleles:
                        genotype_count_dict[allele] = genotype_count_dict.get(allele, 0) + c
                
                possible_genotypes = np.array(list(genotype_count_dict.keys()))
                genotype_counts = np.array(list(genotype_count_dict.values()))
            
            total_count = np.sum(genotype_counts)
            if total_count == 0:
                continue
            
            # Use vectorized operations for sorting and cumulative computation
            sorted_indices = np.argsort(genotype_counts)[::-1]
            sorted_genotypes = possible_genotypes[sorted_indices]
            cumulative_proportion = np.cumsum(genotype_counts[sorted_indices]) / total_count
            
            best_idx = np.argmax(cumulative_proportion >= threshold)
            
            if best_idx == 0:
                new_genotype = str(sorted_genotypes[0])
            else:
                new_genotype = "/".join(str(g) for g in sorted_genotypes[:best_idx + 1])
            
            # Direct assignment using iloc for better performance
            genotype_col_name = genotype_labels[genotype_idx]
            to_genotype.obsm['genotype'].iloc[cell_idx, to_genotype.obsm['genotype'].columns.get_loc(genotype_col_name)] = new_genotype
            to_genotype.obsm['genotype_certainty'].iloc[cell_idx, to_genotype.obsm['genotype_certainty'].columns.get_loc(genotype_col_name)] = float(cumulative_proportion[best_idx])

    # Validate the imputed genotypes
    correct_genotypes = 0.
    if mask is not None:
        # Vectorized comparison
        mask_subset = mask[has_genotype]
        new_masked = to_genotype.obsm['genotype'].values[mask_subset]
        old_masked = original_genotypes.values[has_genotype][mask_subset]
        
        correct_genotypes = np.sum(new_masked == old_masked)
        
        if not impute_all:
            to_genotype.obsm['genotype'].loc[mask_subset] = original_genotypes.loc[has_genotype][mask_subset]

    # Recompile the adata by adding the non-imputed cells back
    adata = ad.concat([to_genotype, adata[unable_to_impute]], axis=0)
    # Reorder the adata to match the original order
    adata = adata[orig_index, :]

    # Return the genotype and genotype_certainty dataframes
    return (adata.obsm['genotype'].loc[adata.obs_names],
            adata.obsm['genotype_certainty'].loc[adata.obs_names],
            correct_genotypes)


def correct_off_by_one_gapfills(adata: ad.AnnData,
                                probes: list[str] = None,
                                cores: int = 1) -> ad.AnnData:
    """
    It has been observed that some gapfills may have a single base deletion randomly at the ligation junction between
    the end of the gapfilled sequence and the start of the RHS. This function attempts to compute this correction by
    comparing all collected gapfill sequences for each probe within its cell and collapsing any single nucleotide
    truncated gapfills into the most likely full-length gapfill sequence.

    If there are multiple candidates for the full-length gapfill sequence, we will redistribute the UMIs approximately
    equally according to the relative proportions of those sequences in the cell.

    Note that this works with raw probe sequences, not collapsed genotypes!
    :param adata: The AnnData object containing the gapfills.
    :param probes: A whitelist of probes to correct. If None, all probes will be corrected.
    :param cores: The number of cores to use for parallel processing. If <1, uses all available cores.
    :return: The AnnData object with the corrected gapfills. Note that probe/gapfill combinations that no longer have
        valid cells will be removed from the AnnData object.
    """
    if cores < 1:
        cores = os.cpu_count()
    elif cores == 1:
        print("Info: if correction takes too long, consider setting cores > 1.")

    probes = adata.var["probe"].unique().tolist() if probes is None else probes
    mp = maybe_multiprocess(cores)

    if cores == 1:
        adata_list = [adata]
    else:
        split_indices = np.array_split(np.arange(adata.shape[0]), cores)
        adata_list = [adata[adata.obs_names[indices], :] for indices in split_indices]

    with mp as pool:
        results = pool.starmap(
            _correct_off_by_one_job,
            tqdm([(adata_chunk.copy(), probes) for adata_chunk in adata_list], desc="Correcting off-by-one gapfills", total=len(adata_list), unit="chunk")
        )

    # Combine the results
    combined_adata = ad.concat(results, axis=0, merge='unique', uns_merge='unique', index_unique=None)
    # Remove probes/gapfill pairs that no longer have valid cells
    original_n_vars = combined_adata.var.shape[0]
    combined_adata = combined_adata[:, combined_adata.X.sum(axis=0) > 0].copy()
    new_n_vars = combined_adata.var.shape[0]

    print(f"Removed {original_n_vars - new_n_vars} probes/gapfill pairs that no longer have valid cells after correction.")

    return combined_adata


def _correct_off_by_one_job(adata: ad.AnnData, probes: list[str]) -> ad.AnnData:
    """
    For every cell, compute an alignment of all gapfills for each probe and identify gapfills which may have had a
    single base deletion at the ligation junction.
    """
    import rapidfuzz

    # Create a copy to avoid modifying views
    adata = adata.copy()

    for cell_idx, cell in enumerate(adata.obs_names):
        for probe in probes:
            probe_mask = (adata.var["probe"] == probe).values
            subset_var_names = adata.var_names[probe_mask]

            # Get UMI counts for this cell and probe
            if scipy.sparse.issparse(adata.X):
                cell_counts = adata.X[cell_idx, probe_mask].toarray().flatten()
            else:
                cell_counts = adata.X[cell_idx, probe_mask].flatten()

            # Remove gapfills with no UMIs
            nonzero_mask = cell_counts > 0
            if nonzero_mask.sum() < 2:
                continue  # There aren't multiple gapfills for this probe in this cell, skip it

            active_var_names = subset_var_names[nonzero_mask]
            active_counts = cell_counts[nonzero_mask]

            # Get the length of the gapfills
            gapfill_lengths = np.array([len(adata.var.loc[var_name, 'gapfill']) for var_name in active_var_names])

            # Only retain gapfills that are pairwise off-by-one
            gapfills_to_check = list()
            for length in np.unique(gapfill_lengths):
                # Find the gapfills are one base shorter than the current length
                longer_mask = gapfill_lengths == length
                shorter_mask = gapfill_lengths == length - 1

                if not shorter_mask.any():
                    continue

                longer_vars = active_var_names[longer_mask]
                shorter_vars = active_var_names[shorter_mask]
                gapfills_to_check.append((longer_vars, shorter_vars))

            if len(gapfills_to_check) == 0:
                continue

            # Finally we perform potential corrections
            for longer_gapfills, shorter_gapfills in gapfills_to_check:
                # If there is 1 longer gapfill and multiple shorter gapfills, we can no longer assume that the
                # shorter gapfills are truncated versions of the longer gapfill, so we skip this case
                if len(longer_gapfills) == 1 and len(shorter_gapfills) > 1:
                    continue

                # Get counts and sequences
                longer_indices = [np.where(active_var_names == var)[0][0] for var in longer_gapfills]
                shorter_indices = [np.where(active_var_names == var)[0][0] for var in shorter_gapfills]

                longer_counts = active_counts[longer_indices]
                shorter_counts = active_counts[shorter_indices]

                longer_sequences = [adata.var.loc[var, 'gapfill'] for var in longer_gapfills]
                shorter_sequences = [adata.var.loc[var, 'gapfill'] for var in shorter_gapfills]

                # The longer counts must be greater than the shorter counts if this was truly a rare base deletion error
                if longer_counts.sum() <= shorter_counts.sum():
                    continue

                # Finally, we can compute alignments
                # Map each gapfill to its count - convert numpy types to Python native types
                longer_gapfill_counts = {seq: float(count) for seq, count in zip(longer_sequences, longer_counts)}
                shorter_gapfill_counts = {seq: float(count) for seq, count in zip(shorter_sequences, shorter_counts)}

                # Create mappings from original sequences to variable names
                longer_seq_to_var = {seq: var for var, seq in zip(longer_gapfills, longer_sequences)}
                shorter_seq_to_var = {seq: var for var, seq in zip(shorter_gapfills, shorter_sequences)}

                aligned_longer, aligned_shorter = _compute_alignments(
                    ref_frequencies=longer_gapfill_counts,
                    alt_frequencies=shorter_gapfill_counts,
                    align=True,  # Align the motifs
                    threads=1,  # Use a single thread for alignment
                )

                # Create mappings from aligned sequences back to original sequences
                longer_aligned_to_original = {}
                shorter_aligned_to_original = {}

                for aligned_seq in aligned_longer.keys():
                    # Remove gaps to get original sequence
                    original_seq = aligned_seq.replace('-', '')
                    longer_aligned_to_original[aligned_seq] = original_seq

                for aligned_seq in aligned_shorter.keys():
                    # Remove gaps to get original sequence
                    original_seq = aligned_seq.replace('-', '')
                    shorter_aligned_to_original[aligned_seq] = original_seq

                # We will be conservative and only correct if there were gaps introduced into the longer sequence
                # since that would indicate there is too much uncertainty in the gapfill sequence.
                # the specific correction we are performing requires that the shorter gapfill is truncated by one base
                # exactly
                if any('-' in motif for motif in aligned_longer.keys()):
                    continue

                # Finally, we can try to correct the gapfills
                # If there is exactly one shorter and one longer gapfill, we can directly correct
                if len(aligned_longer) == 1 and len(aligned_shorter) == 1:
                    aligned_longer_gapfill = next(iter(aligned_longer.keys()))
                    aligned_shorter_gapfill = next(iter(aligned_shorter.keys()))

                    # Map back to original sequences and then to variable names
                    original_longer_seq = longer_aligned_to_original[aligned_longer_gapfill]
                    original_shorter_seq = shorter_aligned_to_original[aligned_shorter_gapfill]
                    longer_var_name = longer_seq_to_var[original_longer_seq]
                    shorter_var_name = shorter_seq_to_var[original_shorter_seq]

                    # Get the column indices for modification
                    shorter_col_idx = np.where(adata.var_names == shorter_var_name)[0][0]
                    longer_col_idx = np.where(adata.var_names == longer_var_name)[0][0]

                    # Correct the shorter gapfill to the longer gapfill
                    shorter_count = aligned_shorter[aligned_shorter_gapfill]

                    # Modify the sparse matrix directly
                    if scipy.sparse.issparse(adata.X):
                        adata.X[cell_idx, shorter_col_idx] = 0
                        current_longer_count = adata.X[cell_idx, longer_col_idx]
                        adata.X[cell_idx, longer_col_idx] = current_longer_count + shorter_count
                    else:
                        adata.X[cell_idx, shorter_col_idx] = 0.
                        adata.X[cell_idx, longer_col_idx] += shorter_count

                elif len(aligned_longer) == 1 and len(aligned_shorter) > 1:
                    # Should never happen
                    raise ValueError("There are multiple shorter gapfills for a single longer gapfill, this should not happen.")
                else:
                    # We will map each short gapfill to its longest counterpart
                    # When multiple have the same distance, we will redistribute the counts
                    for aligned_shorter_gapfill, shorter_count in aligned_shorter.items():
                        original_shorter_seq = shorter_aligned_to_original[aligned_shorter_gapfill]
                        shorter_var_name = shorter_seq_to_var[original_shorter_seq]

                        # Compute the edit distance to the longer gapfills
                        min_distance = float('inf')
                        best_longer_vars = []
                        for aligned_longer_gapfill, longer_count in aligned_longer.items():
                            distance = rapidfuzz.distance.Levenshtein.distance(
                                aligned_shorter_gapfill, aligned_longer_gapfill
                            )
                            if distance < min_distance:
                                min_distance = distance
                                original_longer_seq = longer_aligned_to_original[aligned_longer_gapfill]
                                best_longer_vars = [longer_seq_to_var[original_longer_seq]]
                            elif distance == min_distance:
                                original_longer_seq = longer_aligned_to_original[aligned_longer_gapfill]
                                best_longer_vars.append(longer_seq_to_var[original_longer_seq])

                        if len(best_longer_vars) == 0:
                            raise ValueError("We lost our longer gapfills, this should not happen.")
                        elif len(best_longer_vars) == 1:
                            # Exactly one, directly convert
                            longer_var_name = best_longer_vars[0]

                            # Get column indices
                            shorter_col_idx = np.where(adata.var_names == shorter_var_name)[0][0]
                            longer_col_idx = np.where(adata.var_names == longer_var_name)[0][0]

                            # Modify the sparse matrix directly
                            if scipy.sparse.issparse(adata.X):
                                adata.X[cell_idx, shorter_col_idx] = 0
                                current_longer_count = adata.X[cell_idx, longer_col_idx]
                                adata.X[cell_idx, longer_col_idx] = current_longer_count + shorter_count
                            else:
                                adata.X[cell_idx, shorter_col_idx] = 0.
                                adata.X[cell_idx, longer_col_idx] += shorter_count
                        else:
                            # Redistribute relative to the counts of the longer gapfills
                            longer_counts_array = np.array([
                                aligned_longer[aligned_seq] for aligned_seq in aligned_longer.keys()
                                if longer_seq_to_var[longer_aligned_to_original[aligned_seq]] in best_longer_vars
                            ])
                            relative_counts = longer_counts_array / longer_counts_array.sum()
                            new_counts = (relative_counts * shorter_count).round().astype(int)

                            # Get column indices
                            shorter_col_idx = np.where(adata.var_names == shorter_var_name)[0][0]

                            # Set the counts for the shorter gapfill to 0
                            if scipy.sparse.issparse(adata.X):
                                adata.X[cell_idx, shorter_col_idx] = 0
                            else:
                                adata.X[cell_idx, shorter_col_idx] = 0.

                            # Add the new counts to the longer gapfills
                            for longer_var_name, new_count in zip(best_longer_vars, new_counts):
                                longer_col_idx = np.where(adata.var_names == longer_var_name)[0][0]
                                if scipy.sparse.issparse(adata.X):
                                    current_count = adata.X[cell_idx, longer_col_idx]
                                    adata.X[cell_idx, longer_col_idx] = current_count + new_count
                                else:
                                    adata.X[cell_idx, longer_col_idx] += new_count

    # Finally, we can return the corrected adata
    return adata


# Utility functions for generating genotype frequencies and aligning gapfill motifs

def _generate_genotype_frequencies(gapfill_adata: ad.AnnData,
                                   probe: str,
                                   genotype_mode: str | None) -> tuple[str, dict[str, float]]:
    """
    Generate genotype frequencies for a given probe in the gapfill adata object.
    """
    final_genotypes_available = 'genotype' in gapfill_adata.obsm
    if not final_genotypes_available:
        print("Warning: No genotypes called in gapfill_adata. Using raw probe frequencies.")

    genotype2count = defaultdict(int)
    if final_genotypes_available:
        if genotype_mode is None or genotype_mode == 'genotype':  # From genotype obsm just count number of cells
            # Count the number of cells for each genotype
            value_counts = gapfill_adata.obsm['genotype'][probe].value_counts()
            for genotype, count in value_counts.items():
                if genotype != 'nan' and (genotype == genotype):
                    if "/" in genotype:  # If the genotype is a composite, split it
                        splitted = genotype.split('/')
                        for sub_genotype in splitted:
                            genotype2count[sub_genotype] += count  # We don't divide by the number of sub-genotypes because they are still positive for a given genotype
                    else:
                        genotype2count[genotype] += count
            frequency_name = 'Cells with Genotype'
        else:  # From genotype obsm, collect number of supporting reads
            # Count the number of UMIs for each genotype
            genotypes = gapfill_adata.obsm['genotype'][probe]
            counts = gapfill_adata.obsm['genotype_counts'][probe]
            for genotype, count in zip(genotypes, counts):
                if genotype != 'nan' and (genotype == genotype):
                    if "/" in genotype:  # If the genotype is a composite, split it
                        splitted = genotype.split('/')
                        for sub_genotype in splitted:
                            genotype2count[sub_genotype] += count / len(splitted)
                    else:
                        genotype2count[genotype] += count
            frequency_name = 'UMIs with Genotype'
    else:
        var_mask = (gapfill_adata.var['probe'] == probe).values
        gapfills = gapfill_adata[:, var_mask].var['gapfill'].values
        if genotype_mode is None or genotype_mode == 'raw':  # From raw probe frequencies, count the umis
            for mask, gapfill in zip(gapfill_adata.obs_names, gapfills):
                if gapfill != 'nan' and (gapfill == gapfill):
                    genotype2count[gapfill] += gapfill_adata[mask, var_mask].X.sum()
            frequency_name = 'UMIs with Gapfill'
        else:  # From raw probes, split captured genotypes by the relative abundance of the gapfills per cell.
            # Normalize counts
            normalization_constant = gapfill_adata[:, var_mask].X.sum(axis=1)
            for i, (mask, gapfill) in enumerate(zip(gapfill_adata.obs_names, gapfills)):
                if gapfill != 'nan' and (gapfill == gapfill):
                    # Get the counts for this cell
                    counts = gapfill_adata[mask, var_mask].X.toarray().flatten()
                    # Normalize by the total number of UMIs in the cell
                    normalized_counts = counts / normalization_constant[i]
                    for genotype, count in zip(gapfill_adata.var_names[var_mask], normalized_counts):
                        if genotype != 'nan' and (genotype == genotype):
                            genotype2count[genotype] += count
            frequency_name = 'Relative Gapfill Frequencies'

    return frequency_name, genotype2count


def _compute_alignments(
        ref_frequencies: dict[str, float],
        alt_frequencies: dict[str, float] | None,
        align: bool = True,
        threads: int = 1,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """
    Compute alignments for the motif plot.
    :param ref_frequencies: A dictionary of reference frequencies for the probe.
    :param alt_frequencies: If provided, a dictionary of alternative frequencies for the probe.
    :param align: Whether to align the motifs using pyFAMSA.
    :param threads: The number of threads to use for alignment.
    :return: A tuple containing the aligned reference frequencies and the aligned alternative frequencies (if provided).
    """
    if align:
        try:
            import pyfamsa
            import scoring_matrices
        except ImportError:
            print("pyFAMSA is not installed. Skipping motif alignment.")
            align = False
    if not align:  # No alignment, just return the frequencies as they are padded to the same length
        # Pad the motifs with gaps to the same length
        if alt_frequencies is None:
            max_length = max(len(motif) for motif in ref_frequencies.keys())
            ref_frequencies = {motif.ljust(max_length, '-'): freq for motif, freq in ref_frequencies.items()}
            return ref_frequencies, None
        else:
            max_length = max(max(len(motif) for motif in ref_frequencies.keys()),
                             max(len(motif) for motif in alt_frequencies.keys()))
            ref_frequencies = {motif.ljust(max_length, '-'): freq for motif, freq in ref_frequencies.items()}
            alt_frequencies = {motif.ljust(max_length, '-'): freq for motif, freq in alt_frequencies.items()}
            return ref_frequencies, alt_frequencies


    # Align the motifs using pyFAMSA
    aligner = pyfamsa.Aligner(
        threads=threads,
        guide_tree='nj',  # Use neighbor-joining to build the guide tree
        tree_heuristic=None,  # We don't need a heuristic for the tree since it should be small
        keep_duplicates=True,
        refine=True,  # Refine the alignment
        scoring_matrix=scoring_matrices.ScoringMatrix.from_name('BLOSUM62'),
    )
    # Sort the references by frequency (descending)
    sorted_ref = dict(sorted(ref_frequencies.items(), key=lambda x: x[1], reverse=True))

    if alt_frequencies is None:  # Only need to worry about one group
        if len(sorted_ref) < 2:  # Skip alignment
            print("Not enough motifs to align. Skipping alignment.")
            return _compute_alignments(
                ref_frequencies=sorted_ref,
                alt_frequencies=None,
                align=False,
            )
        # Align the motifs
        sequences = [
            pyfamsa.Sequence(f">{i}${motif}${freq}".encode(), motif.encode())
            for i, (motif, freq) in enumerate(sorted_ref.items())
        ]
        msa = aligner.align(sequences)
        return {
            seq.sequence.decode(): float(seq.id.decode().split("$")[2])
            for seq in msa
        }, None  # No alternative frequencies to return
    else: # Need to align both groups
        sorted_alt = dict(sorted(alt_frequencies.items(), key=lambda x: x[1], reverse=True))
        distinct_motifs = set(sorted_ref.keys()).union(set(sorted_alt.keys()))
        if len(distinct_motifs) < 2:
            print("Not enough motifs to align. Skipping alignment.")
            return _compute_alignments(
                ref_frequencies=sorted_ref,
                alt_frequencies=sorted_alt,
                align=False,
            )
        if len(sorted_ref) < 2 or len(sorted_alt) < 2:  # We need to align them together to be able to get matching lengths
            all_sequences = [
                pyfamsa.Sequence(f">ref_{i} {freq}".encode(), seq.encode())
                for i, (seq, freq) in enumerate(sorted_ref.items())
            ] + [
                pyfamsa.Sequence(f">alt_{i} {freq}".encode(), seq.encode())
                for i, (seq, freq) in enumerate(sorted_alt.items())
            ]
            # Align the motifs
            msa = aligner.align(all_sequences)
            # Parse the aligned sequences back into dictionaries
            ref_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[-1])
                for seq in msa if seq.id.decode().startswith('>ref_')
            }
            alt_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[-1])
                for seq in msa if seq.id.decode().startswith('>alt_')
            }

            return ref_frequencies_aligned, alt_frequencies_aligned
        else:  # Align individually, then align the two together
            ref_sequences = [
                pyfamsa.Sequence(f">ref_{i} {freq}".encode(), seq.encode())
                for i, (seq, freq) in enumerate(sorted_ref.items())
            ]
            alt_sequences = [
                pyfamsa.Sequence(f">alt_{i} {freq}".encode(), seq.encode())
                for i, (seq, freq) in enumerate(sorted_alt.items())
            ]

            ref_msa = aligner.align(ref_sequences)
            alt_msa = aligner.align(alt_sequences)

            total_msa = aligner.align_profiles(
                ref_msa,
                alt_msa
            )

            ref_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[-1])
                for seq in total_msa if seq.id.decode().startswith('>ref_')
            }
            alt_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[-1])
                for seq in total_msa if seq.id.decode().startswith('>alt_')
            }
            return ref_frequencies_aligned, alt_frequencies_aligned
