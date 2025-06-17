from __future__ import annotations

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import giftwrap.analysis.tools as tl
try:
    import spatialdata as sd
    import geopandas as gpd
except ImportError:
    sd = None

try:
    import spatialdata_plot
except ImportError:
    spatialdata_plot = None

try:
    import squidpy as sq
    import scanpy as sc
except ImportError:
    sq = None


def assert_spatial(adata: ad.AnnData):
    """
    Assert that spatialdata is installed.
    """
    if sd is None:
        raise ImportError("spatialdata is not installed. Please install it to use this function.")
    if 'array_row' not in adata.obs or 'array_col' not in adata.obs:
        raise ValueError("This function is currently only applicable to Visium HD data.")


def check_squidpy():
    if sq is None:
        raise ImportError("squidpy is not installed. Please install it to use this function.")


def check_plotting():
    if spatialdata_plot is None:
        raise ImportError("spatialdata_plot is not installed. Please install it to use this function.")


def bin(adata: ad.AnnData, resolution: int = 8) -> ad.AnnData:
    """
    This is ONLY APPLICABLE FOR VISIUM-HD.
    This bins/aggregates data from 2 micron resolution to any other resolution (must be a power of 2).
    Note that this is a simple aggregation intending for dealing with counts in the X matrix (i.e. summing).
    :param adata: The Spatial gapfill data.
    :param resolution: The resolution to aggregate into in microns. Spaceranger typically aggregates to 8um and 16 um.
    :return: The binned data.
    """
    assert resolution % 2 == 0, "Resolution must be a power of 2."
    assert_spatial(adata)
    if resolution == 2:
        return adata  # No need to bin
    effective_resolution = resolution // 2  # Original resolution is 2um

    max_row = adata.obs['array_row'].max() + 1
    max_col = adata.obs['array_col'].max() + 1
    new_nrow = max_row // effective_resolution  # Max Y
    new_ncol = max_col // effective_resolution  # Max X

    # Integer-division to find which bin each spot belongs to
    row_bin = adata.obs['array_row'].values // effective_resolution
    col_bin = adata.obs['array_col'].values // effective_resolution

    # Flatten to a single bin index (so we can group easily)
    # bin_idx will be in range [0, new_nrow * new_ncol)
    bin_idx = row_bin * new_ncol + col_bin

    # Get unique bin IDs and an array telling us which bin each row belongs to
    unique_bins, inverse_idx = np.unique(bin_idx, return_inverse=True)

    X_summed = np.zeros((len(unique_bins), adata.shape[1]))

    # Accumulate sums for each group: np.add.at does this in-place
    #   X_summed[i, :] += X[j, :] for all j that have inverse_idx[j] = i
    if scipy.sparse.issparse(adata.X):
        for i in range(adata.X.shape[0]):
            np.add.at(X_summed, inverse_idx[i], adata.X[i].toarray().ravel())
    else:
        np.add.at(X_summed, inverse_idx, adata.X)

    # Build obs_names for each unique bin
    obs_names = []
    for b in unique_bins:
        new_y = b // new_ncol
        new_x = b % new_ncol
        obs_names.append(f's_{resolution:03d}um_{new_y:05d}_{new_x:05d}-1')

    new_adata = ad.AnnData(
        X=X_summed,
        obs=pd.DataFrame(index=obs_names),
        var=adata.var.copy(),
        varm=adata.varm.copy(),
        uns=dict(adata.uns)
    )

    if 'genotype' in adata.obsm:
        print("Info: Calling genotypes for the binned data using the previous arguments:")
        print("\n".join([f"{k}: {v}" for k, v in adata.uns['genotype_call_args'].items()]))
        tl.call_genotypes(new_adata, **adata.uns['genotype_call_args'])

    return new_adata


def join_with_wta(wta: 'sd.SpatialData', gf_adata: ad.AnnData) -> 'sd.SpatialData':
    """
    Join the spatial data with the whole transcriptome data. Adds additional gapfill tables.
    :param wta: The whole transcriptome data.
    :param gf_adata: The spatial gapfill data.
    :return: The joined data.
    """
    assert_spatial(gf_adata)

    def _build_adata(_wta, _gf, resolution):
        _gf = bin(_gf, resolution)
        # Re-order and filter the "cells" in my adata object to match the 2 micron
        # resolution barcode labels.
        _gf = _gf[_gf.obs.index.isin(_wta.obs.index), :]

        # Fill in missing cells with zeros
        missing_cells = _wta.obs.index.difference(_gf.obs.index).values
        if len(missing_cells) > 0:
            # Concatenate and fill in missing values with nan
            missing_adata = ad.AnnData(X=np.zeros((len(missing_cells), _gf.shape[1])),
                                       obs=pd.DataFrame(index=missing_cells),
                                       var=_gf.var.copy(),
                                       varm={k: v.copy() for k,v in _gf.varm.items()},
                                       uns=dict(_gf.uns),
                                       obsm={k: pd.DataFrame(index=missing_cells, columns=v.columns) for k,v in _gf.obsm.items()},
                                       layers={k: np.zeros((len(missing_cells), _gf.shape[1])) for k in _gf.layers})
            _gf = ad.concat([_gf, missing_adata], axis=0)

        # Re-order the cells to match the original data.
        return _gf[_wta.obs.index]

    # Find all the coordinate systems associated with the original data.
    for table in ['square_002um', 'square_008um', 'square_016um']:
        if table not in wta.tables:
            continue
        wta_table = wta.tables[table]
        gf_table = _build_adata(wta_table, gf_adata, int(table.split('_')[-1].replace('um', '')))
        # Copy over all metadata since the cells should be consistent
        gf_table.uns[sd.models.TableModel.ATTRS_KEY] = wta_table.uns[sd.models.TableModel.ATTRS_KEY].copy()
        gf_table.obsm['spatial'] = wta_table.obsm['spatial'].copy()
        gf_table.obs['region'] = wta_table.obs['region'].copy()
        gf_table.obs['location_id'] = wta_table.obs['location_id'].copy()

        wta.tables['gf_' + table] = gf_table

    return wta


def plot_genotypes(sdata: 'sd.SpatialData',
                   probe: str,
                   dataset_id: str = "",
                   image_name: str = "hires_image",
                   resolution: int = 2) -> 'plt.Axes':
    # Plot the data
    check_plotting()

    res_name = f"square_{resolution:03d}um"

    # Create points for the genotype where not NA
    genotype = sdata.tables[f'gf_{res_name}'].obsm['genotype'][probe].fillna("NA")
    sdata[res_name].obs['giftwrap_genotype'] = genotype

    ax = sdata.pl.render_images(f"{dataset_id}_{image_name}", alpha=0.8) \
        .pl.render_shapes(element=f'{dataset_id}_{res_name}', color='giftwrap_genotype', method='matplotlib', na_color=None) \
        .pl.show(coordinate_systems="global", figsize=(25, 25), na_in_legend=False, title=probe, return_ax=True)

    del sdata[res_name].obs['giftwrap_genotype']

    # Remove the x and y ticks, tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    # Rename the x axis and y axis
    ax.set_xlabel("Spatial 1")
    ax.set_ylabel("Spatial 2")

    return ax


def impute_genotypes(sdata: 'sd.SpatialData',
                     cluster_key: str,
                     resolution: str = None,
                     k: int = None,
                     threshold: float = None,
                     impute_all: bool = None,
                     hold_out: bool = None,
                     cores: int = None
                     ) -> 'sd.SpatialData':
    """
    Wrapper around giftwrap.tl.impute_genotypes to impute genotypes in a SpatialData object across all resolutions.
    :param sdata: The SpatialData object containing the spatial data.
    :param cluster_key: The key in the obs of the SpatialData table that contains the clustering information. Consider
        running giftwrap.sp.recipe_spatial_expression_coclustering first to generate this.
    :param resolution: The resolution to impute genotypes at. If None, will impute at all resolutions.
    :param k: The number of neighbors to use for imputation. If None, uses the default value from giftwrap.tl.impute_genotypes.
    :param threshold: The threshold to use for imputation. If None, uses the default value from giftwrap.tl.impute_genotypes.
    :param impute_all: If True, will impute all genotypes, even those that are already called. If False, will only
        impute genotypes that are not already called. If None, uses the default value from giftwrap.tl.impute_genotypes.
    :param hold_out: If True, will hold out a portion of the data for validation. If False, will not hold out any data.
        If None, uses the default value from giftwrap.tl.impute_genotypes.
    :param cores: The number of cores to use for imputation. If None, uses the default value from giftwrap.tl.impute_genotypes.
    :return: The updated SpatialData object with the imputed genotypes.
    """
    if resolution is None:  # Compute for all resolutions
        for resolution in ['square_002um', 'square_008um', 'square_016um']:
            if resolution in sdata.tables:
                sdata = impute_genotypes(sdata, cluster_key, resolution,
                    k, threshold, impute_all, hold_out, cores)
        return sdata

    if resolution not in sdata.tables:
        raise ValueError(f"Resolution {resolution} not found in SpatialData object. Available resolutions: {list(sdata.tables.keys())}")

    table = sdata.tables[resolution]
    if cluster_key not in table.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in table '{resolution}'. Please run "
                         f"recipe_spatial_expression_coclustering first to generate this.")

    print(f"Imputing genotypes for resolution {resolution} with cluster key '{cluster_key}'...")
    kwargs = {}
    if k is not None:
        kwargs['k'] = k
    if threshold is not None:
        kwargs['threshold'] = threshold
    if impute_all is not None:
        kwargs['impute_all'] = impute_all
    if hold_out is not None:
        kwargs['hold_out'] = hold_out
    if cores is not None:
        kwargs['cores'] = cores
    sdata.tables[resolution] = tl.impute_genotypes(
        table, cluster_key, **kwargs
    )

    return sdata


def recipe_spatial_expression_coclustering(
        sdata: 'sd.SpatialData',
        table_name: str = 'square_002um',
        cluster_key: str = 'spatio_expression_cluster',
        combination_weight: float = 0.5,
        n_highly_variable_genes: int = None,
        hvg_kwargs: dict = None,
        n_neighbors: int = 15,
        n_expression_neighbors: int = None,
        n_spatial_neighbors: int = None,
        leiden_kwargs: dict = None,
) -> 'sd.SpatialData':
    """
    This is a recipe for spatial expression co-clustering. This requires the squidpy package to be installed.
    This function is useful for spatially-informed imputation (i.e. imputation with spatial context in impute_genotypes).
    :param sdata: The SpatialData object containing the spatial data.
    :param table_name: The name of the table to use for co-clustering. Default is 'square_002um'.
    :param cluster_key: The key to use for storing the clustering results.
    :param combination_weight: The weight to use for co-clustering
        (i.e. how much to weight the spatial neighbors vs expression neighbors). Should be between 0 and 1, where
        0 means only spatial neighbors are used, and 1 means only expression neighbors are used.
    :param n_highly_variable_genes: The number of highly variable genes to use for clustering. If None, uses all genes.
    :param hvg_kwargs: Additional keyword arguments for the highly variable genes selection.
    :param n_neighbors: The number of neighbors to use for co-clustering.
    :param n_expression_neighbors: The number of neighbors to use for expression-based KNN connectivities.
    :param n_spatial_neighbors: The number of neighbors to use for spatial-based KNN connectivities.
    :param leiden_kwargs: Additional keyword arguments for the Leiden clustering algorithm.
    :return: The updated SpatialData object with the co-clustering results.
    """
    from spatialdata_io.experimental import to_legacy_anndata
    check_squidpy()

    if n_expression_neighbors is None:
        n_expression_neighbors = n_neighbors
    if n_spatial_neighbors is None:
        n_spatial_neighbors = n_neighbors

    adata = to_legacy_anndata(sdata, table_name=table_name, include_images=False)

    # Run the co-clustering
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_highly_variable_genes, **(hvg_kwargs or {}))
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_expression_neighbors, use_rep='X_pca', key_added='expr_connectivities')
    sq.gr.spatial_neighbors(
        adata,
        coord_type='grid',
        n_neighs=n_spatial_neighbors,
        key_added='spatial_connectivities'
    )

    # Join the connectivities
    exp_connectivities = adata.obsp['expr_connectivities']
    spatial_connectivities = adata.obsp['spatial_connectivities']
    joined_connectivities = (combination_weight * exp_connectivities) + ((1 - combination_weight) * spatial_connectivities)
    adata.obsp['combined_connectivities'] = joined_connectivities

    # Run clustering
    sc.tl.leiden(
        adata,
        obsp='combined_connectivities',
        key_added=cluster_key,
        **(leiden_kwargs or {})
    )

    # Add the clustering results back to the SpatialData object
    sdata[table_name].obs[cluster_key] = adata.obs[cluster_key]
    sdata[table_name].uns[cluster_key] = adata.uns[cluster_key].copy()
    sdata[table_name].obsp['combined_connectivities'] = adata.obsp['combined_connectivities']

    return sdata
