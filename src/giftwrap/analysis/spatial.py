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


def assert_spatial(adata: ad.AnnData):
    """
    Assert that spatialdata is installed.
    """
    if sd is None:
        raise ImportError("spatialdata is not installed. Please install it to use this function.")
    if 'array_row' not in adata.obs or 'array_col' not in adata.obs:
        raise ValueError("This function is currently only applicable to Visium HD data.")


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


