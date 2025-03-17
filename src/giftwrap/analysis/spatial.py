from __future__ import annotations

import anndata as ad
import pandas as pd
import numpy as np
try:
    import spatialdata as sd
except ImportError:
    sd = None


def assert_spatial(adata: ad.AnnData):
    """
    Assert that spatialdata is installed.
    """
    if sd is None:
        raise ImportError("spatialdata is not installed. Please install it to use this function.")
    if 'array_row' not in adata.obs or 'array_col' not in adata.obs:
        raise ValueError("This function is currently only applicable to Visium HD data.")

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

    curr_dim = adata.obs['array_row'].max() + 1, adata.obs['array_col'].max() + 1
    adjusted_dim = curr_dim[0] // resolution, curr_dim[1] // resolution


    # Starting at the top left corner, we will iterate through the data and aggregate the data.

    # curr_n_bins = curr_dim[0] * curr_dim[1]
    new_n_bins = adjusted_dim[0] * adjusted_dim[1]

    X = np.zeros((new_n_bins, adata.shape[1]))
    obs_names = []
    missed = []
    i = 0
    for x in range(adjusted_dim[0]):
        for y in range(adjusted_dim[1]):
            # Indices of cells in the current bin
            curr_bin = np.logical_and(adata.obs['array_row'] >= x * resolution,
                                        adata.obs['array_row'] < (x + 1) * resolution)
            curr_bin = np.logical_and(curr_bin, adata.obs['array_col'] >= y * resolution)
            curr_bin = np.logical_and(curr_bin, adata.obs['array_col'] < (y + 1) * resolution)
            if curr_bin.shape[0] == 0:
                missed.append(i)
                i += 1
                continue
            X[i] = adata[curr_bin].X.sum(axis=0)
            obs_names.append(f's_{x:03d}_{y:03d}um')
            i += 1
    if len(missed) > 0:
        # Remove the missed bins
        X = np.delete(X, missed, axis=0)

    # Create the new AnnData object
    new_adata = ad.AnnData(X=X,
                            obs=pd.DataFrame(index=obs_names),
                            var=adata.var if adata.var is not None else None,
                            varm=adata.varm if adata.varm is not None else None,
                            uns=adata.uns.copy() if adata.uns is not None else None,
                           )

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
        missing_cells = _wta.obs.index.difference(_gf.obs.index)
        if len(missing_cells) > 0:
            # Concatenate and fill in missing values with nan
            missing_adata = ad.AnnData(X=np.zeros((len(missing_cells), _gf.shape[1])),
                                       obs=pd.DataFrame(index=missing_cells),
                                       var=_gf.var,
                                       varm=_gf.varm,
                                       uns=_gf.uns,
                                       obsm=_gf.obsm,
                                       layers={k: np.zeros((len(missing_cells), _gf.shape[1])) for k in _gf.layers})
            _gf = ad.concat([_gf, missing_adata], axis=0)

        # Re-order the cells to match the original data.
        return _gf[_wta.obs.index]

    # Re-order and filter the "cells" in my adata object to match the 2 micron
    # resolution barcode labels.
    gf_adata = gf_adata[gf_adata.obs.index.isin(wta.obs.index), :]

    # Fill in missing cells with zeros
    missing_cells = wta.obs.index.difference(gf_adata.obs.index)
    if len(missing_cells) > 0:
        # Concatenate and fill in missing values with nan
        missing_adata = ad.AnnData(X=np.zeros((len(missing_cells), gf_adata.shape[1])),
                                   obs=pd.DataFrame(index=missing_cells),
                                   var=gf_adata.var,
                                   varm=gf_adata.varm,
                                   uns=gf_adata.uns,
                                   obsm=gf_adata.obsm,
                                   layers={k: np.zeros((len(missing_cells), gf_adata.shape[1])) for k in gf_adata.layers})
        gf_adata = ad.concat([gf_adata, missing_adata], axis=0)

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

        wta.tables['gf_' + table] = gf_table

    return wta
