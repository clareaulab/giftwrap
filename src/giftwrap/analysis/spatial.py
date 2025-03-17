from __future__ import annotations

import anndata as ad
import numpy as np
try:
    import spatialdata as sd
except ImportError:
    sd = None


def assert_spatial():
    """
    Assert that spatialdata is installed.
    """
    if sd is None:
        raise ImportError("spatialdata is not installed. Please install it to use this function.")


def bin(adata: ad.AnnData, resolution: int = 8) -> ad.AnnData:
    """
    This is ONLY APPLICABLE FOR VISIUM-HD.
    This bins/aggregates data from 2 micron resolution to any other resolution (must be a power of 2).
    :param adata: The Spatial gapfill data.
    :param resolution: The resolution to aggregate into in microns. Spaceranger typically aggregates to 8um and 16 um.
    :return: The binned data.
    """
    assert resolution % 2 == 0, "Resolution must be a power of 2."
    assert resolution > 2, "Resolution must be greater than 2."
    assert_spatial()


def join_with_wta(wta: 'sd.SpatialData', gf_adata: ad.AnnData) -> 'sd.SpatialData':
    """
    Join the spatial data with the whole transcriptome data.
    :param wta: The whole transcriptome data.
    :param gf_adata: The spatial gapfill data.
    :return: The joined data.
    """
    assert_spatial()

    wta.set_table_annotates_spatialelement()

    # All we need to do is add a new table to the spatialdata object.
    wta.tables['gf_square_002um'] = gf_adata
    wta.tables['gf_square_008um'] = bin(gf_adata, 8)
    wta.tables['gf_square_016um'] = bin(gf_adata, 16)

    # Find all the coordinate systems associated with the original data.
    for table in ['square_002um', 'square_008um', 'square_016um']:
        if table not in wta.tables:
            continue
        gf_name = 'gf_' + table
        wta.set_table_annotates_spatialelement(gf_name,
                                               wta.tables[table].uns[sd.models.TableModel.ATTRS_KEY]['region']
        # Copy the coordinate system from the original data.
        wta.tables[gf_name].uns[sd.models.TableModel.ATTRS_KEY] = wta.tables[table].uns[sd.models.TableModel.ATTRS_KEY].copy()