import contextlib
import itertools
import multiprocessing
from pathlib import Path
from typing import Iterator, Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy


class DummyResult:

    def __init__(self, res):
        self.res = res

    def get(self, *args, **kwargs):
        return self.res

    def wait(self, *args, **kwargs):
        pass

    def ready(self, *args, **kwargs):
        return True

    def successful(self, *args, **kwargs):
        return True


class ItertoolsWrapper:

    def starmap(self, *args, **kwargs):
        return itertools.starmap(*args, **kwargs)

    def starmap_async(self, *args, **kwargs):
        return DummyResult(itertools.starmap(*args, **kwargs))


def maybe_multiprocess(cores: int) -> multiprocessing.Pool:
    """
    Return a context manager that will either return the multiprocessing module or a dummy module depending on if there
    are more than 1 core reqeusted.
    :param cores: The number of cores.
    :return: The multiprocessing module or a dummy module.
    """
    if cores > 1:
        mp = multiprocessing.Pool(cores)
    else:
        mp = contextlib.nullcontext(ItertoolsWrapper())  # No multiprocessing
    return mp


def read_sparse_matrix(grp: h5py.Group, name: str) -> scipy.sparse.csr_matrix:
    """
    Read a sparse matrix from a group.
    :param grp: The group.
    :param name: The name of the dataset.
    :return: The sparse matrix.
    """
    matrix_grp = grp[name]
    shape = matrix_grp.attrs['shape']
    return scipy.sparse.csr_matrix((matrix_grp['data'], matrix_grp['indices'], matrix_grp['indptr']), shape=shape)


def iter_layers(adata: ad.AnnData) -> Iterator[tuple[str, Any]]:
    """
    Iterate over the (name, matrix) pairs of the real layers of an AnnData object.

    anndata >= 0.13 stores X as ``layers[None]``, so ``adata.layers`` always yields an extra
    ``None`` key aliasing X. Copying that key into a new ``AnnData(X=..., layers=...)`` raises
    "If you provide `layers[None]` and `X`, they must be identical". Always iterate layers
    through this helper instead of ``adata.layers.items()``.

    :param adata: The AnnData object.
    :return: An iterator over the (layer name, layer matrix) pairs, excluding the X alias.
    """
    for name, matrix in adata.layers.items():
        if name is None:
            continue
        yield name, matrix


def read_h5_file(filename: str | Path) -> ad.AnnData:
    """
    Read a generated h5 file and return an AnnData object.
    :param filename: The filename.
    :return: The AnnData object.
    """
    with h5py.File(filename, 'r') as f:
        X = read_sparse_matrix(f['matrix'], 'data')
        layers = {
            'total_reads': read_sparse_matrix(f['matrix'], 'total_reads'),  # Total umis encountered
            'percent_supporting': read_sparse_matrix(f['matrix'], 'percent_supporting'),  # Avg percent of umis supporting the gapfill call
        }
        var_df = pd.DataFrame({
            'probe': f['matrix']['probe'][:, 0].astype(str),
            'gapfill': f['matrix']['probe'][:, 1].astype(str),
        })

        # Add original probe indices if available
        if 'probe_index' in f['matrix']:
            var_df['probe_index'] = f['matrix']['probe_index'][:].astype(int)

        obs_df = pd.DataFrame({
            'barcode': f['matrix']['barcode'][:].astype(str),
        }).set_index('barcode')

        # Add original cell indices if available
        if 'cell_index' in f['matrix']:
            obs_df['cell_index'] = f['matrix']['cell_index'][:].astype(int)

        # Read the obs metadata
        obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)
        obs_meta_df = dict()
        for column in obs_meta_columns:
            values = f['cell_metadata'][column][:]
            if column == 'barcode':
                values = values.astype(str)
            else:
                try:
                    values = values.astype(int)  # Most metadata are ints
                except:  # If that doesn't work, try string
                    try:
                        values = values.astype(str)
                    except:
                        values = np.zeros_like(values, dtype=int)  # Give up
            obs_meta_df[column] = values
        obs_meta_df = pd.DataFrame(obs_meta_df).set_index("barcode")

        obs_df = obs_df.merge(obs_meta_df, on='barcode', how='left')

        manifest = pd.DataFrame({
            'probe': f['probe_metadata']['name'][:].astype(str),
            'lhs_probe': f['probe_metadata']['lhs_probe'][:].astype(str),
            'rhs_probe': f['probe_metadata']['rhs_probe'][:].astype(str),
            'gap_probe_sequence': f['probe_metadata']['gap_probe_sequence'][:].astype(str),
            'original_gap_probe_sequence': f['probe_metadata']['original_sequence'][:].astype(str),
        })
        if 'gene' in f['probe_metadata']:
            manifest['gene'] = f['probe_metadata']['gene'][:].astype(str)

        # Check if probe names are unique on the manifest
        if len(manifest['probe'].unique()) != len(manifest):
            raise ValueError("Probe names are not unique.")

        # Add reference to var_df
        var_df = var_df.merge(manifest, on='probe', how='left')
        var_df = var_df.rename(columns={'gap_probe_sequence': 'expected_gapfill', 'original_gap_probe_sequence': 'reference_gapfill'})
        var_df['probe_gapfill'] = var_df['probe'].str.cat(var_df['gapfill'], sep='|')
        var_df = var_df.set_index('probe_gapfill', drop=True)

        adata = ad.AnnData(X,
                           layers=layers,
                           obs=obs_df,
                           var=var_df,
                           uns={
                                "probe_metadata": manifest,
                                "plex": f.attrs['plex'],
                                "project": f.attrs['project'],
                                "created_date": f.attrs['created_date'], #pd.Timestamp(f.attrs['created_date']),
                                "n_cells": f.attrs['n_cells'],
                                "n_probes": f.attrs['n_probes'],
                                "n_probe_gapfill_combinations": f.attrs['n_probe_gapfill_combinations'],
                                "max_pcr_duplicates": f.attrs['max_pcr_duplicates'] if 'max_pcr_duplicates' in f.attrs else -1,
                           })

        if 'max_pcr_duplicates' in f.attrs and int(f.attrs['max_pcr_duplicates']) > 1:
            # We must read the pcr thresholds save the counts matrices for each threshold to the layers
            dup_grp = f['pcr_thresholded_counts']
            for threshold in range(1, f.attrs['max_pcr_duplicates']):
                adata.layers[f'X_pcr_threshold_{threshold}'] = read_sparse_matrix(dup_grp, f'pcr{threshold}')

    # Check if array_col and array_row exist in obs
    # If present, verify that all are integers
    if 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
        col_mask = adata.obs['array_col'].isnull() | (~pd.api.types.is_integer_dtype(adata.obs['array_col'].dtype))
        row_mask = adata.obs['array_row'].isnull() | (~pd.api.types.is_integer_dtype(adata.obs['array_row'].dtype))
        if col_mask.any() or row_mask.any():
            # We will need to regenerate only the problematic array_col and array_row values
            print("Warning: 'array_col' and 'array_row' in obs contain non-integer or null values. Regenerating problematic values.")
            # Create masks for problematic values
            problematic_mask = col_mask | row_mask

            if problematic_mask.any():
                # Vectorized parse from index -> base part before '-'
                idx_series = pd.Series(adata.obs.index.astype(str), index=adata.obs.index)
                base = idx_series.str.split('-', n=1).str[0]
                # Extract last two underscore-delimited tokens
                parts = base.str.rsplit('_', n=2, expand=True)
                if parts.shape[1] < 3:
                    parts = parts.reindex(columns=range(3))

                array_row_parsed = pd.to_numeric(parts.iloc[:, -2], errors='coerce').fillna(-1).astype(int)
                array_col_parsed = pd.to_numeric(parts.iloc[:, -1], errors='coerce').fillna(-1).astype(int)

                need_col = problematic_mask & col_mask
                need_row = problematic_mask & row_mask

                if need_col.any():
                    adata.obs.loc[need_col, 'array_col'] = array_col_parsed.loc[need_col].to_numpy()
                if need_row.any():
                    adata.obs.loc[need_row, 'array_row'] = array_row_parsed.loc[need_row].to_numpy()
            # Ensure columns are integer type
            adata.obs['array_col'] = adata.obs['array_col'].astype(int)
            adata.obs['array_row'] = adata.obs['array_row'].astype(int)

    return adata