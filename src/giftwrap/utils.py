import functools
import gzip
import io
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
import itertools
import contextlib
import multiprocessing
from typing import Literal, Optional

import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scipy


#Based on: https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterator, n):
    """
    Returns a generator that yields lists of n elements from the input iterable.
    The final list may have fewer than n elements.
    """
    while True:
        chunk = list(itertools.islice(iterator, n))
        if not chunk:
            return
        yield chunk


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


# Inject starmap_async
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


def read_manifest(output_dir: Path) -> pd.DataFrame:
    """
    Read the manifest file. This is a TSV file with the following columns:
    - index: The index for the probe
    - name: The name of the probe
    - lhs_probe: The left hand side probe sequence
    - rhs_probe: The right hand side probe sequence
    - gap_probe_sequence: The sequence the probe was designed against
    - original_gap_probe_sequence: The expected WT gap probe sequence
    :param output_dir: The pipeline output directory.
    :return: The parsed dataframe which should be indexed by the index.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    return pd.read_table(output_dir / "manifest.tsv")


class TechnologyFormatInfo(ABC):
    """
    Generic class to hold metadata related to parsing Read1 and Read2.
    """

    def __init__(self, barcode_dir: Optional[str] = None, read1_length: Optional[int] = None, read2_length: Optional[int] = None):
        self._read1_length = read1_length
        self._read2_length = read2_length

        if barcode_dir:
            self._barcode_dir = Path(barcode_dir)
        else:
            # Fallback to our resources directory
            self._barcode_dir = Path(__file__).parent / "resources"


    @property
    def read1_length(self) -> Optional[int]:
        """
        This is the expected length of each R1 read, if defined the pipeline can improve performance.
        :return: The length, or None if not defined.
        """
        return self._read1_length

    @property
    def read2_length(self) -> Optional[int]:
        """
        This is the expected length of each R2 read, if defined the pipeline can improve performance.
        :return: The length, or None if not defined.
        """
        return self._read2_length

    @property
    @abstractmethod
    def umi_start(self) -> int:
        """
        The start position of the UMI sequence in R1.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def umi_length(self) -> int:
        """
        The length of the UMI sequence on R1.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def cell_barcodes(self) -> list[str]:
        """
        The list of potential barcodes.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def cell_barcode_start(self) -> int:
        """
        The start position of the cell barcode in the read.
        """
        raise NotImplementedError()

    @functools.lru_cache(maxsize=1)
    def length2barcodes(self) -> dict[int, set[str]]:
        """
        Returns a dictionary mapping barcode lengths to the list of potential barcodes. Should be sorted by largest to
            smallest.
        """
        l2bc = dict()
        for bc in sorted(self.cell_barcodes, key=len, reverse=True):
            if len(bc) not in l2bc:
                l2bc[len(bc)] = set()
            l2bc[len(bc)].add(bc)
        return l2bc

    @property
    def max_cell_barcode_length(self) -> int:
        """
        Returns the maximum length of a cell barcode.
        """
        return max(self.length2barcodes().keys())

    @functools.lru_cache(maxsize=1000)
    def barcode2coordinates(self, barcode: str) -> tuple[int, int]:
        """
        Returns the X and Y coordinates of a barcode.
        :param barcode: The barcode.
        """
        return self.barcode_coordinates[self.cell_barcodes.index(barcode)]

    @property
    @abstractmethod
    def is_spatial(self) -> bool:
        """
        Whether the technology is spatial. If true, then barcode_coordinates() must be defined.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def barcode_coordinates(self) -> list[tuple[int, int]]:
        """
        The x and y coordinates of the barcode in the read.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def constant_sequence(self) -> str:
        """
        The constant sequence that is expected in the read.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def constant_sequence_start(self) -> int:
        """
        The start position of the constant sequence in the read. Note that this should be relative to the end of the read
            insert. For example, in 10X flex, 0 would be the first base after the LHS + gapfill + RHS.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_constant_sequence(self) -> bool:
        """
        Whether the read has a constant sequence.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcodes(self) -> list[str]:
        """
        The list of potential probe barcodes.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcode_start(self) -> int:
        """
        The start position of the probe barcode in the read. Note that this should be relative to the end of the constant
            sequence insert. For example, in 10X flex, 2 would be the first base after the constant sequence+NN.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcode_length(self) -> int:
        """
        The length of the probe barcode.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_probe_barcode(self) -> bool:
        """
        Whether the read has a probe barcode.
        """
        raise NotImplementedError()

    @abstractmethod
    def probe_barcode_index(self, bc: str) -> int:
        """
        Convert a probe barcode to an index.
        """
        raise NotImplementedError()

    def make_barcode_string(self, cell_barcode: str, plex: int = 1, x_coord: Optional[int] = None, y_coord: Optional[int] = None) -> str:
        """
        Format a cell barcode into a string.
        :param cell_barcode: The barcode.
        :param plex: The bc index for representing demultiplexed cells.
        :param x_coord: The x coordinate.
        :param y_coord: The y coordinate.
        """
        return f"{cell_barcode}-{plex}"


class FlexFormatInfo(TechnologyFormatInfo):
    """
    Describes the format of a 10X Flex run.
    """

    def __init__(self,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 28,
                 read2_length: Optional[int] = 90):
        super().__init__(barcode_dir, read1_length, read2_length)
        # Load the barcodes
        barcodes = pd.read_table(self._barcode_dir / "737K-fixed-rna-profiling.txt.gz", header=None, names=["barcode"],
                                 compression="gzip")
        # Strip the -Number from the barcode
        barcodes["barcode"] = barcodes["barcode"].str.split("-").str[0]
        # Collect the universe of barcodes
        self._barcodes = list(barcodes["barcode"].unique())

        self._probe_barcodes = {s: (i+1) for i, s in enumerate([
                "ACTTTAGG",
                "AACGGGAA",
                "AGTAGGCT",
                "ATGTTGAC",
                "ACAGACCT",
                "ATCCCAAC",
                "AAGTAGAG",
                "AGCTGTGA",
                "ACAGTCTG",
                "AGTGAGTG",
                "AGAGGCAA",
                "ACTACTCA",
                "ATACGTCA",
                "ATCATGTG",
                "AACGCCGA",
                "ATTCGGTT"
            ]) }


    @property
    def umi_start(self) -> int:
        return 16

    @property
    def umi_length(self) -> int:
        return 12

    @property
    def cell_barcodes(self) -> list[str]:
        return self._barcodes

    @property
    def cell_barcode_start(self) -> int:
        return 0

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def constant_sequence(self) -> str:
        return "ACGCGGTTAGCACGTA"

    @property
    def constant_sequence_start(self) -> int:
        return 0

    @property
    def has_constant_sequence(self) -> bool:
        return True

    @property
    @functools.lru_cache(1)
    def probe_barcodes(self) -> list[str]:
        return list(self._probe_barcodes.keys())

    @property
    def probe_barcode_start(self) -> int:
        return 2  # There is an NN between the constant sequence and the probe barcode

    @property
    def probe_barcode_length(self) -> int:
        return 8

    @property
    def has_probe_barcode(self) -> bool:
        return True

    def probe_barcode_index(self, bc: str):
        return self._probe_barcodes[bc]

    @property
    def barcode_coordinates(self) -> list[tuple[int, int]]:
        raise NotImplementedError()


class VisiumFormatInfo(TechnologyFormatInfo):

    def __init__(self,
                 version: int = 5,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 28,
                 read2_length: Optional[int] = 90):
        super().__init__(barcode_dir, read1_length, read2_length)
        # Load the barcodes
        # TODO: I am assuming that X is first and Y is second
        barcodes = pd.read_table(self._barcode_dir / f"visium-v{version}_coordinates.txt", header=None, names=["barcode", 'x', 'y'])
        self._barcodes = barcodes["barcode"].tolist()
        self._barcode_coordinates = barcodes[['x', 'y']].values.tolist()

    @property
    def umi_start(self) -> int:
        return 0

    @property
    def umi_length(self) -> int:
        return 12

    @property
    def cell_barcodes(self) -> list[str]:
        return self._barcodes

    @property
    def cell_barcode_start(self) -> int:
        return 12

    @property
    def is_spatial(self) -> bool:
        return True

    @property
    def barcode_coordinates(self) -> list[tuple[int, int]]:
        return self._barcode_coordinates

    @property
    def has_constant_sequence(self) -> bool:
        return False

    @property
    def has_probe_barcode(self) -> bool:
        return False

    @property
    def constant_sequence(self) -> str:
        raise NotImplementedError()

    @property
    def constant_sequence_start(self) -> int:
        raise NotImplementedError()

    @property
    def probe_barcodes(self) -> list[str]:
        raise NotImplementedError()

    @property
    def probe_barcode_start(self) -> int:
        raise NotImplementedError()

    def probe_barcode_index(self, bc: str):
        raise NotImplementedError()

    @property
    def probe_barcode_length(self) -> int:
        raise NotImplementedError()


class VisiumHDFormatInfo(TechnologyFormatInfo):

    def __init__(self,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 43,
                 read2_length: Optional[int] = 50):
        super().__init__(barcode_dir, read1_length, read2_length)
        # Load the barcodes, note that this REQUIRES spaceranger to be installed
        import shutil
        import importlib
        import sys
        # Find spaceranger
        spaceranger = shutil.which("spaceranger")
        if not spaceranger:
            raise FileNotFoundError("spaceranger not found on PATH.")
        spaceranger_path = Path(spaceranger)
        # Import the protobuf schema
        sys.path.append(str(spaceranger_path.parent / "lib" / "python" / "cellranger" / "spatial"))
        schema_def = importlib.import_module("visium_hd_schema_pb2")
        # Parse the schema
        slide_def = schema_def.VisiumHdSlideDesign()
        with open(self._barcode_dir / "visium_hd_v1.slide", 'rb') as f:
            slide_def.ParseFromString(f.read())
        # Assemble all possible barcodes
        self._barcode_coordinates = []
        self._barcodes = []
        for y, bc1 in enumerate(slide_def.two_part.bc1_oligos):
            for x, bc2 in enumerate(slide_def.two_part.bc2_oligos):
                self._barcode_coordinates.append((x, y))
                self._barcodes.append(bc1 + bc2)

    @property
    def umi_start(self) -> int:
        return 0

    @property
    def umi_length(self) -> int:
        return 9

    @property
    def cell_barcodes(self) -> list[str]:
        return self._barcodes

    @property
    def cell_barcode_start(self) -> int:
        return 9

    @property
    def is_spatial(self) -> bool:
        return True

    @property
    def barcode_coordinates(self) -> list[tuple[int, int]]:
        return self._barcode_coordinates

    @property
    def has_constant_sequence(self) -> bool:
        return False

    @property
    def has_probe_barcode(self) -> bool:
        return False

    # Cell barcodes will be the 2um "binned" output
    def make_barcode_string(self, cell_barcode: str, plex: int = 1, x_coord: Optional[int] = None, y_coord: Optional[int] = None) -> str:
        return f"s_002um_{x_coord}_{y_coord}-{plex}"

    @property
    def constant_sequence(self) -> str:
        raise NotImplementedError()

    @property
    def constant_sequence_start(self) -> int:
        raise NotImplementedError()

    @property
    def probe_barcodes(self) -> list[str]:
        raise NotImplementedError()

    @property
    def probe_barcode_start(self) -> int:
        raise NotImplementedError()

    def probe_barcode_index(self, bc: str):
        raise NotImplementedError()

    @property
    def probe_barcode_length(self) -> int:
        raise NotImplementedError()


def read_barcodes(output_dir: Path) -> pd.DataFrame:
    """
    Read the list of cell barcodes.
    :param output_dir: The output directory.
    :return: The list of cell barcodes.
    """
    # FIXME: TURN INTO DATAFRAME WITH SPATIAL COORDS
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if (output_dir / "barcodes.tsv").exists():
        return pd.read_table(output_dir / "barcodes.tsv")
    elif (output_dir / "barcodes.tsv.gz").exists():
        return pd.read_table(output_dir / "barcodes.tsv.gz")
    else:
        raise FileNotFoundError("Barcodes file not found.")


# Create a file writer handler that wraps gzip and NamedTemporaryFile
class GzipNamedTemporaryFile:

    def __init__(self):
        self.temp_file = NamedTemporaryFile(mode="w+b", delete=False)
        self.gzip_file = gzip.GzipFile(fileobj=self.temp_file, mode="w")
        # Note that GzipFile only supports binary mode:
        self.gzip_file = io.TextIOWrapper(self.gzip_file)

    def __enter__(self):
        self.temp_file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gzip_file.close()
        return self.temp_file.__exit__(exc_type, exc_val, exc_tb)

    @property
    def name(self):
        return self.temp_file.name

    def write(self, data: str):
        self.gzip_file.write(data)


def maybe_gzip(file: Path | None, mode: Literal["r"] | Literal["w"] = "r"):
    """
    Return a file handle. If the file is gzipped, then we will use gzip.open, otherwise we will use open.
    :param file: The file path. If None, this will return a temporary file.
    :param mode: The mode.
    :return: The file handle.
    """
    file = Path(file)
    if mode == 'r':
        if "gz" in file.suffix:
            return gzip.open(file, 'rt')
        else:
            return open(file, mode)
    else:
        if "gz" in file.suffix:
            return gzip.open(file, 'wt')
        else:
            return open(file, mode)


def sort_tsv_file(file: Path, columns: list[int], cores: int):
    """
    Sort a written tsv file in-place. Will either use a single core or multiple cores depending on the cores argument.
        Note, this will attempt to defer to the unix sort command if cores is > 1.
    :param file: The file. May be gzipped.
    :param columns: The columns indices to sort by.
    :param cores: The number of cores to use.
    """
    if cores > 1:
        # Check for the sort command
        sort_avail = shutil.which("sort")
        if sort_avail:
            # Use the unix sort command
            # First move to a temporary file
            os.rename(file, file.with_suffix(".tmp"))
            # Then sort (Ignore locale for all commands for speed)
            sort_command = "export LC_ALL=C; "
            # First open the file
            if 'gz' in file.suffix:
                sort_command += f"zcat {file.with_suffix('.tmp')} | "
            else:
                sort_command += f"cat {file.with_suffix('.tmp')} | "
            # Note that we need to skip the first line: https://unix.stackexchange.com/a/11857
            sort_command += '(IFS= read -r REPLY; printf "%s\\n" "$REPLY"; '
            # Then sort
            sort_command += f"sort -t \"$(printf '\\t')\" --parallel={cores} --numeric-sort"
            # Note that sort doesn't parallelize piped input since it assumes its a small file so we will give it a
            # large buffer size (1 GB per core)
            sort_command += f" --buffer-size={cores}G"
            # Note that we need to add 1 to the column index since sort is 1-indexed
            for col in columns:
                sort_command += f" -k{col + 1},{col + 1}"
            # Close the parenthesis
            sort_command += ")"

            # if the file is gzipped, then we need to gzip the output
            if ".gz" in file.suffix:
                sort_command += f" | gzip > {file}"
            else:
                sort_command += f" > {file}"

            result = subprocess.run(sort_command, shell=True)
            if result.returncode != 0:
                # Move the file back
                os.rename(file.with_suffix(".tmp"), file)
                raise RuntimeError("Failed to sort the file.")
            # Delete the backup file
            os.remove(file.with_suffix(".tmp"))
            return

    # Not able to use the sort command so fallback to python
    df = pd.read_table(file, sep="\t", compression="gzip" if "gz" in file.suffix else None)
    df = df.sort_values(df.columns[columns].tolist())
    df.to_csv(file, sep="\t", index=False, compression="gzip" if "gz" in file.suffix else None)


def filter_h5_file(input_file: Path, output_file: Path, barcodes_list: list[str], pad_matrix: bool = True):
    """
    Given a counts h5 file and a list of barcodes, filter the barcodes to only include the ones in the list.
    :param input_file: The input h5 file.
    :param output_file: The output h5 file.
    :param barcodes_list: The barcodes list.
    :param pad_matrix: Whether to pad the matrix with zeros if there are barcodes provided that don't exist in the file.
    """
    # First, copy the file
    shutil.copy(input_file, output_file)
    # Then open the file
    with h5py.File(output_file, 'r+') as f:
        barcodes = f['matrix']['barcode'][:].astype(str)
        # Get the filtered list of barcodes indices and re-order them
        barcode_indices = np.array([i for i, bc in enumerate(barcodes) if bc in barcodes_list])
        # Check if the we need to filter the data
        if len(barcode_indices) == len(barcodes):
            return  # Equal size, no point in filtering

        if len(barcode_indices) == 0:
            raise ValueError("No barcodes found in the file.")

        if pad_matrix and len(barcodes_list) > len(barcode_indices):
            padded_barcodes = [bc for bc in barcodes_list if bc not in barcodes]
            print(f"Padding {len(padded_barcodes)} unseen cells with zeroes.")
        else:
            padded_barcodes = []

        # Filter the data
        del f['matrix']['barcode']
        f['matrix'].create_dataset("barcode",
                                  data=np.concatenate([barcodes[barcode_indices].astype('S'), np.array(padded_barcodes, dtype='S')]),
                                  compression='gzip')

        for layer_name in ['data', 'total_reads', 'percent_supporting']:
            data = read_sparse_matrix(f['matrix'], layer_name)
            data = data[barcode_indices, :]
            # Add padding
            if len(padded_barcodes) > 0:
                data = scipy.sparse.vstack([data, scipy.sparse.csr_matrix((len(padded_barcodes), data.shape[1]))])
            del f['matrix'][layer_name]
            write_sparse_matrix(f['matrix'], layer_name, data)

        obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)
        obs_meta_df = pd.DataFrame({
            col: (f['cell_metadata'][col][:].astype(str) if col == 'barcode' else f['cell_metadata'][col][:].astype(int)) for col in obs_meta_columns
        }).set_index('barcode')
        obs_meta_df = obs_meta_df.loc[barcodes[barcode_indices]]
        # Move the index back to a column
        obs_meta_df = obs_meta_df.reset_index()
        # Add padding
        if len(padded_barcodes) > 0:
            obs_meta_df = pd.concat([obs_meta_df, pd.DataFrame({col: (([pd.NA] * len(padded_barcodes)) if col != "barcode" else padded_barcodes) for col in obs_meta_df.columns})])
        del f['cell_metadata']
        cell_metadata_grp = f.create_group('cell_metadata')
        cell_metadata_grp.create_dataset('columns', data=np.array(obs_meta_df.columns, dtype='S'), compression='gzip')
        for col in obs_meta_df.columns:
            values = obs_meta_df[col].values
            cell_metadata_grp.create_dataset(col, data=values, compression='gzip')

        del f.attrs['n_cells']
        f.attrs['n_cells'] = len(barcodes_list) + len(padded_barcodes)

        # Done


def write_sparse_matrix(grp: h5py.Group, name: str, sp_matrix):
    """
    Write a sparse matrix to a group.
    :param grp: The group.
    :param name: The name of the dataset.
    :param sp_matrix: The sparse matrix.
    """
    if not scipy.sparse.isspmatrix_csr(sp_matrix):
        sp_matrix = sp_matrix.tocsr()

    matrix_grp = grp.create_group(name)
    matrix_grp.create_dataset("data", data=sp_matrix.data, compression='gzip', shuffle=True)
    matrix_grp.create_dataset("indices", data=sp_matrix.indices, compression='gzip', shuffle=True)
    matrix_grp.create_dataset("indptr", data=sp_matrix.indptr, compression='gzip', shuffle=True)
    # Shape
    matrix_grp.attrs['shape'] = sp_matrix.shape


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


def read_h5_file(filename: str) -> ad.AnnData:
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
        obs_df = pd.DataFrame({
            'barcode': f['matrix']['barcode'][:].astype(str),
        }).set_index('barcode')

        # Read the obs metadata
        obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)
        obs_meta_df = pd.DataFrame({
            col: (f['cell_metadata'][col][:].astype(str) if col == 'barcode' else f['cell_metadata'][col][:].astype(int)) for col in obs_meta_columns
        }).set_index('barcode')
        obs_df = obs_df.merge(obs_meta_df, on='barcode', how='left')

        manifest = pd.DataFrame({
            'probe': f['probe_metadata']['name'][:].astype(str),
            'lhs_probe': f['probe_metadata']['lhs_probe'][:].astype(str),
            'rhs_probe': f['probe_metadata']['rhs_probe'][:].astype(str),
            'gap_probe_sequence': f['probe_metadata']['gap_probe_sequence'][:].astype(str),
            'original_gap_probe_sequence': f['probe_metadata']['gap_probe_sequence'][:].astype(str),
        })
        if 'gene' in f['probe_metadata']:
            manifest['gene'] = f['probe_metadata']['gene'][:].astype(str)

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
                                "created_date": pd.Timestamp(f.attrs['created_date']),
                                "n_cells": f.attrs['n_cells'],
                                "n_probes": f.attrs['n_probes'],
                                "n_probe_gapfill_combinations": f.attrs['n_probe_gapfill_combinations'],
                           })

    return adata


def merge_anndatas(adata_expression: ad.AnnData, adata_gapfill: ad.AnnData) -> ad.AnnData:
    """
    Merge two AnnData objects. The adata_gapfill should have the same barcodes as the adata_expression.
    :param adata_expression: The expression data.
    :param adata_gapfill: The gapfill data.
    :return: The merged AnnData object.
    """
    # This will attempt to merge the two AnnData objects.
    # Note that they have two completely different sets of vars so we will have to merge them manually by concatenating.

    # First we will concatenate the expression data
    X = scipy.sparse.hstack([adata_expression.X, adata_gapfill.X])
    # For each layer, we will concatenate with empty matrices
    layers = \
        {k: scipy.sparse.hstack([v, scipy.sparse.csr_matrix((v.shape[0], adata_gapfill.X.shape[1]))]) for k, v in adata_expression.layers.items()} \
        + {k: scipy.sparse.csr_matrix((adata_gapfill.X.shape[0], v.shape[1])) for k, v in adata_gapfill.layers.items()}
    # Should be the same cells, so join the obs and fill in with NaNs for missing data
    obs = pd.merge(adata_expression.obs, adata_gapfill.obs, how='outer', left_index=True, right_index=True)
    # Concatenate the var data
    # For each var, concatenate nan for filled in data
    var = dict()
    for column in adata_expression.var.columns:
        var[column] = np.concatenate([adata_expression.var[column].values, np.full(adata_gapfill.X.shape[1], np.nan)])
    for column in adata_gapfill.var.columns:
        var[column] = np.concatenate([np.full(adata_expression.X.shape[1], np.nan), adata_gapfill.var[column].values])
    var = pd.DataFrame(var)

    uns = dict()
    # Merge the uns data
    for key in adata_expression.uns.keys():
        uns[key] = adata_expression.uns[key]
    for key in adata_gapfill.uns.keys():
        uns[key] = adata_gapfill.uns[key]

    # There may be varm or obsm data in the expression anndata, so we will have to merge them as well
    varm = dict()
    for key in adata_expression.varm.keys():
        varm[key] = pd.concat([adata_expression.varm[key], pd.DataFrame(index=adata_gapfill.var.index)], axis=0)
    for key in adata_gapfill.varm.keys():
        varm[key] = pd.concat([pd.DataFrame(index=adata_expression.var.index), adata_gapfill.varm[key]], axis=0)

    obsm = dict()
    for key in adata_expression.obsm.keys():
        obsm[key] = pd.concat([adata_expression.obsm[key], pd.DataFrame(index=adata_gapfill.obs.index)], axis=1)
    for key in adata_gapfill.obsm.keys():
        obsm[key] = pd.concat([pd.DataFrame(index=adata_expression.obs.index), adata_gapfill.obsm[key]], axis=1)

    adata = ad.AnnData(X, layers=layers, obs=obs, var=var, uns=uns, varm=varm, obsm=obsm)
    return adata


def interpret_phred_letter(quality: str, base: Literal['sanger'] | Literal['illumina'] = 'illumina') -> float:
    """
    Convert a phred quality letter to a score.
    :param quality: The quality letter.
    :param base: The base quality system. Either 'sanger' or 'illumina'.
    :return: The probability of the base being incorrect.
    """
    assert len(quality) == 1, "Quality must be a single character."
    # Convert the character to a number
    score = ord(quality) - (33 if base == 'illumina' else 64)
    # Convert to P(error)
    return 10 ** (-score / 10)


def phred_string_to_probs(quality: str, base: Literal['sanger'] | Literal['illumina'] = 'illumina') -> list[float]:
    """
    Convert a phred quality string to a list of probabilities.
    :param quality: The quality string.
    :param base: The base quality system. Either 'sanger' or 'illumina'.
    :return: The list of probabilities.
    """
    return [interpret_phred_letter(q, base) for q in quality]


def permute_bases(seq: str, pos: list[int]) -> str:
    # Compute all possible sequences
    for combination in itertools.product("ACGT", repeat=len(pos)):
        curr_seq = seq
        for i, base in zip(pos, combination):
            curr_seq = curr_seq[:i] + base + curr_seq[i+1:]
        yield curr_seq
