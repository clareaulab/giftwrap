from collections import defaultdict
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
try:
    import scanpy as sc
except ImportError:
    sc = None


def _check_genotypes(adata: ad.AnnData):
    if 'genotype' not in adata.obsm:
        raise ValueError("Genotypes not found in adata. Please run call_genotypes first.")


# Gapfill-adata plots

def _generate_genotype_frequencies(gapfill_adata: ad.AnnData,
                                   probe: str,
                                   final_genotypes_available: bool,
                                   genotype_mode: str | None) -> tuple[str, dict[str, float]]:
    """
    Generate genotype frequencies for a given probe in the gapfill adata object.
    """
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
            for mask, gapfill in zip(gapfill_adata.obs_names, gapfills):
                if gapfill != 'nan' and (gapfill == gapfill):
                    # Get the counts for this cell
                    counts = gapfill_adata[mask, var_mask].X.toarray().flatten()
                    # Normalize by the total number of UMIs in the cell
                    normalized_counts = counts / normalization_constant[mask]
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
            ref_frequencies = {motif.ljust(max_length, '-') : freq for motif, freq in ref_frequencies.items()}
            return ref_frequencies, None
        else:
            max_length = max(max(len(motif) for motif in ref_frequencies.keys()),
                             max(len(motif) for motif in alt_frequencies.keys()))
            ref_frequencies = {motif.ljust(max_length, '-') : freq for motif, freq in ref_frequencies.items()}
            alt_frequencies = {motif.ljust(max_length, '-') : freq for motif, freq in alt_frequencies.items()}
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
            pyfamsa.Sequence(f">{i} {freq}".encode(), seq.encode())
            for i, (seq, freq) in enumerate(sorted_ref.items())
        ]
        msa = aligner.align(sequences)
        return {
            seq.sequence.decode(): float(seq.id.decode().split(" ")[1])
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
                seq.sequence.decode(): float(seq.id.decode().split(" ")[1])
                for seq in msa if seq.id.decode().startswith('ref_')
            }
            alt_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[1])
                for seq in msa if seq.id.decode().startswith('alt_')
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
                seq.sequence.decode(): float(seq.id.decode().split(" ")[1])
                for seq in total_msa if seq.id.decode().startswith('ref_')
            }
            alt_frequencies_aligned = {
                seq.sequence.decode(): float(seq.id.decode().split(" ")[1])
                for seq in total_msa if seq.id.decode().startswith('alt_')
            }
            return ref_frequencies_aligned, alt_frequencies_aligned


def plot_logo(gapfill_adata: ad.AnnData,
               probe: str,
               groupby: str = None,
               group: str = None,
               compare_to: str = None,
               genotype_mode: Literal['genotype', 'raw'] = None,
               align: bool = True,
               threads: int = 1) -> tuple['logomaker.Logo', plt.Axes]:
    """
    Generate a logo plot for a single probe in the gapfill adata object.

    The final plot depends on the combination of input parameters given.

    - When alignments are disabled, sequences are padded to the same length with gaps naively.
    - When genotype_mode is set to 'genotype', the logo frequencies are plotted by the number of cells for each genotype.
    - When genotype_mode is set to 'raw', the logo frequencies are plotted by the total number of UMIs for each genotype.
    - If a compared group is provided, the logo frequencies are first normalized to the total frequencies per position,
        then the alt frequencies are subtracted from the ref frequencies.

    :param gapfill_adata: The gapfill adata object. If there is an obsm['genotype'] slot, called genotypes will be used.
        If not, raw probe gapfill frequencies will be used. Note that this can generate some unexpected results in this
        case due to rare genotypes captured with varying gap lengths.
    :param probe: The probe name.
    :param groupby: The groupby column to use for subsetting the adata object. If None, the entire adata object is used.
    :param group: A specific group to plot. If provided, this will subset the adata object to only include cells
    :param compare_to: If provided, a second group to compare the logo frequencies against.
    :param genotype_mode: The mode for genotypes. If 'genotype', requires that genotypes were previously called.
        the logo frequency will be plotted by the number of cells for each genotype. When 'raw', the logo frequency
        will be plotted by the total number of UMIs for each genotype. By default, we use 'genotype' if genotypes
        have been called, otherwise 'raw'.
    :param align: Whether to align the logos using pyFAMSA.
    :param threads: The number of threads to use for alignment. Default is 1.
    :return: The logo object and its matplotlib axes.
    """
    final_genotypes_available = 'genotype' in gapfill_adata.obsm
    if not final_genotypes_available:
        print("Warning: No genotypes called in gapfill_adata. Using raw probe frequencies.")

    # Subset the adata to the specified group
    if groupby is not None and group is not None:
        if groupby not in gapfill_adata.obs:
            raise ValueError(f"Groupby {groupby} not found in gapfill_adata.obs.")
        group_mask = gapfill_adata.obs[groupby] == group
        if compare_to is not None:
            compared_mask = gapfill_adata.obs[groupby] == compare_to
            compared_adata = gapfill_adata[compared_mask, :]
        else:
            compared_adata = None
        gapfill_adata = gapfill_adata[group_mask, :]
    else:
        compared_adata = None

    import logomaker

    ref_group_frequency_name, ref_genotype2count = _generate_genotype_frequencies(
        gapfill_adata,
        probe,
        final_genotypes_available,
        genotype_mode
    )

    if compared_adata is not None:
        ref_group_frequency_name += " (Normalized)"

        _, alt_genotype2count = _generate_genotype_frequencies(
            compared_adata,
            probe,
            final_genotypes_available,
            genotype_mode
        )
        ref_genotype2count, alt_genotype2count = _compute_alignments(
            ref_frequencies=ref_genotype2count,
            alt_frequencies=alt_genotype2count,
            align=align,
            threads=threads
        )
        # Prepare the data for logomaker (index = Position, columns = Nucleotides, Values = Frequencies)
        seq_length = max(len(logo) for logo in ref_genotype2count.keys())
        ref_data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        for logo, freq in ref_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in ref_data:
                    ref_data[nucleotide][i] += freq
        ref_data = pd.DataFrame(ref_data).set_index('relative_position')
        alt_data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        # Subtract from the alt frequencies
        for logo, freq in alt_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in alt_data:
                    alt_data[nucleotide][i] += freq
        alt_data = pd.DataFrame(alt_data).set_index('relative_position')

        # Normalize by total frequencies per position then subtract alt data from ref data
        ref_data = ref_data.div(ref_data.sum(axis=1), axis=0)
        alt_data = alt_data.div(alt_data.sum(axis=1), axis=0)
        data = ref_data - alt_data
    else:
        ref_genotype2count, _ = _compute_alignments(
            ref_frequencies=ref_genotype2count,
            alt_frequencies=None,
            align=align,
            threads=threads
        )
        # Prepare the data for logomaker (index = Position, columns = Nucleotides, Values = Frequencies)
        seq_length = max(len(logo) for logo in ref_genotype2count.keys())
        data = {
            'Relative Position (bp)': list(range(seq_length)),
            'A': [0. for _ in range(seq_length)],
            'C': [0. for _ in range(seq_length)],
            'G': [0. for _ in range(seq_length)],
            'T': [0. for _ in range(seq_length)],
        }
        for logo, freq in ref_genotype2count.items():
            for i, nucleotide in enumerate(logo):
                if nucleotide in data:
                    data[nucleotide][i] += freq
        data = pd.DataFrame(data).set_index('relative_position')

    # Now we need to plot the logo with logomaker
    logo = logomaker.Logo(
        data,
        shade_below=0.25,
        fade_below=0.25,
        stack_order='small_on_top',
    )

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(fmt='%d', anchor=0, rotation=45)

    logo.ax.set_xlabel('Relative Position (bp)')
    logo.ax.xaxis.set_ticks_position('none')
    logo.ax.xaxis.set_tick_params(pad=-1)

    logo.ax.set_ylabel(ref_group_frequency_name)

    return logo, logo.ax


def dendrogram(gapfill_adata: ad.AnnData, groupby: str, **kwargs):
    """
    Generate a dendrogram of the gapfills. Similar to dendrograms in sc.pl.dendrogram.
        Note, this requires sc.tl.dendrogram to be run first.
    :param gapfill_adata: The gapfill adata object.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.dendrogram.
    :return: The figure/axes.
    """
    return sc.pl.dendrogram(gapfill_adata, groupby, **kwargs)


def dotplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a dotplot of the gapfills for a single probe. Similar to dotplots in sc.pl.dotplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.dotplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.dotplot(gapfill_adata, var_names, groupby, **kwargs)


def tracksplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a tracksplot of the gapfills for a single probe. Similar to tracksplots in sc.pl.tracksplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.tracksplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.tracksplot(gapfill_adata, var_names, groupby, **kwargs)


def matrixplot(gapfill_adata: ad.AnnData, probe: str, groupby: str, **kwargs):
    """
    Generate a matrixplot of the gapfills for a single probe. Similar to matrixplots in sc.pl.matrixplot.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param groupby: The groupby column to use.
    :param kwargs: Arguments passed to sc.pl.matrixplot.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.matrixplot(gapfill_adata, var_names, groupby, **kwargs)


def violin(gapfill_adata: ad.AnnData, probe: str, **kwargs):
    """
    Generate a violin plot of the gapfills for a single probe. Similar to violin plots in sc.pl.violin.
    :param gapfill_adata: The gapfill adata object.
    :param probe: The probe name.
    :param kwargs: Arguments passed to sc.pl.violin.
    :return: The figure/axes.
    """
    if probe not in gapfill_adata.var['probe']:
        raise ValueError(f"Probe {probe} not found in gapfill_adata.")

    var_names = gapfill_adata.var_names[gapfill_adata.var['probe'] == probe].tolist()

    return sc.pl.violin(gapfill_adata, var_names, **kwargs)

# Genotyped-adata plots

def clustermap(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a clustermap of the genotypes. Similar to clustermaps in sc.pl.clustermap.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.clustermap.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    # Move the genotype to obs
    if 'genotype' in adata.obs:
        print("Warning: Overwriting existing genotype column in adata.obs.")

    adata.obs['genotype'] = adata.obsm['genotype'][genotype]

    return_val = sc.pl.clustermap(adata, **kwargs)

    # Drop the fake column
    adata.obs.drop(columns=['genotype'], inplace=True)

    return return_val


def tsne(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a t-SNE plot colored by the specified genotype.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.tsne.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    if 'genotype' in adata.obs or 'genotype_proportion' in adata.obs:
        print("Warning: Overwriting existing genotype and genotype_proportion columns in adata.obs.")

    # Add fake obs columns so that we may plot the genotype and its probability
    adata.obs['genotype'] = adata.obsm['genotype'][genotype]
    adata.obs['genotype_proportion'] = adata.obsm['genotype_proportion'][genotype]

    return_val = sc.pl.tsne(adata, color=['genotype', 'genotype_proportion'], **kwargs)

    # Drop the fake columns
    adata.obs.drop(columns=['genotype', 'genotype_proportion'], inplace=True)

    return return_val


def umap(adata: ad.AnnData, genotype: str, **kwargs):
    """
    Generate a UMAP plot colored by the specified genotype.
    :param adata: The adata object with genotypic information.
    :param genotype: The genotype to plot. Can be a single genotype or a list of genotypes.
    :param kwargs: Additional arguments to pass to sc.pl.umap.
    :return: The figure/axes.
    """
    _check_genotypes(adata)

    if genotype not in adata.obsm['genotype'].columns:
        raise ValueError(f"Genotype {genotype} not found in adata.")

    if 'genotype' in adata.obs or 'genotype_proportion' in adata.obs:
        print("Warning: Overwriting existing genotype and genotype_proportion columns in adata.obs.")

    # Add fake obs columns so that we may plot the genotype and its probability
    adata.obs['genotype'] = adata.obsm['genotype'][genotype]
    adata.obs['genotype_proportion'] = adata.obsm['genotype_proportion'][genotype]

    return_val = sc.pl.umap(adata, color=['genotype', 'genotype_proportion'], **kwargs)

    # Drop the fake columns
    adata.obs.drop(columns=['genotype', 'genotype_proportion'], inplace=True)

    return return_val
