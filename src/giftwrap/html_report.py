"""
giftwrap_html_report_v2.py
--------------------------
Redesigned HTML QC report — professional light theme, scientific instrument aesthetic.
Matches the visual authority of CellRanger web_summary.html.
"""

from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .utils import real_gapfill_mask

# Third-party web assets are vendored under giftwrap/assets/ and inlined into the
# report so it renders with no network access — these reports are routinely opened
# on offline laptops and cluster nodes where a CDN fetch would silently yield a
# page with no figures at all.
_ASSETS_DIR = Path(__file__).parent / "assets"

# Kept in sync with the vendored plotly.min.js; used only if that file is missing.
_PLOTLY_VERSION = "2.32.0"


def _read_asset(name: str) -> Optional[str]:
    """Return the text of a vendored asset, or None if it isn't available."""
    try:
        return (_ASSETS_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return None


def _to_list(arr) -> list:
    return np.asarray(arr).flatten().tolist()


def _saturation_curve(total_reads_layer, n_points: int = 100) -> tuple[list, list]:
    mat = np.asarray(
        total_reads_layer.todense() if hasattr(total_reads_layer, "todense") else total_reads_layer
    )
    # n_cells = number of barcodes (rows), used as denominator for "mean reads per cell"
    n_cells = mat.shape[0]
    arr = mat.flatten()
    arr = arr[arr > 0].astype(int)
    total = arr.sum()
    probs = arr / total  # loop-invariant — compute once, not per fraction
    fractions = np.linspace(0.05, 1.0, n_points)
    xs, ys = [], []
    rng = np.random.default_rng(42)
    for frac in fractions:
        target = int(total * frac)
        sampled = rng.multinomial(target, probs)
        sat = 1 - (sampled > 0).sum() / max(target, 1)
        xs.append(round(target / n_cells, 2))
        ys.append(round(float(sat), 4))
    return xs, ys


def _fig_sankey(fastq: dict, counts: dict) -> dict:
    label = [
        "All Reads", "Exact Matches", "Corrected LHS", "Corrected RHS", "Corrected Barcode",
        "Filtered", "No LHS", "No RHS", "No Cell Barcode", "No Constant Seq", "No Probe BC",
        "Mapped Reads",
    ]
    idx = {l: i for i, l in enumerate(label)}

    # One blue for the whole mapped side, one red for the whole filtered side.
    node_colors = [
        "#2563EB",  # All Reads — blue
        "#2563EB",  # Exact Matches
        "#2563EB",  # Corrected LHS
        "#2563EB",  # Corrected RHS
        "#2563EB",  # Corrected Barcode
        "#DC2626",  # Filtered — red
        "#DC2626",  # No LHS
        "#DC2626",  # No RHS
        "#DC2626",  # No Cell Barcode
        "#DC2626",  # No Constant Seq
        "#DC2626",  # No Probe BC
        "#2563EB",  # Mapped Reads — blue
    ]

    # Both halves read the same way: All Reads → reason (layer 2) → aggregate
    # (layer 3). Mapped reasons converge into "Mapped Reads"; filtered reasons
    # converge into "Filtered".
    raw_flows = [
        ("All Reads", "Exact Matches",     fastq.get("EXACT", 0)),
        ("All Reads", "Corrected LHS",     fastq.get("CORRECTED_LHS", 0)),
        ("All Reads", "Corrected RHS",     fastq.get("CORRECTED_RHS", 0)),
        ("All Reads", "Corrected Barcode", fastq.get("CORRECTED_BARCODE", 0)),
        ("Exact Matches",    "Mapped Reads", fastq.get("EXACT", 0)),
        ("Corrected LHS",    "Mapped Reads", fastq.get("CORRECTED_LHS", 0)),
        ("Corrected RHS",    "Mapped Reads", fastq.get("CORRECTED_RHS", 0)),
        ("Corrected Barcode","Mapped Reads", fastq.get("CORRECTED_BARCODE", 0)),
        ("All Reads", "No LHS",          fastq.get("FILTERED_NO_LHS", 0)),
        ("All Reads", "No RHS",          fastq.get("FILTERED_NO_RHS", 0)),
        ("All Reads", "No Cell Barcode", fastq.get("FILTERED_NO_CELL_BARCODE", 0)),
        ("All Reads", "No Constant Seq", fastq.get("FILTERED_NO_CONSTANT", 0)),
        ("All Reads", "No Probe BC",     fastq.get("FILTERED_NO_PROBE_BARCODE", 0)),
        ("No LHS",          "Filtered", fastq.get("FILTERED_NO_LHS", 0)),
        ("No RHS",          "Filtered", fastq.get("FILTERED_NO_RHS", 0)),
        ("No Cell Barcode", "Filtered", fastq.get("FILTERED_NO_CELL_BARCODE", 0)),
        ("No Constant Seq", "Filtered", fastq.get("FILTERED_NO_CONSTANT", 0)),
        ("No Probe BC",     "Filtered", fastq.get("FILTERED_NO_PROBE_BARCODE", 0)),
    ]
    flows = [(s, t, v) for s, t, v in raw_flows if v > 0]

    # Nodes on the filtered side; a link is pink if it touches one of these.
    # (Can't use idx >= 5: Mapped Reads is idx 11 but belongs to the mapped side.)
    filt_idx = {idx[n] for n in
                ("Filtered", "No LHS", "No RHS", "No Cell Barcode",
                 "No Constant Seq", "No Probe BC")}

    # Per-link percentage of the source node's total outgoing reads (P2.3).
    outgoing_total: dict = {}
    for s, _, v in flows:
        outgoing_total[s] = outgoing_total.get(s, 0) + v
    link_pct = [100.0 * v / outgoing_total[s] if outgoing_total.get(s) else 0.0
                for s, _, v in flows]

    return {
        "data": [{
            "type": "sankey", "orientation": "h",
            "arrangement": "snap",
            "node": {
                "pad": 15, "thickness": 18,
                "line": {"color": "rgba(0,0,0,0.1)", "width": 0.5},
                "label": label, "color": node_colors,
                "hovertemplate": "<b>%{label}</b><br>%{value:,} reads<extra></extra>",
            },
            "link": {
                "source": [idx[s] for s, _, _ in flows],
                "target": [idx[t] for _, t, _ in flows],
                "value":  [v       for _, _, v in flows],
                "customdata": link_pct,
                # Pink if either endpoint is on the filtered side, else blue.
                "color":  ["rgba(220,38,38,0.12)" if (idx[s] in filt_idx or idx[t] in filt_idx)
                           else "rgba(37,99,235,0.15)" for s, t, _ in flows],
                "hovertemplate": "%{source.label} → %{target.label}<br><b>%{value:,}</b> reads (%{customdata:.1f}% of %{source.label})<extra></extra>",
            },
        }],
        "layout": {
            "font": {"size": 11, "family": "Figtree, sans-serif", "color": "#374151"},
            "height": 420,
            "margin": {"l": 10, "r": 10, "t": 16, "b": 10},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
        }
    }


def _fig_barcode_rank(gapfill_adata) -> dict:
    umis = _to_list(gapfill_adata.X.sum(axis=1))
    umis_sorted = sorted(umis, reverse=True)
    ranks = list(range(1, len(umis_sorted) + 1))
    if len(umis_sorted) > 2000:
        # Prepend index 0 so the top (rank-1) barcode — the most informative
        # point of a knee plot — is never dropped by the log-spacing.
        indices = np.unique(np.concatenate(([0], np.round(
            np.logspace(0, np.log10(len(umis_sorted) - 1), 1000)).astype(int))))
        umis_sorted = [umis_sorted[i] for i in indices]
        ranks = [ranks[i] for i in indices]
    return {
        "data": [{"type": "scatter", "mode": "lines",
                  "x": ranks, "y": umis_sorted,
                  "line": {"color": "#1D4ED8", "width": 2},
                  "fill": "tozeroy", "fillcolor": "rgba(29,78,216,0.06)",
                  "hovertemplate": "Rank %{x}<br><b>%{y:,}</b> UMIs<extra></extra>"}],
        "layout": {
            "xaxis": {"title": "Barcode Rank", "type": "log", "showgrid": True},
            "yaxis": {"title": "UMI Counts", "type": "log", "showgrid": True},
            "height": 340,
            "margin": {"l": 60, "r": 20, "t": 16, "b": 50},
        }
    }


def _fig_saturation_curve(gapfill_adata) -> dict:
    try:
        from .analysis.tools import collapse_gapfills
        layer = collapse_gapfills(gapfill_adata).layers["total_reads"]
    except Exception:
        layer = gapfill_adata.layers.get("total_reads", gapfill_adata.X)
    xs, ys = _saturation_curve(layer)
    return {
        "data": [{"type": "scatter", "mode": "lines",
                  "x": xs, "y": ys,
                  "line": {"color": "#0369A1", "width": 2},
                  "hovertemplate": "Mean reads/cell: %{x}<br>Saturation: %{y:.1%}<extra></extra>"}],
        "layout": {
            "xaxis": {"title": "Mean Reads per Cell", "showgrid": True},
            "yaxis": {"title": "Sequencing Saturation", "tickformat": ".0%", "range": [0, 1], "showgrid": True},
            "height": 340,
            "margin": {"l": 70, "r": 20, "t": 16, "b": 50},
        }
    }


def _fig_pcr_histogram(probe_reads_file, filter_cutoff: int) -> dict:
    from .utils import maybe_gzip
    duplicates = []
    try:
        with maybe_gzip(probe_reads_file, "r") as f:
            first = True
            for line in f:
                if first: first = False; continue
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    duplicates.append(int(parts[5]))
    except Exception:
        pass
    if not duplicates:
        return {}
    log_dups = np.log10(np.array(duplicates, dtype=float) + 1)
    counts, edges = np.histogram(log_dups, bins=80)
    mid = ((edges[:-1] + edges[1:]) / 2).tolist()
    shapes = []
    if filter_cutoff > 0:
        xval = float(np.log10(filter_cutoff + 1))
        shapes.append({"type": "line", "x0": xval, "x1": xval,
                       "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": "#DC2626", "dash": "dash", "width": 1.5}})
    return {
        "data": [{"type": "bar", "x": mid, "y": counts.tolist(),
                  "marker": {"color": "#0EA5E9"},
                  "hovertemplate": "Reads/UMI ≈ %{x:.2f} (log₁₀)<br>UMIs: %{y:,}<extra></extra>"}],
        "layout": {
            "xaxis": {"title": "Reads per UMI (log₁₀ scale)", "showgrid": True},
            "yaxis": {"title": "# of UMIs", "showgrid": True},
            "shapes": shapes, "height": 340,
            "margin": {"l": 60, "r": 20, "t": 16, "b": 50},
        }
    }


def _fig_umis_per_cell(gapfill_adata) -> dict:
    umis = _to_list(gapfill_adata.X.sum(axis=1))
    counts, edges = np.histogram(umis, bins=80)
    mid = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {
        "data": [{"type": "bar", "x": mid, "y": counts.tolist(),
                  "marker": {"color": "#2563EB", "opacity": 0.8},
                  "hovertemplate": "UMIs: %{x:.0f}<br>Cells: %{y:,}<extra></extra>"}],
        "layout": {
            "xaxis": {"title": "UMIs per Cell", "showgrid": True},
            "yaxis": {"title": "# of Cells (log scale)", "type": "log", "showgrid": True},
            "height": 340, "margin": {"l": 60, "r": 20, "t": 16, "b": 50},
        }
    }


def _fig_cells_per_gapfill(gapfill_adata) -> dict:
    gapfill_adata = gapfill_adata[:, real_gapfill_mask(gapfill_adata)]
    cpg = _to_list((gapfill_adata.X > 0).sum(axis=0))
    counts, edges = np.histogram(cpg, bins=min(50, len(cpg)))
    mid = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {
        "data": [{"type": "bar", "x": mid, "y": counts.tolist(),
                  "marker": {"color": "#059669", "opacity": 0.8},
                  "hovertemplate": "%{y:,} variants detected in ~%{x:.0f} cells<extra></extra>"}],
        "layout": {
            "xaxis": {"title": "Number of cells a variant is detected in", "showgrid": True},
            "yaxis": {"title": "Number of gapfill variants (log)", "type": "log", "showgrid": True},
            "height": 340, "margin": {"l": 60, "r": 20, "t": 16, "b": 50},
        }
    }


def _fig_probe_boxplot(gapfill_adata, top_n: int = 15) -> dict:
    """Top gapfill variants (probe + gap sequence) ranked by cell prevalence.

    Per-cell UMI counts for a probe are near-binary (almost always 0 or 1), so a
    boxplot collapses to a flat line and conveys nothing. Instead, rank each
    variant column by the number of cells that carry it and draw a horizontal bar
    sized by that variant's prevalence (% of cells), so how common each genotype
    is reads at a glance.
    """
    if "probe" not in gapfill_adata.var.columns:
        return {}
    gapfill_adata = gapfill_adata[:, real_gapfill_mask(gapfill_adata)]
    probe_arr = gapfill_adata.var["probe"].values
    seq_col = "gapfill" if "gapfill" in gapfill_adata.var.columns else None
    seq_arr = (gapfill_adata.var[seq_col].values if seq_col
               else gapfill_adata.var_names.values)
    # One column = one (probe, gap sequence) variant; rank by the number of cells
    # detecting it (UMI >= 1), not total UMIs (P4.1).
    cells_per_variant = np.asarray((gapfill_adata.X > 0).sum(axis=0)).flatten()
    n_cells = gapfill_adata.shape[0]
    if n_cells <= 0:
        return {}
    # plotly draws the first y-category at the bottom, so reverse the descending
    # order to put the largest bar at the top of the list.
    order = np.argsort(cells_per_variant)[::-1][:top_n][::-1]
    counts = cells_per_variant[order]
    pcts = (counts / n_cells * 100.0)

    def _short(s, n=28):
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    labels = [f"{probe_arr[j]} · {_short(seq_arr[j])}" for j in order]
    customdata = [[str(probe_arr[j]), str(seq_arr[j]), int(cells_per_variant[j])]
                  for j in order]
    return {
        "data": [{
            "type": "bar", "orientation": "h",
            "x": pcts.tolist(), "y": labels,
            "customdata": customdata,
            "marker": {"color": "#1D4ED8", "opacity": 0.85},
            "text": [f"{p:.1f}%" for p in pcts],
            "textposition": "outside", "cliponaxis": False,
            "hovertemplate": ("<b>%{customdata[0]}</b><br>gap: %{customdata[1]}"
                              "<br>%{customdata[2]:,} cells (%{x:.2f}%)<extra></extra>"),
        }],
        "layout": {
            "xaxis": {"title": "Cells carrying variant (% of cells)", "showgrid": True},
            "yaxis": {"automargin": True, "tickfont": {"size": 10}},
            "height": 460,
            "margin": {"l": 200, "r": 48, "t": 16, "b": 50},
            "showlegend": False,
        }
    }


def _fig_reads_per_gapfill(gapfill_adata) -> dict:
    if "total_reads" not in gapfill_adata.layers:
        return {}
    gapfill_adata = gapfill_adata[:, real_gapfill_mask(gapfill_adata)]
    layer = gapfill_adata.layers["total_reads"]
    reads = np.asarray(_to_list(
        layer.todense().__array__().sum(0) if hasattr(layer, "todense") else np.asarray(layer).sum(0)
    ))
    probe_arr = (gapfill_adata.var["probe"].values if "probe" in gapfill_adata.var.columns
                 else gapfill_adata.var_names.values)
    # argsort (not sorted()) so each bar keeps the probe identity it came from,
    # through both the descending sort and the log-spaced subsampling below.
    order = np.argsort(reads)[::-1]
    reads_sorted = reads[order]
    labels_sorted = [str(p) for p in probe_arr[order]]
    detected = reads_sorted[reads_sorted > 0]
    median_reads = float(np.median(detected)) if detected.size else 0.0
    # One bar per gapfill variant; panels can have hundreds of thousands, so
    # subsample the rank curve (log-spaced to preserve the head) for display.
    n_total = len(reads_sorted)
    ranks = list(range(n_total))
    if n_total > 2000:
        idx = np.unique(np.concatenate(([0], np.round(
            np.geomspace(1, n_total - 1, 1000)).astype(int))))
        reads_sorted = reads_sorted[idx]
        labels_sorted = [labels_sorted[i] for i in idx]
        ranks = [int(i) for i in idx]
    return {
        "data": [
            {"type": "bar", "x": ranks, "y": reads_sorted.tolist(),
             "customdata": labels_sorted,
             "marker": {"color": "#0369A1"},
             "hovertemplate": "<b>%{customdata}</b><br>Gapfill rank %{x}<br>Reads: %{y:,}<extra></extra>"},
            {"type": "scatter", "mode": "lines",
             "x": [0, n_total - 1], "y": [median_reads, median_reads],
             "line": {"color": "#DC2626", "dash": "dash", "width": 1.5},
             "name": f"Median ({median_reads:,.0f} reads)"},
        ],
        "layout": {
            "xaxis": {"title": "Gapfill Rank", "showticklabels": False},
            "yaxis": {"title": "Reads (log scale)", "type": "log", "showgrid": True},
            "height": 340, "margin": {"l": 70, "r": 20, "t": 16, "b": 50},
            "showlegend": True,
        }
    }


def _fig_wta_correlation(gapfill_adata, adata) -> list[dict]:
    if adata is None:
        return []
    # GIFTwrap barcodes embed a plex_seq before the suffix (CELLBC{plex_seq}-1)
    # while CellRanger barcodes are plain 16bp (CELLBC-1). Strip the plex_seq when
    # prefix lengths differ so the two sets line up — mirrors the PDF path in
    # step5_summarize_counts.py. Without this, indexing adata by GIFTwrap
    # barcodes selects the wrong cells (or raises).
    gw_bcs = gapfill_adata.obs.index
    cr_bcs = gw_bcs
    if len(gw_bcs) > 0 and len(adata.obs_names) > 0:
        gw_plen = len(gw_bcs[0].rsplit("-", 1)[0])
        cr_plen = len(adata.obs_names[0].rsplit("-", 1)[0])
        if gw_plen > cr_plen:
            plex_len = gw_plen - cr_plen
            cr_bcs = pd.Index([
                bc.rsplit("-", 1)[0][:-plex_len] + "-" + bc.rsplit("-", 1)[1]
                for bc in gw_bcs
            ])
    adata = adata[cr_bcs, :]
    if "gene" not in gapfill_adata.var.columns:
        gapfill_adata.var["gene"] = [p.replace(" ", "_").split("_")[0] for p in gapfill_adata.var["probe"]]
    adata.var["gene"] = adata.var_names.values
    matched = set(gapfill_adata.var["gene"]) & set(adata.var["gene"])
    if not matched:
        return []
    probes = gapfill_adata.var["probe"].unique().tolist()
    n = len(probes)
    wta_sc  = np.zeros((gapfill_adata.shape[0], n))
    gap_sc  = np.zeros((gapfill_adata.shape[0], n))
    genes = []
    for i, probe in enumerate(probes):
        gene = gapfill_adata.var["gene"][gapfill_adata.var["probe"] == probe].values[0]
        genes.append(str(gene))
        if gene in adata.var_names:
            wta_sc[:, i] = adata[:, adata.var_names == gene].X.toarray().flatten()
        gap_sc[:, i] = gapfill_adata[:, gapfill_adata.var["probe"] == probe].X.toarray().sum(axis=1).flatten()
    sc_rho, sc_p = spearmanr(gap_sc.flatten(), wta_sc.flatten())
    # [probe, gene] per column i, for hover identity — aligned with column i of
    # gap_sc/wta_sc so it can be tiled/indexed the same way as those matrices.
    pair_labels = np.array([[str(probes[i]), genes[i]] for i in range(n)], dtype=object)

    def _scatter(x, y, title, xlabel, ylabel, customdata, max_points: Optional[int] = None):
        # The correlation (title) is always computed on the full data by the
        # caller; here we only thin what gets embedded as scatter markers so the
        # report does not balloon to hundreds of MB. The all-zero (0,0) pairs are
        # overplotted and uninformative, so drop them first, then cap.
        x = np.asarray(x); y = np.asarray(y)
        customdata = np.asarray(customdata, dtype=object)
        if max_points is not None:
            nz = (x > 0) | (y > 0)
            x, y, customdata = x[nz], y[nz], customdata[nz]
            if x.size > max_points:
                sel = np.random.default_rng(42).choice(x.size, max_points, replace=False)
                x, y, customdata = x[sel], y[sel], customdata[sel]
        return {
            "data": [{"type": "scatter", "mode": "markers",
                      "x": x.tolist(), "y": y.tolist(),
                      "customdata": customdata.tolist(),
                      "marker": {"size": 4, "color": "#2563EB", "opacity": 0.5},
                      "hovertemplate": ("<b>%{customdata[0]}</b> (%{customdata[1]})<br>" +
                                        xlabel + ": %{x}<br>" + ylabel + ": %{y}<extra></extra>")}],
            "layout": {
                "title": {"text": title, "font": {"size": 12}},
                "xaxis": {"title": xlabel, "showgrid": True},
                "yaxis": {"title": ylabel, "showgrid": True},
                "height": 340, "margin": {"l": 70, "r": 20, "t": 50, "b": 50},
            }
        }

    # Flattening (n_cells, n) matrices in C order visits column i (probe i) for
    # every row before moving to the next row, so tiling pair_labels n_cells
    # times lines each label up with the same flat index as gap_sc/wta_sc.
    sc_customdata = np.tile(pair_labels, (gapfill_adata.shape[0], 1))
    figs = [_scatter(gap_sc.flatten(), wta_sc.flatten(),
                     f"Single-cell: Spearman ρ = {sc_rho:.2f} (p = {sc_p:.2e})",
                     "Gapfill probe UMIs (per cell)",
                     "WTA gene UMIs (per cell)",
                     sc_customdata,
                     max_points=50_000)]
    pb_gap = gap_sc.sum(axis=0)
    pb_wta = wta_sc.sum(axis=0)
    pb_rho, pb_p = spearmanr(pb_gap, pb_wta)
    figs.append(_scatter(pb_gap, pb_wta,
                         f"Pseudobulk: Spearman ρ = {pb_rho:.2f} (p = {pb_p:.2e})",
                         "Gapfill probe UMIs (summed across cells)",
                         "WTA gene UMIs (summed across cells)",
                         pair_labels))
    return figs


def _build_hero_metrics(fastq: dict, counts: dict, gapfill_adata=None) -> list[dict]:
    total = fastq.get("TOTAL_READS", 0)
    probe = fastq.get("PROBE_CONTAINING_READS", 0)
    pct_mapped = 100 * probe / total if total else 0
    sat = float(counts.get("SEQUENCING_SATURATION", 0)) * 100
    cells_w_probes = int(counts.get("GAPFILL_CONTAINING_CELLS", 0))
    est_cells = int(counts.get("TOTAL_CELLS", 0))
    # The detection rate is only meaningful when TOTAL_CELLS comes from a real
    # cell-calling denominator (e.g. WTA/cellranger barcodes). When no separate
    # cell list is provided, TOTAL_CELLS == GAPFILL_CONTAINING_CELLS and the
    # ratio is a meaningless 100% — show n/a instead (P2.6).
    has_real_total = est_cells > cells_w_probes
    if has_real_total:
        detection_rate_val = f"{cells_w_probes / est_cells:.1%}"
        detection_rate_sub = f"{cells_w_probes:,} of {est_cells:,} estimated"
    else:
        detection_rate_val = "n/a"
        detection_rate_sub = "no separate cell-calling denominator provided"

    encountered = int(fastq.get("PROBES_ENCOUNTERED", 0))
    possible = int(fastq.get("POSSIBLE_PROBES", 0))
    probes_pct = f" ({100 * encountered / possible:.1f}%)" if possible else ""

    if gapfill_adata is not None:
        probes_per_cell = float(np.asarray((gapfill_adata.X > 0).sum(axis=1)).flatten().mean())
    else:
        probes_per_cell = 0.0

    return [
        {"label": "Cells with Probes",          "value": f"{cells_w_probes:,}",                           "sub": "gapfill-containing barcodes",
         "tip": "Number of called cells with at least one detected gapfill probe (UMI > 0)."},
        {"label": "Mean UMIs / Cell",          "value": f"{counts.get('UMIS_PER_CELL_MEAN', 0):.1f}",   "sub": "gapfill UMIs",
         "tip": "Average number of unique gapfill molecules (UMIs) detected per called cell."},
        {"label": "Probes / Cell",             "value": f"{probes_per_cell:.1f}",                        "sub": "mean gapfills detected per cell",
         "tip": "Average number of distinct gapfill probes detected per called cell. A cell can have several UMIs from one probe, so this is always ≤ Mean UMIs / Cell."},
        {"label": "Encountered / Possible Probes", "value": f"{encountered:,} / {possible:,}",            "sub": f"probes detected / panel size{probes_pct}",
         "tip": "Distinct probes detected in this dataset versus the total number of probes in the reference panel."},
        {"label": "Cells with Probes / Total Cells", "value": detection_rate_val,                          "sub": detection_rate_sub,
         "tip": "Fraction of all called cells that contain at least one gapfill probe. Requires a separate cell-calling denominator (e.g. WTA/cellranger barcodes)."},
        {"label": "Total Reads",               "value": f"{int(total):,}",                              "sub": "sequenced read pairs",
         "tip": "Total input read pairs before any filtering."},
        {"label": "Reads Mapped",              "value": f"{pct_mapped:.1f}%",                           "sub": "probe-containing reads",
         "tip": "Percentage of input reads where a gapfill probe was successfully identified and assigned to a cell barcode."},
        {"label": "Sequencing Saturation",     "value": f"{sat:.1f}%",                                  "sub": "PCR duplicate fraction",
         "tip": "Fraction of reads that are PCR duplicates (already-seen UMIs). Low values mean deeper sequencing would yield more unique UMIs."},
        {"label": "Total UMIs",                "value": f"{int(counts.get('TOTAL_UMIS', 0)):,}",        "sub": "across all cells",
         "tip": "Sum of all gapfill UMI counts across every cell."},
    ]


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GIFTwrap QC Report — {plex}</title>
{fonts_tag}
{plotly_tag}
<style>
:root {{
  --bg:         #F8FAFC;
  --surface:    #FFFFFF;
  --surface2:   #F1F5F9;
  --border:     #E2E8F0;
  --border2:    #CBD5E1;
  --text:       #0F172A;
  --text2:      #475569;
  --text3:      #94A3B8;
  --accent:     #2563EB;
  --accent-lt:  #DBEAFE;
  --success:    #059669;
  --success-lt: #D1FAE5;
  --warn:       #D97706;
  --warn-lt:    #FEF3C7;
  --mono: "JetBrains Mono", "Fira Code", monospace;
  --sans: "Figtree", "Segoe UI", system-ui, sans-serif;
  --radius: 8px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}

/* ── Top bar ── */
.topbar {{
  background: #1E293B;
  height: 44px;
  display: flex;
  align-items: center;
  padding: 0 32px;
  gap: 16px;
}}
.topbar-brand {{
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  color: #F1F5F9;
  letter-spacing: 0.02em;
}}
.topbar-brand span {{ color: #38BDF8; }}
.topbar-sep {{ color: #475569; font-size: 11px; }}
.topbar-meta {{ font-size: 12px; color: #94A3B8; }}
.topbar-right {{ margin-left: auto; font-size: 12px; color: #94A3B8; font-family: var(--mono); }}

/* ── Run info header ── */
.run-header {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 24px 32px 20px;
}}
.run-header h1 {{
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.3px;
  margin-bottom: 4px;
}}
.run-header .run-meta {{
  font-size: 12px;
  color: var(--text3);
  display: flex;
  gap: 20px;
}}
.run-header .run-meta span {{ display: flex; gap: 5px; align-items: center; }}
.run-header .run-meta b {{ color: var(--text2); font-weight: 500; }}

/* ── Alert strip ── */
.alert-strip {{
  background: var(--success-lt);
  border-bottom: 1px solid #A7F3D0;
  padding: 10px 32px;
  font-size: 13px;
  color: #065F46;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.alert-strip svg {{ flex-shrink: 0; }}

/* ── Nav tabs ── */
nav {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  display: flex;
  gap: 0;
  overflow-x: auto;
  position: sticky;
  top: 0;
  z-index: 50;
  box-shadow: var(--shadow);
}}
nav button {{
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text2);
  cursor: pointer;
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 500;
  padding: 12px 18px;
  white-space: nowrap;
  transition: color 0.12s, border-color 0.12s;
}}
nav button:hover {{ color: var(--text); }}
nav button.active {{
  color: var(--accent);
  border-bottom-color: var(--accent);
  font-weight: 600;
}}

/* ── Main ── */
main {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 28px 32px 60px;
}}

.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

/* ── Section label ── */
.section-label {{
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text3);
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}

/* ── Hero metrics ── */
.hero-metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}}
.hero-metric {{
  background: var(--surface);
  padding: 20px 22px;
  transition: background 0.12s;
}}
.hero-metric:hover {{ background: #F8FAFC; }}
.hero-metric .val {{
  font-size: 28px;
  font-weight: 700;
  color: var(--accent);
  font-family: var(--mono);
  letter-spacing: -1px;
  line-height: 1.1;
  margin-bottom: 4px;
}}
.hero-metric .lbl {{
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 2px;
}}
.hero-metric .sub {{
  font-size: 11px;
  color: var(--text3);
}}

/* ── Card ── */
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.card-header {{
  padding: 14px 18px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.card-title {{
  font-size: 12px;
  font-weight: 600;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.card-body {{
  padding: 12px;
  min-height: 0;
}}
.card-body > div[id^="plot-"] {{
  min-height: 340px;
}}

/* ── Plot grids ── */
.plot-2col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}}
.plot-1col {{
  margin-bottom: 16px;
}}
@media (max-width: 900px) {{
  .plot-2col {{ grid-template-columns: 1fr; }}
  .hero-metrics {{ grid-template-columns: repeat(2, 1fr); }}
  main {{ padding: 16px; }}
}}

/* ── Metrics table ── */
.metrics-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.metrics-table thead th {{
  background: var(--surface2);
  padding: 9px 14px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text3);
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
.metrics-table tbody td {{
  padding: 9px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}}
.metrics-table tbody td:last-child {{
  font-family: var(--mono);
  font-size: 12px;
  color: var(--accent);
  text-align: right;
}}
.metrics-table tbody tr:hover td {{ background: var(--surface2); }}
.metrics-table tbody tr:last-child td {{ border-bottom: none; }}

/* ── Footer ── */
footer {{
  border-top: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text3);
  background: var(--surface);
}}
footer .brand {{ font-family: var(--mono); font-weight: 500; color: var(--accent); }}

/* ── Help tooltips ── */
.help-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--text3);
  font-size: 9px;
  font-weight: 700;
  cursor: help;
  margin-left: 5px;
  vertical-align: middle;
  line-height: 1;
  flex-shrink: 0;
}}
#tooltip-box {{
  position: fixed;
  background: #1E293B;
  color: #F1F5F9;
  font-size: 12px;
  line-height: 1.55;
  padding: 9px 13px;
  border-radius: 6px;
  max-width: 300px;
  white-space: pre-line;
  z-index: 9999;
  pointer-events: none;
  display: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  font-family: var(--sans);
  font-weight: 400;
}}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <span class="topbar-brand"><span>GIFT</span>wrap QC</span>
  <span class="topbar-sep">|</span>
  <span class="topbar-meta">Gapfill Integrated Feature-level Transcriptomic Workup &amp; Report</span>
  <span class="topbar-right">plex: {plex}</span>
</div>

<!-- Run header -->
<div class="run-header">
  <h1>{plex_display}</h1>
  <div class="run-meta">
    <span><b>Generated</b> {timestamp}</span>
    <span><b>Pipeline</b> <a href="https://doi.org/10.64898/2026.04.11.717967" target="_blank" rel="noopener">GIFTwrap</a></span>
    <span><b>Chemistry</b> 10x Flex</span>
    {ilab_meta_span}
    {sample_meta_span}
    {metrics_download}
  </div>
</div>

<!-- Nav -->
<nav>
  <button class="active" onclick="showTab('overview', this)">Summary</button>
  <button onclick="showTab('cells', this)">Cell QC</button>
  <button onclick="showTab('gapfills', this)">Gapfill Analysis</button>
  {wta_tab_btn}
  <button onclick="showTab('metrics', this)">Metrics</button>
</nav>

<main>

<!-- SUMMARY TAB -->
<div id="tab-overview" class="tab-panel active">
  <div class="hero-metrics">
    {hero_html}
  </div>
  <div class="plot-2col">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Read Processing Flow</span>
        <span class="help-btn" data-tip="Exact Match: probe + barcode matched the reference with no errors
Corrected LHS/RHS: 1–2 errors in a probe arm fixed by the aligner
Corrected Barcode: 1 barcode mismatch corrected against the Flex whitelist
No Constant Seq: spacer between barcode and probe was missing
No Cell Barcode: barcode unrecognized even after correction
No LHS/RHS: a probe arm was absent, so the probe can't be located
No Probe BC: gapfill not found in the reference panel">?</span>
      </div>
      <div class="card-body">
        <div id="plot-sankey-overview"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Barcode Rank Plot (Knee Plot)</span>
        <span class="help-btn" data-tip="Each barcode ranked by its total gapfill UMI count (both axes log scale). Real cells cluster on the left as a flat plateau (many UMIs); empty droplets and background noise trail off on the right (few UMIs).
The knee between them is where the cell caller draws the line between cells and noise. A sharp knee means clean cell/background separation; a gradual slope may suggest ambient RNA contamination or many low-RNA cells.">?</span>
      </div>
      <div class="card-body">
        <div id="plot-barcode-rank-overview"></div>
      </div>
    </div>
  </div>
  <div class="plot-2col">
    <div class="card">
      <div class="card-header">
        <span class="card-title">PCR Duplicate Distribution</span>
        <span class="help-btn" data-tip="Distribution of reads per UMI (PCR duplicate level), log scale. Each UMI is one original molecule; bar height is how many UMIs were sequenced at that duplication level.
Shifted right = many reads are PCR duplicates (high saturation); near 1 = most reads are unique molecules, so deeper sequencing would still recover new UMIs.">?</span>
      </div>
      <div class="card-body"><div id="plot-pcr"></div></div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">Sequencing Saturation</span></div>
      <div class="card-body"><div id="plot-saturation"></div></div>
    </div>
  </div>
</div>

<!-- CELL QC TAB -->
<div id="tab-cells" class="tab-panel">
  <div class="plot-1col">
    <div class="card">
      <div class="card-header"><span class="card-title">UMIs per Cell</span></div>
      <div class="card-body"><div id="plot-umis-per-cell"></div></div>
    </div>
  </div>
  <div class="plot-2col">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Detection breadth of gapfill variants</span>
        <span class="help-btn" data-tip="How many cells each distinct gapfill variant is detected in (x-axis), counted across all variants (y-axis, log scale).

        GOOD: variants detected across many cells (mass toward the right). Broadly-shared gapfills are reproducible and likely real genotypes.

        BAD: nearly everything piled at 1–2 cells with nothing broad. Single-cell variants are usually PCR/sequencing artifacts; a long singleton tail is normal, but without some broadly-detected variants there's no reproducible signal.">?</span>
      </div>
      <div class="card-body"><div id="plot-cells-per-gapfill"></div></div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Supporting Reads per Gapfill</span>
        <span class="help-btn" data-tip="Gapfill Rank orders every detected gapfill from most-supported to least-supported by total read count — rank 1 is the best-supported gapfill, and the curve falls off toward the rarely-observed ones on the right.
The y-axis (log scale) is the number of supporting reads. A long flat head followed by a steep drop indicates a few dominant, well-supported gapfills.">?</span>
      </div>
      <div class="card-body"><div id="plot-reads-per-gapfill"></div></div>
    </div>
  </div>
</div>

<!-- GAPFILL TAB -->
<div id="tab-gapfills" class="tab-panel">
  <div class="plot-1col">
    <div class="card">
      <div class="card-header"><span class="card-title">Top Gapfill Variants by Cell Prevalence</span></div>
      <div class="card-body"><div id="plot-probe-boxplot"></div></div>
    </div>
  </div>
</div>

<!-- WTA TAB -->
{wta_tab_html}

<!-- FULL METRICS TAB -->
<div id="tab-metrics" class="tab-panel">
  <div class="plot-2col">
    <div class="card">
      <div class="card-header"><span class="card-title">Counts Summary</span></div>
      <div class="card-body">
        <table class="metrics-table">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{counts_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">FASTQ Metrics</span></div>
      <div class="card-body">
        <table class="metrics-table">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{fastq_rows}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>

</main>

<footer>
  <span class="brand">GIFTwrap</span>
  <span>Generated {timestamp}</span>
</footer>

<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
  const panel = document.getElementById('tab-' + id);
  panel.classList.add('active');
  btn.classList.add('active');
  panel.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
}}

const cfg = {{
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d','select2d','zoom2d'],
  toImageButtonOptions: {{format: 'png', scale: 2}}
}};

// Light theme applied to all plots
const theme = {{
  autosize: true,
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: {{color: '#374151', family: 'Figtree, system-ui, sans-serif', size: 12}},
  xaxis: {{gridcolor: '#F1F5F9', linecolor: '#E2E8F0', zerolinecolor: '#E2E8F0',
           tickfont: {{size: 11}}, titlefont: {{size: 12, color: '#475569'}}}},
  yaxis: {{gridcolor: '#F1F5F9', linecolor: '#E2E8F0', zerolinecolor: '#E2E8F0',
           tickfont: {{size: 11}}, titlefont: {{size: 12, color: '#475569'}}}},
}};

function plot(id, fig) {{
  if (!fig || !fig.data) return;
  const layout = Object.assign({{}}, theme, fig.layout);
  if (layout.xaxis) layout.xaxis = Object.assign({{}}, theme.xaxis, fig.layout.xaxis || {{}});
  if (layout.yaxis) layout.yaxis = Object.assign({{}}, theme.yaxis, fig.layout.yaxis || {{}});
  if (layout.title) layout.title = Object.assign({{font: {{size: 13, color: '#374151', family: 'Figtree, sans-serif'}}}}, fig.layout.title || {{}});
  Plotly.newPlot(id, fig.data, layout, cfg);
}}

const FIGS = {figs_json};

plot('plot-sankey-overview',       FIGS.sankey);
plot('plot-barcode-rank-overview', FIGS.barcode_rank);
plot('plot-saturation',      FIGS.saturation);
plot('plot-pcr',             FIGS.pcr);
plot('plot-umis-per-cell',   FIGS.umis_per_cell);
plot('plot-cells-per-gapfill', FIGS.cells_per_gapfill);
plot('plot-probe-boxplot',   FIGS.probe_boxplot);
plot('plot-reads-per-gapfill', FIGS.reads_per_gapfill);
{wta_plot_js}

// Help tooltips
const _tt = document.createElement('div');
_tt.id = 'tooltip-box';
document.body.appendChild(_tt);
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mouseenter', () => {{
    _tt.textContent = el.dataset.tip;
    _tt.style.display = 'block';
    const r = el.getBoundingClientRect();
    _tt.style.left = Math.max(8, Math.min(r.left, window.innerWidth - _tt.offsetWidth - 8)) + 'px';
    const above = r.top - _tt.offsetHeight - 8;
    _tt.style.top = (above < 8 ? r.bottom + 8 : above) + 'px';
  }});
  el.addEventListener('mouseleave', () => {{ _tt.style.display = 'none'; }});
}});
</script>
</body>
</html>
"""


def make_html_report(
    output_file: Path,
    gapfill_adata,
    adata,
    fastq_metrics_path: Path,
    counts_metrics_path: Path,
    probe_reads_path: Path,
    filter_cutoff: int = 0,
    sample_id: Optional[str] = None,
    ilab: Optional[str] = None,
    plex_id: Optional[str] = None,
) -> None:
    import datetime
    if plex_id is None:
        parts = output_file.name.split(".")
        plex_id = parts[1] if len(parts) > 2 else output_file.stem

    fastq: dict = {}
    counts: dict = {}
    try:
        fastq = pd.read_table(fastq_metrics_path).set_index("metric")["value"].to_dict()
    except Exception as e:
        print(f"  Warning: could not read fastq_metrics.tsv: {e}")
    try:
        counts = pd.read_table(counts_metrics_path).set_index("statistic")["value"].to_dict()
    except Exception as e:
        print(f"  Warning: could not read summary.tsv: {e}")

    sankey_fig          = _fig_sankey(fastq, counts)
    barcode_rank_fig    = _fig_barcode_rank(gapfill_adata)
    saturation_fig      = _fig_saturation_curve(gapfill_adata)
    pcr_fig             = _fig_pcr_histogram(probe_reads_path, filter_cutoff)
    umis_fig            = _fig_umis_per_cell(gapfill_adata)
    cpg_fig             = _fig_cells_per_gapfill(gapfill_adata)
    probe_box_fig       = _fig_probe_boxplot(gapfill_adata)
    reads_gapfill_fig   = _fig_reads_per_gapfill(gapfill_adata)
    wta_figs            = _fig_wta_correlation(gapfill_adata, adata)

    all_figs = {
        "sankey": sankey_fig, "barcode_rank": barcode_rank_fig,
        "saturation": saturation_fig, "pcr": pcr_fig,
        "umis_per_cell": umis_fig, "cells_per_gapfill": cpg_fig,
        "probe_boxplot": probe_box_fig, "reads_per_gapfill": reads_gapfill_fig,
    }
    if wta_figs:
        all_figs["wta_sc"] = wta_figs[0]
        if len(wta_figs) > 1:
            all_figs["wta_pb"] = wta_figs[1]

    wta_tab_btn  = '<button onclick="showTab(\'wta\', this)">WTA Correlation</button>' if wta_figs else ""
    wta_tab_html = """
<div id="tab-wta" class="tab-panel">
  <div class="plot-2col">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Single-cell Correlation</span>
        <span class="help-btn" data-tip="One point per cell × probe: gapfill UMIs (x) vs WTA expression (y) for the same gene in the same cell. Tests whether gapfill tracks WTA cell-by-cell — the stringent check, naturally noisy due to per-cell dropout. A positive Spearman ρ means probes report the right signal at single-cell resolution.">?</span>
      </div>
      <div class="card-body"><div id="plot-wta-sc"></div></div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Pseudobulk Correlation</span>
        <span class="help-btn" data-tip="One point per probe, counts summed across all cells: total gapfill (x) vs total WTA (y). Tests whether gapfill tracks overall gene abundance with single-cell noise averaged out — usually a much higher ρ. Confirms probe design and quantification are sound at the assay level.">?</span>
      </div>
      <div class="card-body"><div id="plot-wta-pb"></div></div>
    </div>
  </div>
</div>""" if wta_figs else ""
    wta_plot_js = "plot('plot-wta-sc', FIGS.wta_sc);\nplot('plot-wta-pb', FIGS.wta_pb);" if wta_figs else ""

    hero = _build_hero_metrics(fastq, counts, gapfill_adata)

    def _hero_lbl(m: dict) -> str:
        tip = m.get("tip", "").replace('"', "&quot;")
        help_html = f' <span class="help-btn" data-tip="{tip}">?</span>' if tip else ""
        return f'{m["label"]}{help_html}'

    hero_html = "".join(
        f'<div class="hero-metric"><div class="val">{m["value"]}</div>'
        f'<div class="lbl">{_hero_lbl(m)}</div>'
        f'<div class="sub">{m["sub"]}</div></div>'
        for m in hero
    )

    _counts_help = {
        "TOTAL_CELLS":                   "Number of barcodes called as real cells (at least 1 gapfill UMI).",
        "TOTAL_UMIS":                    "Sum of all gapfill UMI counts across every cell.",
        "GAPFILL_CONTAINING_CELLS":      "Cells with at least one detected gapfill variant. Equals TOTAL_CELLS unless a separate cell barcode list was provided.",
        "UMIS_PER_CELL_MEAN":            "Average gapfill UMI count per cell.",
        "UMIS_PER_CELL_MEDIAN":          "Median gapfill UMI count per cell. More robust than the mean for skewed distributions.",
        "UMIS_PER_CELL_STD":             "Standard deviation of UMIs per cell. High values indicate uneven probe detection across the population.",
        "UMIS_PER_CELL_MIN":             "Minimum UMIs seen in any called cell (can be 0 if the cell has no gapfill reads).",
        "UMIS_PER_CELL_MIN_EXCLUDING_ZERO": "Minimum UMIs among cells that have at least 1 count. Useful when many cells have 0.",
        "UMIS_PER_CELL_MAX":             "Maximum UMIs seen in any single cell.",
        "CELLS_PER_GAPFILL_MEAN":        "Average number of cells that carry each distinct gapfill variant.",
        "CELLS_PER_GAPFILL_MEDIAN":      "Median cells per gapfill. A high value suggests one dominant or clonal variant.",
        "CELLS_PER_GAPFILL_STD":         "Spread in how widely gapfills are distributed across cells.",
        "CELLS_PER_GAPFILL_MIN":         "The rarest gapfill — detected in this many cells.",
        "CELLS_PER_GAPFILL_MAX":         "The most common gapfill — detected in this many cells.",
        "SEQUENCING_SATURATION":         "Fraction of reads that are PCR duplicates (already-seen UMIs). 0 = no duplicates, 1 = fully saturated. Low values mean deeper sequencing would yield more unique UMIs.",
    }
    _fastq_help = {
        "PROBE_CONTAINING_READS":        "Reads where a gapfill probe sequence was successfully identified and assigned to a cell barcode.",
        "POSSIBLE_PROBES":               "Total number of distinct probe sequences in the reference panel file.",
        "PROBES_ENCOUNTERED":            "Number of distinct probes actually detected in this dataset (out of all possible probes).",
        "TOTAL_READS":                   "Total input read pairs before any filtering.",
        "FILTERED_NO_CONSTANT":          "Reads discarded because the constant flanking sequence (spacer between barcode and probe) could not be found. Often indicates an adapter trimming or chemistry mismatch.",
        "CORRECTED_BARCODE":             "Reads where the cell barcode had exactly 1 mismatch corrected against the 10x Flex whitelist.",
        "CORRECTED_LHS":                 "Reads where the left-hand side anchor sequence had sequencing errors that were corrected.",
        "FILTERED_NO_CELL_BARCODE":      "Reads discarded because the barcode did not match any known cell, even after correction attempts.",
        "EXACT":                         "Reads that matched the reference exactly — no corrections needed for barcode or probe.",
        "FILTERED_NO_RHS":               "Reads discarded because the right-hand side flanking sequence (downstream of the probe) was absent or unrecognizable.",
        "CORRECTED_RHS":                 "Reads where the right-hand side anchor sequence had sequencing errors that were corrected.",
        "FILTERED_NO_LHS":               "Reads discarded because the left-hand side anchor sequence was absent.",
        "FILTERED_NO_PROBE_BARCODE":     "Reads discarded because the gapfill sequence was not found in the reference panel.",
    }

    def _rows(d: dict, help_dict: dict = {}) -> str:
        rows = []
        for k, v in d.items():
            if isinstance(v, (float, np.floating)):
                v_str = f"{v:.4f}" if abs(v) < 1 else f"{v:,.2f}"
            elif isinstance(v, (int, np.integer)) or (isinstance(v, str) and str(v).isdigit()):
                v_str = f"{int(v):,}"
            else:
                v_str = str(v)
            tip = help_dict.get(k, "").replace('"', '&quot;')
            tip_html = f' <span class="help-btn" data-tip="{tip}">?</span>' if tip else ""
            rows.append(f"<tr><td>{k}{tip_html}</td><td>{v_str}</td></tr>")
        return "\n".join(rows)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if sample_id:
        plex_display = f"{sample_id} — Plex {plex_id}"
        sample_meta_span = f"<span><b>Library</b> {sample_id}</span>"
    else:
        plex_display = f"Plex {plex_id}"
        sample_meta_span = ""
    ilab_meta_span = f"<span><b>iLab</b> {ilab}</span>" if ilab else ""

    # Emit a flat metrics.csv mirror of the Metrics tab next to the report (P5.3),
    # sharing the report's basename (e.g. IL-1234_CLL02.metrics.csv).
    if output_file.name.endswith(".summary.html"):
        metrics_csv_path = output_file.with_name(
            output_file.name[: -len(".summary.html")] + ".metrics.csv"
        )
    else:
        metrics_csv_path = output_file.with_name(f"{plex_id}.metrics.csv")
    metrics_download = ""
    try:
        metric_rows = (
            [("category", "metric", "value")]
            + [("counts", k, v) for k, v in counts.items()]
            + [("fastq", k, v) for k, v in fastq.items()]
        )
        csv_text = pd.DataFrame(
            metric_rows[1:], columns=metric_rows[0]
        ).to_csv(index=False)
        # Still emit the sidecar file for pipeline consumers (P5.3) ...
        pd.DataFrame(metric_rows[1:], columns=metric_rows[0]).to_csv(metrics_csv_path, index=False)
        print(f"  Metrics CSV written → {metrics_csv_path}")
        # ... but embed the same CSV as a base64 data URI so the download works
        # even if the HTML is moved away from the sidecar file (self-contained).
        b64 = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
        metrics_download = (
            f'<span><a href="data:text/csv;base64,{b64}" '
            f'download="{metrics_csv_path.name}">⤓ metrics.csv</a></span>'
        )
    except Exception as e:
        print(f"  Warning: could not build metrics.csv: {e}")

    # Inline the vendored assets. If they're missing (e.g. a source tree without
    # the assets dir), fall back to the CDN so the report still works online.
    plotly_js = _read_asset("plotly.min.js")
    if plotly_js is not None:
        plotly_tag = f"<script>{plotly_js}</script>"
    else:
        print("  Warning: vendored plotly.min.js not found; falling back to CDN "
              "(report will require internet access to render figures).")
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{_PLOTLY_VERSION}.min.js"></script>'

    fonts_css = _read_asset("fonts.css")
    if fonts_css is not None:
        fonts_tag = f"<style>{fonts_css}</style>"
    else:
        fonts_tag = (
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700'
            '&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
        )

    html = _HTML_TEMPLATE.format(
        plex=plex_id,
        fonts_tag=fonts_tag,
        plotly_tag=plotly_tag,
        plex_display=plex_display,
        sample_meta_span=sample_meta_span,
        ilab_meta_span=ilab_meta_span,
        metrics_download=metrics_download,
        hero_html=hero_html,
        counts_rows=_rows(counts, _counts_help),
        fastq_rows=_rows(fastq, _fastq_help),
        figs_json=json.dumps(all_figs, allow_nan=False, default=lambda x: None),
        wta_tab_btn=wta_tab_btn,
        wta_tab_html=wta_tab_html,
        wta_plot_js=wta_plot_js,
        timestamp=timestamp,
    )
    output_file.write_text(html, encoding="utf-8")
    print(f"  HTML report written → {output_file}")