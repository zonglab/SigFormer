#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s04_Util_apply.py

Application and evaluation utilities for SigFormer.

Shared application and analysis utilities for SigFormer. The module contains
metrics, plotting, UMAP/Leiden helpers, reference harmonization, replacement tests,
and output helpers without importing the training loop.
"""

from __future__ import annotations
import os, re, copy, json, math, time, pickle, hashlib, inspect, random, warnings
import datetime as _dt
from typing import Any, List, Dict, Union, Tuple, Optional, Sequence, Iterable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.colors import to_hex, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering, dendrogram
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, eye, save_npz
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12
GLOBAL_SEED = 717
COLORS_plot = [color for color in ['#1e90ff', '#000000', '#ff0000', '#b3b3b3', '#9acd32', '#eeaeee'] for _ in range(16)]
ntcomp = {'T': 'A', 'G': 'C', 'C': 'G', 'A': 'T', 'N': 'M'}
VEC_substit = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
VEC_context = [f"{x}-{y}" for x in "ACGT" for y in "ACGT"]
VEC_sub_ctx = [f"{sub},{ctx}" for sub in VEC_substit for ctx in VEC_context]
XTIC_3ntctx = [f"{ctx[4]}{ctx[0]}{ctx[6]}" for ctx in VEC_sub_ctx]

def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Dataset inference / cache handling
# ============================================================================
def run_methods(df_3nt_raw, df_ref_sig, runners, cache_paths=None, df_refmask=None, chunk_size=500, method_names=None, save_cache=True, verbose=True):
    """Load cached decompositions when available; otherwise run wrappers in sample chunks."""
    cache_paths = {} if cache_paths is None else {k: Path(v) for k, v in cache_paths.items()}
    method_names = {} if method_names is None else dict(method_names)
    results, pending = {}, []
    for tag, runner in runners.items():
        name = method_names.get(tag, tag)
        path = cache_paths.get(tag)
        if path is not None and path.exists():
            compo = pd.read_csv(path, sep="\t", index_col=0).reindex(df_3nt_raw.index).fillna(0.0).clip(lower=0.0)
            if (compo.sum(axis=1) > 2).any():
                compo = compo.div(compo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            reference = df_ref_sig.reindex(compo.columns)
            results[name] = {"tag": tag, "compo": compo, "recon": compo @ reference, "ood": (1 - compo.sum(axis=1)).clip(lower=0.0), "source": "cache"}
            if verbose:
                print(f"{tag:>8}:{name:<22} loaded cache {path}")
        else:
            print(f"{tag:>8}:{name:<22} needs to be run")
            pending.append((tag, name, runner, path))
    if not pending:
        return results
    pieces = {name: {"compo": [], "recon": [], "ood": [], "diagnostics": []} for _, name, _, _ in pending}
    step = max(1, int(chunk_size or len(df_3nt_raw)))
    for start in range(0, len(df_3nt_raw), step):
        index = df_3nt_raw.index[start:start + step]
        X = df_3nt_raw.loc[index]
        mask = None if df_refmask is None else df_refmask.reindex(index=index, columns=df_ref_sig.index, fill_value=False)
        if verbose:
            print(f"inference chunk {start // step + 1}: samples {start + 1}-{start + len(index)} / {len(df_3nt_raw)}")
        for tag, name, runner, _ in pending:
            out = runner(X, df_ref_sig, mask)
            if isinstance(out, dict):
                compo, recon, ood = out.get("compo", out.get("composition")), out.get("recon"), out.get("ood")
            else: compo, recon, ood = out[0], out[1], out[2] if len(out) > 2 else None
            compo = compo.reindex(index=index, columns=df_ref_sig.index, fill_value=0.0)
            recon = recon.reindex(index=index, columns=df_ref_sig.columns, fill_value=0.0)
            if ood is None:
                ood = pd.Series(0.0, index=index, name="OOD")
            elif isinstance(ood, pd.DataFrame):
                ood = ood.reindex(index=index, columns=["OOD"], fill_value=0.0)["OOD"]
            else: ood = pd.Series(ood, index=index, name="OOD").reindex(index)
            pieces[name]["compo"].append(compo)
            pieces[name]["recon"].append(recon)
            pieces[name]["ood"].append(ood)
            diagnostics = getattr(runner, "last_diagnostics", None)
            if isinstance(diagnostics, pd.DataFrame) and len(diagnostics):
                pieces[name]["diagnostics"].append(diagnostics.copy())
    for tag, name, _, path in pending:
        compo = pd.concat(pieces[name]["compo"]).reindex(df_3nt_raw.index)
        recon = pd.concat(pieces[name]["recon"]).reindex(df_3nt_raw.index)
        ood = pd.concat(pieces[name]["ood"]).reindex(df_3nt_raw.index)
        results[name] = {"tag": tag, "compo": compo, "recon": recon, "ood": ood, "source": "inference"}
        if pieces[name]["diagnostics"]:
            results[name]["diagnostics"] = pd.concat(pieces[name]["diagnostics"]).reindex(df_3nt_raw.index)
        if save_cache and path is not None:
            ensure_dir(path.parent)
            compo.to_csv(path, sep="\t")
    return results

def _as_np(a) -> np.ndarray:
    """Convert a Series/DataFrame/array-like to a finite float numpy array."""
    if isinstance(a, (pd.DataFrame, pd.Series)):
        a = a.to_numpy(dtype=float)
    arr = np.asarray(a, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def sum_scale(data, by=0, axis=None):
    """Normalize rows (default) or columns, preserving the established row/column semantics."""
    if axis is not None:
        by = 0 if axis == 1 else 1
    assert by in {0, 1, "0", "1", "row", "col"}
    by = 0 if by in {0, "0", "row"} else 1
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame().T if by == 1 else data.to_frame()
    return data.div(data.sum(axis=1 - by), axis=by)


def calc_sens(y_true, y_pred, thr_active: float = 0.01, eps: float = 1e-12) -> float:
    """Sensitivity/recall for active components after thresholding."""
    yt = _as_np(y_true).ravel() > thr_active
    yp = _as_np(y_pred).ravel() > thr_active
    tp = float(np.sum(yt & yp))
    fn = float(np.sum(yt & ~yp))
    return tp / max(tp + fn, eps)


def calc_spec(y_true, y_pred, thr_active: float = 0.01, eps: float = 1e-12) -> float:
    """Specificity for inactive components after thresholding."""
    yt = _as_np(y_true).ravel() > thr_active
    yp = _as_np(y_pred).ravel() > thr_active
    tn = float(np.sum(~yt & ~yp))
    fp = float(np.sum(~yt & yp))
    return tn / max(tn + fp, eps)


def calc___F1(y_true, y_pred, thr_active: float = 0.01, eps: float = 1e-12) -> float:
    """F1 score for active/inactive component calls after thresholding."""
    yt = _as_np(y_true).ravel() > thr_active
    yp = _as_np(y_pred).ravel() > thr_active
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return 2.0 * precision * recall / max(precision + recall, eps)


def calc___R2(y_true, y_pred, eps: float = 1e-12) -> float:
    """Coefficient of determination over flattened arrays."""
    yt = _as_np(y_true).ravel()
    yp = _as_np(y_pred).ravel()
    if yt.size == 0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < eps:
        return 1.0 if ss_res < eps else 0.0
    return 1.0 - ss_res / ss_tot


def calc__cos(y_true, y_pred, eps: float = 1e-12) -> float:
    """Cosine similarity over flattened arrays."""
    yt = _as_np(y_true).ravel()
    yp = _as_np(y_pred).ravel()
    denom = float(np.linalg.norm(yt) * np.linalg.norm(yp))
    if denom < eps:
        return 1.0 if np.linalg.norm(yt - yp) < eps else 0.0
    return float(np.dot(yt, yp) / denom)


def _compute_metrics(y_true, y_pred, thr_active: float = 0.01) -> Dict[str, float]:
    """Compute metrics with strict >0.01 active threshold by default."""
    return {
        "sensitivity": calc_sens(y_true, y_pred, thr_active=thr_active),
        "specificity": calc_spec(y_true, y_pred, thr_active=thr_active),
        "F1": calc___F1(y_true, y_pred, thr_active=thr_active),
        "R2": calc___R2(y_true, y_pred),
        "Cosine": calc__cos(y_true, y_pred),
        "MSE": float(np.mean((_as_np(y_true).ravel() - _as_np(y_pred).ravel()) ** 2)),
    }


def plot_scatter_compo(df_true: pd.DataFrame,
                       df_pred: pd.DataFrame,
                       max_points: int = 8000, thr_active: float = 0.01,
                       dot_color="#1976d2", dot_alpha=0.5, dot_sizes=5,
                       title: str = "Composition scatterplot",
                       fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    df_pred = df_pred.reindex_like(df_true).fillna(0.0)

    # Convert to dense matrices after alignment.
    true_mat = df_true.to_numpy(dtype=float)
    pred_mat = df_pred.to_numpy(dtype=float)
    n_samples, n_sigs = true_mat.shape

    # Flatten all sample-signature cells for a global metric block.
    true_all = true_mat.ravel()
    pred_all = pred_mat.ravel()

    metrics: Dict[str, Dict[str, float]] = {}
    metrics["all"] = _compute_metrics(true_all, pred_all, thr_active)

    # Binary active/inactive labels are used only for FP/FN highlighting.
    true_vals = true_all
    pred_vals = pred_all
    n_cells = true_vals.size
    true_active = true_vals > thr_active
    pred_active = pred_vals > thr_active
    fp_mask = (~true_active) & pred_active
    fn_mask = true_active & (~pred_active)
    border_mask = fp_mask | fn_mask
    normal_mask = ~border_mask

    # Downsample points without biasing metrics. Metrics stay full-resolution.
    if n_cells > max_points:
        rng = np.random.default_rng(20260604)
        idx = rng.choice(n_cells, size=max_points, replace=False)
        true_vals_plot = true_vals[idx]
        pred_vals_plot = pred_vals[idx]
        border_mask_plot = border_mask[idx]
        normal_mask_plot = normal_mask[idx]
    else:
        true_vals_plot = true_vals
        pred_vals_plot = pred_vals
        border_mask_plot = border_mask
        normal_mask_plot = normal_mask

    # Draw into caller-provided axes when available. This avoids tight_layout
    # complaints when figures are assembled as grids.
    created_fig = (fig is None) or (ax is None)
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1, alpha=0.7)
    ax.scatter(true_vals_plot[normal_mask_plot],
               pred_vals_plot[normal_mask_plot],
               c=dot_color, s=dot_sizes, alpha=dot_alpha, label="normal",)

    if bool(np.any(border_mask_plot)):
        ax.scatter(true_vals_plot[border_mask_plot],
                   pred_vals_plot[border_mask_plot],
                   s=10, alpha=0.9, facecolors="none", edgecolors="darkred",
                   linewidths=0.7, label="FP/FN",)

    m_all = metrics["all"]
    # Legend/text block format requested for benchmark readability:
    # sens/spec/F1, Cos/R2, MSE. Metrics are computed on all cells; plotting
    # may be downsampled, because apparently pixels are finite even when ambition is not.
    txt = (f"sens/spec/F1\n"
           f"{m_all['sensitivity']:.3f}/{m_all['specificity']:.3f}/{m_all['F1']:.3f}\n"
           f"Cos/R2\n"
           f"{m_all['Cosine']:.3f}/{m_all['R2']:.3f}\n"
           f"MSE\n"
           f"{m_all['MSE']:.3e}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),)
    ax.set(title=title,
           xlim=(-0.05, 1.05), xlabel="true composition",
           ylim=(-0.05, 1.05), ylabel="pred composition",)
    ax.set_aspect("equal", adjustable="box")
    return fig, metrics


# SBS96 plotting utilities
def stack_inputs(*args):
    # 找到所有有列名的对象的列名集合
    colnames = [tuple(x.columns) if isinstance(x, pd.DataFrame) else
                tuple(x.index) if isinstance(x, pd.Series) and not isinstance(x.index, pd.RangeIndex) else None
                for x in args]
    # 过滤掉None，只看有名对象
    colnames = [c for c in colnames if c is not None]
    unified_cols = list(colnames[0]) if colnames and all(c == colnames[0] for c in colnames) else None
    if colnames and unified_cols is None:
        raise ValueError("Not all named inputs have identical colnames.")

    processed, index = [], []
    for i, x in enumerate(args):
        if isinstance(x, pd.DataFrame):
            processed.append(x.copy())
            index.extend(x.index)
        else:
            # Series
            if isinstance(x, pd.Series) and not isinstance(x.index, pd.RangeIndex):
                df1 = pd.DataFrame([x.values], columns=x.index)
            else:
                arr = np.asarray(x)
                if unified_cols:
                    if len(arr) != len(unified_cols):
                        raise ValueError("Input length does not match colnames length.")
                    df1 = pd.DataFrame([arr], columns=unified_cols)
                else:
                    df1 = pd.DataFrame([arr])
            processed.append(df1)
            name = getattr(x, 'name', None)
            index.append(name if name is not None else f"row{len(index)}")
    df = pd.concat(processed, ignore_index=True)
    if len(index) == len(df):
        df.index = index
    return df


def CALC_cos_sim(*args):
    """计算所有输入行之间的cosine similarity，返回DataFrame"""
    df = stack_inputs(*args)
    if df.shape[0] < 2:
        raise ValueError("至少需要两行")

    mat = cosine_similarity(df.values)
    res = pd.DataFrame(mat, index=df.index, columns=df.index)
    return res


def _nice_tick_step(ymax, n_ticks=6):
    ymax = float(ymax) if np.isfinite(ymax) else 1.0
    if ymax <= 0:
        return 1.0
    raw = ymax / max(n_ticks, 1)
    exp = np.floor(np.log10(raw))
    base = raw / (10 ** exp)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)



def _auto_symlog_linthresh(values, mode_numb="count"):
    """
    Pick a safe symlog linear window from the data itself.

    Why not log-transform the data manually?
    - symlog keeps zero finite;
    - sign is handled by the axis transform;
    - tiny mutation rates stay tiny in raw units instead of becoming negative log numbers.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.abs(arr[np.isfinite(arr) & (arr != 0)])
    if arr.size == 0:
        return 1.0

    min_nonzero = float(np.nanmin(arr))
    max_abs = float(np.nanmax(arr))
    if mode_numb == "count" and max_abs >= 1:
        return max(1.0, min_nonzero / 2.0)
    return max(min_nonzero / 2.0, max_abs * 1e-9, np.finfo(float).tiny)


def _format_3nt_tick_value(y, digit=3):
    if not np.isfinite(y):
        return ""
    if y == 0:
        return "0"
    ay = abs(y)
    if (ay >= 1e4) or (ay < 1e-3):
        return f"{y:.2e}"
    return f"{y:.{digit}f}".rstrip("0").rstrip(".")


def _decorate_3nt_axis(
    ax,
    maxy,
    title,
    mode_numb,
    plusminus=False,
    abs_ytick=False,
    maxy_minus=None,
    yscale="linear",
    symlog_linthresh=None,
    symlog_linscale=1.0,
    symlog_base=10,
    tick_inverse=None,
):
    maxy = max(float(maxy), 1e-12)
    maxy_minus = maxy if maxy_minus is None else max(float(maxy_minus), 1e-12)
    yscale = (yscale or "linear").lower()
    if yscale not in {"linear", "symlog"}:
        raise ValueError("yscale must be 'linear' or 'symlog'.")

    yr_rec, yr_hei, yr_txt, yr_upp = 1.15, 0.12, 1.32, 1.40
    yc_rec, yh_hei, yc_txt, yc_upp = yr_rec * maxy, yr_hei * maxy, yr_txt * maxy, yr_upp * maxy
    rec_par = {"width": 16, "height": yh_hei, "linewidth": 2}
    txt_par = {"ha": "center", "va": "center", "fontsize": 10, "color": "black"}
    for stt, col, lbl in [(0,"#1e90ff","C>A"), (16,"#000000","C>G"), (32,"#ff0000","C>T"),
                          (48,"#b3b3b3","T>A"), (64,"#9acd32","T>C"), (80,"#eeaeee","T>G")]:
        ax.add_patch(patches.Rectangle((stt-0.5, yc_rec), facecolor=col, **rec_par))
        ax.text(stt+8, yc_txt, lbl, **txt_par)

    ylow = -1.15 * maxy_minus if plusminus else 0
    yhigh = yc_upp

    ylabel = "Ratio" if mode_numb == "ratio" else "Count"
    ax.set(title=title, ylabel=ylabel, xlim=[-1, 96], ylim=[ylow, yhigh])
    ax.set_xticks(range(96), XTIC_3ntctx, rotation=90, fontsize=10)

    if yscale == "symlog":
        if symlog_linthresh is None:
            symlog_linthresh = _auto_symlog_linthresh([ylow, yhigh], mode_numb=mode_numb)
        symlog_linthresh = max(float(symlog_linthresh), np.finfo(float).tiny)
        ax.set_yscale("symlog", linthresh=symlog_linthresh, linscale=symlog_linscale, base=symlog_base)
        ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(base=symlog_base, linthresh=symlog_linthresh))
        digit = 6 if mode_numb == "ratio" else 3
    else:
        step = _nice_tick_step(max(yhigh, abs(ylow)))
        digit = max(int(np.ceil(-np.log10(step))), 0)
        if plusminus:
            yticks = np.arange(np.floor(ylow / step) * step, yhigh + step, step)
        else:
            yticks = np.arange(0, yhigh + step, step)
        ax.set_yticks(yticks)

    def _fmt_tick(y, pos=None):
        raw_y = tick_inverse(y) if callable(tick_inverse) else y
        yy = abs(raw_y) if abs_ytick else raw_y
        return _format_3nt_tick_value(yy, digit=digit)

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    return ax


def _infer_plus_minus_groups(index, plus_group=None, minus_group=None):
    idx = list(index)
    if plus_group is not None and minus_group is not None:
        return plus_group, minus_group

    lowered = {str(x).lower(): x for x in idx}
    if plus_group is None:
        pyr_hits = [x for x in idx if "pyr" in str(x).lower()]
        plus_group = pyr_hits[0] if pyr_hits else None
    if minus_group is None:
        pur_hits = [x for x in idx if "pur" in str(x).lower()]
        minus_group = pur_hits[0] if pur_hits else None

    if plus_group is None or minus_group is None:
        if len(idx) == 2:
            plus_group = idx[0] if plus_group is None else plus_group
            minus_group = idx[1] if minus_group is None else minus_group
        else:
            raise ValueError("mode_plot='plusminus' needs plus_group/minus_group, or exactly two rows, or row labels containing pyr/pur.")
    return plus_group, minus_group


def PLOT_3nt_patterns(
    df_3nt,
    sample=None,
    title="3nt_profile",
    mode_numb="count",
    maxy=None,
    ax=None,
    figsize=(12, 2.5),
    mode_plot="one",
    error_bar=False,
    group_order=None,
    hatch_list=None,
    plus_group=None,
    minus_group=None,
    legend=True,
    yscale="linear",
    symlog_linthresh=None,
    symlog_linscale=1.0,
    symlog_base=10,
    plusminus_scale="shared",
    maxy_minus=None,
):
    """
    Plot 96D trinucleotide profiles.

    mode_plot:
        - "one"      : one row as the standard 96D bar plot
        - "mean"     : mean profile across rows, optionally with error_bar=True
        - "split"    : one mini-bar per df_3nt row within each 96D channel, using hatches
        - "plusminus": plus row upward, minus row downward; useful for pyr/pur-centric profiles

    yscale:
        - "linear"   : regular axis
        - "symlog"   : signed log-like axis that keeps zero valid. Do not pre-log mutation rates.

    plusminus_scale:
        - "shared"   : plus/minus use the same y data scale; old behavior
        - "separate" : plus/minus use independent data scales but comparable visual height.
                       Negative tick labels are converted back to the original minus-group units.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    assert mode_numb in {"count", "ratio"}, "mode_numb should be either count/ratio"
    assert mode_plot in {"one", "mean", "violin", "split", "plusminus"}, \
        "mode_plot must be one of: one/mean/violin/split/plusminus"

    yscale = (yscale or "linear").lower()
    plusminus_scale = (plusminus_scale or "shared").lower()
    if plusminus_scale not in {"shared", "separate"}:
        raise ValueError("plusminus_scale must be 'shared' or 'separate'.")

    if hatch_list is None:
        hatch_list = ["", "///", "\\\\", "---", "|||", "xxx", "++", "..."]

    df_plt = df_3nt.loc[:, VEC_sub_ctx].copy()
    if group_order is not None:
        df_plt = df_plt.loc[group_order, :]
    if mode_numb == "ratio":
        df_plt = sum_scale(df_plt).fillna(0.0)

    x = np.arange(96)
    plusminus = mode_plot == "plusminus"
    minus_display_factor = 1.0
    axis_maxy_minus = None
    tick_inverse = None
    symlog_ref_values = []

    if mode_plot == "one":
        if sample is None:
            sample = df_plt.index[0]
        values = df_plt.loc[sample, :]
        if maxy is None:
            maxy = values.max()
        symlog_ref_values.append(values.values)
        ax.bar(x, values, color=COLORS_plot, edgecolor="none")

    elif mode_plot == "mean":
        values, errors = df_plt.mean(axis=0), df_plt.std(axis=0)
        if maxy is None:
            maxy = (values + errors.fillna(0)).max()
        symlog_ref_values.extend([values.values, errors.fillna(0).values])
        ax.bar(
            x,
            values,
            color=COLORS_plot,
            edgecolor="none",
            yerr=errors if error_bar else None,
            capsize=2 if error_bar else 0,
            linewidth=0.7,
        )

    elif mode_plot == "split":
        n_group = df_plt.shape[0]
        width = 0.82 / max(n_group, 1)
        offsets = (np.arange(n_group) - (n_group - 1) / 2) * width
        if maxy is None:
            maxy = df_plt.max().max()
        symlog_ref_values.append(df_plt.values)

        for j, idx in enumerate(df_plt.index):
            hatch = hatch_list[j % len(hatch_list)]
            ax.bar(
                x + offsets[j],
                df_plt.loc[idx, :].values,
                width=width * 0.95,
                color=COLORS_plot,
                edgecolor="black" if hatch else "none",
                linewidth=0.25 if hatch else 0,
                hatch=hatch,
                label=str(idx),
            )

        if legend:
            handles = [patches.Patch(facecolor="white", edgecolor="black", hatch=hatch_list[j % len(hatch_list)], label=str(idx))
                       for j, idx in enumerate(df_plt.index)]
            ax.legend(handles=handles, title="group", frameon=False, ncol=min(4, n_group))

    elif mode_plot == "plusminus":
        plus_group, minus_group = _infer_plus_minus_groups(df_plt.index, plus_group, minus_group)
        plus_values = df_plt.loc[plus_group, :]
        minus_values = df_plt.loc[minus_group, :]
        plus_data_max = max(float(plus_values.max()), 1e-12)
        minus_data_max = max(float(minus_values.max()), 1e-12)

        if plusminus_scale == "shared":
            if maxy is None:
                maxy = max(plus_data_max, minus_data_max)
            axis_maxy_minus = maxy if maxy_minus is None else maxy_minus
            minus_plot_values = minus_values.values
        else:
            if maxy is None:
                maxy = plus_data_max
            if maxy_minus is None:
                maxy_minus = minus_data_max
            maxy_minus = max(float(maxy_minus), 1e-12)
            minus_display_factor = float(maxy) / maxy_minus
            axis_maxy_minus = maxy
            minus_plot_values = minus_values.values * minus_display_factor

            def tick_inverse(y):
                return y if y >= 0 else y / minus_display_factor

        symlog_ref_values.extend([plus_values.values, -minus_plot_values])
        ax.bar(x, plus_values.values, color=COLORS_plot, edgecolor="none", label=str(plus_group))
        ax.bar(x, -minus_plot_values, color=COLORS_plot, edgecolor="black", linewidth=0.25,
               hatch="///", label=str(minus_group))
        ax.axhline(0, color="black", linewidth=0.8)

        if legend:
            handles = [
                patches.Patch(facecolor="white", edgecolor="black", label=f"+ {plus_group}"),
                patches.Patch(facecolor="white", edgecolor="black", hatch="///", label=f"- {minus_group}"),
            ]
            if plusminus_scale == "separate":
                handles.append(patches.Patch(facecolor="white", edgecolor="none", label="separate +/- y scale"))
            ax.legend(handles=handles, frameon=False)

    if yscale == "symlog" and symlog_linthresh is None:
        symlog_linthresh = _auto_symlog_linthresh(np.concatenate([np.ravel(v) for v in symlog_ref_values]), mode_numb=mode_numb)

    _decorate_3nt_axis(
        ax,
        maxy=maxy,
        title=title,
        mode_numb=mode_numb,
        plusminus=plusminus,
        abs_ytick=plusminus,
        maxy_minus=axis_maxy_minus,
        yscale=yscale,
        symlog_linthresh=symlog_linthresh,
        symlog_linscale=symlog_linscale,
        symlog_base=symlog_base,
        tick_inverse=tick_inverse,
    )

    ax._yz_3nt_plot_meta = {
        "mode_plot": mode_plot,
        "mode_numb": mode_numb,
        "yscale": yscale,
        "maxy": maxy,
        "maxy_minus": maxy_minus,
        "plus_group": plus_group,
        "minus_group": minus_group,
        "plusminus_scale": plusminus_scale,
        "minus_display_factor": minus_display_factor,
        "symlog_linthresh": symlog_linthresh,
        "symlog_linscale": symlog_linscale,
        "symlog_base": symlog_base,
    }
    return ax, maxy


def PLOT_pred_outline(df_3nt, ax, maxy=None,sample=None, label="prediction",
                      mode_numb="inherit", mode_plot="inherit", plus_group=None, minus_group=None,
                      edgecolor="#4B0082", linewidth=1.2, legend=True,):
    """
    Overlay prediction outlines on PLOT_3nt_patterns().

    - mode_plot="inherit" reads plotting metadata from the target axis when possible.
    - Supports "one", "mean", and "plusminus".
    - For plusminus_scale="separate", the minus outline is scaled exactly like PLOT_3nt_patterns().
    """
    assert ax is not None
    meta = getattr(ax, "_yz_3nt_plot_meta", {}) or {}
    if mode_numb == "inherit":
        mode_numb = meta.get("mode_numb", "count")
    assert mode_numb in {"count", "ratio"}, "mode_numb should be either inherit/count/ratio"

    if mode_plot == "inherit":
        inherited_mode = meta.get("mode_plot", "one")
        mode_plot = inherited_mode if inherited_mode in {"one", "mean", "plusminus"} else ("mean" if df_3nt.shape[0] > 1 else "one")
    assert mode_plot in {"one", "mean", "plusminus"}, "mode_plot must be one of: inherit/one/mean/plusminus"

    if meta and mode_numb != meta.get("mode_numb", mode_numb):
        warnings.warn(f"PLOT_pred_outline mode_numb={mode_numb!r} differs from axis mode_numb={meta.get('mode_numb')!r}. "
                      "Make sure observed and predicted profiles are on the same scale.",
                       UserWarning,)

    df_plt = df_3nt.loc[:, VEC_sub_ctx].copy()
    if mode_numb == "ratio":
        df_plt = sum_scale(df_plt).fillna(0.0)

    x = np.arange(96)
    plusminus_scale = meta.get("plusminus_scale", "shared")
    minus_display_factor = float(meta.get("minus_display_factor", 1.0))

    # -------- single input: transparent bar outline --------
    if mode_plot == "one":
        if sample is None:
            sample = df_plt.index[0]
        ax.bar(x, df_plt.loc[sample, :].values, color=(0, 0, 0, 0),
               edgecolor=edgecolor, label=label, linewidth=linewidth)

    # -------- multiple inputs: boxplot per 96D channel; single row falls back to outline --------
    elif mode_plot == "mean":
        if df_plt.shape[0] <= 1:
            values = df_plt.iloc[0, :].values
            ax.bar(x, values, color=(0, 0, 0, 0), edgecolor=edgecolor, label=label, linewidth=linewidth)
        else:
            for i in range(96):
                ax.boxplot(
                    df_plt.iloc[:, i].values,
                    positions=[i],
                    widths=0.8,
                    patch_artist=True,
                    showmeans=False,
                    boxprops={"facecolor": "none", "edgecolor": edgecolor, "linewidth": linewidth},
                    flierprops={"marker": "o", "markersize": 3, "color": edgecolor, "alpha": 0.6, "markerfacecolor": "none"},
                    whiskerprops={"color": edgecolor, "linewidth": linewidth},
                    capprops={"color": edgecolor, "linewidth": linewidth},
                    medianprops={"color": edgecolor, "linewidth": linewidth},
                )
            ax.plot([], [], color=edgecolor, label=label, linewidth=linewidth)

    # -------- plus/minus overlay: use the same negative transform as the background axis --------
    elif mode_plot == "plusminus":
        plus_group = plus_group if plus_group is not None else meta.get("plus_group", None)
        minus_group = minus_group if minus_group is not None else meta.get("minus_group", None)
        plus_group, minus_group = _infer_plus_minus_groups(df_plt.index, plus_group, minus_group)

        plus_values = df_plt.loc[plus_group, :].values
        minus_values = df_plt.loc[minus_group, :].values
        if plusminus_scale == "separate":
            minus_values = minus_values * minus_display_factor

        ax.bar(x, plus_values, color=(0, 0, 0, 0), edgecolor=edgecolor,
               label=f"{label} +", linewidth=linewidth)
        ax.bar(x, -minus_values, color=(0, 0, 0, 0), edgecolor=edgecolor,
               label=f"{label} -", linewidth=linewidth, linestyle="--")

    # Keep PLOT_3nt_patterns' axis limits/transforms when metadata exists.
    ax.set_xticks(range(96), XTIC_3ntctx, rotation=90, fontsize=10)
    if not meta and maxy is not None:
        if mode_plot == "plusminus":
            ax.set_ylim(-1.15 * maxy, maxy * 1.4)
        else:
            ax.set_ylim(0, maxy * 1.4)
    if legend:
        ax.legend(frameon=False)
    return ax


##############################################################################################################################
### align references
##############################################################################################################################

def align_ref_names(source, canonical, source_label, same_threshold=0.98, like_threshold=0.90):
    """Align a source reference to canonical names and return reference, log, and mapping."""
    if not source.columns.equals(canonical.columns):
        raise ValueError("source and canonical must have identical columns and order")
    similarity = cosine_similarity(source.to_numpy(float), canonical.to_numpy(float))
    best = similarity.argmax(axis=1)
    score = similarity[np.arange(len(source)), best]
    original = source.index.astype(str)
    matched = canonical.index.astype(str)[best]
    aligned, category = [], []
    for old, hit, cosine in zip(original, matched, score):
        if cosine >= same_threshold:
            new, kind = hit, "same"
        elif cosine >= like_threshold:
            new, kind = f"{hit}_like_{source_label}_{old}", "like"
        else: 
            clean = re.sub(r"\W+", "_", old).strip("_")
            new, kind = f"{clean}_{source_label}_denovo", "denovo"
        aligned.append(new)
        category.append(kind)
    renamed = source.copy()
    renamed.index = aligned
    renamed.index.name = source.index.name
    records = pd.DataFrame({"original": original, "aligned": aligned, "matched": matched,
                            "cosine": score, "category": category, "source": source_label})
    return renamed, records, dict(zip(original, aligned))


def merge_signature_dfs(*frames):
    """Merge references in priority order, retain first duplicates, and natural-sort names."""
    merged = pd.concat(frames).loc[lambda x: ~x.index.duplicated(keep="first")]
    return merged.loc[sorted(merged.index, key=sig_natkey)]




##############################################################################################################################
### reconstruction violin
##############################################################################################################################
def plot_recon_violin(cosine_results, method_order, method_colors, ax=None, figsize=(6, 4.2)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    cosine_df = pd.DataFrame(cosine_results)
    methods = [m for m in method_order if m in cosine_df.columns]
    data = [cosine_df[m].dropna().to_numpy() for m in methods]
    pos = np.arange(len(methods))

    vp = ax.violinplot(data, positions=pos, widths=0.78,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, m in zip(vp["bodies"], methods):
        body.set(facecolor=method_colors[m], edgecolor="none", alpha=0.65)

    bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.3),
                    boxprops=dict(edgecolor="black", linewidth=0.8),
                    whiskerprops=dict(color="black", linewidth=0.8),
                    capprops=dict(color="black", linewidth=0.8))
    for box, m in zip(bp["boxes"], methods):
        box.set(facecolor=method_colors[m], alpha=0.95)

    ax.set(ylim=(0.8, 1.0), ylabel="Cosine similarity",
           xticks=pos, xticklabels=methods)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)

##############################################################################################################################
### UMAP
##############################################################################################################################



def _seed_everything(seed: int | None):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)



def _intersect_indices(dfs: Sequence[pd.DataFrame]) -> pd.Index:
    idx = None
    for df in dfs:
        if df is None:
            continue
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None:
        raise ValueError("No valid dataframe provided.")
    return idx.sort_values()



def _assign_categorical_column(
    df: pd.DataFrame, idx: pd.Index, key: str, values: np.ndarray, *, as_string_with_prefix: str = "C"
) -> None:
    s = pd.Series([f"{as_string_with_prefix}{int(v)}" for v in values], index=idx, name=key)
    if key in df.columns and isinstance(df[key].dtype, CategoricalDtype):
        exist_cats = df[key].cat.categories.astype(str)
        new_cats = pd.Index(pd.unique(s)).astype(str)
        union_cats = exist_cats.union(new_cats)
        df[key] = df[key].cat.set_categories(union_cats)
        df.loc[idx, key] = pd.Categorical(s, categories=union_cats)
    else:
        df.loc[idx, key] = s.astype(object)
        df[key] = df[key].astype("category")



def _pdist_probability_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    if df.min().min() < 0:
        raise ValueError(f"Metric '{metric}' requires non-negative inputs.")
    X = np.clip(df.values, 1e-12, None)
    X = (X.T / X.sum(axis=1)).T
    if metric == "hellinger":
        dist = pdist(np.sqrt(X), metric="euclidean") / np.sqrt(2.0)
        return squareform(dist)
    if metric == "js":
        P = X[:, None, :]
        Q = X[None, :, :]
        M = 0.5 * (P + Q)
        return 0.5 * (np.sum(P * np.log(P / M), axis=2) + np.sum(Q * np.log(Q / M), axis=2))
    if metric == "wasserstein":
        from scipy.stats import wasserstein_distance
        n = X.shape[0]
        out = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                out[i, j] = out[j, i] = wasserstein_distance(X[i], X[j])
        return out
    raise ValueError(f"Unsupported probability metric: {metric}")



def compute_pairwise_distance(df_data: pd.DataFrame, df_feats=None, *, metric="cosine",
                              preprocess=None, combine="concat", weights=None, dtype="float32") -> Tuple[np.ndarray, pd.Index]:
    if df_feats is not None:
        blocks = [df_data] + (list(df_feats.values()) if isinstance(df_feats, dict) else list(df_feats))
        idx = _intersect_indices(blocks)
        blocks = [b.loc[idx] for b in blocks]
    else:
        idx = df_data.index
        blocks = [df_data.loc[idx]]

    def _prep(block):
        if preprocess == "zscore":
            return (block - block.mean()) / (block.std(ddof=0) + 1e-12)
        if preprocess == "row_norm":
            X = np.clip(block.values, 1e-12, None)
            X = (X.T / X.sum(axis=1)).T
            return pd.DataFrame(X, index=block.index, columns=block.columns)
        return block

    blocks = [_prep(b) for b in blocks]
    if combine == "concat":
        X = pd.concat(blocks, axis=1).astype(dtype)
        if metric in ("cosine", "euclidean"):
            dist = squareform(pdist(X.values, metric=metric))
        elif metric in ("js", "hellinger", "wasserstein"):
            dist = _pdist_probability_metric(X, metric)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    elif combine == "weighted_sum":
        if weights is None:
            weights = [1.0] * len(blocks)
        if len(weights) != len(blocks):
            raise ValueError("weights length mismatch")
        dist = None
        for w, blk in zip(weights, blocks):
            if metric in ("cosine", "euclidean"):
                d = squareform(pdist(blk.values.astype(dtype), metric=metric))
            elif metric in ("js", "hellinger", "wasserstein"):
                d = _pdist_probability_metric(blk, metric)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            dist = d * w if dist is None else dist + d * w
    else:
        raise ValueError("combine must be 'concat' or 'weighted_sum'")

    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    return dist.astype(dtype, copy=False), idx



def build_knn_graph(dist: np.ndarray, n_neighbors: int = 25, include_self: bool = False) -> Tuple[csr_matrix, csr_matrix]:
    n = dist.shape[0]
    k = max(1, min(n_neighbors, n - 1 if not include_self else n))
    nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nbrs.fit(dist)

    def _kneighbors_graph_compat(nbrs, X, mode, include_self):
        try:
            return nbrs.kneighbors_graph(X, mode=mode, include_self=include_self)
        except TypeError:
            graph = nbrs.kneighbors_graph(X, mode=mode)
            n = X.shape[0]
            if include_self:
                if mode == "connectivity":
                    graph = (graph + eye(n, format="csr")).astype(int)
                else:
                    graph = graph.tolil()
                    graph.setdiag(0.0)
                    graph = graph.tocsr()
            else:
                graph = graph.tolil()
                graph.setdiag(0.0)
                graph = graph.tocsr()
            return graph

    conn = _kneighbors_graph_compat(nbrs, dist, mode="connectivity", include_self=include_self).tocsr()
    dmat = _kneighbors_graph_compat(nbrs, dist, mode="distance", include_self=include_self).tocsr()
    conn = ((conn + conn.T) > 0).astype(int).tocsr()
    dmat = dmat.minimum(dmat.T)
    dmat = dmat.tolil()
    dmat.setdiag(0.0)
    dmat = dmat.tocsr()
    return conn, dmat



def run_umap(dist: np.ndarray, df_anno: pd.DataFrame, *, used_index=None, n_neighbors=25,
             min_dist=0.1, spread=1.0, init="spectral", max_iter=None, random_state=None,
             tag: str = "default"):
    try:
        import umap as umap_learn
    except ImportError as exc:
        raise ImportError("run_umap requires umap-learn") from exc
    reducer = umap_learn.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
        metric="precomputed", init=init, n_epochs=max_iter,
        random_state=random_state, verbose=False,
    )
    coords = reducer.fit_transform(dist)
    if used_index is None:
        used_index = df_anno.index
    df_anno.loc[used_index, f"UMAP_{tag}_1"] = coords[:, 0]
    df_anno.loc[used_index, f"UMAP_{tag}_2"] = coords[:, 1]
    return coords, reducer



def run_leiden(conn_graph: csr_matrix, *, resolution=1.0, tag="default",
               df_anno=None, used_index=None, random_state=None) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError("run_leiden requires python-igraph and leidenalg") from exc
    _seed_everything(random_state)
    sources, targets = conn_graph.nonzero()
    graph = ig.Graph(n=conn_graph.shape[0], edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
    kwargs = {"resolution_parameter": resolution}
    try:
        signature = inspect.signature(leidenalg.find_partition)
        if "seed" in signature.parameters and random_state is not None:
            kwargs["seed"] = int(random_state)
    except Exception:
        pass
    try:
        if hasattr(leidenalg, "set_rng_seed") and random_state is not None:
            leidenalg.set_rng_seed(int(random_state))
    except Exception:
        pass
    part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, **kwargs)
    labels = np.asarray(part.membership, dtype=int)
    key = f"Leiden_{tag}_{resolution}"
    if df_anno is not None:
        if used_index is None:
            used_index = df_anno.index
        _assign_categorical_column(df_anno, used_index, key, labels, as_string_with_prefix="C")
    return labels



def umap_pipeline(df_data: pd.DataFrame, df_anno: pd.DataFrame, *,
                  df_feats=None, metric="cosine", preprocess=None,
                  combine="concat", weights=None, n_neighbors=25,
                  min_dist=0.1, spread=1.0, init="spectral", max_iter=None,
                  random_state=None, leiden_resolution: Optional[float] = 1.0,
                  tag: str = "default") -> Dict[str, object]:
    dist, used_idx = compute_pairwise_distance(df_data, df_feats, metric=metric,
                                               preprocess=preprocess, combine=combine, weights=weights)
    conn, dmat = build_knn_graph(dist, n_neighbors=n_neighbors, include_self=False)
    coords, reducer = run_umap(dist, df_anno, used_index=used_idx, n_neighbors=n_neighbors,
                               min_dist=min_dist, spread=spread, init=init, max_iter=max_iter,
                               random_state=random_state, tag=tag)
    labels = None
    if leiden_resolution is not None:
        labels = run_leiden(conn, resolution=leiden_resolution, tag=tag,
                            df_anno=df_anno, used_index=used_idx,
                            random_state=random_state)
    return dict(dist_matrix=dist, used_index=used_idx,
                knn_connectivity=conn, knn_distance=dmat,
                umap_coords=coords, umap_reducer=reducer,
                leiden_labels=labels)


##############################################################################################################################
### signature stacked barplot
##############################################################################################################################

# Stable signature and tissue palettes
DEFAULT_SIGNATURE_COLORS = {'SBS1':'#782d2d',  'SBS2':'#fdc508',  'SBS3':'#879673',  'SBS4':'#f06e4b',  'SBS5':'#d09073',  'SBS6':'#557882',  'SBS7a':'#b91941',  'SBS7b':'#f0466e',  'SBS7c':'#9b5a64',  'SBS7d':'#6e3680',
                    'SBS8':'#8b7621',  'SBS9':'#33ccff',  'SBS10a':'#88419d',  'SBS10b':'#8c6bb1',  'SBS10c':'#557882',  'SBS10d':'#a5236e',  'SBS11':'#a6bddb',  'SBS12':'#3690c0',  'SBS13':'#0986e1',
                    'SBS14':'#74a9cf',  'SBS15':'#0570b0',  'SBS16':'#506e00',  'SBS17a':'#a1d99b',  'SBS17b':'#879673',  'SBS18':'#376ea5',  'SBS19':'#980a88',  'SBS20':'#7bccc4',  'SBS21':'#2d969b',
                    'SBS22a':'#ef8363',  'SBS22b':'#49636d',  'SBS23':'#c7e9b4',  'SBS24':'#41ab5d',  'SBS25':'#f3a4c7',  'SBS26':'#238b45',
                    'SBS27':'#006d2c',  'SBS28':'#00441b',  'SBS29':'#fabe0f',  'SBS30':'#fec44f',  'SBS31':'#d95f0e',  'SBS32':'#6e5a7d',  'SBS33':'#993404',
                    'SBS34':'#8c2d04',  'SBS35':'#cc4c02',  'SBS36':'#c86469',  'SBS37':'#ec7014',  'SBS38':'#fe9929',  'SBS39':'#fec44f',
                    'SBS40a': '#f5d7af',  'SBS40a_like_PCAWG_SBS40': '#f0d0a3',  'SBS40b': '#9bd2d2',  'SBS40b_like_MuSiCal_SBS40': '#b7dfdf',  'SBS40c': '#66c2a4',
                    'SBS41': '#c994c7',  'SBS42': '#df65b0',  'SBS43': '#e7298a',  'SBS44': '#ce1256',  'SBS45': '#980043',  'SBS46': '#67001f',  'SBS47': '#54278f',  'SBS48': '#756bb1',
                    'SBS49': '#9e9ac8',  'SBS50': '#cbc9e2',  'SBS51': '#6a51a3',  'SBS52': '#807dba',  'SBS53': '#9e9ac8',  'SBS54': '#bcbddc',  'SBS55': '#dadaeb',
                    'SBS56': '#f2f0f7',  'SBS57': '#e6550d',  'SBS58': '#fd8d3c',  'SBS59': '#fdae6b',  'SBS60': '#fdd0a2',  'SBS84': '#3c5a82',  'SBS85': '#ff4164',
                    'SBS86': '#2d969b',  'SBS87': '#f06937',  'SBS88': '#ef8892',  'SBS89': '#64508c',  'SBS90': '#557882',  'SBS91': '#a5236e',  'SBS92': '#05d7a0',  'SBS93': '#b44641',
                    'SBS94': '#5f7896',  'SBS95_MuSiCal_denovo': '#f06937',  'SBS95': '#373737',  'SBS96_MuSiCal_denovo': '#af9114',  'SBS96': '#0a509b',  'SBS97': '#a5236e',
                    'SBS97_MuSiCal_denovo': '#3c5a82',  'SBS98_MuSiCal_denovo': '#b44641',  'SBS98': '#2d969b',  'SBS99_MuSiCal_denovo': '#f06937',  'SBS99': '#6e5a7d',  'SBS100_MuSiCal_denovo': '#af9114',
                    'OOD': '#000000', 'Other': '#e2e2df',  'Other known': '#d9d9d9'}

SIG_COLORS = DEFAULT_SIGNATURE_COLORS.copy()  # legacy read-only compatibility; pass sig_colors= explicitly

# Exact aliases share colors across both notebooks.  This avoids cohort-dependent
# palette shifts when one cohort contains a different set of tissue labels.
TISSUE_COLORS = {
    "Biliary": "#8c510a", "Bladder": "#01665e", "Bone": "#35978f", "Breast": "#c51b7d",
    "CNS": "#5e3c99", "Cervix": "#e66101", "ColoRect": "#1b9e77", "colon": "#1b9e77",
    "intestine": "#66a61e", "Eso": "#d95f02", "Head": "#7570b3", "Kidney": "#a6761d",
    "Liver": "#e6ab02", "liver": "#e6ab02", "Lung": "#666666", "Lymph": "#1f78b4",
    "blood": "#1f78b4", "bonemarrow": "#6baed6", "spleen": "#9ecae1", "tonsil": "#c6dbef",
    "Myeloid": "#08519c", "Ovary": "#e7298a", "Panc": "#e31a1c", "Prost": "#fb9a99",
    "Skin": "#fdbf6f", "SoftTissue": "#cab2d6", "Stomach": "#b15928", "Thy": "#33a02c",
    "Uterus": "#ff7f00", "breast": "#c51b7d", "invitro": "#969696",
}

# Analysis functions default to all available methods when this tuple is empty.
ANALYSIS_NAMES = ()
OOD_MASS_METHODS = set()
GLOBAL_SIG_COLORS = {}  # legacy compatibility only; stacked bars never mutate or read this mapping


def make_signature_colors(signatures, base_colors=None, alignment=None, like_mix=0.28):
    """Create stable colors; source-like signatures inherit a tinted canonical color."""
    base_colors = {} if base_colors is None else dict(base_colors)
    alignment = pd.DataFrame() if alignment is None else alignment.copy()
    palette = {}
    fallback = ["#3c5a82", "#b44641", "#2d969b", "#f06937", "#6e5a7d", "#af9114", "#557882", "#a5236e"]
    for position, signature in enumerate(map(str, signatures)):
        if signature in base_colors:
            palette[signature] = base_colors[signature]
            continue
        match = alignment.loc[alignment["aligned_signature"].astype(str).eq(signature)] if len(alignment) else pd.DataFrame()
        parent = str(match.iloc[0]["matched_signature"]) if len(match) else None
        if parent in base_colors and "_like" in signature:
            rgb = np.asarray(to_rgb(base_colors[parent]))
            tint = np.ones(3) if "MuSiCal" in signature else np.zeros(3)
            palette[signature] = to_hex((1.0 - like_mix) * rgb + like_mix * tint)
        else:
            palette[signature] = fallback[position % len(fallback)]
    palette.setdefault("OOD", "#bdbdbd")
    palette.setdefault("Other", "#e2e2df")
    return palette


def geometry_metrics(raw_profile, DICT_res, k=35, max_pairs=200000, random_state=717, knn_graphs=None):
    raw_frame = raw_profile.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    raw_positive = raw_frame.sum(axis=1) > EPS
    raw_frame = raw_frame.loc[raw_positive]
    rng = np.random.default_rng(random_state)
    rows = []

    def _neighbors(values, n_neighbors):
        return NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(values).kneighbors(
            values, return_distance=False
        )[:, 1:]

    for method in DICT_res.keys():
        sig_frame = DICT_res[method]["compo"]
        exposure = sig_frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        exposure = exposure.loc[exposure.sum(axis=1) > EPS]
        common = raw_frame.index.intersection(exposure.index, sort=False)
        if len(common) < 3:
            rows.append(dict(method=method, n_samples=len(common), pairwise_spearman=np.nan,
                             knn_jaccard=np.nan, hellinger_neighbor_distance=np.nan,
                             median_active_signatures=np.nan, median_effective_signatures=np.nan))
            continue

        raw = sum_scale(raw_frame.loc[common])
        exposure = sum_scale(exposure.loc[common])
        raw_sqrt = np.sqrt(raw.to_numpy(float))
        exposure_sqrt = np.sqrt(exposure.to_numpy(float))
        k_eff = min(int(k), len(common) - 1)
        raw_neighbors = _neighbors(raw_sqrt, k_eff)
        neighbors = _neighbors(exposure_sqrt, k_eff)

        n_possible = len(common) * (len(common) - 1) // 2
        n_pairs = min(int(max_pairs), n_possible)
        if n_pairs == n_possible and n_possible <= int(max_pairs):
            first, second = np.triu_indices(len(common), 1)
        else:
            first, second = [], []
            while len(first) < n_pairs:
                a = rng.integers(0, len(common), size=max(2 * (n_pairs - len(first)), 1000))
                b = rng.integers(0, len(common), size=len(a))
                keep = a != b
                first.extend(a[keep].tolist())
                second.extend(b[keep].tolist())
            first = np.asarray(first[:n_pairs], dtype=int)
            second = np.asarray(second[:n_pairs], dtype=int)

        raw_distance = np.linalg.norm(raw_sqrt[first] - raw_sqrt[second], axis=1) / np.sqrt(2.0)
        exposure_distance = np.linalg.norm(exposure_sqrt[first] - exposure_sqrt[second], axis=1) / np.sqrt(2.0)
        rho = spearmanr(raw_distance, exposure_distance).statistic
        jaccard = [len(set(a).intersection(b)) / max(len(set(a).union(b)), 1)
                   for a, b in zip(raw_neighbors, neighbors)]
        raw_neighbor_distance = np.linalg.norm(raw_sqrt[:, None, :] - raw_sqrt[neighbors], axis=2) / np.sqrt(2.0)
        rows.append(dict(
            method=method, n_samples=len(common), pairwise_spearman=float(rho),
            knn_jaccard=float(np.mean(jaccard)),
            hellinger_neighbor_distance=float(np.mean(raw_neighbor_distance)),
            median_active_signatures=float((exposure > 0.01).sum(axis=1).median()),
            median_effective_signatures=float(np.median(
                1.0 / np.maximum((exposure * exposure).sum(axis=1), EPS))),
        ))
    return pd.DataFrame(rows).set_index("method")


def dataframe_fingerprint(frame, digits=12):
    """Stable SHA256 fingerprint for labels plus rounded numeric contents."""
    frame = frame.copy()
    payload = ["\x1f".join(map(str, frame.index)), "\x1f".join(map(str, frame.columns))]
    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    payload.append(np.round(numeric, int(digits)).tobytes().hex())
    return hashlib.sha256("\x1e".join(payload).encode("utf-8")).hexdigest()


def save_figure(fig, stem, dpi=220, formats=("png", "pdf"), close=False):
    """Save a figure without silently destroying the live notebook object.

    PNG is included for quick viewing. PDF and SVG remain vector outputs as long
    as the plotted artists were not explicitly created with ``rasterized=True``.
    ``close=False`` is intentional: callers can show the figure first and close
    it explicitly afterwards, which makes ``plt.show()`` reliable in notebooks.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        path = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved[fmt] = path
    if close:
        plt.close(fig)
    return saved


def run_umap_representation(frame, metadata, tag, output_dir, params, reference_tag=None):
    """Run the internal UMAP pipeline and persist its direct outputs.

    The input frame is row-normalized and all samples are processed in the supplied
    order.  Coordinates are optionally aligned to a previously computed reference
    embedding by a centred orthogonal Procrustes transform; distances, KNN graph,
    Leiden clustering, and UMAP fitting remain those of the original representation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = metadata.index[metadata.index.isin(frame.index)]
    data = frame.loc[index].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    zero_rows = data.sum(axis=1) <= EPS
    if zero_rows.any():
        if 'OOD' not in data.columns:
            data['OOD'] = 0.0
        data.loc[zero_rows, 'OOD'] = 1.0
    data = data.div(data.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    local_meta = metadata.loc[index].copy()
    params = dict(params)
    params.setdefault("init", "random")
    result = umap_pipeline(data, local_meta, tag=tag, **params)
    result["input_fingerprint"] = dataframe_fingerprint(data)
    xcol, ycol = f"UMAP_{tag}_1", f"UMAP_{tag}_2"
    coords = local_meta[[xcol, ycol]].to_numpy(float)
    aligned = coords.copy()
    if reference_tag is not None:
        rx, ry = f"UMAP_{reference_tag}_aligned_1", f"UMAP_{reference_tag}_aligned_2"
        if rx not in metadata.columns or ry not in metadata.columns:
            raise KeyError(f"Reference embedding '{reference_tag}' has not been computed.")
        target = metadata.loc[index, [rx, ry]].to_numpy(float)
        source_center = coords.mean(axis=0)
        target_center = target.mean(axis=0)
        source0 = coords - source_center
        target0 = target - target_center
        u, _, vt = np.linalg.svd(source0.T @ target0, full_matrices=False)
        rotation = u @ vt
        rotated = source0 @ rotation
        scale = np.sqrt(np.sum(target0 * target0) / max(np.sum(rotated * rotated), EPS))
        aligned = rotated * scale + target_center
    metadata.loc[index, xcol] = coords[:, 0]
    metadata.loc[index, ycol] = coords[:, 1]
    metadata.loc[index, f"UMAP_{tag}_aligned_1"] = aligned[:, 0]
    metadata.loc[index, f"UMAP_{tag}_aligned_2"] = aligned[:, 1]
    leiden_cols = [c for c in local_meta.columns if c.startswith(f"Leiden_{tag}_")]
    for col in leiden_cols:
        metadata.loc[index, col] = local_meta[col].astype(str)
    pd.DataFrame(coords, index=index, columns=[xcol, ycol]).assign(
        **{f"UMAP_{tag}_aligned_1": aligned[:, 0], f"UMAP_{tag}_aligned_2": aligned[:, 1]}
    ).to_csv(output_dir / f"UMAP_{tag}.tsv", sep="\t")
    save_npz(output_dir / f"KNN_{tag}.npz", result["knn_connectivity"])
    save_npz(output_dir / f"KNN_distance_{tag}.npz", result["knn_distance"])
    return result


def plot_umap_grid(metadata, panels, color_values, categorical=True, palette=None,
                   point_size=5, legend_max=35, panel_size=4,
                   title=None, rasterized=False):
    """Plot square UMAP panels using coordinates already computed in metadata."""
    n_panels = len(panels)
    ncols = 4 if n_panels >= 8 else 3
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*panel_size, nrows*panel_size),
                             constrained_layout=True, subplot_kw={"box_aspect": 1})
    axes = np.atleast_1d(axes).ravel()
    values = pd.Series(color_values, index=metadata.index)
    scatter = None
    if categorical:
        categories = values.fillna("NA").astype(str)
        order = sorted(categories.unique())
        cmap = plt.get_cmap("tab20")
        colors = {category: (palette or {}).get(category, cmap(i % 20)) for i, category in enumerate(order)}
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        vmin, vmax = float(numeric.quantile(0.02)), float(numeric.quantile(0.98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(numeric.min()), float(numeric.max() + EPS)
        norm = plt.Normalize(vmin, vmax)
    for ax, (label, tag) in zip(axes, panels):
        xcol, ycol = f"UMAP_{tag}_aligned_1", f"UMAP_{tag}_aligned_2"
        ok = metadata[[xcol, ycol]].notna().all(axis=1)
        if categorical:
            ax.scatter(metadata.loc[ok, xcol], metadata.loc[ok, ycol],
                       c=[colors[x] for x in categories.loc[ok]], s=point_size,
                       linewidths=0, rasterized=rasterized)
        else:
            scatter = ax.scatter(metadata.loc[ok, xcol], metadata.loc[ok, ycol],
                                 c=numeric.loc[ok], s=point_size, linewidths=0,
                                 cmap="viridis", norm=norm, rasterized=rasterized)
        x = metadata.loc[ok, xcol].to_numpy(float)
        y = metadata.loc[ok, ycol].to_numpy(float)
        span = max(np.ptp(x), np.ptp(y), EPS) * 1.10
        ax.set_xlim((x.min() + x.max() - span) / 2, (x.min() + x.max() + span) / 2)
        ax.set_ylim((y.min() + y.max() - span) / 2, (y.min() + y.max() + span) / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=7, length=2)
    for ax in axes[n_panels:]:
        ax.axis("off")
    if categorical:
        handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[c],
                          markeredgecolor="none", label=c, markersize=5) for c in order[:legend_max]]
        if len(order) > legend_max:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
                                  markeredgecolor="grey", label=f"... +{len(order) - legend_max}", markersize=5))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=7)
    elif scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[:n_panels].tolist(), fraction=0.018, pad=0.01,
                                label=values.name or "compo")
        if getattr(colorbar, "solids", None) is not None:
            colorbar.solids.set_rasterized(False)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig,axes


def _composition_from_obj(obj):
    """Return the canonical composition DataFrame, accepting legacy ``compo``."""
    if obj is None:
        return None
    frame = obj.get("composition")
    if frame is None:
        frame = obj.get("compo")
    return frame


def _resolve_methods(res, methods):
    methods = list(methods) if methods else list(res)
    return [name for name in methods if name in res]


def plot_cluster_dotplot(ax, res, locs, methods=ANALYSIS_NAMES, min_sample_frac=0.03, min_weight=0.03,
                         size_lim=(4, 90), cmap="viridis", size_by="pct_active", color_by="avg_active",
                         size_range=None, color_range=None, show_grid=True):
    """Reproduce the requested method-by-signature cluster dotplot.

    The visual encoding is intentionally unchanged. The only data-structure
    adaptation is that ``res[method]['composition']`` is canonical, while the
    historical ``'compo'`` key remains accepted.
    """
    methods = _resolve_methods(res, methods)
    locs = pd.Index(locs)
    rows, sigs = [], set()
    metric_labels = {"pct_active": "% active samples", "avg_active": "mean active level"}
    if size_by not in metric_labels or color_by not in metric_labels:
        raise ValueError(f"size_by/color_by must be one of {sorted(metric_labels)}")

    for name in methods:
        Y = _composition_from_obj(res.get(name, {}))
        if Y is None:
            continue
        Y = Y.reindex(locs).dropna(how="all").fillna(0.0)
        active = Y > min_weight
        for sig in Y.columns:
            pct = float(active[sig].mean()) if len(Y) else 0.0
            avg = float(Y.loc[active[sig], sig].mean()) if active[sig].any() else 0.0
            if pct >= min_sample_frac:
                sigs.add(sig)
            rows.append(dict(method=name, signature=sig, pct_active=pct, avg_active=avg))
    key = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", re.sub(r"[_-](PCAWG|MuSiCal)[_-](like|denovo)$", "", str(s)))]
    sigs = sorted(sigs, key=key)
    df = pd.DataFrame(rows)
    if not sigs or df.empty:
        ax.text(0.5, 0.5, "No active signatures", ha="center", va="center")
        return df
    df = df[df["signature"].isin(sigs) & df["method"].isin(methods)].copy()
    xmap = {s:i for i,s in enumerate(sigs)}
    y_methods = [m for m in methods if m in df["method"].unique()]
    ymap = {m:i for i,m in enumerate(y_methods)}
    d = df[df["pct_active"] > 0].copy()
    d["x"], d["y"] = d["signature"].map(xmap), d["method"].map(ymap)

    if size_range is None:
        size_range = (0.0, 1.0)
    lo, hi = map(float, size_range)
    denom = max(hi - lo, 1e-12)
    size_norm = np.clip((d[size_by].astype(float) - lo) / denom, 0, 1)
    d["size"] = size_lim[0] + (size_lim[1] - size_lim[0]) * size_norm

    if color_range is None:
        cmin, cmax = float(d[color_by].min()), float(d[color_by].max())
        if not np.isfinite(cmin) or not np.isfinite(cmax) or abs(cmax - cmin) < 1e-12:
            cmin, cmax = 0.0, 1.0
    else:
        cmin, cmax = map(float, color_range)

    sc = ax.scatter(d["x"], d["y"], s=d["size"], c=d[color_by], cmap=cmap,
                    vmin=cmin, vmax=cmax, linewidths=0, rasterized=False)
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels(sigs, rotation=80, ha="right", fontsize=8)
    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(list(ymap.keys()), fontsize=9)
    ax.set_xlabel("active signatures")
    ax.set_ylabel("methods")
    ax.set_xlim(-0.5, len(sigs)-0.5)
    ax.set_ylim(-0.5, len(ymap)-0.5)

    if show_grid:
        ax.set_xticks(np.arange(-0.5, len(sigs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ymap), 1), minor=True)
        ax.grid(which="minor", linewidth=0.4, alpha=0.35)
        ax.tick_params(which="minor", length=0)

    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(max(len(ymap), 1) / max(len(sigs), 1))
    except Exception:
        pass

    colorbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=metric_labels[color_by])
    if getattr(colorbar, "solids", None) is not None:
        colorbar.solids.set_rasterized(False)
    ax.set_title(f"dot size = {metric_labels[size_by]}; color = {metric_labels[color_by]}", fontsize=10)

    mass_rows = []
    for name in methods:
        Y = _composition_from_obj(res.get(name, {}))
        if Y is None:
            continue
        Y = Y.reindex(locs).dropna(how="all").fillna(0.0)
        denom_mass = Y.sum(axis=1).replace(0, np.nan)
        mass_rows.append(dict(method=name, n=len(Y), displayed_mass=float((Y.reindex(columns=sigs, fill_value=0).sum(axis=1)/denom_mass).mean())))
    display(pd.DataFrame(mass_rows).T)
    return df


def sig_natkey(s):
    s = re.sub(r"[_-](PCAWG|MuSiCal)[_-](like|denovo)$", "", str(s))
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def stack_sig_colors(res, sigs=None, base_colors=None):
    pal = sum((list(plt.get_cmap(x).colors) for x in ["tab20", "tab20b", "tab20c"]), [])
    base = dict(DEFAULT_SIGNATURE_COLORS)
    base.update({} if base_colors is None else base_colors)
    sigs = sigs or [c for o in res.values() if _composition_from_obj(o) is not None for c in _composition_from_obj(o).columns]
    colors = {}
    for i, signature in enumerate(sorted(map(str, set(sigs)), key=sig_natkey)):
        clean = re.sub(r"[_-](PCAWG|MuSiCal)[_-](like|denovo)$", "", signature)
        colors[signature] = base.get(signature, base.get(clean, pal[i % len(pal)]))
    colors.update({key: value for key, value in base.items() if key in {"OOD", "Other", "Other known"}})
    return colors


def stack_filter_sigs(res, locs, methods, min_sample_frac, min_weight):
    sigs, mats = set(), []
    for m in methods:
        Y = _composition_from_obj(res.get(m, {}))
        if Y is None:
            continue
        Y = Y.reindex(locs).dropna(how="all").fillna(0)
        Y.columns = Y.columns.map(str)
        sigs |= set(Y.columns[(Y > min_weight).mean(0) >= min_sample_frac])
        mats.append(Y)
    sigs = sorted(sigs, key=sig_natkey)
    avg = pd.concat(mats).reindex(columns=sigs, fill_value=0).mean().sort_values(ascending=False) if mats and sigs else pd.Series(dtype=float)
    return sigs, avg


def stack_ood_mass(obj, Y, ood_key="ood"):
    o = obj.get(ood_key)
    if o is None:
        return np.clip(1 - Y.sum(1).to_numpy(float), 0, None) if obj.get("tag") in OOD_MASS_METHODS else np.zeros(len(Y))
    o = o.reindex(Y.index)
    if isinstance(o, pd.DataFrame):
        o = o.apply(pd.to_numeric, errors="coerce").sum(1)
    else:
        o = pd.to_numeric(o, errors="coerce")
    return np.clip(np.nan_to_num(o.to_numpy(float), nan=0, posinf=0, neginf=0), 0, None)


def stack_thin(order, n=None, how="even", groups=None, min_per_group=8):
    order = list(order)
    if not n or n <= 0 or len(order) <= n:
        return order
    if groups is not None:
        groups = pd.Series(groups).reindex(order).astype(str)
        counts = groups.value_counts()
        alloc = {g: min(int(total), max(int(min_per_group), int(round(n * total / len(order))))) for g, total in counts.items()}
        while sum(alloc.values()) > n:
            candidates = [g for g in alloc if alloc[g] > min(int(min_per_group), int(counts[g]))]
            if not candidates:
                break
            alloc[max(candidates, key=lambda g: alloc[g])] -= 1
        while sum(alloc.values()) < n:
            candidates = [g for g in alloc if alloc[g] < counts[g]]
            if not candidates:
                break
            alloc[max(candidates, key=lambda g: counts[g] - alloc[g])] += 1
        keep = set()
        for group, size in alloc.items():
            loc = [sample for sample in order if groups.loc[sample] == group]
            pos = np.unique(np.rint(np.linspace(0, len(loc) - 1, size)).astype(int))
            keep.update(loc[i] for i in pos)
        return [sample for sample in order if sample in keep]
    if how == "first":
        return order[:n]
    if how == "random":
        keep = set(np.random.default_rng(GLOBAL_SEED).choice(order, n, replace=False))
        return [x for x in order if x in keep]
    idx = np.unique(np.rint(np.linspace(0, len(order) - 1, n)).astype(int))
    idx = sorted(list(idx) + [i for i in range(len(order)) if i not in set(idx)][:n - len(idx)])
    return [order[i] for i in idx[:n]]


def stack_umap_order(M, samples, UMAP_tag="SgF", use_aligned=True):
    a = [f"UMAP_{UMAP_tag}_aligned_1", f"UMAP_{UMAP_tag}_aligned_2"]
    r = [f"UMAP_{UMAP_tag}_1", f"UMAP_{UMAP_tag}_2"]
    cols = a if use_aligned and set(a) <= set(M.columns) else r if set(r) <= set(M.columns) else None
    if cols is None:
        return list(samples), {"mode": "input-fallback", "source": f"UMAP:{UMAP_tag}", "coord": "missing"}

    C = M.reindex(samples)[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(C) < 3:
        return list(C.index) + [x for x in samples if x not in C.index], {"mode": "input-fallback", "source": f"UMAP:{UMAP_tag}"}

    G = minimum_spanning_tree(squareform(pdist(C.to_numpy(float), "euclidean"))).toarray()
    G = G + G.T
    a0 = int(dijkstra(G, directed=False, indices=0).argmax())
    b0 = int(dijkstra(G, directed=False, indices=a0).argmax())
    start = b0 if str(C.index[b0]) < str(C.index[a0]) else a0
    score = dijkstra(G, directed=False, indices=start)
    order = list(C.index[np.argsort(score)])
    return order + [x for x in samples if x not in order], {
        "mode": "UMAP-MST-trajectory",
        "source": f"UMAP:{UMAP_tag}",
        "coord": "aligned" if cols == a else "raw",
    }


def stack_hclust_order(res, samples, sigs, hclust_tag="SigFormer", hclust_dist="cosine", hclust_met="average", hclust_opt="OLO", anchor=None, return_linkage=False):
    Y = _composition_from_obj(res.get(hclust_tag, {}))
    if Y is None:
        value = (list(samples), {"mode": "input-fallback", "source": hclust_tag})
        return (*value, None) if return_linkage else value
    Z = Y.reindex(samples).dropna(how="all").fillna(0)
    Z.columns = Z.columns.map(str)
    A = np.nan_to_num(Z.reindex(columns=sigs, fill_value=0).to_numpy(float), nan=0, posinf=0, neginf=0)
    if len(A) <= 2:
        value = (list(Z.index), {"mode": "input-fallback", "source": hclust_tag})
        return (*value, None) if return_linkage else value
    met = "ward" if hclust_met == "ward.D2" else hclust_met
    dist = "euclidean" if met == "ward" else hclust_dist
    A2 = pd.DataFrame(A).rank(axis=1).to_numpy(float) if dist == "spearman" else A
    d = np.nan_to_num(pdist(A2, "correlation" if dist == "pearson" else dist), nan=1, posinf=1, neginf=1)
    if len(d) == 0 or np.allclose(d, 0):
        value = (list(Z.index), {"mode": "input-fallback", "source": hclust_tag})
        return (*value, None) if return_linkage else value
    L = linkage(A, method="ward") if met == "ward" else linkage(d, method=met)
    if str(hclust_opt).upper() == "OLO":
        L = optimal_leaf_ordering(L, d)
    order = list(Z.index[leaves_list(L)])
    if anchor in Z.columns and len(order) and Z.loc[order[0], anchor] > Z.loc[order[-1], anchor]:
        order = order[::-1]
    value = (order, {"mode": f"hclust-{hclust_met}-{dist}-{hclust_opt}", "source": hclust_tag})
    return (*value, L) if return_linkage else value


def plot_cluster_stackbar(axes, res, M, locs, methods=ANALYSIS_NAMES, min_sample_frac=0.03, min_weight=0.03, sort="UMAP", UMAP_tag="SgF", hclust_tag="SigFormer", hclust_dist="cosine", hclust_met="average", hclust_opt="OLO", max_samples=30, sample_subset="even", use_aligned_umap=True, xtick_every=None, xtick_rotation=90, xtick_fontsize=6, legend_ncol=3, size_sig_legend=7, ood_key="ood", verbose=True, sig_colors=None, section_values=None, section_colors=None, section_labels=None, sigs=None):
    """Reproduce the requested per-method stacked-composition plot."""
    methods = _resolve_methods(res, methods)
    axes, locs = np.atleast_1d(axes).ravel(), pd.Index(locs)
    if sigs is None:
        sigs, sig_avg = stack_filter_sigs(res, locs, methods, min_sample_frac, min_weight)
    else:
        sigs = [str(sig) for sig in sigs]
        mats = [_composition_from_obj(res.get(method, {})).reindex(locs).fillna(0).reindex(columns=sigs, fill_value=0) for method in methods if _composition_from_obj(res.get(method, {})) is not None]
        sig_avg = pd.concat(mats).mean().sort_values(ascending=False) if mats else pd.Series(dtype=float)
    colors, sort0 = stack_sig_colors(res, sigs=sigs, base_colors=sig_colors), str(sort).lower()
    fallback_colors = dict(DEFAULT_SIGNATURE_COLORS)
    if sig_colors is not None:
        fallback_colors.update(sig_colors)

    if sort0 == "input":
        full, info = list(locs), {"mode": "input", "source": "locs"}
    elif sort0 == "hclust":
        full, info = stack_hclust_order(res, locs, sigs, hclust_tag, hclust_dist, hclust_met, hclust_opt)
    elif sort0 == "umap":
        full, info = stack_umap_order(M, locs, UMAP_tag, use_aligned_umap)
    else:
        raise ValueError('sort must be "input", "hclust", or "UMAP"')

    order, used = stack_thin(full, max_samples, sample_subset), []

    for ax, name in zip(axes, methods):
        obj = res.get(name, {})
        Y = _composition_from_obj(obj)
        if Y is None:
            ax.axis("off")
            continue

        Y = Y.reindex(order).fillna(0)
        Y.columns = Y.columns.map(str)
        x, bottom = np.arange(len(Y)), np.zeros(len(Y))

        for s in sigs:
            v = Y.get(s, pd.Series(0, index=Y.index)).to_numpy(float)
            if len(v) and np.nanmax(v) > 1e-12:
                h = ax.bar(x, v, bottom=bottom, width=1, color=colors.get(s, "#ccc"), linewidth=0)
                bottom += v
                used.append((s, h[0]))

        other = np.clip(Y.sum(1).to_numpy(float) - bottom, 0, None)
        if len(other) and other.max() > 1e-4:
            h = ax.bar(x, other, bottom=bottom, width=1, color=fallback_colors.get("Other known", fallback_colors.get("Other", "#d9d9d9")), linewidth=0)
            bottom += other
            used.append(("Other known", h[0]))

        ood = stack_ood_mass(obj, Y, ood_key)
        if len(ood) and ood.max() > 1e-4:
            h = ax.bar(x, ood, bottom=bottom, width=1, color=fallback_colors.get("OOD", "#000000"), linewidth=0)
            bottom += ood
            used.append(("OOD", h[0]))

        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xlim(-.5, max(len(Y)-.5, .5))
        ax.set_ylim(0, max(1, bottom.max()*1.02 if len(Y) else 1))
        ax.margins(x=0)
        ax.set_xticks([])

    for ax in axes[len(methods):]:
        ax.axis("off")

    if xtick_every and len(order):
        ax = axes[min(len(methods), len(axes))-1]
        ticks = np.arange(0, len(order), int(xtick_every))
        ax.set_xticks(ticks)
        ax.set_xticklabels([order[i] for i in ticks], rotation=xtick_rotation, ha="right", fontsize=xtick_fontsize)

    if section_values is not None and len(order):
        section_values = pd.Series(section_values).reindex(order).astype(str)
        section_colors = {} if section_colors is None else dict(section_colors)
        section_labels = {} if section_labels is None else dict(section_labels)
        strip = axes[0].inset_axes([0, 1.01, 1, 0.055])
        strip.imshow(np.array([[to_rgba(section_colors.get(value, "#cccccc")) for value in section_values]]), aspect="auto")
        strip.set_axis_off()
        bounds = section_values.ne(section_values.shift()).to_numpy().nonzero()[0]
        for boundary in bounds[1:]: axes[0].axvline(boundary - 0.5, color="white", lw=1)
        for section, index in section_values.groupby(section_values).groups.items():
            positions = np.array([order.index(sample) for sample in index])
            axes[0].text((positions.min() + positions.max()) / 2, 1.075, section_labels.get(section, section), ha="center", va="bottom", fontsize=7, linespacing=0.9, transform=axes[0].get_xaxis_transform())

    legend = {k: v for k, v in used}
    if legend:
        labs = sorted(legend, key=lambda z: (z not in ["OOD", "Other known"], sig_natkey(z)))
        axes[0].figure.legend(
            [legend[x] for x in labs], labs,
            bbox_to_anchor=(1.01, .5), loc="center left",
            frameon=False, fontsize=size_sig_legend, ncol=legend_ncol,
            columnspacing=.9, handlelength=1.2, handletextpad=.4,
        )

    if verbose:
        print(f"Stackbar config: sort={sort} | {info}")
        print(f"Stackbar samples: total={len(full)}, shown={len(order)}, max_samples={max_samples}, subset={sample_subset}")
        print(f"Stackbar signatures pass filter={len(sigs)} | natural order:", sigs)
        print("Stackbar signatures avg desc:", [(k, round(float(v), 4)) for k, v in sig_avg.items()])

    meta = {
        "sigs": sigs,
        "sig_avg_desc": sig_avg,
        "shown_order": order,
        "full_order": full,
        "order_info": info,
        "sig_colors": {s: colors[s] for s in sigs if s in colors},
    }
    return meta




####################################################################################################
### Replacement test
####################################################################################################
EPS = 1e-12
MM = 1 / 25.4

# Three 60-mm panels occupy 180 mm, leaving ~30 mm for margins/gaps on A4.
PANEL_W = 60 * MM
PANEL_SIZE = (PANEL_W, 48 * MM)
PANEL_SCATTER = (PANEL_W, 42 * MM)
PANEL_TALL = (PANEL_W, 58 * MM)
A4_TRIPLE_W = 180 * MM

FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND = 5.5, 6.0, 6.5, 5.2
POINT_SIZE, LINE_WIDTH = 6, 0.5

METHOD_COLORS = {
    "PCAWG official": "#bdbdbd", "SigFormer_raw": "#9ecae1", "SigFormer raw": "#9ecae1", "SigFormer": "#6da6ce", "SigFormer_hard_0.01": "#4f7f9f",
    "MuSiCal published": "#7f7f7f", "MuSiCal": "#ffac62", "MuSiCal refit": "#ffac62",
    "SigProfilerAssignment": "#e47273", "SigProfiler": "#e47273", "sigfit": "#b4918a",
    "sigLASSO": "#eda6d7", "SigLASSO": "#eda6d7", "signature.tools.lib": "#d3d46f", "sig.tool.lib": "#d3d46f",
}

METHOD_SHORT = {"PCAWG official": "PCAWG",
                "SigFormer_raw": "SigFormer raw",
                "SigFormer raw": "SigFormer raw",
                "SigFormer": "SigFormer",
                "SigFormer_hard_0.01": "SigFormer hard",
                "MuSiCal published": "MuSiCal pub.",
                "MuSiCal": "MuSiCal",
                "MuSiCal refit": "MuSiCal refit",
                "SigProfilerAssignment": "SigProfilerAssignment",
                "SigProfiler": "SigProfiler",
                "sigfit": "sigfit",
                "sigLASSO": "sigLASSO",
                "SigLASSO": "SigLASSO",
                "signature.tools.lib": "signature.tools.lib",
                "sig.tool.lib": "sig.tool.lib",}


_safe_name = lambda text: text.replace(" ", "_").replace(".", "_")


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)



def _normalize_ref(df_ref):
    ref = df_ref.astype(float).copy()
    return ref.div(ref.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def _normalize_q(q):
    q = np.clip(np.asarray(q, float), EPS, None)
    return q / q.sum()


def _multinomial_deviance(counts, q, scale=1e6):
    n = np.asarray(counts, float)
    if n.sum() <= 0:
        return np.nan

    p = np.clip(n / n.sum(), EPS, None)
    q = _normalize_q(q)
    return float(2 * np.sum(n * np.log(p / q)) / n.sum() * scale)


def _train_nll(counts, q, mask):
    n = np.asarray(counts, float)
    q = np.clip(np.asarray(q, float)[mask], EPS, None)
    q = q / q.sum()
    return float(-np.sum(n[mask] * np.log(q)))


def _fit_pool(counts, fixed, pool_profiles, pool_mass, mask=None, init=None):
    """Optimize nonnegative pool weights while preserving total pool mass."""
    pool_profiles = np.asarray(pool_profiles, float)
    k = pool_profiles.shape[0]

    if pool_mass <= EPS:
        return np.zeros(k)
    if k == 1:
        return np.array([pool_mass])
    if mask is None:
        mask = np.ones(len(counts), bool)

    if init is None or len(init) != k or np.sum(init) <= EPS:
        init = np.full(k, pool_mass / k)
    else:
        init = np.clip(np.asarray(init, float), 0, None)
        init = init / init.sum() * pool_mass

    def objective(w):
        return _train_nll(counts, fixed + w @ pool_profiles, mask)

    kwargs = dict(
        method="SLSQP",
        bounds=[(0, pool_mass)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - pool_mass},
        options={"maxiter": 80, "ftol": 1e-9},
    )
    fit = minimize(objective, init, **kwargs)

    if not fit.success or not np.isfinite(fit.fun):
        kwargs["options"] = {"maxiter": 160, "ftol": 1e-9}
        fit = minimize(objective, np.full(k, pool_mass / k), **kwargs)

    w = np.clip(fit.x, 0, None)
    return w / max(w.sum(), EPS) * pool_mass


# ============================================================
# Necessity test
# ============================================================
def replacement_test(df_profile,
                     df_compo,
                     df_refseq,
                     sig_source,
                     sig_target,
                     presence_threshold=1e-4,
                     score_scale=1e6,
):
    """
    Restricted model: source only.
    Full model: source + target.

    Positive score:
        deleting an originally present target worsens fit.

    Negative score:
        adding an originally absent target improves fit.
    """
    source, target = _as_list(sig_source), _as_list(sig_target)
    ref = _normalize_ref(df_refseq)

    missing = [s for s in source + target if s not in ref.index]
    if missing:
        raise KeyError(f"Missing reference signatures: {missing}")

    samples = df_profile.index.intersection(df_compo.index)
    profile = df_profile.loc[samples].astype(float)

    missing_channels = profile.columns.difference(ref.columns)
    if len(missing_channels):
        raise KeyError(
            f"Reference is missing mutation channels: "
            f"{missing_channels.tolist()}"
        )

    ref = ref.reindex(columns=profile.columns)
    compo = (
        df_compo.reindex(index=samples, columns=ref.index, fill_value=0)
        .fillna(0)
        .astype(float)
    )

    R = ref.to_numpy()
    pos = {s: i for i, s in enumerate(ref.index)}
    source_idx = [pos[s] for s in source]
    target_idx = [pos[s] for s in target]

    pool = source + target
    pool_idx = [pos[s] for s in pool]
    rows = []

    for smp in samples:
        n = profile.loc[smp].to_numpy(float)
        c = compo.loc[smp].to_numpy(float)

        original_target = c[target_idx].sum()
        present = original_target > presence_threshold
        pool_mass = c[pool_idx].sum()

        if n.sum() <= 0 or pool_mass <= EPS:
            rows.append({
                "sample": smp,
                "present": present,
                "testable": False,
                "original_target": original_target,
                "n_mut": n.sum(),
            })
            continue

        fixed = c @ R - c[pool_idx] @ ref.loc[pool].to_numpy()

        w0 = _fit_pool(
            n,
            fixed,
            ref.loc[source].to_numpy(),
            pool_mass,
            init=c[source_idx],
        )
        q0 = fixed + w0 @ ref.loc[source].to_numpy()

        init1 = np.r_[
            w0 * 0.9,
            np.full(len(target), pool_mass * 0.1 / len(target)),
        ]
        if present:
            init1 = c[pool_idx]

        w1 = _fit_pool(
            n,
            fixed,
            ref.loc[pool].to_numpy(),
            pool_mass,
            init=init1,
        )
        q1 = fixed + w1 @ ref.loc[pool].to_numpy()

        d0 = _multinomial_deviance(n, q0, score_scale)
        d1 = _multinomial_deviance(n, q1, score_scale)
        gain = max(0.0, d0 - d1)

        rows.append({
            "sample": smp,
            "present": present,
            "testable": True,
            "n_mut": n.sum(),
            "known_mass": c.sum(),
            "original_source": c[source_idx].sum(),
            "original_target": original_target,
            "optimized_source": w1[:len(source)].sum(),
            "optimized_target": w1[len(source):].sum(),
            "x_composition": (
                original_target if present else w1[len(source):].sum()
            ),
            "restricted_deviance": d0,
            "full_deviance": d1,
            "signed_deviance_gain": (1 if present else -1) * gain,
            **{f"optimized_{s}": w1[j] for j, s in enumerate(pool)},
        })

    return pd.DataFrame(rows).set_index("sample")


# ============================================================
# Specificity test
# ============================================================
def _fit_all_single_candidates(
    counts,
    fixed,
    source_shape,
    pool_mass,
    candidate_profiles,
    mask,
    grid,
):
    """Fit one source-to-candidate transfer fraction for each candidate."""
    f = grid[None, :, None]

    q = fixed[None, None, :] + pool_mass * (
        (1 - f) * source_shape[None, None, :]
        + f * candidate_profiles[:, None, :]
    )
    q = np.clip(q, EPS, None)

    n = np.asarray(counts, float)
    q_train = q[:, :, mask]
    q_train = q_train / q_train.sum(axis=2, keepdims=True)

    loss = -np.sum(n[None, None, mask] * np.log(q_train), axis=2)
    best = np.argmin(loss, axis=1)

    return grid[best], q[np.arange(len(candidate_profiles)), best]


def _fit_pair_candidate(
    counts,
    fixed,
    source_shape,
    pool_mass,
    target_profiles,
    mask,
    pair_grid,
):
    """Fit two-dimensional transfer for SBS7c + SBS7d-like targets."""
    f1, f2 = pair_grid[:, 0, None], pair_grid[:, 1, None]

    q = fixed[None, :] + pool_mass * (
        (1 - f1 - f2) * source_shape[None, :]
        + f1 * target_profiles[0][None, :]
        + f2 * target_profiles[1][None, :]
    )
    q = np.clip(q, EPS, None)

    n = np.asarray(counts, float)
    q_train = q[:, mask]
    q_train = q_train / q_train.sum(axis=1, keepdims=True)

    loss = -np.sum(n[None, mask] * np.log(q_train), axis=1)
    j = int(np.argmin(loss))

    return pair_grid[j], q[j]


def test_specificity(
    df_profile,
    df_compo,
    df_refseq,
    sig_source,
    sig_target,
    candidates=None,
    n_folds=8,
    presence_threshold=1e-4,
    recover_present=False,
    target_equivalents=None,
    score_scale=1e6,
    transfer_grid_size=51,
):
    """Mutation-channel-held-out predictive specificity test."""
    source, target = _as_list(sig_source), _as_list(sig_target)
    target_equivalents = list(
        dict.fromkeys(_as_list(target_equivalents or target))
    )
    ref = _normalize_ref(df_refseq)

    required = source + target + target_equivalents
    missing = [x for x in required if x not in ref.index]
    if missing:
        raise KeyError(f"Missing reference signatures: {missing}")

    if candidates is None:
        candidates = [
            x for x in ref.index
            if x not in source
            and "_like_" not in x
            and "_MuSiCal_" not in x
        ]
    candidates = list(dict.fromkeys(candidates))

    samples = df_profile.index.intersection(df_compo.index)
    profile = df_profile.loc[samples].astype(float)

    missing_channels = profile.columns.difference(ref.columns)
    if len(missing_channels):
        raise KeyError(
            f"Reference is missing mutation channels: "
            f"{missing_channels.tolist()}"
        )

    ref = ref.reindex(columns=profile.columns)
    compo = (
        df_compo.reindex(index=samples, columns=ref.index, fill_value=0)
        .fillna(0)
        .astype(float)
    )

    cand_profiles = ref.loc[candidates].to_numpy()
    pair_target = len(target) == 2
    target_label = "+".join(target)

    R = ref.to_numpy()
    pos = {x: i for i, x in enumerate(ref.index)}
    source_idx = [pos[x] for x in source]
    target_idx = [pos[x] for x in target]
    equiv_idx = [pos[x] for x in target_equivalents]

    fold_id = np.arange(profile.shape[1]) % n_folds
    grid = np.linspace(0, 1, transfer_grid_size)

    pair_axis = np.linspace(
        0,
        1,
        max(15, transfer_grid_size // 2),
    )
    pair_grid = np.array([
        (a, b)
        for a in pair_axis
        for b in pair_axis
        if a + b <= 1 + 1e-12
    ])

    rows = []

    for smp in samples:
        n = profile.loc[smp].to_numpy(float)
        c = compo.loc[smp].to_numpy(float)

        exact_target_mass = c[target_idx].sum()
        if not recover_present and exact_target_mass > presence_threshold:
            continue

        removed = list(dict.fromkeys(
            source + (target_equivalents if recover_present else [])
        ))
        removed_idx = [pos[x] for x in removed]
        pool_mass = c[removed_idx].sum()

        if pool_mass <= EPS or n.sum() <= 0:
            continue

        fixed = c @ R - c[removed_idx] @ ref.loc[removed].to_numpy()
        gain_num = np.zeros(len(candidates))
        pair_gain_num = 0.0

        for fold in range(n_folds):
            test_mask = fold_id == fold
            train_mask = fold_id != fold

            w0 = _fit_pool(
                n,
                fixed,
                ref.loc[source].to_numpy(),
                pool_mass,
                mask=train_mask,
                init=c[source_idx],
            )
            source_shape = (
                w0 @ ref.loc[source].to_numpy() / pool_mass
            )
            q0 = np.clip(
                fixed + pool_mass * source_shape,
                EPS,
                None,
            )

            _, q1 = _fit_all_single_candidates(
                n,
                fixed,
                source_shape,
                pool_mass,
                cand_profiles,
                train_mask,
                grid,
            )

            q0_test = q0[test_mask] / q0[test_mask].sum()
            q1_test = (
                q1[:, test_mask]
                / q1[:, test_mask].sum(axis=1, keepdims=True)
            )

            gain_num += 2 * np.sum(
                n[None, test_mask]
                * np.log(q1_test / q0_test[None, :]),
                axis=1,
            )

            if pair_target:
                _, q_pair = _fit_pair_candidate(
                    n,
                    fixed,
                    source_shape,
                    pool_mass,
                    ref.loc[target].to_numpy(),
                    train_mask,
                    pair_grid,
                )
                q_pair_test = q_pair[test_mask] / q_pair[test_mask].sum()
                pair_gain_num += 2 * np.sum(
                    n[test_mask] * np.log(q_pair_test / q0_test)
                )

        w0_all = _fit_pool(
            n,
            fixed,
            ref.loc[source].to_numpy(),
            pool_mass,
            init=c[source_idx],
        )
        source_shape_all = (
            w0_all @ ref.loc[source].to_numpy() / pool_mass
        )

        frac_all, _ = _fit_all_single_candidates(
            n,
            fixed,
            source_shape_all,
            pool_mass,
            cand_profiles,
            np.ones(len(n), bool),
            grid,
        )
        cv_gain = gain_num / n.sum() * score_scale

        for j, candidate in enumerate(candidates):
            rows.append({
                "sample": smp,
                "candidate": candidate,
                "cv_deviance_gain": cv_gain[j],
                "optimized_mass": pool_mass * frac_all[j],
                "original_exact_target": exact_target_mass,
                "original_target_equivalent": c[equiv_idx].sum(),
                "pool_mass": pool_mass,
            })

        if pair_target:
            frac_pair, _ = _fit_pair_candidate(
                n,
                fixed,
                source_shape_all,
                pool_mass,
                ref.loc[target].to_numpy(),
                np.ones(len(n), bool),
                pair_grid,
            )
            rows.append({
                "sample": smp,
                "candidate": target_label,
                "cv_deviance_gain": (
                    pair_gain_num / n.sum() * score_scale
                ),
                "optimized_mass": pool_mass * frac_pair.sum(),
                "original_exact_target": exact_target_mass,
                "original_target_equivalent": c[equiv_idx].sum(),
                "pool_mass": pool_mass,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    detail["rank"] = (
        detail.groupby("sample")["cv_deviance_gain"]
        .rank(ascending=False, method="average")
    )
    detail["percentile"] = detail.groupby("sample")["rank"].transform(
        lambda x: 1 - (x - 1) / max(len(x) - 1, 1)
    )

    summary = detail.groupby("candidate").agg(
        n=("sample", "size"),
        median_cv_gain=("cv_deviance_gain", "median"),
        mean_cv_gain=("cv_deviance_gain", "mean"),
        positive_fraction=("cv_deviance_gain", lambda x: np.mean(x > 0)),
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        top1_fraction=("rank", lambda x: np.mean(x <= 1)),
        top3_fraction=("rank", lambda x: np.mean(x <= 3)),
        mean_percentile=("percentile", "mean"),
        median_optimized_mass=("optimized_mass", "median"),
    ).sort_values(
        ["mean_rank", "median_cv_gain"],
        ascending=[True, False],
    )

    return detail, summary


# ============================================================
# Compact plotting utilities
# ============================================================
def _nice_ticks(vmin, vmax, min_ticks=5, max_ticks=8):
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        return np.array([vmin, vmax])

    ideal = (vmax - vmin) / 6
    center = int(np.floor(np.log10(ideal))) if ideal > 0 else 0
    choices = []

    for exp in range(center - 2, center + 3):
        for base in (1, 2, 5):
            step = base * 10.0 ** exp
            lo = np.floor(vmin / step) * step
            hi = np.ceil(vmax / step) * step
            n = int(round((hi - lo) / step)) + 1

            if 1 < n < 1000:
                penalty = (
                    0
                    if min_ticks <= n <= max_ticks
                    else min(
                        abs(n - min_ticks),
                        abs(n - max_ticks),
                    ) + 10
                )
                choices.append((
                    penalty,
                    abs(n - 6),
                    step,
                    lo + np.arange(n) * step,
                ))

    return min(
        choices,
        key=lambda x: (x[0], x[1], x[2]),
    )[3]


def _style_axis(ax):
    if hasattr(ax, "ax") and hasattr(ax, "outline"):
        ax.ax.tick_params(labelsize=FS_TICK, width=0.4, length=2, pad=1.5)
        ax.set_label(ax.ax.get_ylabel(), fontsize=FS_LABEL)
        ax.outline.set_linewidth(0.45)
        return
    ax.tick_params(
        axis="both",
        labelsize=FS_TICK,
        width=0.45,
        length=2.2,
        pad=1.5,
    )
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    ax.title.set_size(FS_TITLE)
    ax.xaxis.get_offset_text().set_fontsize(FS_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FS_TICK)

    for spine in ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)


_style_colorbar = lambda cbar: _style_axis(cbar)


def residual_cluster_summary(composition, residuals, metadata, cluster_col, tissue_col="tissue", background_sigs=("SBS1", "SBS5", "SBS40a"), no_split_clusters=("C8",), tissue_split_cutoff=0.60, distinct_cosine_cutoff=0.85, distinct_excess_cutoff=0.05, signature_label_cutoff=0.03, label_mode="compact"):
    """Aggregate positive residuals with the normal-tissue subgrouping rules used by the demos."""
    cluster_series = metadata[cluster_col].astype("string")
    statistics = {}
    for cluster in sorted(cluster_series.dropna().unique(), key=sig_natkey):
        locs = metadata.index[cluster_series.eq(cluster).fillna(False)]
        tissue = metadata.loc[locs, tissue_col].fillna("unknown").astype(str).value_counts(normalize=True)
        mean_composition = composition.loc[locs].mean()
        nonbackground = mean_composition.drop(labels=list(background_sigs), errors="ignore").clip(lower=0)
        nonbackground = nonbackground / nonbackground.sum() if nonbackground.sum() > 0 else nonbackground * 0
        statistics[str(cluster)] = {"locs": locs, "dominant_tissue": tissue.index[0], "dominant_fraction": float(tissue.iloc[0]), "nonbackground": nonbackground}
    groups, split_info = {}, {}
    for cluster, info in statistics.items():
        references = [item["nonbackground"] for name, item in statistics.items() if name != cluster and item["dominant_tissue"] == info["dominant_tissue"] and item["dominant_fraction"] >= tissue_split_cutoff]
        nearest_cosine, maximum_excess, distinct = np.nan, 0.0, False
        if info["dominant_fraction"] < tissue_split_cutoff and references:
            query = info["nonbackground"]
            ref = pd.DataFrame([item.reindex(query.index, fill_value=0) for item in references])
            q = query.to_numpy(float)
            qn = np.linalg.norm(q)
            sims = [float(np.dot(q, row) / (qn * np.linalg.norm(row))) if qn * np.linalg.norm(row) > 0 else 0.0 for row in ref.to_numpy(float)]
            nearest_cosine = max(sims)
            maximum_excess = float((query - ref.max(axis=0)).max())
            distinct = nearest_cosine < distinct_cosine_cutoff and maximum_excess >= distinct_excess_cutoff
        locs = info["locs"]
        tissue_values = metadata.loc[locs, tissue_col].fillna("unknown").astype(str)
        dominant_locs = locs[tissue_values.eq(info["dominant_tissue"])]
        other_locs = locs.difference(dominant_locs)
        split = cluster not in set(no_split_clusters) and info["dominant_fraction"] < tissue_split_cutoff and not distinct and len(dominant_locs) > 0 and len(other_locs) > 0
        if split:
            groups[f"{cluster}.{info['dominant_tissue']}"] = dominant_locs
            groups[f"{cluster}.other"] = other_locs
        else: groups[cluster] = locs
        split_info[cluster] = {"parent_cluster": cluster, "parent_dominant_tissue": info["dominant_tissue"], "parent_dominant_fraction": info["dominant_fraction"], "distinct_nonbackground_pattern": distinct, "nearest_same_tissue_cosine": nearest_cosine, "maximum_signature_excess": maximum_excess, "split": split, "split_blocked": cluster in set(no_split_clusters)}
    aggregate, labels, records = {}, {}, []
    for subgroup, locs in groups.items():
        parent = subgroup.split(".", 1)[0]
        positive = sum_scale(residuals.loc[locs].clip(lower=0))
        mean_positive = positive.mean(axis=0)
        mean_composition = composition.loc[locs].mean()
        tissue = metadata.loc[locs, tissue_col].fillna("unknown").astype(str).value_counts(normalize=True)
        selected = mean_composition.drop(labels=list(background_sigs), errors="ignore")
        selected = selected[selected >= signature_label_cutoff].nlargest(3)
        signature_label = ", ".join(f"{signature} {value:.0%}" for signature, value in selected.items())
        dominant_signature = signature_label.split(",", 1)[0].split()[0] if signature_label else None
        detailed = " | ".join([subgroup, f"{tissue.index[0].title()} {float(tissue.iloc[0]):.0%}"] + ([signature_label] if signature_label else []))
        compact = f"{subgroup} | {'/'.join(filter(None, [tissue.index[0].title(), dominant_signature]))}"
        aggregate[subgroup] = mean_positive
        labels[subgroup] = detailed if label_mode == "detailed" else compact
        records.append({"cluster": subgroup, "parent_cluster": parent, "n": len(locs), "dominant_tissue": tissue.index[0], "dominant_tissue_fraction": float(tissue.iloc[0]), "composition_label": signature_label, "label_detailed": detailed, "label_compact": compact, **split_info[parent]})
    aggregate = sum_scale(pd.DataFrame(aggregate).T)
    normalized = aggregate.to_numpy(float)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), EPS)
    similarity_full = np.clip(normalized @ normalized.T, -1, 1)
    distance = np.clip(1 - similarity_full, 0, 2)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    if len(aggregate) > 1:
        condensed = squareform(distance, checks=False)
        tree = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
        order = leaves_list(tree)
    else: order = np.array([0])
    ordered = aggregate.index[order]
    similarity = pd.DataFrame(similarity_full[np.ix_(order, order)], index=ordered, columns=ordered)
    return aggregate, similarity, pd.DataFrame(records), pd.Series(labels), pd.Index(ordered)


def _savefig(fig, fig_path):
    if fig_path:
        fig.savefig(
            fig_path,
            bbox_inches="tight",
            pad_inches=0.02,
        )


# ============================================================
# Necessity plots
# ============================================================
def plot_replacement_scatter(
    results,
    title,
    fig_path=None,
    method_colors=METHOD_COLORS,
):
    frames = (
        {"Result": results}
        if isinstance(results, pd.DataFrame)
        else results
    )
    multiple = len(frames) > 1

    fig, ax = plt.subplots(
        figsize=PANEL_SCATTER,
        constrained_layout=True,
    )
    all_x, all_y = [], []

    for method, df in frames.items():
        z = df[
            df["testable"].fillna(False)
        ].dropna(subset=[
            "x_composition",
            "signed_deviance_gain",
        ])

        for present, marker in [(True, "o"), (False, "X")]:
            g = z[z["present"].eq(present)]
            if g.empty:
                continue

            color = (
                method_colors.get(method, "#7f7f7f")
                if multiple
                else ("#bdbdbd" if present else "#3084b7")
            )
            label = (
                method
                if multiple and present
                else (
                    None
                    if multiple
                    else (
                        "Originally present"
                        if present
                        else "Originally absent"
                    )
                )
            )

            ax.scatter(
                g["x_composition"],
                g["signed_deviance_gain"],
                s=POINT_SIZE,
                marker=marker,
                c=color,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.2,
                label=label,
                zorder=3,
            )

            all_x.extend(g["x_composition"])
            all_y.extend(g["signed_deviance_gain"])

    ax.axhline(
        0,
        color="0.55",
        ls="--",
        lw=0.7,
    )

    xmax = max(all_x) if all_x else 1
    ymin, ymax = (
        (min(all_y), max(all_y))
        if all_y
        else (-1, 1)
    )
    pad = max((ymax - ymin) * 0.05, 1e-9)

    # X-axis is never allowed to extend beyond 1.
    x_upper = min(max(xmax * 1.05, 0.05), 1.0)

    ax.set_xlim(0, x_upper)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xticks(_nice_ticks(0, x_upper))
    ax.set_yticks(_nice_ticks(ymin - pad, ymax + pad))

    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(-3, 4),
        useMathText=True,
    )

    ax.set_xlabel("Target exposure (original/optimized)")
    ax.set_ylabel("Signed deviance gain / $10^6$")
    ax.set_title(title)
    _style_axis(ax)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            frameon=False,
            fontsize=FS_LEGEND,
            ncol=2 if multiple else 1,
            markerscale=0.85,
            handletextpad=0.25,
            columnspacing=0.55,
            borderaxespad=0.2,
        )

    _savefig(fig, fig_path)
    return fig, ax


def plot_specificity(
    summary,
    target_label,
    title,
    fig_path=None,
    top_n=12,
):
    z = (
        summary.sort_values("median_cv_gain", ascending=False)
        .head(top_n)
        .copy()
    )

    if target_label in summary.index and target_label not in z.index:
        z = pd.concat([
            z.iloc[:-1],
            summary.loc[[target_label]],
        ])

    z = z.sort_values("median_cv_gain")
    colors = [
        "#3084b7" if x == target_label else "#bdbdbd"
        for x in z.index
    ]

    fig, ax = plt.subplots(
        figsize=PANEL_TALL,
        constrained_layout=True,
    )

    ax.barh(
        z.index,
        z["median_cv_gain"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axvline(0, color="0.55", ls="--", lw=0.7,)

    ax.set_xlabel("Median held-out deviance gain / $10^6$")
    ax.set_title(title)

    span = max(np.nanmax(np.abs(z["median_cv_gain"].to_numpy())), EPS,)

    for y, (_, row) in enumerate(z.iterrows()):
        value = row["median_cv_gain"]
        ax.text(value + np.sign(value or 1) * span * 0.02, y,
                f"rank {row['mean_rank']:.1f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=FS_TICK,)

    ax.margins(x=0.18)
    _style_axis(ax)
    _savefig(fig, fig_path)

    return fig, ax


def plot_optimized_vs_sigformer(method_results, df_sigformer, sig_target, title, fig_path=None,):
    target = _as_list(sig_target)
    sf = (df_sigformer.reindex(columns=target, fill_value=0).sum(axis=1))

    methods = [method
               for method in method_results
               if method != "SigFormer"]

    ncols = 3
    nrows = math.ceil(len(methods) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(A4_TRIPLE_W, nrows * 48 * MM),
                             constrained_layout=True, squeeze=False,)
    stats = []

    for ax, method in zip(axes.flat, methods):
        z = method_results[method]
        est = z["original_target"].where(z["present"], z["optimized_target"],)

        idx = sf.index.intersection(est.dropna().index)
        x = sf.loc[idx].to_numpy()
        y = est.loc[idx].to_numpy()

        lim = max(np.max(x) if len(x) else 0,
                  np.max(y) if len(y) else 0,) * 1.05 or 1

        ax.scatter(x, y, s=POINT_SIZE,
                   alpha=0.8, c=METHOD_COLORS.get(method, "#7f7f7f"),
                   edgecolors="black", linewidths=0.2,)
        ax.plot([0, lim], [0, lim], color="0.55", ls="--", lw=0.7,)

        ax.set(xlim=(0, lim), xlabel="SigFormer",
               ylim=(0, lim), ylabel="Observed/optimized",
               title=METHOD_SHORT.get(method, method),)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(_nice_ticks(0, lim, 3, 5))
        ax.set_yticks(_nice_ticks(0, lim, 3, 5))
        _style_axis(ax)

        rho, p = (spearmanr(x, y)
                  if len(x) > 2
                  else (np.nan, np.nan))
        diff = y - x
        pw = (wilcoxon(diff).pvalue
              if len(diff) and np.any(np.abs(diff) > 0)
              else np.nan)

        stats.append({"method": method,
            "n": len(x),
            "spearman_rho": rho,
            "spearman_p": p,
            "median_difference": (
                np.median(diff) if len(diff) else np.nan
            ),
            "wilcoxon_p": pw,
        })

    for ax in axes.flat[len(methods):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=7)
    _savefig(fig, fig_path)

    return (fig, axes, pd.DataFrame(stats).set_index("method"),)




