#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s05_Train.py

SigFormer training script.

This file is the executable training entry point. The sections run top-to-bottom,
and each training run snapshots the relevant source files into ``out_dir/0_scripts``
for reproducibility.
"""

# ============================================================
# library session
# ============================================================

from __future__ import annotations

import argparse, contextlib, copy, datetime as dt, gc, json, math, os, pickle, random, shutil, sys, subprocess, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s01_Core import SigFormerCore
from . import s02_Data as data_v9
from .s03_Util_train import (
    ProgressMeter, StepTimer, curriculum, str2bool, current_stamp, seed_everything,
    copy_run_sources, save_pickle, load_pickle, model_without_parallel,
    should_use_data_parallel, model_parameter_summary, build_ref_bank, training_ref_bank,
    batch_from_curriculum, build_eval_data, ensure_dir, json_safe, format_seconds,
    loss__mse, plot_learn_curve, print_log,
)
from .s04_Util_apply import (
    calc___R2,
    calc__cos,
    plot_scatter_compo,
    sum_scale,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# CLI helpers and training utilities
# ============================================================


# ============================================================
# Curriculum
# ============================================================


# ============================================================
# define classes and functions: reference bank / data generation
# ============================================================


# COSMIC can be built from one or more bundled/user paths. Duplicate
# signature IDs are dropped after the first occurrence, so mixing genome/version
# files does not silently create repeated SBS1/SBS2/etc. clones.


# ============================================================
# define classes and functions: tensor conversion / losses
# ============================================================


def batch_to_tensors(batch: Dict[str, Any], device: torch.device, mode: str = "high_prior") -> Dict[str, torch.Tensor]:
    """Convert a generated batch into model tensors.

    ``mode='high_prior'`` uses sample-specific ``Y_compo_mask``. ``mode='full'``
    makes every reference visible and sets target OOD mass to zero, useful for
    the full-spectrum evaluation requested by the user.
    """
    X = torch.as_tensor(batch["X_profl_data"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).contiguous()
    R = torch.as_tensor(batch["R_refsigpool"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).contiguous()
    B = X.shape[0]
    R_batched = R.unsqueeze(0).expand(B, -1, -1).contiguous()
    y_true = torch.as_tensor(batch["Y_compo_true"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).contiguous()
    y_mask = torch.as_tensor(batch["Y_compo_mask"].to_numpy(dtype=bool), dtype=torch.bool, device=device).contiguous()
    y_ood_mask = torch.as_tensor(batch["Y__OOD__mask"].to_numpy(dtype=bool), dtype=torch.bool, device=device).contiguous()
    if mode == "full":
        ref_mask = torch.ones_like(y_mask, dtype=torch.bool, device=device)
        y_known = y_true
        y_ood_mass = torch.zeros(B, dtype=torch.float32, device=device)
    else:
        ref_mask = y_mask
        y_known = y_true * y_mask.float()
        y_ood_mass = (y_true * y_ood_mask.float()).sum(dim=1)
    y_ext = torch.cat([y_known, y_ood_mass.unsqueeze(1)], dim=1).contiguous()
    target_recon_known = torch.bmm(y_known.unsqueeze(1), R_batched).squeeze(1).contiguous()
    target_profile_full = torch.as_tensor(batch["X_profl_true"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).contiguous()
    return {
        "X": X,
        "R": R,
        "R_batched": R_batched,
        "ref_mask": ref_mask,
        "y_true": y_true,
        "y_known": y_known,
        "y_ext": y_ext,
        "y_ood_mass": y_ood_mass,
        "y_ood_mask": y_ood_mask,
        "target_recon_known": target_recon_known,
        "target_profile_full": target_profile_full,
    }


def known_composition_cell_mask(batch: Dict[str, Any], mode: str) -> pd.DataFrame:
    """Cells that belong in known-signature composition evaluation.

    OOD-held-out references are excluded even in full-reference plots. Otherwise
    the figure happily draws true-composition=1 OOD cells in a panel named
    "known composition", because apparently labels enjoy gaslighting too.
    """
    y_true = batch["Y_compo_true"]
    not_ood = ~batch["Y__OOD__mask"].astype(bool).reindex_like(y_true).fillna(False)
    if mode == "high_prior":
        visible = batch["Y_compo_mask"].astype(bool).reindex_like(y_true).fillna(False)
        mask = visible & not_ood
    else:
        mask = not_ood
    return mask.astype(bool)


def true_known_composition(batch: Dict[str, Any], mode: str) -> pd.DataFrame:
    """Ground-truth known composition for a mode, with OOD cells removed."""
    y_true = batch["Y_compo_true"].copy()
    mask = known_composition_cell_mask(batch, mode)
    return y_true.where(mask, 0.0).astype(float)


def pred_known_for_eval(df_pred: pd.DataFrame, cell_mask: pd.DataFrame) -> pd.DataFrame:
    """Predicted known composition aligned to an evaluation cell mask."""
    aligned = df_pred.reindex_like(cell_mask).fillna(0.0)
    return aligned.where(cell_mask.astype(bool), 0.0).astype(float)


def known_reconstruction_from_composition(df_compo: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Known-only profile reconstruction from a composition matrix."""
    R_use = R.reindex(df_compo.columns).fillna(0.0)
    vals = df_compo.to_numpy(dtype=float) @ R_use.to_numpy(dtype=float)
    return pd.DataFrame(vals, index=df_compo.index.copy(), columns=R_use.columns.copy())

def infer_batch(model: nn.Module, batch: Dict[str, Any], device: torch.device, mode: str, simplex: str = "softmax") -> Dict[str, Any]:
    """Run one batch through SigFormer and return tensors/dataframes."""
    tensors = batch_to_tensors(batch, device=device, mode=mode)
    with torch.no_grad():
        extra = model(tensors["X"], tensors["R_batched"], simplex=simplex, ref_mask=tensors["ref_mask"], return_extra=True)
        pred_known = extra["known_composition"].detach()
        ood_mass = extra["ood_mass"].detach()
        recon_known = torch.bmm(pred_known.unsqueeze(1), tensors["R_batched"]).squeeze(1)
    refs = batch["R_refsigpool"].index.copy()
    samples = batch["X_profl_data"].index.copy()
    ctx = batch["R_refsigpool"].columns.copy()
    df_pred = pd.DataFrame(pred_known.cpu().numpy(), index=samples, columns=refs)
    df_recon = pd.DataFrame(recon_known.cpu().numpy(), index=samples, columns=ctx)
    df_ood = pd.DataFrame({"ood_mass_pred": ood_mass.cpu().numpy(), "ood_mass_true": tensors["y_ood_mass"].detach().cpu().numpy()}, index=samples)
    return {"pred_compo": df_pred, "recon": df_recon, "ood": df_ood, "tensors": tensors}


def compute_losses_from_extra(extra: Dict[str, torch.Tensor], tensors: Dict[str, torch.Tensor], lmda_compo: float, lmda_recon: float, lmda_ood: float) -> Dict[str, torch.Tensor]:
    """Compute training loss components.

    Values stored in logs are raw components. Only ``loss_total`` applies lambda
    weights. This is the small mercy that saves future plotting from guessing
    whether a number has been multiplied already.
    """
    pred_known = extra["known_composition"]
    pred_ood = extra["ood_mass"]
    pred_ext = torch.cat([pred_known, pred_ood.unsqueeze(1)], dim=1)
    pred_recon_known = torch.bmm(pred_known.unsqueeze(1), tensors["R_batched"]).squeeze(1)
    loss_compo = loss__mse(pred_ext, tensors["y_ext"])
    loss_recon = loss__mse(pred_recon_known, tensors["target_recon_known"])
    no_ood = (tensors["y_ood_mass"] <= 1e-8).float()
    fp_ood = ((pred_ood ** 2) * no_ood).mean()
    mse_ood = loss__mse(pred_ood, tensors["y_ood_mass"])
    loss_ood = mse_ood + 0.25 * fp_ood
    loss_total = lmda_compo * loss_compo + lmda_recon * loss_recon + lmda_ood * loss_ood
    return {"loss_total": loss_total, "loss_compo": loss_compo, "loss_recon": loss_recon, "loss_ood": loss_ood, "loss_ood_mse": mse_ood, "loss_ood_fp": fp_ood}


def grad_norm(model: nn.Module) -> float:
    """Compute global L2 gradient norm."""
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        val = float(p.grad.detach().data.norm(2).cpu())
        total += val * val
    return math.sqrt(total)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


# ============================================================
# define classes and functions: evaluation metrics and plots
# ============================================================


def df_metrics_by_sample(df_true: pd.DataFrame, df_pred: pd.DataFrame, df_recon_true: Optional[pd.DataFrame] = None, df_recon_pred: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-sample composition/reconstruction metrics."""
    df_pred = df_pred.reindex_like(df_true).fillna(0.0)
    rows = []
    for sid in df_true.index:
        row = {"sample_id": sid}
        row["composition_MSE"] = float(np.mean((df_true.loc[sid].values - df_pred.loc[sid].values) ** 2))
        row["composition_Cosine"] = calc__cos(df_true.loc[[sid]], df_pred.loc[[sid]])
        row["composition_R2"] = calc___R2(df_true.loc[[sid]], df_pred.loc[[sid]])
        if df_recon_true is not None and df_recon_pred is not None:
            rp = df_recon_pred.reindex_like(df_recon_true).fillna(0.0)
            row["reconstruction_MSE"] = float(np.mean((df_recon_true.loc[sid].values - rp.loc[sid].values) ** 2))
            row["reconstruction_Cosine"] = calc__cos(df_recon_true.loc[[sid]], rp.loc[[sid]])
            row["reconstruction_R2"] = calc___R2(df_recon_true.loc[[sid]], rp.loc[[sid]])
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id")


def f1_flat(y_true, y_pred, thr: float = 0.01) -> float:
    """F1 with strict >0.01 positive threshold."""
    yt = np.asarray(y_true, dtype=float).ravel() > float(thr)
    yp = np.asarray(y_pred, dtype=float).ravel() > float(thr)
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def summarize_eval_pair(batch: Dict[str, Any], res: Dict[str, Any], depth: str, noise: str, mode: str, ood_source: str) -> Dict[str, Any]:
    """Summary metrics for one eval batch/mode using known-composition cells only."""
    cell_mask = known_composition_cell_mask(batch, mode)
    true_compo = true_known_composition(batch, mode)
    pred_compo = pred_known_for_eval(res["pred_compo"], cell_mask)
    ood_true = res["ood"]["ood_mass_true"].to_numpy(dtype=float)
    ood_pred = res["ood"]["ood_mass_pred"].to_numpy(dtype=float)
    has_ood = ood_true > 0.01
    keep_vals = cell_mask.to_numpy(dtype=bool)
    tvals = true_compo.to_numpy(dtype=float)[keep_vals]
    pvals = pred_compo.to_numpy(dtype=float)[keep_vals]
    return {
        "depth": depth,
        "noise": noise,
        "ood_source": ood_source,
        "mode": mode,
        "n_samples": int(true_compo.shape[0]),
        "n_with_ood": int(np.sum(has_ood)),
        "MSE": float(np.mean((tvals - pvals) ** 2)) if tvals.size else float("nan"),
        "R2": calc___R2(tvals, pvals),
        "F1": f1_flat(tvals, pvals, thr=0.01),
        "Cosine": calc__cos(tvals, pvals),
        "ood_MSE": float(np.mean((ood_true - ood_pred) ** 2)),
        "ood_R2": calc___R2(ood_true, ood_pred),
        "ood_F1": f1_flat(ood_true, ood_pred, thr=0.01),
        "ood_pred_mean": float(np.mean(ood_pred)),
        "ood_true_mean": float(np.mean(ood_true)),
    }


def grouped_composition_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame, has_ood: np.ndarray, cell_mask: Optional[pd.DataFrame] = None) -> Dict[str, Tuple[float, float]]:
    """Compute separate R2/F1 for no-OOD and OOD sample rows."""
    out: Dict[str, Tuple[float, float]] = {}
    if cell_mask is None:
        cell_mask = pd.DataFrame(True, index=df_true.index, columns=df_true.columns)
    cell_mask = cell_mask.reindex_like(df_true).fillna(False).astype(bool)
    df_pred = df_pred.reindex_like(df_true).fillna(0.0)
    for label, row_mask in [("noOOD", ~has_ood), ("OOD", has_ood)]:
        if not np.any(row_mask):
            out[label] = (float("nan"), float("nan"))
            continue
        rows = np.where(row_mask)[0]
        keep = cell_mask.iloc[rows].to_numpy(dtype=bool)
        t = df_true.iloc[rows].to_numpy(dtype=float)[keep]
        p = df_pred.iloc[rows].to_numpy(dtype=float)[keep]
        out[label] = (calc___R2(t, p), f1_flat(t, p, thr=0.01)) if t.size else (float("nan"), float("nan"))
    return out


def plot_grid_compo_scatter(
    all_records: List[Dict[str, Any]],
    out_png: Path,
    mode: str,
    title: str,
    max_points_per_group: int = 80000,
) -> None:
    """Grid scatter with OOD cells excluded from known-composition panels."""
    depth_bins = ["100-400", "401-2000", "2001-7000", "7000-100000"]
    noise_bins = ["0.85-0.90", "0.90-0.95", "0.95-1.00"]
    fig, axes = plt.subplots(len(depth_bins), len(noise_bins), figsize=(14, 16), constrained_layout=False)
    for i, depth in enumerate(depth_bins):
        for j, noise in enumerate(noise_bins):
            ax = axes[i, j]
            subset = [r for r in all_records if r["depth"] == depth and r["noise"] == noise and r["mode"] == mode]
            if not subset:
                ax.set_axis_off()
                continue
            true_list, pred_list, meta_list, mask_list = [], [], [], []
            for r in subset:
                true_list.append(r["true_compo"])
                pred_list.append(r["pred_compo"])
                meta_list.append(r["meta"])
                mask_list.append(r.get("cell_mask", pd.DataFrame(True, index=r["true_compo"].index, columns=r["true_compo"].columns)))
            df_true = pd.concat(true_list, axis=0)
            df_pred = pd.concat(pred_list, axis=0).reindex_like(df_true).fillna(0.0)
            cell_mask = pd.concat(mask_list, axis=0).reindex_like(df_true).fillna(False).astype(bool)
            meta = pd.concat(meta_list, axis=0).reindex(df_true.index)
            has_ood = meta["n_ood"].astype(float).to_numpy() > 0
            keep = cell_mask.to_numpy(dtype=bool).ravel()
            x_all = df_true.to_numpy(dtype=float).ravel()
            y_all = df_pred.to_numpy(dtype=float).ravel()
            sample_ood_all = np.repeat(has_ood, df_true.shape[1])
            x = x_all[keep]
            y = y_all[keep]
            sample_ood = sample_ood_all[keep]
            if x.size > max_points_per_group:
                rng = np.random.default_rng(args.seed + i * 10 + j)
                idx = rng.choice(x.size, size=max_points_per_group, replace=False)
                x, y, sample_ood = x[idx], y[idx], sample_ood[idx]
            ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=0.8, alpha=0.6)
            ax.scatter(x[~sample_ood], y[~sample_ood], s=2, alpha=0.16, c="#0a509b", edgecolors="none", linewidths=0, label="w/o OOD")
            if np.any(sample_ood):
                ax.scatter(x[sample_ood], y[sample_ood], s=2, alpha=0.16, c="#a50f14", edgecolors="none", linewidths=0, label="with OOD")
            gm = grouped_composition_metrics(df_true, df_pred, has_ood, cell_mask=cell_mask)
            txt = (
                f"noOOD R2/F1={gm['noOOD'][0]:.3f}/{gm['noOOD'][1]:.3f}\n"
                f"OOD   R2/F1={gm['OOD'][0]:.3f}/{gm['OOD'][1]:.3f}\n"
                f"known cells={int(keep.sum())}"
            )
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.72))
            ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="true known composition", ylabel="pred known composition")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{depth} | {noise}", fontsize=9)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.93, hspace=0.38, wspace=0.28)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _metric_plot_values(vals: pd.Series, col: str) -> Tuple[np.ndarray, str, int, int]:
    """Return capped values, note text, n_low_cap, n_high_cap for histograms."""
    arr = vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return arr, "", 0, 0
    n_low = n_high = 0
    note = ""
    if col.endswith("_R2"):
        n_low = int(np.sum(arr < -1.0))
        arr = np.clip(arr, -1.0, 1.0)
        note = f"R²<-1 capped: {n_low}" if n_low else ""
    elif col.endswith("_Cosine"):
        n_low = int(np.sum(arr < 0.0))
        n_high = int(np.sum(arr > 1.0))
        arr = np.clip(arr, 0.0, 1.0)
        note = f"outside [0,1] capped: {n_low+n_high}" if (n_low+n_high) else ""
    else:
        # MSE can have a long right tail. Cap at a robust high percentile so one
        # catastrophic sample does not flatten the whole panel into a barcode.
        hi = float(np.nanquantile(arr, 0.995)) if arr.size > 20 else float(np.nanmax(arr))
        hi = max(hi, 1e-12)
        n_high = int(np.sum(arr > hi))
        arr = np.clip(arr, 0.0, hi)
        note = f">p99.5 capped: {n_high}" if n_high else ""
    return arr, note, n_low, n_high


def plot_hist_metrics(all_sample_metrics: pd.DataFrame, out_png: Path, title: str = "Per-sample metrics") -> None:
    """Histograms for common per-sample metrics, split by OOD source."""
    cols = [c for c in ["composition_MSE", "composition_Cosine", "composition_R2", "reconstruction_MSE", "reconstruction_Cosine", "reconstruction_R2"] if c in all_sample_metrics.columns]
    if not cols:
        return
    df = all_sample_metrics.copy()
    if "ood_source_label" not in df.columns:
        df["ood_source_label"] = np.where(df.get("has_ood", False), "ood", "none")
    label_order = ["none", "cosmic", "seen_denovo", "leaveout_denovo"]
    pretty = {"none": "w/o OOD", "cosmic": "COSMIC OOD", "seen_denovo": "train-seen OOD", "leaveout_denovo": "leaveout OOD", "ood": "with OOD"}
    colors = {"none": "#777777", "cosmic": "#377eb8", "seen_denovo": "#4daf4a", "leaveout_denovo": "#984ea3", "ood": "#a50f14"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=False)
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        prepared: Dict[str, np.ndarray] = {}
        notes = []
        all_vals = []
        for src in label_order + [x for x in sorted(df["ood_source_label"].dropna().astype(str).unique()) if x not in label_order]:
            vals, note, _, _ = _metric_plot_values(df.loc[df["ood_source_label"].astype(str) == src, col], col)
            if vals.size:
                prepared[src] = vals
                all_vals.append(vals)
                if note:
                    notes.append(f"{pretty.get(src, src)} {note}")
        if not all_vals:
            ax.set_axis_off()
            continue
        concat = np.concatenate(all_vals)
        if col.endswith("_R2"):
            bins = np.linspace(-1.0, 1.0, 61)
            ax.set_xlim(-1.02, 1.02)
        elif col.endswith("_Cosine"):
            bins = np.linspace(0.0, 1.0, 61)
            ax.set_xlim(-0.02, 1.02)
        else:
            hi = max(float(np.nanmax(concat)), 1e-12)
            bins = np.linspace(0.0, hi, 61)
            ax.set_xlim(0.0, hi * 1.02)
        for src in label_order + [x for x in prepared if x not in label_order]:
            vals = prepared.get(src)
            if vals is None or vals.size == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.42, color=colors.get(src, None), label=pretty.get(src, src))
        ax.set_title(col)
        ax.grid(alpha=0.15)
        if notes:
            ax.text(0.98, 0.96, "\n".join(notes[:4]), transform=ax.transAxes, va="top", ha="right", fontsize=6.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.70))
        ax.legend(fontsize=7)
    for ax in axes[len(cols):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.90, hspace=0.36, wspace=0.25)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_grid_ood_scatter(all_records: List[Dict[str, Any]], out_png: Path) -> None:
    """OOD true/pred scatter by source: COSMIC, seen de novo, leaveout de novo."""
    rows = []
    for r in all_records:
        if r["mode"] != "high_prior":
            continue
        ood = r["ood"].copy()
        meta = r["meta"].copy()
        for sid in ood.index:
            src = str(meta.loc[sid, "ood_source_label"]) if "ood_source_label" in meta.columns else ("ood" if float(meta.loc[sid, "n_ood"]) > 0 else "none")
            rows.append({
                "sample_id": sid,
                "depth": r["depth"],
                "noise": r["noise"],
                "ood_source": src,
                "true_ood_mass": float(ood.loc[sid, "ood_mass_true"]),
                "pred_ood_mass": float(ood.loc[sid, "ood_mass_pred"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    depth_bins = ["100-400", "401-2000", "2001-7000", "7000-100000"]
    noise_bins = ["0.85-0.90", "0.90-0.95", "0.95-1.00"]
    colors = {"none": "#777777", "cosmic": "#377eb8", "seen_denovo": "#4daf4a", "leaveout_denovo": "#984ea3"}
    labels = {"none": "no OOD", "cosmic": "OOD: COSMIC", "seen_denovo": "OOD: train-seen de novo", "leaveout_denovo": "OOD: leaveout de novo"}
    fig, axes = plt.subplots(len(depth_bins), len(noise_bins), figsize=(14, 16), constrained_layout=False)
    for i, depth in enumerate(depth_bins):
        for j, noise in enumerate(noise_bins):
            ax = axes[i, j]
            g0 = df[(df["depth"] == depth) & (df["noise"] == noise)]
            if g0.empty:
                ax.set_axis_off()
                continue
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.65)
            txt_lines = []
            for src in ["none", "cosmic", "seen_denovo", "leaveout_denovo"]:
                g = g0[g0["ood_source"] == src]
                if g.empty:
                    continue
                ax.scatter(g["true_ood_mass"], g["pred_ood_mass"], s=8, alpha=0.36, c=colors[src], edgecolors="none", linewidths=0, label=labels[src])
                if src != "none":
                    r2 = calc___R2(g["true_ood_mass"].values, g["pred_ood_mass"].values)
                    f1 = f1_flat(g["true_ood_mass"].values, g["pred_ood_mass"].values, thr=0.01)
                    txt_lines.append(f"{src}: R2={r2:.3f}, F1={f1:.3f}")
            if txt_lines:
                ax.text(0.02, 0.98, "\n".join(txt_lines), transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.74))
            ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="true OOD mass", ylabel="pred OOD mass")
            ax.set_title(f"{depth} | {noise}", fontsize=9)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("OOD residual mass calibration by OOD source | high-prior mode", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.93, hspace=0.38, wspace=0.28)
    fig.savefig(out_png, dpi=150)
    df.to_csv(out_png.with_suffix(".tsv"), sep="\t", index=False)
    plt.close(fig)


def eval_one_epoch(
    model: nn.Module,
    eval_paths: List[Dict[str, Any]],
    device: torch.device,
    out_result_epoch: Path,
    out_plot_dir: Path,
    path_log: str,
    epoch: int,
    sample_seen: int,
    simplex: str = "softmax",
    timer: Optional[StepTimer] = None,
) -> pd.DataFrame:
    """Run complete evaluation and save tables/plots."""
    model.eval()
    ensure_dir(out_result_epoch)
    print_log(f"eval epoch={epoch:04d}: loading {len(eval_paths)} batches", path_log, print_time=True)
    meter = ProgressMeter(total=len(eval_paths), label=f"eval epoch={epoch:04d}", path_log=path_log, every=max(1, len(eval_paths) // 10))
    summary_rows: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    sample_metric_rows: List[pd.DataFrame] = []

    for i, rec in enumerate(eval_paths, start=1):
        with (timer.block("s05_Train.py", "eval_one_epoch", "load_eval_pkl", epoch=epoch, batch=i, mode=str(rec.get("ood_source", "mixed"))) if timer else contextlib.nullcontext()):
            batch = load_pickle(Path(rec["pkl"]))
        for mode in ["full", "high_prior"]:
            with (timer.block("s05_Train.py", "infer_batch", "forward_eval", epoch=epoch, batch=i, mode=mode, n_ref=batch["R_refsigpool"].shape[0]) if timer else contextlib.nullcontext()):
                res = infer_batch(model, batch, device=device, mode=mode, simplex=simplex)
            with (timer.block("s05_Train.py", "eval_one_epoch", "metrics_eval", epoch=epoch, batch=i, mode=mode) if timer else contextlib.nullcontext()):
                cell_mask = known_composition_cell_mask(batch, mode)
                true_compo = true_known_composition(batch, mode)
                pred_compo_eval = pred_known_for_eval(res["pred_compo"], cell_mask)
                summ = summarize_eval_pair(batch, res, depth=str(rec["depth"]), noise=str(rec["noise"]), mode=mode, ood_source=str(rec.get("ood_source", "mixed")))
                summ.update({"epoch": int(epoch), "sample_seen": int(sample_seen), "batch_id": rec["batch_id"]})
                summary_rows.append(summ)
                all_records.append({
                    "depth": str(rec["depth"]),
                    "noise": str(rec["noise"]),
                    "mode": mode,
                    "ood_source": str(rec.get("ood_source", "mixed")),
                    "true_compo": true_compo,
                    "pred_compo": pred_compo_eval,
                    "cell_mask": cell_mask,
                    "ood": res["ood"],
                    "meta": batch["M_sampl_meta"],
                })
                true_recon_known = known_reconstruction_from_composition(true_compo, batch["R_refsigpool"])
                pred_recon_known = known_reconstruction_from_composition(pred_compo_eval, batch["R_refsigpool"])
                sm = df_metrics_by_sample(true_compo, pred_compo_eval, true_recon_known, pred_recon_known)
                meta_sm = batch["M_sampl_meta"].reindex(sm.index)
                sm["has_ood"] = meta_sm["n_ood"].astype(float) > 0
                default_src = pd.Series(np.where(sm["has_ood"].to_numpy(dtype=bool), "ood", "none"), index=sm.index)
                sm["ood_source_label"] = meta_sm["ood_source_label"].astype(object).where(meta_sm["ood_source_label"].notna(), default_src).astype(str).to_numpy()
                sm["depth"], sm["noise"], sm["mode"], sm["epoch"] = str(rec["depth"]), str(rec["noise"]), mode, int(epoch)
                sample_metric_rows.append(sm)
        meter.update(i)

    with (timer.block("s05_Train.py", "eval_one_epoch", "write_eval_tables", epoch=epoch) if timer else contextlib.nullcontext()):
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(out_result_epoch / "eval_summary_by_batch.tsv", sep="\t", index=False)
        df_group = df_summary.groupby(["epoch", "sample_seen", "mode", "depth", "noise", "ood_source"], as_index=False).agg(
            n_samples=("n_samples", "sum"),
            n_with_ood=("n_with_ood", "sum"),
            MSE=("MSE", "mean"),
            R2=("R2", "mean"),
            F1=("F1", "mean"),
            Cosine=("Cosine", "mean"),
            ood_MSE=("ood_MSE", "mean"),
            ood_R2=("ood_R2", "mean"),
            ood_F1=("ood_F1", "mean"),
            ood_pred_mean=("ood_pred_mean", "mean"),
            ood_true_mean=("ood_true_mean", "mean"),
        )
        df_group["group"] = df_group["depth"].astype(str) + "|" + df_group["noise"].astype(str) + "|" + df_group["ood_source"].astype(str) + "|" + df_group["mode"].astype(str)
        df_group.to_csv(out_result_epoch / "eval_summary_grouped.tsv", sep="\t", index=False)
        if sample_metric_rows:
            df_sm = pd.concat(sample_metric_rows, axis=0)
            df_sm.to_csv(out_result_epoch / "eval_sample_metrics.tsv", sep="\t")
        else:
            df_sm = pd.DataFrame()

    with (timer.block("s05_Train.py", "eval_one_epoch", "plot_eval", epoch=epoch) if timer else contextlib.nullcontext()):
        plot_grid_compo_scatter(
            all_records,
            out_plot_dir / f"epoch_{epoch:04d}_1_compo_scatter_full_spectra.png",
            mode="full",
            title=f"SigFormer full-reference composition scatter | epoch {epoch:04d}",
        )
        plot_grid_compo_scatter(
            all_records,
            out_plot_dir / f"epoch_{epoch:04d}_2_compo_scatter_high_prior.png",
            mode="high_prior",
            title=f"SigFormer high-prior composition scatter | epoch {epoch:04d}",
        )
        if not df_sm.empty:
            plot_hist_metrics(
                df_sm[df_sm["mode"] == "high_prior"],
                out_plot_dir / f"epoch_{epoch:04d}_3_sample_metric_hist_high_prior.png",
                title=f"Per-sample metrics, high prior | epoch {epoch:04d}",
            )
        plot_grid_ood_scatter(all_records, out_plot_dir / f"epoch_{epoch:04d}_4_ood_pred_true_scatter_by_source.png")
    meter.update(len(eval_paths), force=True)
    model.train()
    return df_group


# ============================================================
# training setup
# ============================================================


# ============================================================
# parse arguments
# ============================================================
from datetime import datetime
import random

default_seed = (int(datetime.now().strftime("%Y%m%d%H%M%S"))
                + random.randint(0, 999999)) % (2**32)

parser = argparse.ArgumentParser(description="Train SigFormer on synthetic mutational-signature data")
parser.add_argument("--out_dir", default="run_09_sigformer_v9")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=default_seed)
parser.add_argument("--cosmic_path", default="")
parser.add_argument("--cosmic_paths", default="", help="Comma-separated COSMIC files; default uses the newest bundled hg38 reference. Use 'all' only when multiple bundled versions are intentionally required.")
parser.add_argument("--n_eps", type=int, default=500)
parser.add_argument("--n_bch", type=int, default=1000)
parser.add_argument("--bch_size", type=int, default=64)
parser.add_argument("--lr_base", type=float, default=4e-4)
parser.add_argument("--lr_warm_ep", type=int, default=5)
parser.add_argument("--lr_cool_ep", type=int, default=150)
parser.add_argument("--lambda_compo", type=float, default=3.0)
parser.add_argument("--lambda_recon", type=float, default=1.0)
parser.add_argument("--lambda_ood", type=float, default=0.20)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--simplex", default="entmax", choices=["softmax", "entmax", "entmax15", "entmax1.5", "sparsemax"])
parser.add_argument("--model_d_model", type=int, default=192)
parser.add_argument("--model_n_heads", type=int, default=8)
parser.add_argument("--model_smp_n_lyr", type=int, default=1)
parser.add_argument("--model_ref_n_lyr", type=int, default=2)
parser.add_argument("--model_smp_ref_n_lyr", type=int, default=1)
parser.add_argument("--model_dropout", type=float, default=0.10)
parser.add_argument("--model_mlp_ratio", type=float, default=4.0)
parser.add_argument("--residual_logit_bias_init", type=float, default=-2.0)
parser.add_argument("--residual_evidence", type=str2bool, default=True, help="Accepted for run-script compatibility; residual token is controlled by model use_tok_ood.")
parser.add_argument("--residual_evidence_scale", type=float, default=50000.0, help="Accepted for run-script compatibility; current core uses internal depth scaling.")
parser.add_argument("--pretrained", default="")
parser.add_argument("--use_data_parallel", type=str2bool, default=False, help="True uses DataParallel only when CUDA and >1 visible GPUs are available.")
parser.add_argument("--amp", type=str2bool, default=True)
parser.add_argument("--log_points_per_epoch", type=int, default=200)
parser.add_argument("--profile_timing", type=str2bool, default=False, help="Record train/eval timing by script/function/stage.")
parser.add_argument("--profile_batch_every", type=int, default=1, help="Record timing every N batches when profiling is enabled.")
parser.add_argument("--eval_every", type=int, default=10)
parser.add_argument("--eval_n_bch", type=int, default=15)
parser.add_argument("--eval_bch_size", type=int, default=0)
parser.add_argument("--eval_quick", type=str2bool, default=False, help="Quick eval: one depth/noise tile but all OOD-source categories.")
parser.add_argument("--rebuild_eval", type=str2bool, default=False)
parser.add_argument("--n_mock", type=int, default=64, help="train-seen mock de novo signatures")
parser.add_argument("--n_mock_leaveout", type=int, default=16, help="leaveout mock de novo reserved for OOD evaluation only")
parser.add_argument("--n_cosmic_per_batch", type=int, default=65, help="COSMIC refs sampled per train/eval batch.")
parser.add_argument("--n_seen_denovo_per_batch", type=int, default=65, help="seen de novo refs sampled per train/eval batch.")
parser.add_argument("--mock_max_trials", type=int, default=250000)
parser.add_argument("--mock_combo_bank_size", type=int, default=1024)
parser.add_argument("--mock_cosine_max", type=float, default=0.80)
parser.add_argument("--mock_combo_cosine_max", type=float, default=0.88)
parser.add_argument("--perturb_basis", type=str2bool, default=True)
parser.add_argument("--eval_perturb_basis", type=str2bool, default=False)
parser.add_argument("--basis_perturb_conc", type=float, default=50000.0)
parser.add_argument("--basis_perturb_mix", type=float, default=0.020)
parser.add_argument("--basis_perturb_mix_denovo", type=float, default=None, help="Optional separate perturb mix for de novo refs.")
parser.add_argument("--ood_clean_ep", type=int, default=80)
parser.add_argument("--ood_ramp_end_ep", type=int, default=200)
parser.add_argument("--ood_stable_rate", type=float, default=0.50)
parser.add_argument("--pcOOD_cosmic_fraction", type=float, default=0.20, help="Fraction of training OOD leave-outs drawn from masked COSMIC; 0.20 gives COSMIC:seen-denovo = 1:4.")
parser.add_argument("--train_ood_cosmic_fraction", type=float, default=None, help="Alias for --pcOOD_cosmic_fraction used by older run scripts.")
parser.add_argument("--ood_min_compo", type=float, default=0.05, help="Minimum forced OOD mass for generated OOD samples.")
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--num_threads", type=int, default=8)
parser.add_argument("--smoke", type=str2bool, default=False)
args = parser.parse_args()
if args.train_ood_cosmic_fraction is not None:
    args.pcOOD_cosmic_fraction = float(args.train_ood_cosmic_fraction)


# ============================================================
# setup output directories and log
# ============================================================

if args.smoke:
    args.n_eps = min(args.n_eps, 1)
    args.n_bch = min(args.n_bch, 1)
    args.bch_size = min(args.bch_size, 5)
    args.eval_every = 1
    args.eval_n_bch = min(args.eval_n_bch, 1)
    args.eval_bch_size = min(int(args.eval_bch_size or args.bch_size), 5)
    args.eval_quick = True
    args.n_mock = min(args.n_mock, 8)
    args.n_mock_leaveout = min(args.n_mock_leaveout, 3)
    args.model_d_model = min(args.model_d_model, 32)
    args.model_n_heads = min(args.model_n_heads, 4)
    args.model_smp_n_lyr = min(args.model_smp_n_lyr, 1)
    args.model_ref_n_lyr = min(args.model_ref_n_lyr, 1)
    args.model_smp_ref_n_lyr = min(args.model_smp_ref_n_lyr, 1)

seed_everything(int(args.seed))
torch.set_num_threads(max(1, int(args.num_threads)))

OUT_DIR = Path(args.out_dir).resolve()
PATH_LOG = str(OUT_DIR / "train.log")
OUT_SCRIPTS = OUT_DIR / "0_scripts"
OUT_EVAL_DATA = OUT_DIR / "1_eval_data"
OUT_WEIGHTS = OUT_DIR / "3_model_weights"
OUT_EVAL_RESULT = OUT_DIR / "4_eval_result"
OUT_EVAL_PLOTS = OUT_DIR / "5_eval_plots"
for p in [OUT_DIR, OUT_SCRIPTS, OUT_EVAL_DATA, OUT_WEIGHTS, OUT_EVAL_RESULT, OUT_EVAL_PLOTS]:
    ensure_dir(p)

TIMER = StepTimer(
    enabled=bool(args.profile_timing),
    out_dir=OUT_DIR,
    path_log=PATH_LOG,
    device=None,
    batch_every=max(1, int(args.profile_batch_every)),
)

print_log("=" * 88, PATH_LOG, print_time=True)
print_log("SigFormer training run started", PATH_LOG, print_time=True)
print_log("=" * 88, PATH_LOG, print_time=True)
print_log(f"random seed is {args.seed}", PATH_LOG, print_time=True)
print_log(f"args: {json.dumps(json_safe(vars(args)), ensure_ascii=False, indent=2)}", PATH_LOG, print_time=False)
with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(json_safe(vars(args)), f, indent=2, ensure_ascii=False)
copy_run_sources(OUT_SCRIPTS, PATH_LOG)


# ============================================================
# build ref bank and eval data
# ============================================================

R_GRAND, M_GRAND = build_ref_bank(args, OUT_DIR, PATH_LOG, timer=TIMER)
R_TRAIN, M_TRAIN = training_ref_bank(R_GRAND, M_GRAND)
print_log(f"training ref bank: n_ref={R_TRAIN.shape[0]} (leaveout de novo excluded)", PATH_LOG, print_time=True)
EVAL_PATHS = build_eval_data(R_GRAND, M_GRAND, OUT_EVAL_DATA, args, PATH_LOG, timer=TIMER)


# ============================================================
# initialize model / optimizer
# ============================================================

if args.device == "cuda" and not torch.cuda.is_available():
    print_log("requested cuda but no CUDA is visible; falling back to CPU", PATH_LOG, print_time=True)
DEVICE = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
TIMER.set_device(DEVICE)

MODEL = SigFormerCore(
    n_chann=96,
    d_model=int(args.model_d_model),
    n_heads=int(args.model_n_heads),
    n_L_smp=int(args.model_smp_n_lyr),
    n_L_ref=int(args.model_ref_n_lyr),
    n_L_smp_ref=int(args.model_smp_ref_n_lyr),
    mlp_ratio=float(args.model_mlp_ratio),
    dropout=float(args.model_dropout),
    simplex=args.simplex,
    use_tok_ood=bool(args.residual_evidence),
    residual_init="zero_sample_depth",
    ood_lg_bias_init=float(args.residual_logit_bias_init),
)

if args.pretrained:
    ckpt = torch.load(args.pretrained, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    MODEL.load_state_dict(state, strict=False)
    print_log(f"loaded pretrained weights from {args.pretrained}", PATH_LOG, print_time=True)

MODEL = MODEL.to(DEVICE)
_use_dp, _dp_reason = should_use_data_parallel(args.use_data_parallel, DEVICE)
if _use_dp:
    MODEL = nn.DataParallel(MODEL)
    print_log(f"DataParallel activated ({_dp_reason})", PATH_LOG, print_time=True)
else:
    print_log(f"DataParallel not activated ({_dp_reason})", PATH_LOG, print_time=True)

OPTIM = torch.optim.AdamW(MODEL.parameters(), lr=float(args.lr_base), weight_decay=float(args.weight_decay))
use_amp = bool(args.amp and DEVICE.type == "cuda")
SCALER = torch.cuda.amp.GradScaler(enabled=use_amp)
CUR = curriculum(
    ep_total=int(args.n_eps),
    bch_per_ep=int(args.n_bch),
    bch_size=int(args.bch_size),
    lr_base=float(args.lr_base),
    lr_warm_ep=int(args.lr_warm_ep),
    lr_cool_ep=int(args.lr_cool_ep),
    lmda_compo=float(args.lambda_compo),
    lmda_recon=float(args.lambda_recon),
    lmda_ood=float(args.lambda_ood),
    ood_clean_ep=int(args.ood_clean_ep),
    ood_ramp_end_ep=int(args.ood_ramp_end_ep),
    ood_stable_rate=float(args.ood_stable_rate),
    pcOOD_cosmic_fraction=float(args.pcOOD_cosmic_fraction),
    ood_min_compo=float(args.ood_min_compo),
    perturb_basis=bool(args.perturb_basis),
    basis_perturb_conc=float(args.basis_perturb_conc),
    basis_perturb_mix=float(args.basis_perturb_mix),
    basis_perturb_mix_denovo=args.basis_perturb_mix_denovo,
    n_cosmic_per_batch=int(args.n_cosmic_per_batch),
    n_seen_denovo_per_batch=int(args.n_seen_denovo_per_batch),
)

print_log("model initialized", PATH_LOG, print_time=True)
print_log(str(model_without_parallel(MODEL)), PATH_LOG, print_time=False)
param_df = model_parameter_summary(MODEL)
param_df.to_csv(OUT_DIR / "model_parameter_summary.tsv", sep="\t", index=False)
trainable = int(param_df.loc[param_df["trainable"], "n_parameters"].sum())
total = int(param_df["n_parameters"].sum())
size_mb = float(param_df["size_MB_fp32"].sum())
print_log(f"model parameters: total={total:,}, trainable={trainable:,}, non_trainable={total-trainable:,}, fp32_size={size_mb:.2f} MB", PATH_LOG, print_time=True)


# ============================================================
# train
# ============================================================

curve_rows: List[Dict[str, Any]] = []
eval_history: List[pd.DataFrame] = []
sample_seen = 0
run_t0 = time.time()
log_every = max(1, int(args.n_bch) // 4)
record_every = max(1, int(args.n_bch) // max(1, int(args.log_points_per_epoch)))

header = "time                 epoch       batch       pct         loss      compo      recon        ood       grad         lr    ood%"
print_log(header, PATH_LOG, print_time=False)

for epoch in range(1, int(args.n_eps) + 1):
    MODEL.train()
    cur_cfg = CUR(epoch)
    set_optimizer_lr(OPTIM, cur_cfg["lr"])
    ep_t0 = time.time()
    running = {"loss_total": [], "loss_compo": [], "loss_recon": [], "loss_ood": [], "grad_norm": []}

    for b in range(1, int(args.n_bch) + 1):
        batch_seed = int(args.seed) + epoch * 100000 + b
        with TIMER.block("s05_Train.py", "batch_from_curriculum", "total", epoch=epoch, batch=b):
            batch = batch_from_curriculum(
                R_TRAIN,
                M_TRAIN,
                cur_cfg,
                seed=batch_seed,
                batch_id=f"train_ep{epoch:04d}_b{b:05d}",
                timer=TIMER,
                epoch=epoch,
                batch_num=b,
            )
        with TIMER.block("s05_Train.py", "batch_to_tensors", "dataframe_to_cuda_tensors", epoch=epoch, batch=b, n_ref=batch["R_refsigpool"].shape[0]):
            tensors = batch_to_tensors(batch, DEVICE, mode="high_prior")
        with TIMER.block("s05_Train.py", "optimizer", "zero_grad", epoch=epoch, batch=b):
            OPTIM.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            with TIMER.block("s01_Core.py", "SigFormerCore.forward", "forward_train", epoch=epoch, batch=b, n_ref=batch["R_refsigpool"].shape[0]):
                extra = MODEL(tensors["X"], tensors["R_batched"], simplex=args.simplex, ref_mask=tensors["ref_mask"], return_extra=True)
            with TIMER.block("s05_Train.py", "compute_losses_from_extra", "loss", epoch=epoch, batch=b):
                losses = compute_losses_from_extra(
                    extra,
                    tensors,
                    lmda_compo=float(cur_cfg["lmda_compo"]),
                    lmda_recon=float(cur_cfg["lmda_recon"]),
                    lmda_ood=float(cur_cfg["lmda_ood"]),
                )
        with TIMER.block("s05_Train.py", "train_step", "backward", epoch=epoch, batch=b):
            SCALER.scale(losses["loss_total"]).backward()
        with TIMER.block("s05_Train.py", "train_step", "grad_clip_and_norm", epoch=epoch, batch=b):
            if float(args.grad_clip) > 0:
                SCALER.unscale_(OPTIM)
                torch.nn.utils.clip_grad_norm_(MODEL.parameters(), float(args.grad_clip))
            gnorm = grad_norm(MODEL)
        with TIMER.block("s05_Train.py", "optimizer", "step", epoch=epoch, batch=b):
            SCALER.step(OPTIM)
            SCALER.update()

        bsz = int(tensors["X"].shape[0])
        sample_seen += bsz
        row = {
            "epoch": epoch,
            "batch": b,
            "sample_seen": sample_seen,
            "wall_time": dt.datetime.now().strftime("%H:%M:%S"),
            "time_elapsed_sec": time.time() - run_t0,
            "lr": cur_cfg["lr"],
            "pcOOD": cur_cfg["data_config"]["pcOOD"],
            "grad_norm": gnorm,
            "loss_total": float(losses["loss_total"].detach().cpu()),
            "loss_compo": float(losses["loss_compo"].detach().cpu()),
            "loss_recon": float(losses["loss_recon"].detach().cpu()),
            "loss_ood": float(losses["loss_ood"].detach().cpu()),
            "loss_ood_mse": float(losses["loss_ood_mse"].detach().cpu()),
            "loss_ood_fp": float(losses["loss_ood_fp"].detach().cpu()),
        }
        for k in running:
            running[k].append(float(row[k]))
        if b % record_every == 0 or b == 1 or b == int(args.n_bch):
            curve_rows.append(row)
        if b % log_every == 0 or b == 1 or b == int(args.n_bch):
            pct = 100.0 * b / max(1, int(args.n_bch))
            print_log(
                f"{current_stamp():<20} {epoch:>4}/{args.n_eps:<4} {b:>6}/{args.n_bch:<6} {pct:>6.2f}% "
                f"{np.mean(running['loss_total']):>10.6f} {np.mean(running['loss_compo']):>10.6f} "
                f"{np.mean(running['loss_recon']):>10.6f} {np.mean(running['loss_ood']):>10.6f} "
                f"{np.mean(running['grad_norm']):>10.4f} {cur_cfg['lr']:>10.3g} {100*cur_cfg['data_config']['pcOOD']:>6.1f}",
                PATH_LOG,
                print_time=False,
            )

        # Drop references promptly; long eval runs should not become memory hoarders.
        del batch, tensors, extra, losses
        if b % 25 == 0:
            gc.collect()

    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(OUT_DIR / "learning_curve.tsv", sep="\t", index=False)
    eval_df_for_curve = pd.concat(eval_history, axis=0, ignore_index=True) if eval_history else None
    plot_learn_curve(
        df_curve,
        OUT_DIR / "learning_curve.png",
        df_eval=eval_df_for_curve,
        finished_ep=epoch,
        total_samples=int(args.n_eps) * int(args.n_bch) * int(args.bch_size),
    )
    plt.close("all")
    print_log(f"epoch {epoch:04d} finished in {format_seconds(time.time() - ep_t0)}", PATH_LOG, print_time=True)
    TIMER.flush()
    TIMER.log_epoch_summary(epoch)

    if epoch % int(args.save_every) == 0 or epoch == int(args.n_eps):
        ckpt = {
            "epoch": epoch,
            "sample_seen": sample_seen,
            "model_state": model_without_parallel(MODEL).state_dict(),
            "optimizer_state": OPTIM.state_dict(),
            "args": json_safe(vars(args)),
            "curriculum": asdict(CUR.cfg),
            "ref_bank_path": str(OUT_DIR / "ref_bank.pkl"),
        }
        torch.save(ckpt, OUT_WEIGHTS / f"sigformer_v9_epoch_{epoch:04d}.pt")
        save_pickle(ckpt, OUT_WEIGHTS / f"sigformer_v9_epoch_{epoch:04d}.pkl")
        print_log(f"checkpoint saved for epoch={epoch:04d}", PATH_LOG, print_time=True)

    if epoch % int(args.eval_every) == 0 or epoch == int(args.n_eps):
        out_result_epoch = OUT_EVAL_RESULT / f"epoch_{epoch:04d}"
        df_eval = eval_one_epoch(
            MODEL,
            EVAL_PATHS,
            DEVICE,
            out_result_epoch,
            OUT_EVAL_PLOTS,
            PATH_LOG,
            epoch,
            sample_seen,
            simplex=args.simplex,
            timer=TIMER,
        )
        eval_history.append(df_eval)
        TIMER.flush()
        TIMER.log_epoch_summary(epoch)
        pd.concat(eval_history, axis=0, ignore_index=True).to_csv(OUT_DIR / "eval_history.tsv", sep="\t", index=False)
        df_curve = pd.DataFrame(curve_rows)
        plot_learn_curve(
            df_curve,
            OUT_DIR / "learning_curve.png",
            df_eval=pd.concat(eval_history, axis=0, ignore_index=True),
            finished_ep=epoch,
            total_samples=int(args.n_eps) * int(args.n_bch) * int(args.bch_size),
        )
        plt.close("all")
        print_log(header, PATH_LOG, print_time=False)

print_log("training completed", PATH_LOG, print_time=True)
