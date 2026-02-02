#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SigFormer training utilities (V5_6 refactor)

This module intentionally keeps "boring but essential" parts:
- logging helper
- small math helpers (R2, EMA smoothing, epoch-block sizing)
- plotting:
    * global loss / grad / lr curve
    * evaluation composition grid + confidence grid

It is designed to be imported by the training script.
"""

import os
import time
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


# ============================================================
# Global log handle / session
# ============================================================

LOG_FH = None
_LOG_SESSION = None


def bind_log_file(log_fh):
    """Bind a file handle for print_log to mirror stdout into a log file."""
    global LOG_FH
    LOG_FH = log_fh


def print_log(msg: str, session: str = None, log_fh=None):
    """
    Print to stdout and (optionally) to a log file.
    When session changes, insert a timestamp header.
    """
    global LOG_FH, _LOG_SESSION
    if log_fh is None:
        log_fh = LOG_FH

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if session is not None and session != _LOG_SESSION:
        header = f"[{now_str}] [SESSION {session}]"
        print(header)
        if log_fh is not None:
            log_fh.write(header + "\n")
            log_fh.flush()
        _LOG_SESSION = session

    print(msg)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


# ============================================================
# Small helpers
# ============================================================


# Create a directory if it doesn't exist (safe to call repeatedly).
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



# R² helper used for quick sanity checks in eval plots (not a fancy stats package).
def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard R² = 1 - SSE/SST.
    If variance of y_true is ~0, avoid divide-by-zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_mean = y_true.mean()
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_pred - y_true) ** 2)
    sst = max(float(sst), 1e-10)
    return 1.0 - float(sse) / sst



# Exponential moving average for prettier curves (and calmer humans).
def smooth_ema(values: List[float], alpha: float = 0.1) -> List[float]:
    """Simple EMA smoothing for curves."""
    smoothed = []
    avg = None
    for v in values:
        if avg is None:
            avg = v
        else:
            avg = alpha * v + (1.0 - alpha) * avg
        smoothed.append(avg)
    return smoothed



# Pick epoch shading blocks with a human-friendly {1,2,5}×10^k cadence.
def choose_epoch_block_size(epoch_done: int, max_blocks: int = 10) -> int:
    """
    Choose a block size from {1,2,5} * 10^k so that
    ceil(epoch_done / block_size) <= max_blocks, while block_size is as large as possible.
    """
    if epoch_done <= 0:
        return 1

    base_sizes = [1, 2, 5]
    candidates: List[int] = []
    k = 0
    while True:
        any_added = False
        for b in base_sizes:
            size = b * (10 ** k)
            if size <= epoch_done:
                candidates.append(size)
                any_added = True
        if not any_added:
            break
        k += 1

    candidates = sorted(set(candidates))
    chosen = 1
    for size in candidates:
        n_blocks = math.ceil(epoch_done / size)
        if n_blocks <= max_blocks:
            chosen = size
    return chosen


# ============================================================
# Plotting: global curves
# ============================================================


# Plot training traces from the main loop.
# Assumptions:
# - LOSS_record_Xs and loss_* traces are aligned (same length).
# - grad_trace_steps / lr_trace_steps are global step indices (same x-axis space).
def plot_global_loss_grad_lr(
    out_dir: str,
    epoch_idx: int,
    args,
    LOSS_record_Xs: List[int],
    loss_trace_values: List[float],
    loss_comp_trace_values: List[float],
    loss_recon_trace_values: List[float],
    loss_conf_trace_values: List[float],
    grad_trace_steps: List[int],
    grad_trace_values: List[float],
    lr_trace_steps: List[int],
    lr_trace_values: List[float],
):
    """
    Plot a single global figure combining:
    - total loss (raw + EMA)
    - lambda-weighted component losses
    - grad norm (EMA, secondary axis)
    - LR trace (optional, light overlay)

    The logic mirrors your original script but is packaged cleanly.
    """
    if len(LOSS_record_Xs) == 0:
        return

    # Convert python lists to numpy arrays once to keep plotting code simple.
    xs = np.array(LOSS_record_Xs, dtype=float)
    ys_total = np.array(loss_trace_values, dtype=float)
    ys_smoth = np.array(smooth_ema(loss_trace_values, alpha=0.1), dtype=float)
    ys_compo = np.array(loss_comp_trace_values, dtype=float)
    ys_recon = np.array(loss_recon_trace_values, dtype=float)
    ys_confi = np.array(loss_conf_trace_values, dtype=float)

    xs_grad = np.array(grad_trace_steps, dtype=float)
    ys_grad = np.array(grad_trace_values, dtype=float)
    ys_grad_s = np.array(smooth_ema(grad_trace_values, alpha=0.1), dtype=float)

    xs_lr = np.array(lr_trace_steps, dtype=float) if len(lr_trace_steps) else None
    ys_lr = np.array(lr_trace_values, dtype=float) if len(lr_trace_values) else None

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(xs, ys_total, alpha=0.25, label="total_loss (raw)")
    ax1.plot(xs, ys_smoth, alpha=0.9,  label="total_loss (smooth)")
    ax1.plot(xs, ys_compo, alpha=0.7,  label="lambda * loss_comp")
    ax1.plot(xs, ys_recon, alpha=0.7,  label="lambda * loss_recon")
    ax1.plot(xs, ys_confi, alpha=0.7,  label="lambda * loss_conf")

    ax1.set_xlabel("samples seen")
    ax1.set_ylabel("loss (lambda-weighted)")
    ax1.set_title("Global loss / grad / LR curves")

    # epoch blocks
    ax1.relim()
    ax1.autoscale()
    ymin, ymax = ax1.get_ylim()
    xs_min, xs_max = float(xs.min()), float(xs.max())

    epoch_done = epoch_idx + 1
    block_size = choose_epoch_block_size(epoch_done, max_blocks=10)
    # Map epoch indices onto the same x-axis ("samples seen") used for loss curves.
    # This assumes each epoch processes n_batches * batch_size samples.
    samples_per_epoch = args.n_batches * args.batch_size
    n_blocks = math.ceil(epoch_done / block_size)

    epoch_colors = ["#ffe5b4", "#cce5ff", "#e0ffcc", "#ffd6d6"]

    for i_block in range(n_blocks):
        e_start = i_block * block_size + 1
        e_end = min((i_block + 1) * block_size, epoch_done)
        x_start = (e_start - 1) * samples_per_epoch
        x_end = e_end * samples_per_epoch

        if x_end < xs_min or x_start > xs_max:
            continue

        color_block = epoch_colors[i_block % len(epoch_colors)]
        ax1.axvspan(x_start, x_end, alpha=0.25, color=color_block, zorder=-1)

        text_x = x_start + 0.01 * (xs_max - xs_min)
        text_y = ymax - 0.03 * (ymax - ymin)
        ax1.text(text_x, text_y, f"e{e_start}", fontsize=7, ha="left", va="top")

    # secondary axis: grad norm
    ax2 = ax1.twinx()
    ax2.plot(xs_grad, ys_grad_s, alpha=0.5, label="grad_norm (smooth)")
    ax2.set_ylabel("grad_norm")

    # optional LR overlay (third axis would be messy, so reuse ax2 scale lightly)
    if xs_lr is not None and ys_lr is not None and len(xs_lr):
        # Normalize LR into the grad axis range so it's visible but not obnoxious.
        # This is purely a visualization trick (do not read physics into it).
        try:
            lr_min, lr_max = float(np.min(ys_lr)), float(np.max(ys_lr))
            if lr_max > lr_min:
                gmin, gmax = ax2.get_ylim()
                ys_lr_scaled = (ys_lr - lr_min) / (lr_max - lr_min + 1e-12)
                ys_lr_scaled = gmin + ys_lr_scaled * (gmax - gmin)
                ax2.plot(xs_lr, ys_lr_scaled, alpha=0.25, label="lr (scaled)")
        except Exception:
            pass

    # merge legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    ensure_dir(out_dir)
    fig_path = os.path.join(out_dir, "loss_grad_curve_global.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print_log(f"[PLOT] Saved global loss/grad curves to {fig_path}", session="EVAL")


# ============================================================
# Eval plotting
# ============================================================


# End-to-end eval helper used by training/benchmark scripts.
# It expects pre-binned cell data and produces two figures per depth_category:
# - composition scatter grid
# - confidence vs error grid
def eval_and_plot_grid(
    model,
    device: torch.device,
    depth_category: str,
    cells: Dict[Tuple[int, int], Dict[str, Any]],
    comp_bins: List[Tuple[float, float]],
    profile_concs: List[float],
    fig_out_path: str,
    simplex_for_model,
):
    """
    Evaluate and plot a 4x5 grid for a single depth_category.

    Key refactor difference:
    - Each (noise_idx, comp_idx) cell can carry its OWN reference subset and
      therefore its own K_sub and its own cosmic/denovo membership.
    - This directly fixes the old behavior where a whole depth_category shared
      one reference subset.

    `cells[(j_prof, j_comp)]` must provide:
      comp_full:       [N, K_sub]
      profile_noisy:   [N, 96]
      counts:          [N, 96]
      depths:          [N]
      ref_profiles:    [K_sub, 96]
      is_cosmic_sub:   [K_sub]
      is_denovo_sub:   [K_sub]
    """
    # Switch to eval mode: no dropout/bn updates. Gradients are disabled later.
    model.eval()

    # composition grid
    fig, axes = plt.subplots(
        nrows=4, ncols=5, figsize=(20, 16), sharex=False, sharey=False
    )
    fig.suptitle(f"SigFormer eval: depth_category={depth_category}", fontsize=16)

    # confidence grid
    fig_conf, axes_conf = plt.subplots(
        nrows=4, ncols=5, figsize=(20, 16), sharex=False, sharey=False
    )
    fig_conf.suptitle(f"Confidence vs error: depth_category={depth_category}", fontsize=16)

    r2_cosmic_grid = np.full((4, 5), np.nan, dtype=float)
    r2_denovo_grid = np.full((4, 5), np.nan, dtype=float)

    # Keep eval batches reasonably sized to avoid GPU OOM during plotting.
    batch_size_eval = 256

    for j_prof, prof_conc in enumerate(profile_concs):
        for j_comp, comp_range in enumerate(comp_bins):
            ax1 = axes[j_prof, j_comp]
            axc = axes_conf[j_prof, j_comp]

            cell = cells.get((j_prof, j_comp), None)
            if cell is None:
                ax1.set_title("missing cell")
                ax1.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1)
                axc.set_title("missing cell")
                axc.axvline(0.0, color="gray", lw=1, alpha=0.5)
                continue

            comp_full = cell["comp_full"]
            profile_noisy = cell["profile_noisy"]
            counts = cell["counts"]
            ref_profiles = cell["ref_profiles"]
            is_cosmic_sub = cell["is_cosmic_sub"]
            is_denovo_sub = cell["is_denovo_sub"]

            # Choose model input representation:
            # - "normalized": use noisy normalized profiles
            # - otherwise: use raw counts (float32)
            if depth_category == "normalized":
                sample_profile_in = profile_noisy
            else:
                sample_profile_in = counts.astype(np.float32)

            n_samples = comp_full.shape[0]
            k_ref = comp_full.shape[1]

            # ref tensor (per-cell)
            ref_mat = torch.tensor(ref_profiles, dtype=torch.float32, device=device)  # [K, 96]

            compo_pred_all = np.zeros_like(comp_full, dtype=np.float32)
            confi_pred_all = np.zeros_like(comp_full, dtype=np.float32)

            # Forward pass in chunks; avoids holding the whole eval set on GPU.
            with torch.no_grad():
                for start in range(0, n_samples, batch_size_eval):
                    end = min(start + batch_size_eval, n_samples)
                    X_smp_raw = torch.tensor(sample_profile_in[start:end], dtype=torch.float32, device=device)
                    B = X_smp_raw.size(0)
                    X_ref_sig = ref_mat.unsqueeze(0).expand(B, -1, -1).contiguous()
                    compo_pred, confi_pred = model(X_smp_raw, X_ref_sig, simplex=simplex_for_model)
                    compo_pred_all[start:end] = compo_pred.detach().cpu().numpy()
                    confi_pred_all[start:end] = confi_pred.detach().cpu().numpy()

            # Indices for cosmic/denovo within this cell
            cosmic_idx = np.where(is_cosmic_sub)[0]
            denovo_idx = np.where(is_denovo_sub)[0]

            h_true = comp_full
            h_pred = compo_pred_all

            # Flattened vectors for R2
            if cosmic_idx.size > 0:
                y_true_cos = h_true[:, cosmic_idx].ravel()
                y_pred_cos = h_pred[:, cosmic_idx].ravel()
            else:
                y_true_cos = np.array([], dtype=float)
                y_pred_cos = np.array([], dtype=float)

            if denovo_idx.size > 0:
                y_true_den = h_true[:, denovo_idx].ravel()
                y_pred_den = h_pred[:, denovo_idx].ravel()
            else:
                y_true_den = np.array([], dtype=float)
                y_pred_den = np.array([], dtype=float)

            # Scatter masks
            mask_plot_cos = np.maximum(y_true_cos, y_pred_cos) > 0.01 if y_true_cos.size > 0 else np.array([], dtype=bool)
            mask_plot_den = np.maximum(y_true_den, y_pred_den) > 0.01 if y_true_den.size > 0 else np.array([], dtype=bool)

            if y_true_cos.size > 0:
                ax1.scatter(
                    y_true_cos[mask_plot_cos],
                    y_pred_cos[mask_plot_cos],
                    s=3, alpha=0.35, label="COSMIC",
                    color="navy", edgecolors="none",
                )
            if y_true_den.size > 0:
                ax1.scatter(
                    y_true_den[mask_plot_den],
                    y_pred_den[mask_plot_den],
                    s=3, alpha=0.35, label="denovo",
                    color="darkred", edgecolors="none",
                )

            ax1.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1)
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)

            # R2 for cosmic/denovo
            r2_cos = compute_r2(y_true_cos, y_pred_cos) if y_true_cos.size > 0 else float("nan")
            r2_den = compute_r2(y_true_den, y_pred_den) if y_true_den.size > 0 else float("nan")

            r2_cosmic_grid[j_prof, j_comp] = r2_cos
            r2_denovo_grid[j_prof, j_comp] = r2_den

            # Simple "active signature" thresholding for coarse classification metrics.
            # This is mainly for debugging and quick regressions, not a publication-ready metric.
            thr_act = 0.02
            y_true_bin = (h_true > thr_act)
            y_pred_bin = (h_pred > thr_act)

            tp = np.logical_and(y_true_bin, y_pred_bin).sum()
            tn = np.logical_and(~y_true_bin, ~y_pred_bin).sum()
            fp = np.logical_and(~y_true_bin, y_pred_bin).sum()
            fn = np.logical_and(y_true_bin, ~y_pred_bin).sum()

            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            f1 = 2 * prec * sens / (prec + sens) if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0 else np.nan

            r2_all = compute_r2(h_true.ravel(), h_pred.ravel())

            true_act_counts = y_true_bin.sum(axis=1)
            pred_act_counts = y_pred_bin.sum(axis=1)
            mu_t, sd_t = float(true_act_counts.mean()), float(true_act_counts.std())
            mu_p, sd_p = float(pred_act_counts.mean()), float(pred_act_counts.std())

            metrics_text = (
                f"Sens={sens:.2f}, Spec={spec:.2f}\n"
                f"F1={f1:.2f}, R2={r2_all:.2f}\n"
                f"Act T/P={mu_t:.1f}±{sd_t:.1f}/{mu_p:.1f}±{sd_p:.1f}"
            )
            ax1.text(
                0.02, 0.98, metrics_text,
                transform=ax1.transAxes, ha="left", va="top", fontsize=7
            )

            title = (
                f"noise={int(prof_conc)}, comp_bin={j_comp}\n"
                f"R2(C)={r2_cos:.3f}, R2(D)={r2_den:.3f}"
            )
            ax1.set_title(title, fontsize=9)

            if j_prof == 3 and j_comp == 0:
                ax1.legend(loc="lower right", fontsize=8)

            # Confidence grid: x = (pred - true), y = predicted confidence.
            # Downsample points for speed if needed.
            diff = (h_pred - h_true).ravel()
            conf_flat = confi_pred_all.ravel()
            n_points = diff.shape[0]
            max_points = 50000
            if n_points > max_points:
                idx = np.random.choice(n_points, size=max_points, replace=False)
                diff = diff[idx]
                conf_flat = conf_flat[idx]

            axc.scatter(diff, conf_flat, s=2, alpha=0.2)
            axc.axvline(0.0, color="gray", lw=1, alpha=0.5)
            axc.set_title(f"noise={int(prof_conc)}, comp_bin={j_comp}", fontsize=9)
            axc.set_xlim(-1.0, 1.0)
            axc.set_ylim(-0.05, 1.05)

    for ax in axes[-1, :]:
        ax.set_xlabel("true composition")
    for ax in axes[:, 0]:
        ax.set_ylabel("pred composition")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ensure_dir(os.path.dirname(fig_out_path))
    fig.savefig(fig_out_path, dpi=120)
    plt.close(fig)
    print_log(f"[PLOT] Saved eval figure: {fig_out_path}", session="EVAL")

    for ax in axes_conf[-1, :]:
        ax.set_xlabel("pred - true composition")
    for ax in axes_conf[:, 0]:
        ax.set_ylabel("confidence")

    fig_conf.tight_layout(rect=[0, 0.03, 1, 0.95])
    conf_path = fig_out_path.replace(".png", "_confidence_grid.png")
    fig_conf.savefig(conf_path, dpi=120)
    plt.close(fig_conf)
    print_log(f"[PLOT] Saved confidence grid: {conf_path}", session="EVAL")

    df_r2_cosmic = pd.DataFrame(
        r2_cosmic_grid,
        index=[f"noise_{int(x)}" for x in profile_concs],
        columns=[f"comp_bin_{j}" for j in range(len(comp_bins))],
    )
    df_r2_denovo = pd.DataFrame(
        r2_denovo_grid,
        index=[f"noise_{int(x)}" for x in profile_concs],
        columns=[f"comp_bin_{j}" for j in range(len(comp_bins))],
    )

    return {
        "r2_cosmic": df_r2_cosmic,
        "r2_denovo": df_r2_denovo,
    }
