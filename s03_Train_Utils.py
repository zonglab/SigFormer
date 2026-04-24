import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


LOG_FH = None
_LOG_SESSION = None


def bind_log_file(log_fh):
    global LOG_FH
    LOG_FH = log_fh


def print_log(msg: str, session: Optional[str] = None, log_fh=None):
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


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    y_mean = y_true.mean()
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_pred - y_true) ** 2)
    sst = max(float(sst), 1e-10)
    return 1.0 - float(sse) / sst


def smooth_ema(values: List[float], alpha: float = 0.1) -> List[float]:
    out = []
    avg = None
    for value in values:
        if avg is None:
            avg = value
        else:
            avg = alpha * value + (1.0 - alpha) * avg
        out.append(avg)
    return out


def compute_binary_stats(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    tp = np.logical_and(y_true_bin, y_pred_bin).sum()
    tn = np.logical_and(~y_true_bin, ~y_pred_bin).sum()
    fp = np.logical_and(~y_true_bin, y_pred_bin).sum()
    fn = np.logical_and(y_true_bin, ~y_pred_bin).sum()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * prec * sens / (prec + sens) if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0 else np.nan
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "sens": float(sens),
        "spec": float(spec),
        "prec": float(prec),
        "f1": float(f1),
    }


def np_corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# Training losses / targets
# ============================================================


def composition_mse_loss(compo_pred: torch.Tensor, comp_true: torch.Tensor) -> torch.Tensor:
    k_sig = comp_true.size(1)
    return F.mse_loss(compo_pred, comp_true, reduction="mean") * k_sig



def composition_cosine_loss(compo_pred: torch.Tensor, comp_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = torch.clamp(compo_pred, min=0.0)
    true = torch.clamp(comp_true, min=0.0)
    pred = pred / (pred.sum(dim=1, keepdim=True) + eps)
    true = true / (true.sum(dim=1, keepdim=True) + eps)
    cos = F.cosine_similarity(pred, true, dim=1)
    return (1.0 - cos).mean()



def false_positive_weak_loss(
    compo_pred: torch.Tensor,
    comp_true: torch.Tensor,
    thr_act: float = 0.02,
    power: float = 1.0,
) -> torch.Tensor:
    with torch.no_grad():
        mask_fp = (comp_true < thr_act).float()

    pred_pos = torch.clamp(compo_pred, min=0.0)
    if power != 1.0:
        pred_pos = pred_pos ** power
    return (pred_pos * mask_fp).mean()



def build_confidence_target(
    compo_pred_detach: torch.Tensor,
    comp_true: torch.Tensor,
    conf_scale: float = 4.0,
    thr_true: float = 0.02,
    fp_scale: float = 12.0,
    miss_scale: float = 8.0,
) -> torch.Tensor:
    """
    Confidence should be high only when the weight is both numerically close and support-consistent.
    This is stricter than exp(-|pred-true|) and punishes tiny false positives explicitly.
    """
    abs_err = torch.abs(compo_pred_detach - comp_true)
    err_term = torch.exp(-conf_scale * abs_err)

    true_active = (comp_true >= thr_true).float()
    inactive_penalty = torch.exp(-fp_scale * torch.clamp(compo_pred_detach, min=0.0))
    miss_penalty = torch.exp(-miss_scale * torch.relu(thr_true - compo_pred_detach))

    target = torch.where(true_active > 0.5, err_term * miss_penalty, err_term * inactive_penalty)
    return target.clamp(0.0, 1.0)


# ============================================================
# Plotting helpers
# ============================================================


def choose_epoch_block_size(epoch_done: int, max_blocks: int = 10) -> int:
    if epoch_done <= 0:
        return 1
    base_sizes = [1, 2, 5]
    candidates = []
    k = 0
    while True:
        any_added = False
        for base in base_sizes:
            size = base * (10 ** k)
            if size <= epoch_done:
                candidates.append(size)
                any_added = True
        if not any_added:
            break
        k += 1

    chosen = 1
    for size in sorted(set(candidates)):
        if math.ceil(epoch_done / size) <= max_blocks:
            chosen = size
    return chosen



def plot_global_loss_grad_lr(
    out_dir: str,
    epoch_idx: int,
    args,
    LOSS_record_Xs: List[int],
    loss_trace_values: List[float],
    loss_comp_trace_values: List[float],
    loss_recon_trace_values: List[float],
    loss_conf_trace_values: List[float],
    loss_fp_trace_values: List[float],
    grad_trace_steps: List[int],
    grad_trace_values: List[float],
    lr_trace_steps: List[int],
    lr_trace_values: List[float],
):
    if not LOSS_record_Xs:
        return

    xs = np.asarray(LOSS_record_Xs, dtype=float)
    y_total = np.asarray(loss_trace_values, dtype=float)
    y_total_s = np.asarray(smooth_ema(loss_trace_values, alpha=0.1), dtype=float)
    y_comp = np.asarray(loss_comp_trace_values, dtype=float)
    y_recon = np.asarray(loss_recon_trace_values, dtype=float)
    y_conf = np.asarray(loss_conf_trace_values, dtype=float)
    y_fp = np.asarray(loss_fp_trace_values, dtype=float)

    x_grad = np.asarray(grad_trace_steps, dtype=float)
    y_grad = np.asarray(smooth_ema(grad_trace_values, alpha=0.1), dtype=float)
    x_lr = np.asarray(lr_trace_steps, dtype=float) if lr_trace_steps else None
    y_lr = np.asarray(lr_trace_values, dtype=float) if lr_trace_values else None

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(xs, y_total, alpha=0.20, label="total_loss (raw)")
    ax1.plot(xs, y_total_s, alpha=0.90, label="total_loss (smooth)")
    ax1.plot(xs, y_comp, alpha=0.70, label="lambda * loss_comp")
    ax1.plot(xs, y_recon, alpha=0.70, label="lambda * loss_recon")
    ax1.plot(xs, y_conf, alpha=0.70, label="lambda * loss_conf")
    ax1.plot(xs, y_fp, alpha=0.70, label="lambda * loss_fp")
    ax1.set_xlabel("samples seen")
    ax1.set_ylabel("loss")
    ax1.set_title("Global loss / grad / LR curves")

    ax1.relim()
    ax1.autoscale()
    ymin, ymax = ax1.get_ylim()
    xs_min, xs_max = float(xs.min()), float(xs.max())
    epoch_done = epoch_idx + 1
    block_size = choose_epoch_block_size(epoch_done, max_blocks=10)
    samples_per_epoch = args.n_batches * args.batch_size
    epoch_colors = ["#ffe5b4", "#cce5ff", "#e0ffcc", "#ffd6d6"]

    for i_block in range(math.ceil(epoch_done / block_size)):
        e_start = i_block * block_size + 1
        e_end = min((i_block + 1) * block_size, epoch_done)
        x_start = (e_start - 1) * samples_per_epoch
        x_end = e_end * samples_per_epoch
        if x_end < xs_min or x_start > xs_max:
            continue
        ax1.axvspan(x_start, x_end, alpha=0.25, color=epoch_colors[i_block % len(epoch_colors)], zorder=-1)
        ax1.text(
            x_start + 0.01 * (xs_max - xs_min),
            ymax - 0.03 * (ymax - ymin),
            f"e{e_start}",
            fontsize=7,
            ha="left",
            va="top",
        )

    ax2 = ax1.twinx()
    ax2.plot(x_grad, y_grad, alpha=0.5, label="grad_norm (smooth)")
    ax2.set_ylabel("grad_norm")

    if x_lr is not None and y_lr is not None and len(x_lr) > 0:
        try:
            lr_min, lr_max = float(np.min(y_lr)), float(np.max(y_lr))
            if lr_max > lr_min:
                gmin, gmax = ax2.get_ylim()
                y_lr_scaled = (y_lr - lr_min) / (lr_max - lr_min + 1e-12)
                y_lr_scaled = gmin + y_lr_scaled * (gmax - gmin)
                ax2.plot(x_lr, y_lr_scaled, alpha=0.25, label="lr (scaled)")
        except Exception:
            pass

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()

    ensure_dir(out_dir)
    fig_path = os.path.join(out_dir, "loss_grad_curve_global.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print_log(f"[PLOT] Saved global loss/grad curves to {fig_path}", session="EVAL")



def plot_epoch_batch_losses(out_dir: str, epoch_idx: int, batch_df: pd.DataFrame):
    if batch_df.empty:
        return

    ensure_dir(out_dir)
    tsv_path = os.path.join(out_dir, f"epoch{epoch_idx + 1:03d}_batch_losses.tsv")
    batch_df.to_csv(tsv_path, sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(batch_df["batch"], batch_df["loss_total"], alpha=0.8, label="loss_total")
    if "loss_comp" in batch_df:
        ax.plot(batch_df["batch"], batch_df["loss_comp"], alpha=0.7, label="loss_comp")
    if "loss_recon" in batch_df:
        ax.plot(batch_df["batch"], batch_df["loss_recon"], alpha=0.7, label="loss_recon")
    if "loss_conf" in batch_df:
        ax.plot(batch_df["batch"], batch_df["loss_conf"], alpha=0.7, label="loss_conf")
    if "loss_fp" in batch_df:
        ax.plot(batch_df["batch"], batch_df["loss_fp"], alpha=0.7, label="loss_fp")
    ax.set_xlabel("batch")
    ax.set_ylabel("loss")
    ax.set_title(f"Epoch {epoch_idx + 1} per-batch loss")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, f"epoch{epoch_idx + 1:03d}_batch_losses.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


# ============================================================
# Evaluation
# ============================================================


def evaluate_one_cell(
    model,
    device: torch.device,
    depth_category: str,
    cell: Dict[str, Any],
    simplex_for_model,
    batch_size_eval: int = 256,
    activity_threshold: float = 0.02,
    confidence_threshold: float = 0.35,
):
    comp_true = cell["comp_full"]
    profile_noisy = cell["profile_noisy"]
    counts = cell["counts"]
    ref_profiles = cell["ref_profiles"]
    is_cosmic_sub = cell["is_cosmic_sub"]
    is_denovo_sub = cell["is_denovo_sub"]

    if depth_category == "normalized":
        sample_profile_in = profile_noisy
    else:
        sample_profile_in = counts.astype(np.float32)

    n_samples = comp_true.shape[0]
    ref_tensor = torch.tensor(ref_profiles, dtype=torch.float32, device=device)
    comp_pred_all = np.zeros_like(comp_true, dtype=np.float32)
    conf_pred_all = np.zeros_like(comp_true, dtype=np.float32)
    comp_mask_all = np.zeros_like(comp_true, dtype=np.float32)
    novelty_score_all = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size_eval):
            end = min(start + batch_size_eval, n_samples)
            x_smp = torch.tensor(sample_profile_in[start:end], dtype=torch.float32, device=device)
            B = x_smp.size(0)
            x_ref = ref_tensor.unsqueeze(0).expand(B, -1, -1).contiguous()
            comp_pred, conf_pred, aux = model(x_smp, x_ref, simplex=simplex_for_model, return_aux=True)
            comp_mask = model.build_masked_composition(
                comp_pred,
                conf_pred,
                min_composition=1e-3,
                confidence_threshold=confidence_threshold,
                mode="soft",
                soft_power=2.0,
            )
            comp_pred_all[start:end] = comp_pred.detach().cpu().numpy()
            conf_pred_all[start:end] = conf_pred.detach().cpu().numpy()
            comp_mask_all[start:end] = comp_mask.detach().cpu().numpy()
            novelty_score_all[start:end] = aux["residual_cosine"].detach().cpu().numpy()

    cosmic_idx = np.where(is_cosmic_sub)[0]
    denovo_idx = np.where(is_denovo_sub)[0]

    y_true = comp_true
    y_pred = comp_pred_all
    y_mask = comp_mask_all

    y_true_cos = y_true[:, cosmic_idx].ravel() if cosmic_idx.size > 0 else np.array([], dtype=float)
    y_pred_cos = y_pred[:, cosmic_idx].ravel() if cosmic_idx.size > 0 else np.array([], dtype=float)
    y_true_den = y_true[:, denovo_idx].ravel() if denovo_idx.size > 0 else np.array([], dtype=float)
    y_pred_den = y_pred[:, denovo_idx].ravel() if denovo_idx.size > 0 else np.array([], dtype=float)

    y_true_bin = y_true > activity_threshold
    y_pred_bin = y_pred > activity_threshold
    y_mask_bin = y_mask > activity_threshold

    stats_raw = compute_binary_stats(y_true_bin, y_pred_bin)
    stats_mask = compute_binary_stats(y_true_bin, y_mask_bin)
    conf_correct = (y_true_bin == y_pred_bin).astype(float).ravel()
    conf_flat = conf_pred_all.ravel()

    return {
        "comp_true": y_true,
        "comp_pred": y_pred,
        "comp_masked": y_mask,
        "confidence": conf_pred_all,
        "novelty_score": novelty_score_all,
        "r2_all": compute_r2(y_true.ravel(), y_pred.ravel()),
        "r2_all_masked": compute_r2(y_true.ravel(), y_mask.ravel()),
        "r2_cosmic": compute_r2(y_true_cos, y_pred_cos) if y_true_cos.size > 0 else float("nan"),
        "r2_denovo": compute_r2(y_true_den, y_pred_den) if y_true_den.size > 0 else float("nan"),
        "stats_raw": stats_raw,
        "stats_mask": stats_mask,
        "confidence_corr": np_corrcoef_safe(conf_flat, conf_correct),
    }



def eval_and_plot_grid(
    model,
    device: torch.device,
    depth_category: str,
    cells: Dict[Tuple[int, int], Dict[str, Any]],
    comp_bins: List[Tuple[float, float]],
    profile_concs: List[float],
    fig_out_path: str,
    simplex_for_model,
) -> Dict[str, pd.DataFrame]:
    model.eval()

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16), sharex=False, sharey=False)
    fig.suptitle(f"SigFormer eval: depth_category={depth_category}", fontsize=16)

    fig_conf, axes_conf = plt.subplots(nrows=4, ncols=5, figsize=(20, 16), sharex=False, sharey=False)
    fig_conf.suptitle(f"Confidence vs correctness: depth_category={depth_category}", fontsize=16)

    r2_cosmic_grid = np.full((4, 5), np.nan, dtype=float)
    r2_denovo_grid = np.full((4, 5), np.nan, dtype=float)
    f1_raw_grid = np.full((4, 5), np.nan, dtype=float)
    f1_mask_grid = np.full((4, 5), np.nan, dtype=float)
    conf_corr_grid = np.full((4, 5), np.nan, dtype=float)

    for j_prof, prof_conc in enumerate(profile_concs):
        for j_comp, comp_range in enumerate(comp_bins):
            ax = axes[j_prof, j_comp]
            axc = axes_conf[j_prof, j_comp]
            cell = cells.get((j_prof, j_comp))
            if cell is None:
                ax.set_title("missing cell")
                axc.set_title("missing cell")
                continue

            out = evaluate_one_cell(
                model=model,
                device=device,
                depth_category=depth_category,
                cell=cell,
                simplex_for_model=simplex_for_model,
            )
            comp_true = out["comp_true"]
            comp_pred = out["comp_pred"]
            comp_mask = out["comp_masked"]
            confidence = out["confidence"]

            flat_true = comp_true.ravel()
            flat_pred = comp_pred.ravel()
            flat_mask = comp_mask.ravel()
            flat_conf = confidence.ravel()
            plot_mask = np.maximum(flat_true, flat_pred) > 0.01

            ax.scatter(flat_true[plot_mask], flat_pred[plot_mask], s=2, alpha=0.22, label="raw")
            ax.scatter(flat_true[plot_mask], flat_mask[plot_mask], s=2, alpha=0.22, label="masked")
            ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

            stats_raw = out["stats_raw"]
            stats_mask = out["stats_mask"]
            metrics_text = (
                f"R2 raw/mask={out['r2_all']:.2f}/{out['r2_all_masked']:.2f}\n"
                f"F1 raw/mask={stats_raw['f1']:.2f}/{stats_mask['f1']:.2f}\n"
                f"Conf corr={out['confidence_corr']:.2f}"
            )
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, ha="left", va="top", fontsize=7)
            ax.set_title(
                f"noise={int(prof_conc)}, comp_bin={j_comp}\n"
                f"R2(C)={out['r2_cosmic']:.3f}, R2(D)={out['r2_denovo']:.3f}",
                fontsize=9,
            )
            if j_prof == 3 and j_comp == 0:
                ax.legend(loc="lower right", fontsize=8)

            correctness = ((comp_true > 0.02) == (comp_pred > 0.02)).astype(float).ravel()
            if flat_conf.size > 50000:
                idx = np.random.choice(flat_conf.size, size=50000, replace=False)
                conf_plot = flat_conf[idx]
                corr_plot = correctness[idx]
            else:
                conf_plot = flat_conf
                corr_plot = correctness
            axc.scatter(conf_plot, corr_plot + np.random.normal(0, 0.02, size=corr_plot.size), s=2, alpha=0.20)
            axc.set_xlim(-0.05, 1.05)
            axc.set_ylim(-0.05, 1.05)
            axc.set_title(f"noise={int(prof_conc)}, comp_bin={j_comp}", fontsize=9)

            r2_cosmic_grid[j_prof, j_comp] = out["r2_cosmic"]
            r2_denovo_grid[j_prof, j_comp] = out["r2_denovo"]
            f1_raw_grid[j_prof, j_comp] = stats_raw["f1"]
            f1_mask_grid[j_prof, j_comp] = stats_mask["f1"]
            conf_corr_grid[j_prof, j_comp] = out["confidence_corr"]

    for ax in axes[-1, :]:
        ax.set_xlabel("true composition")
    for ax in axes[:, 0]:
        ax.set_ylabel("pred composition")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ensure_dir(os.path.dirname(fig_out_path))
    fig.savefig(fig_out_path, dpi=120)
    plt.close(fig)

    for ax in axes_conf[-1, :]:
        ax.set_xlabel("predicted confidence")
    for ax in axes_conf[:, 0]:
        ax.set_ylabel("support correctness")
    fig_conf.tight_layout(rect=[0, 0.03, 1, 0.95])
    conf_path = fig_out_path.replace(".png", "_confidence_grid.png")
    fig_conf.savefig(conf_path, dpi=120)
    plt.close(fig_conf)

    print_log(f"[PLOT] Saved eval figure: {fig_out_path}", session="EVAL")
    print_log(f"[PLOT] Saved confidence grid: {conf_path}", session="EVAL")

    index = [f"noise_{int(x)}" for x in profile_concs]
    columns = [f"comp_bin_{j}" for j in range(len(comp_bins))]
    return {
        "r2_cosmic": pd.DataFrame(r2_cosmic_grid, index=index, columns=columns),
        "r2_denovo": pd.DataFrame(r2_denovo_grid, index=index, columns=columns),
        "f1_raw": pd.DataFrame(f1_raw_grid, index=index, columns=columns),
        "f1_masked": pd.DataFrame(f1_mask_grid, index=index, columns=columns),
        "confidence_corr": pd.DataFrame(conf_corr_grid, index=index, columns=columns),
    }
