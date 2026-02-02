#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("loading common modules")
import os, time, math, argparse, random
from typing import Dict, Any, Tuple, List, Optional

print("loading maths")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("loading torch")
import torch
import torch.nn as nn
import torch.nn.functional as F

print("loading SigFormer")
from s01_SigFormer_Core import SigFormerCore

print("loading data generator primitives")
from s02_Data_Generation import (get_COSMIC, 
                                 summarize_cosmic_signatures,
                                 build_denovo_signatures,
                                 sample_active_signatures_and_profile,
                                 sample_noisy_counts_from_profile,
                                 _sample_depth_for_profile,)

print("loading utils")
from s03_Train_Utils import (print_log, bind_log_file,
                             ensure_dir,
                             compute_r2, smooth_ema,
                             plot_global_loss_grad_lr,
                             eval_and_plot_grid,)

# ============================================================
# Seeds
# ============================================================
def make_run_seed() -> int:
    t_ms = int(time.time() * 1000.0)
    return t_ms % (2**32 - 1)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Global simulation defaults
# ============================================================

COMP_ALPHA_BINS: List[Tuple[float, float]] = [(0.05, 0.2),
                                              (0.2, 0.8),
                                              (0.8, 2.0),
                                              (2.0, 5.0),
                                              (5.0, 20.0),]
PROFILE_NOISE_LEVELS: List[float] = [40.0, 120.0, 280.0, 2300.0]
DEPTH_MODES: List[str] = ["low", "medium", "high"]
MAX_ACTIVE_BY_DEPTH = {"low": 4, "medium": 10, "high": 16, "mixed": 16}


# ============================================================
# Curriculum (kept minimal)
# ============================================================

def build_default_curriculum(n_epochs: int) -> pd.DataFrame:
    if n_epochs <= 0:
        n_epochs = 1

    p1 = max(1, int(0.3 * n_epochs))
    p2 = max(p1 + 1, int(0.7 * n_epochs))
    p2 = min(p2, n_epochs)

    rows = []
    rows.append(dict(epoch_stt=1, epoch_end=p1, simplex="softmax",
                     depth_low=1.0, depth_mid=1.0, depth_hig=1.0, norm_frac=0.25))
    rows.append(dict(epoch_stt=p1 + 1, epoch_end=p2, simplex="softmax",
                     depth_low=1.0, depth_mid=1.0, depth_hig=1.0, norm_frac=0.25))
    rows.append(dict(epoch_stt=p2 + 1, epoch_end=n_epochs, simplex="sparsemax",
                     depth_low=1.0, depth_mid=1.0, depth_hig=1.0, norm_frac=0.25))

    return pd.DataFrame(rows)


def load_curriculum(args: argparse.Namespace, n_epochs: int, out_dir: str) -> pd.DataFrame:
    if args.curriculum is not None and os.path.isfile(args.curriculum):
        df = pd.read_csv(args.curriculum, sep="\t")
        print_log(f"[CURRICULUM] Loaded curriculum from {args.curriculum}", session="INIT")
    else:
        df = build_default_curriculum(n_epochs)
        cur_path = os.path.join(out_dir, "curriculum.tsv")
        df.to_csv(cur_path, sep="\t", index=False)
        print_log(f"[CURRICULUM] No curriculum TSV provided, wrote default to {cur_path}", session="INIT")
    return df


def get_curriculum_row_for_epoch(curriculum_df: pd.DataFrame, epoch_idx: int) -> pd.Series:
    epoch_num = epoch_idx + 1
    mask = (curriculum_df["epoch_stt"] <= epoch_num) & (curriculum_df["epoch_end"] >= epoch_num)
    if not mask.any():
        return curriculum_df.iloc[-1]
    return curriculum_df[mask].iloc[0]


# ============================================================
# Reference bank: COSMIC + de novo
# ============================================================

def build_reference_bank(
    denovo_n_expect: int = 500,
    denovo_max__cos: float = 0.6,
    denovo_nois_rng: Tuple[float, float] = (0.1, 600.0),
    denovo_max_trial: int = 50000,
    denovo_seed: int = 2025,
):
    print_log("[STEP] Loading COSMIC signatures ...", session="REFBANK")
    df_COSMIC = get_COSMIC()

    print_log("[STEP] Summarizing COSMIC signatures ...", session="REFBANK")
    _ = summarize_cosmic_signatures(df_COSMIC)

    print_log("[STEP] Building de novo signatures ...", session="REFBANK")
    df_denovo = build_denovo_signatures(
        df_COSMIC,
        n_denovo=denovo_n_expect,
        target_cosine_max=denovo_max__cos,
        alpha_scale_range=denovo_nois_rng,
        max_total_trials=denovo_max_trial,
        random_state=denovo_seed,
    )

    df_refsig = pd.concat([df_COSMIC, df_denovo], axis=0)
    k_cosmic = df_COSMIC.shape[0]
    k_ref = df_refsig.shape[0]

    is_cosmic = np.zeros(k_ref, dtype=bool)
    is_denovo = np.zeros(k_ref, dtype=bool)
    is_cosmic[:k_cosmic] = True
    is_denovo[k_cosmic:] = True

    print_log(f"[INFO] df_COSMIC: {df_COSMIC.shape}, df_denovo: {df_denovo.shape}, df_refsig: {df_refsig.shape}",
              session="REFBANK")
    return df_COSMIC, df_denovo, df_refsig, is_cosmic, is_denovo


# ============================================================
# Reference subset sampling
# ============================================================

def sample_ref_subset_indices(
    rng: np.random.Generator,
    n_ref: int,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    k_min: int,
    k_max: int,
) -> np.ndarray:
    """
    Sample a reference subset index list.

    Simplified rules:
    - k uniformly in [k_min, min(k_max, n_ref)]
    - 80% chance: enforce COSMIC >= 30%
    - 20% chance: fully random
    """
    if n_ref <= k_min:
        return np.arange(n_ref, dtype=int)

    k_upper = min(k_max, n_ref)
    k = int(rng.integers(k_min, k_upper + 1))

    idx_cos_all = np.where(is_cosmic)[0]
    idx_den_all = np.where(is_denovo)[0]

    if (len(idx_cos_all) == 0) or (len(idx_den_all) == 0):
        return rng.choice(n_ref, size=k, replace=False)

    if rng.random() < 0.2:
        return rng.choice(n_ref, size=k, replace=False)

    min_cosmic = int(math.ceil(0.3 * k))
    max_cosmic_possible = min(k, len(idx_cos_all))
    min_cosmic = min(min_cosmic, max_cosmic_possible)

    max_cosmic = min(max_cosmic_possible, k - 1) if k > 1 else 1
    if max_cosmic < min_cosmic:
        max_cosmic = min_cosmic

    n_cosmic = int(rng.integers(min_cosmic, max_cosmic + 1))
    n_denovo = k - n_cosmic
    n_denovo = min(n_denovo, len(idx_den_all))
    if n_denovo < k - n_cosmic:
        n_cosmic = k - n_denovo

    idx_cos_chosen = rng.choice(idx_cos_all, size=n_cosmic, replace=False)
    idx_den_chosen = rng.choice(idx_den_all, size=n_denovo, replace=False)
    subset_idx = np.concatenate([idx_cos_chosen, idx_den_chosen])
    rng.shuffle(subset_idx)
    return subset_idx


# ============================================================
# Unified batch simulation
# ============================================================

def _sample_index(rng: np.random.Generator, n: int, weights: Optional[np.ndarray] = None) -> int:
    if weights is None:
        return int(rng.integers(n))
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.all(w <= 0):
        return int(rng.integers(n))
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    return int(rng.choice(n, p=w))


def choose_train_sim_params(epoch_idx: int, total_epochs: int, cur_row: pd.Series) -> Dict[str, Any]:
    """
    Minimal curriculum-aware distributions for training.
    """
    p1 = int(0.3 * total_epochs)

    if epoch_idx < p1:
        n_active_range = (3, 10)
    else:
        n_active_range = (2, 16)

    depth_weights = np.array(
        [
            float(cur_row.get("depth_low", 1.0)),
            float(cur_row.get("depth_mid", 1.0)),
            float(cur_row.get("depth_hig", 1.0)),
        ],
        dtype=float,
    )
    if np.all(depth_weights <= 0):
        depth_weights = None

    norm_frac = float(cur_row.get("norm_frac", 0.25))

    return {
        "comp_alpha_ranges": COMP_ALPHA_BINS,
        "profile_concs": PROFILE_NOISE_LEVELS,
        "depth_modes": DEPTH_MODES,
        "n_active_range": n_active_range,
        "depth_mode_weights": depth_weights,
        "norm_frac": norm_frac,
    }


def simulate_batch_train(
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    sim_params: Dict[str, Any],
    k_min: int,
    k_max: int,
) -> Dict[str, Any]:
    """
    Training batch simulation.
    - Sample ONE reference subset for the batch.
    - For each sample, sample comp_alpha bin, profile_conc, depth_mode.
    """
    n_ref_total = df_refsig.shape[0]
    ctx_cols = df_refsig.columns.to_list()
    k_ctx = len(ctx_cols)

    # Sample a reference subset once per batch, then generate all B samples against it.
    # This matches the idea that a minibatch shares a reference panel, while the panel itself
    # varies across batches to prevent overfitting to a fixed signature bank.
    subset_idx = sample_ref_subset_indices(
        rng=rng,
        n_ref=n_ref_total,
        is_cosmic=is_cosmic,
        is_denovo=is_denovo,
        k_min=k_min,
        k_max=k_max,
    )
    df_ref_sub = df_refsig.iloc[subset_idx].reset_index(drop=True)
    k_sub = df_ref_sub.shape[0]
    ref_profiles_sub = df_ref_sub.values.astype(np.float32)  # [K_sub, 96]

    # Pre-allocate arrays (shapes):
    #   sample_profile_in: [B, 96] model input (either counts or normalized profile)
    #   comp_full:        [B, K_sub] ground-truth mixture weights over the sampled subset
    #   profile_noisy:    [B, 96] noisy profile used for reconstruction monitoring
    #   counts:           [B, 96] integer counts
    #   depths:           [B] sampled sequencing depth / total counts
    sample_profile_in = np.zeros((batch_size, k_ctx), dtype=np.float32)
    comp_full = np.zeros((batch_size, k_sub), dtype=np.float32)
    profile_noisy = np.zeros((batch_size, k_ctx), dtype=np.float32)
    counts = np.zeros((batch_size, k_ctx), dtype=np.int32)
    depths = np.zeros(batch_size, dtype=np.float32)
    is_normalized = np.zeros(batch_size, dtype=bool)

    comp_alpha_ranges = sim_params["comp_alpha_ranges"]
    profile_concs = sim_params["profile_concs"]
    depth_modes = sim_params["depth_modes"]
    n_active_range = sim_params["n_active_range"]
    depth_w = sim_params.get("depth_mode_weights", None)
    norm_frac = float(sim_params.get("norm_frac", 0.25))

    for i in range(batch_size):
        comp_range = comp_alpha_ranges[_sample_index(rng, len(comp_alpha_ranges), None)]
        prof_conc = float(profile_concs[_sample_index(rng, len(profile_concs), None)])
        depth_mode = depth_modes[_sample_index(rng, len(depth_modes), depth_w)]

        max_active_by_depth = MAX_ACTIVE_BY_DEPTH.get(depth_mode, n_active_range[1])
        n_active_min = n_active_range[0]
        n_active_max = min(n_active_range[1], max_active_by_depth, k_sub)
        if n_active_max < n_active_min:
            n_active_min = max(1, n_active_max)
        n_active = int(rng.integers(n_active_min, n_active_max + 1))

        comp_alpha = float(rng.uniform(comp_range[0], comp_range[1]))

        a_out = sample_active_signatures_and_profile(
            df_refsig=df_ref_sub,
            n_active=n_active,
            comp_dirichlet_alpha=comp_alpha,
            min_composition=0.005,
            rng=rng,
        )
        active_idx = a_out["active_idx"]
        comp_vec = a_out["composition"]
        profile_clean = a_out["profile_clean"]

        comp_full[i, active_idx] = comp_vec

        depth = _sample_depth_for_profile(profile_clean, depth_mode, rng)
        depths[i] = depth

        b_out = sample_noisy_counts_from_profile(
            profile_clean=profile_clean,
            depth=int(depth),
            profile_dirichlet_conc=prof_conc,
            rng=rng,
        )
        counts_i = b_out["counts"].astype(np.int32)
        profile_noisy_i = b_out["profile_noisy"].astype(np.float32)

        counts[i] = counts_i
        profile_noisy[i] = profile_noisy_i

        # With probability `norm_frac`, feed the network a normalized profile (sums to ~1).
        # Otherwise feed raw counts. Training on both makes the model robust to upstream
        # preprocessing differences and mixed input conventions.
        if rng.random() < norm_frac:
            sample_profile_in[i] = profile_noisy_i
            is_normalized[i] = True
        else:
            sample_profile_in[i] = counts_i.astype(np.float32)
            is_normalized[i] = False

    # Shuffle the reference subset order to reduce positional shortcuts.
    # Shuffle subset order within batch
    perm = rng.permutation(k_sub)
    ref_profiles_sub = ref_profiles_sub[perm]
    comp_full = comp_full[:, perm]
    is_cosmic_sub = is_cosmic[subset_idx][perm]
    is_denovo_sub = is_denovo[subset_idx][perm]

    ref_profiles_batch = np.tile(ref_profiles_sub[None, :, :], (batch_size, 1, 1))

    return {
        "sample_profile_in": sample_profile_in,
        "comp_full": comp_full,
        "profile_noisy": profile_noisy,
        "counts": counts,
        "depths": depths,
        "is_normalized": is_normalized,
        "ref_profiles": ref_profiles_batch,
        "ref_profiles_sub": ref_profiles_sub,
        "is_cosmic_sub": is_cosmic_sub,
        "is_denovo_sub": is_denovo_sub,
    }


def simulate_batch_fixed(
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    comp_range: Tuple[float, float],
    profile_conc: float,
    depth_mode: str,
    input_mode: str,
    k_fixed: int,
) -> Dict[str, Any]:
    """
    Evaluation (or controlled) batch simulation.
    - Sample ONE reference subset of fixed size k_fixed.
    - Keep comp alpha range / noise / depth_mode fixed for the whole batch.
    - Force input type ("counts" or "normalized").
    """
    n_ref_total = df_refsig.shape[0]
    ctx_cols = df_refsig.columns.to_list()
    k_ctx = len(ctx_cols)

    # Evaluation uses a fixed subset size (k_fixed) so grid comparisons are apples-to-apples.
    subset_idx = sample_ref_subset_indices(
        rng=rng,
        n_ref=n_ref_total,
        is_cosmic=is_cosmic,
        is_denovo=is_denovo,
        k_min=k_fixed,
        k_max=k_fixed,
    )
    df_ref_sub = df_refsig.iloc[subset_idx].reset_index(drop=True)
    k_sub = df_ref_sub.shape[0]
    ref_profiles_sub = df_ref_sub.values.astype(np.float32)

    n_active_min = 1
    max_active = MAX_ACTIVE_BY_DEPTH.get(depth_mode, 16)
    n_active_max = min(max_active, k_sub)

    sample_profile_in = np.zeros((batch_size, k_ctx), dtype=np.float32)
    comp_full = np.zeros((batch_size, k_sub), dtype=np.float32)
    profile_noisy = np.zeros((batch_size, k_ctx), dtype=np.float32)
    counts = np.zeros((batch_size, k_ctx), dtype=np.int32)
    depths = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        n_active = int(rng.integers(n_active_min, n_active_max + 1))
        comp_alpha = float(rng.uniform(comp_range[0], comp_range[1]))

        a_out = sample_active_signatures_and_profile(
            df_refsig=df_ref_sub,
            n_active=n_active,
            comp_dirichlet_alpha=comp_alpha,
            min_composition=0.005,
            rng=rng,
        )
        active_idx = a_out["active_idx"]
        comp_vec = a_out["composition"]
        profile_clean = a_out["profile_clean"]

        comp_full[i, active_idx] = comp_vec

        depth = _sample_depth_for_profile(profile_clean, depth_mode, rng)
        depths[i] = depth

        b_out = sample_noisy_counts_from_profile(
            profile_clean=profile_clean,
            depth=int(depth),
            profile_dirichlet_conc=float(profile_conc),
            rng=rng,
        )
        counts_i = b_out["counts"].astype(np.int32)
        profile_noisy_i = b_out["profile_noisy"].astype(np.float32)

        counts[i] = counts_i
        profile_noisy[i] = profile_noisy_i

        if input_mode == "normalized":
            sample_profile_in[i] = profile_noisy_i
        else:
            sample_profile_in[i] = counts_i.astype(np.float32)

    # Shuffle subset order within batch
    perm = rng.permutation(k_sub)
    ref_profiles_sub = ref_profiles_sub[perm]
    comp_full = comp_full[:, perm]
    is_cosmic_sub = is_cosmic[subset_idx][perm]
    is_denovo_sub = is_denovo[subset_idx][perm]

    ref_profiles_batch = np.tile(ref_profiles_sub[None, :, :], (batch_size, 1, 1))

    return {
        "sample_profile_in": sample_profile_in,
        "comp_full": comp_full,
        "profile_noisy": profile_noisy,
        "counts": counts,
        "depths": depths,
        "ref_profiles": ref_profiles_batch,
        "ref_profiles_sub": ref_profiles_sub,
        "is_cosmic_sub": is_cosmic_sub,
        "is_denovo_sub": is_denovo_sub,
    }


# ============================================================
# LR schedule (kept from original)
# ============================================================


# ============================================================
# Learning-rate schedule
#   Linear warmup -> hold -> linear decay
#   (epoch_pos is fractional: epoch + batch_fraction)
# ============================================================
def compute_lr_factor_for_epoch(
    epoch_pos: float,
    n_epochs: int,
    warmup_epochs: int = 5,
    hold_until_epoch: int = 60,
    min_factor: float = 0.05,
) -> float:
    if n_epochs <= 0:
        return 1.0

    hold_until_epoch = min(hold_until_epoch, n_epochs)

    if epoch_pos <= 0.0:
        return 1e-8

    if epoch_pos < warmup_epochs:
        factor = epoch_pos / float(max(1.0, warmup_epochs))
        return float(max(1e-8, factor))

    if epoch_pos < hold_until_epoch:
        return 1.0

    if hold_until_epoch >= n_epochs:
        return 1.0

    t = (epoch_pos - hold_until_epoch) / float(max(1.0, n_epochs - hold_until_epoch))
    factor = 1.0 - (1.0 - min_factor) * t
    return float(max(min_factor, factor))


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    epoch_idx: int,
    args: argparse.Namespace,
    model: SigFormerCore,
    device: torch.device,
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    sample_seen: int,
    LOSS_record_Xs: List[int],
    loss_trace_values: List[float],
    loss_comp_trace_values: List[float],
    loss_recon_trace_values: List[float],
    loss_conf_trace_values: List[float],
    grad_trace_steps: List[int],
    grad_trace_values: List[float],
    lr_trace_steps: List[int],
    lr_trace_values: List[float],
    cur_row: pd.Series,
    simplex_for_model,
    log_fh,
) -> Tuple[int, int]:
    """
    Train one epoch with batch-wise simulation.
    """
    model.train()
    n_batches = args.n_batches
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    epoch_seed = (args.base_seed + (epoch_idx + 1) * 10007) % (2**32 - 1)
    set_global_seeds(epoch_seed)
    rng = np.random.default_rng(epoch_seed)

    sim_params = choose_train_sim_params(epoch_idx, n_epochs, cur_row)

    run_loss_total = 0.0
    run_loss_compo = 0.0
    run_loss_recon = 0.0
    run_loss_conf = 0.0
    counts_per_run = 0
    current_lr = args.lr_base

    for batch_idx in range(n_batches):
        # LR schedule
        epoch_pos = epoch_idx + batch_idx / max(1, n_batches)
        lr_factor = compute_lr_factor_for_epoch(
            epoch_pos=epoch_pos,
            n_epochs=n_epochs,
            warmup_epochs=args.lr_warmup_epochs,
            hold_until_epoch=args.lr_hold_until_epoch,
            min_factor=args.lr_min_factor,
        )
        base_lr = args.lr_base * lr_factor
        current_lr = base_lr

        # Update LR per parameter group.
        # entmax_alpha (if present) is intentionally frozen (lr=0) to keep simplex behavior stable.
        for pg in optimizer.param_groups:
            name = pg.get("name", "base")
            if name == "base":
                pg["lr"] = base_lr
            elif name == "entmax":
                pg["lr"] = 0.0

        # Simulate training batch
        batch = simulate_batch_train(
            df_refsig=df_refsig,
            is_cosmic=is_cosmic,
            is_denovo=is_denovo,
            batch_size=batch_size,
            rng=rng,
            sim_params=sim_params,
            k_min=args.train_k_min,
            k_max=args.train_k_max,
        )

        x_np = batch["sample_profile_in"]
        comp_true_np = batch["comp_full"]
        profile_noisy_np = batch["profile_noisy"]
        depths_np = batch["depths"]
        ref_profiles_np = batch["ref_profiles"]

        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        comp_true = torch.tensor(comp_true_np, dtype=torch.float32, device=device)
        profile_noisy = torch.tensor(profile_noisy_np, dtype=torch.float32, device=device)
        depths = torch.tensor(depths_np, dtype=torch.float32, device=device)
        ref_profiles_batch = torch.tensor(ref_profiles_np, dtype=torch.float32, device=device)

        B = x.size(0)

        optimizer.zero_grad(set_to_none=True)
        compo_pred, confi_pred = model(x, ref_profiles_batch, simplex=simplex_for_model)

        # composition MSE * k
        k_sig = comp_true.size(1)
        loss_mse_raw = F.mse_loss(compo_pred, comp_true, reduction="mean")
        smp_loss_compo = loss_mse_raw * k_sig

        # reconstruction
        pred_profile = torch.bmm(compo_pred.unsqueeze(1), ref_profiles_batch).squeeze(1)
        diff = pred_profile - profile_noisy
        per_sample_recon = (diff ** 2).mean(dim=1)
        # Scale by depth so deep samples do not dominate purely due to larger absolute counts.
        per_sample_recon = per_sample_recon / (depths + 1e-6)
        smp_loss_recon = per_sample_recon.mean()

        # confidence supervision
        # Confidence supervision: target is high when |pred - true| is small.
        # Exponential keeps the target within (0, 1] with smooth decay.
        with torch.no_grad():
            comp_err = torch.abs(compo_pred.detach() - comp_true)
            conf_target = torch.exp(- args.conf_scale * comp_err)
        smp_loss_conf = F.mse_loss(confi_pred, conf_target, reduction="mean")

        # NOTE: Backprop uses composition + confidence losses.
        # Reconstruction loss is computed for diagnostics/plots only.
        # total for backward (kept consistent with your old intent)
        smp_loss_total = (
            args.lambda_comp * smp_loss_compo
            + args.lambda_conf * smp_loss_conf
        )

        # total for plotting
        loss_total_plot = (
            args.lambda_comp * smp_loss_compo
            + args.lambda_recon * smp_loss_recon
            + args.lambda_conf * smp_loss_conf
        )

        smp_loss_total.backward()

        grad_norm_raw = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_value = float(grad_norm_raw.detach().cpu().item()) if isinstance(grad_norm_raw, torch.Tensor) else float(grad_norm_raw)

        optimizer.step()

        global_step += 1
        sample_seen += B

        run_loss_total += float(smp_loss_total.item())
        run_loss_compo += float(smp_loss_compo.item())
        run_loss_recon += float(smp_loss_recon.item())
        run_loss_conf += float(smp_loss_conf.item())
        counts_per_run += 1

        if ((batch_idx + 1) % args.log_every == 0) or (batch_idx + 1 == n_batches):
            run_loss_total /= max(1, counts_per_run)
            run_loss_compo /= max(1, counts_per_run)
            run_loss_recon /= max(1, counts_per_run)
            run_loss_conf /= max(1, counts_per_run)

            msg = (
                f"[EPOCH {epoch_idx + 1:03d}/{n_epochs:03d}] "
                f"batch {batch_idx + 1:4d}/{n_batches:04d}, "
                f"samples seen:{sample_seen:.3e}, "
                f"lr={current_lr:.2e}, "
                f"loss: total={run_loss_total:.3e}, "
                f"compo={run_loss_compo:.3e}, "
                f"recon={run_loss_recon:.3e}, "
                f"conf={run_loss_conf:.3e}"
            )
            print_log(msg, session="TRAIN", log_fh=log_fh)

            run_loss_total, run_loss_compo, run_loss_recon, run_loss_conf, counts_per_run = 0.0, 0.0, 0.0, 0.0, 0

        # traces
        LOSS_record_Xs.append(sample_seen)
        loss_trace_values.append(float(loss_total_plot.item()))
        loss_comp_trace_values.append(float((args.lambda_comp * smp_loss_compo).item()))
        loss_recon_trace_values.append(float((args.lambda_recon * smp_loss_recon).item()))
        loss_conf_trace_values.append(float((args.lambda_conf * smp_loss_conf).item()))
        grad_trace_steps.append(sample_seen)
        grad_trace_values.append(grad_value)
        lr_trace_steps.append(sample_seen)
        lr_trace_values.append(current_lr)

    return global_step, sample_seen


# ============================================================
# Eval dataset builder (cell-wise subsets)
# ============================================================

def build_eval_cells_for_depth_category(
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    depth_category: str,
    n_per_combo: int,
    rng: np.random.Generator,
    k_eval: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Build evaluation cells for one depth_category.

    Each cell corresponds to:
      (profile_conc_idx=j_prof, comp_bin_idx=j_comp)

    Each cell will sample its own reference subset of fixed size k_eval,
    and then generate n_per_combo samples using simulate_batch_fixed.

    This is the simple, direct fix to the old "one subset per depth category" bug.
    """
    depth_mode_map = {
        "normalized": "mixed",
        "low": "low",
        "sufficient": "medium",
        "deep": "high",
    }
    if depth_category not in depth_mode_map:
        raise ValueError(f"Unknown depth_category: {depth_category}")

    depth_mode = depth_mode_map[depth_category]
    input_mode = "normalized" if depth_category == "normalized" else "counts"

    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for j_prof, prof_conc in enumerate(PROFILE_NOISE_LEVELS):
        for j_comp, comp_range in enumerate(COMP_ALPHA_BINS):
            batch = simulate_batch_fixed(
                df_refsig=df_refsig,
                is_cosmic=is_cosmic,
                is_denovo=is_denovo,
                batch_size=n_per_combo,
                rng=rng,
                comp_range=comp_range,
                profile_conc=prof_conc,
                depth_mode=depth_mode,
                input_mode=input_mode,
                k_fixed=k_eval,
            )

            # store per-cell dataset
            cells[(j_prof, j_comp)] = {
                "comp_full": batch["comp_full"],
                "profile_noisy": batch["profile_noisy"],
                "counts": batch["counts"],
                "depths": batch["depths"],
                "ref_profiles": batch["ref_profiles_sub"],
                "is_cosmic_sub": batch["is_cosmic_sub"],
                "is_denovo_sub": batch["is_denovo_sub"],
                "profile_conc": float(prof_conc),
                "comp_range": comp_range,
            }

    return cells


def build_all_eval_cells(
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    n_per_combo: int,
    base_seed: int,
    k_eval: int,
) -> Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]:
    rng_master = np.random.default_rng(base_seed)
    out = {}

    for depth_category in ["normalized", "low", "sufficient", "deep"]:
        seed = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)

        print_log(f"[EVAL-DATA] Building eval cells for depth_category={depth_category}",
                  session="EVALDATA")

        out[depth_category] = build_eval_cells_for_depth_category(
            df_refsig=df_refsig,
            is_cosmic=is_cosmic,
            is_denovo=is_denovo,
            depth_category=depth_category,
            n_per_combo=n_per_combo,
            rng=rng,
            k_eval=k_eval,
        )

    return out


# ============================================================
# Main
# ============================================================

def main(args):
    print_log("initializing", session="INIT")

    out_dir = args.dir

    # Output layout:
    #   train_log.txt         - training log
    #   summary_epochs.tsv    - per-epoch numeric summary
    #   2_eval_figs/          - evaluation grid figures
    #   3_model_wts/          - model checkpoints
    ensure_dir(out_dir)

    DIR_eval_figs = os.path.join(out_dir, "2_eval_figs")
    DIR_model_wts = os.path.join(out_dir, "3_model_wts")
    ensure_dir(DIR_model_wts)
    ensure_dir(DIR_eval_figs)

    # seed
    if args.base_seed is None:
        args.base_seed = make_run_seed()
    args.base_seed = int(args.base_seed % (2**32 - 1))
    set_global_seeds(args.base_seed)
    print_log(f"[SEED] base_seed={args.base_seed}", session="INIT")

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        print_log("[WARN] CUDA not available, falling back to CPU", session="INIT")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # log file
    log_path = os.path.join(out_dir, "train_log.txt")
    log_fh = open(log_path, "w", buffering=1)
    bind_log_file(log_fh)
    log_fh.write(f"[INFO] Run dir: {out_dir}\n")
    log_fh.write(f"[INFO] base_seed: {args.base_seed}\n")
    log_fh.flush()

    # curriculum
    curriculum_df = load_curriculum(args, n_epochs=args.n_epochs, out_dir=out_dir)
    print_log(f"[CURRICULUM] Using curriculum with {len(curriculum_df)} rows", session="INIT")

    # reference bank
    df_COSMIC, df_denovo, df_refsig, is_cosmic, is_denovo = build_reference_bank()
    n_ref = df_refsig.shape[0]
    n_chann = df_refsig.shape[1]

    # model
    print_log("build model", session="MODEL")
    model = SigFormerCore(
        n_chann=n_chann,
        d_model=args.model_d_model,
        n_heads=args.model_n_heads,
        n_L_smp=args.model_smp_n_lyr,
        n_L_ref=args.model_ref_n_lyr,
        dropout=args.model_dropout,
    )
    model.to(device)

    # freeze entmax_alpha if present
    if hasattr(model, "entmax_alpha"):
        with torch.no_grad():
            try:
                model.entmax_alpha.fill_(1.5)
            except Exception:
                model.entmax_alpha.data[...] = 1.5
        model.entmax_alpha.requires_grad = False

    para_base = []
    para_emax = []
    for name, p in model.named_parameters():
        if "entmax_alpha" in name:
            para_emax.append(p)
        else:
            para_base.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": para_base, "lr": args.lr_base, "weight_decay": args.weight_decay, "name": "base"},
            {"params": para_emax, "lr": 0.0,          "weight_decay": args.weight_decay, "name": "entmax"},
        ]
    )

    # eval cells (cell-wise reference subsets)
    print_log("build eval cells", session="EVALDATA")
    eval_cells_all = build_all_eval_cells(
        df_refsig=df_refsig,
        is_cosmic=is_cosmic,
        is_denovo=is_denovo,
        n_per_combo=args.eval_n_per_combo,
        base_seed=args.base_seed + 2025,
        k_eval=args.eval_k,
    )

    # training traces
    global_step = 0
    sample_seen = 0

    LOSS_record_Xs: List[int] = []
    loss_trace_values: List[float] = []
    loss_comp_trace_values: List[float] = []
    loss_recon_trace_values: List[float] = []
    loss_conf_trace_values: List[float] = []
    grad_trace_steps: List[int] = []
    grad_trace_values: List[float] = []
    lr_trace_steps: List[int] = []
    lr_trace_values: List[float] = []

    summary_tsv_path = os.path.join(out_dir, "summary_epochs.tsv")
    if not os.path.exists(summary_tsv_path):
        with open(summary_tsv_path, "w") as fh:
            fh.write(
                "epoch\t"
                "train_loss_avg\ttrain_loss_last\t"
                "train_conf_loss_avg\ttrain_conf_loss_last\t"
                "r2_cosmic_norm\t"
                "r2_cosmic_low\t"
                "r2_cosmic_sufficient\t"
                "r2_cosmic_deep\t"
                "r2_denovo_norm\t"
                "r2_denovo_low\t"
                "r2_denovo_sufficient\t"
                "r2_denovo_deep\n"
            )

    print_log("start training", session="TRAIN")

    for epoch_idx in range(args.n_epochs):
        print_log(f"[EPOCH] ===== Epoch {epoch_idx + 1}/{args.n_epochs} =====", session="TRAIN")

        cur_row = get_curriculum_row_for_epoch(curriculum_df, epoch_idx)
        simplex_raw = str(cur_row.get("simplex", "softmax")).lower()
        epoch_num = epoch_idx + 1

        # keep your original rule: switch to entmax15 after a threshold
        if epoch_num < args.entmax_fixed_start_epoch:
            simplex_for_model = simplex_raw
        else:
            simplex_for_model = "entmax15"

        # indices for epoch-average loss
        loss_before = len(loss_trace_values)
        loss_conf_before = len(loss_conf_trace_values)

        # train
        global_step, sample_seen = train_one_epoch(
            epoch_idx=epoch_idx,
            args=args,
            model=model,
            device=device,
            df_refsig=df_refsig,
            is_cosmic=is_cosmic,
            is_denovo=is_denovo,
            optimizer=optimizer,
            global_step=global_step,
            sample_seen=sample_seen,
            LOSS_record_Xs=LOSS_record_Xs,
            loss_trace_values=loss_trace_values,
            loss_comp_trace_values=loss_comp_trace_values,
            loss_recon_trace_values=loss_recon_trace_values,
            loss_conf_trace_values=loss_conf_trace_values,
            grad_trace_steps=grad_trace_steps,
            grad_trace_values=grad_trace_values,
            lr_trace_steps=lr_trace_steps,
            lr_trace_values=lr_trace_values,
            cur_row=cur_row,
            simplex_for_model=simplex_for_model,
            log_fh=log_fh,
        )

        loss_after = len(loss_trace_values)
        loss_conf_after = len(loss_conf_trace_values)

        train_loss_epoch = float(np.mean(loss_trace_values[loss_before:loss_after])) if loss_after > loss_before else float("nan")
        train_loss_last = float(loss_trace_values[-1]) if loss_after > 0 else float("nan")

        train_loss_conf_epoch = float(np.mean(loss_conf_trace_values[loss_conf_before:loss_conf_after])) if loss_conf_after > loss_conf_before else float("nan")
        train_loss_conf_last = float(loss_conf_trace_values[-1]) if loss_conf_after > 0 else float("nan")

        # Save a self-contained checkpoint (model + optimizer + counters + args snapshot).
        # save ckpt
        ckpt_path = os.path.join(DIR_model_wts, f"SigFormer_epoch{epoch_idx + 1:03d}.pt")
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "sample_seen": sample_seen,
                "args": vars(args),
            },
            ckpt_path,
        )
        print_log(f"[CKPT] Saved model checkpoint to {ckpt_path}", session="MODEL")

        # eval plots
        print_log("[EVAL] start eval", session="EVAL")
        r2_summary = {}

        for depth_category in ["normalized", "low", "sufficient", "deep"]:
            cells = eval_cells_all[depth_category]
            fig_path = os.path.join(DIR_eval_figs, f"epoch{epoch_idx + 1:03d}_eval_{depth_category}.png")

            out = eval_and_plot_grid(
                model=model,
                device=device,
                depth_category=depth_category,
                cells=cells,
                comp_bins=COMP_ALPHA_BINS,
                profile_concs=PROFILE_NOISE_LEVELS,
                fig_out_path=fig_path,
                simplex_for_model=simplex_for_model,
            )

            r2_cosmic_mean = float(out["r2_cosmic"].values.mean())
            r2_denovo_mean = float(out["r2_denovo"].values.mean())
            r2_summary[(depth_category, "cosmic")] = r2_cosmic_mean
            r2_summary[(depth_category, "denovo")] = r2_denovo_mean

            print_log(
                f"[EVAL] epoch={epoch_idx + 1:03d}, depth={depth_category}, "
                f"R2(COSMIC)={r2_cosmic_mean:.4f}, R2(denovo)={r2_denovo_mean:.4f}",
                session="EVAL",
            )

        # global curves
        plot_global_loss_grad_lr(
            out_dir=out_dir,
            epoch_idx=epoch_idx,
            args=args,
            LOSS_record_Xs=LOSS_record_Xs,
            loss_trace_values=loss_trace_values,
            loss_comp_trace_values=loss_comp_trace_values,
            loss_recon_trace_values=loss_recon_trace_values,
            loss_conf_trace_values=loss_conf_trace_values,
            grad_trace_steps=grad_trace_steps,
            grad_trace_values=grad_trace_values,
            lr_trace_steps=lr_trace_steps,
            lr_trace_values=lr_trace_values,
        )

        # summary TSV
        with open(summary_tsv_path, "a") as fh:
            fh.write(
                f"{epoch_idx + 1}\t"
                f"{train_loss_epoch:.6e}\t{train_loss_last:.6e}\t"
                f"{train_loss_conf_epoch:.6e}\t{train_loss_conf_last:.6e}\t"
                f"{r2_summary.get(('normalized','cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low','cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient','cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep','cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('normalized','denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low','denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient','denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep','denovo'), float('nan')):.6f}\n"
            )
        print_log(f"[SUMMARY] epoch={epoch_idx + 1:03d} summary written to {summary_tsv_path}", session="EVAL")

    log_fh.close()
    print("training finished.")


# ============================================================
# CLI
# ============================================================


# Example:
#   python s03_Train_V5_6_refactor.py --dir ./run_001 --device cuda --n_epochs 200
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigFormer training 2025-11-15 (V5_6 refactor)")

    # basic
    g_basic = parser.add_argument_group("basic")
    g_basic.add_argument("--dir", type=str, required=True, help="output directory for this run")
    g_basic.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    g_basic.add_argument("--base_seed", type=int, default=None, help="base random seed (if None, auto-generate)")
    g_basic.add_argument("--curriculum", type=str, default=None, help="optional curriculum TSV path")

    # model
    g_model = parser.add_argument_group("model")
    g_model.add_argument("--model_d_model", type=int, default=256)
    g_model.add_argument("--model_n_heads", type=int, default=8)
    g_model.add_argument("--model_smp_n_lyr", type=int, default=2)
    g_model.add_argument("--model_ref_n_lyr", type=int, default=4)
    g_model.add_argument("--model_dropout", type=float, default=0.1)

    # training
    g_train = parser.add_argument_group("train")
    g_train.add_argument("--n_epochs", type=int, default=200)
    g_train.add_argument("--n_batches", type=int, default=4000)
    g_train.add_argument("--batch_size", type=int, default=64)

    g_train.add_argument("--lr_base", type=float, default=3e-4)
    g_train.add_argument("--weight_decay", type=float, default=1e-4)
    g_train.add_argument("--grad_clip", type=float, default=1.0)

    # LR schedule knobs (were hard-coded before)
    g_train.add_argument("--lr_warmup_epochs", type=int, default=5)
    g_train.add_argument("--lr_hold_until_epoch", type=int, default=60)
    g_train.add_argument("--lr_min_factor", type=float, default=0.05)

    # reference subset range for training
    g_train.add_argument("--train_k_min", type=int, default=10)
    g_train.add_argument("--train_k_max", type=int, default=90)

    # losses
    g_train.add_argument("--lambda_comp", type=float, default=1.0)
    g_train.add_argument("--lambda_recon", type=float, default=0.1)
    g_train.add_argument("--lambda_conf", type=float, default=0.001, help="weight for confidence loss")
    g_train.add_argument("--conf_scale", type=float, default=4.0, help="scale factor in exp(-conf_scale * |pred-true|)")

    # entmax switching
    g_train.add_argument("--entmax_fixed_start_epoch", type=int, default=30, help="epoch to switch to entmax15")

    # logging frequency
    g_train.add_argument("--log_every", type=int, default=100)

    # eval
    g_eval = parser.add_argument_group("eval")
    g_eval.add_argument("--eval_n_per_combo", type=int, default=1200,
                        help="per (comp_bin, noise_level) samples for eval")
    g_eval.add_argument("--eval_k", type=int, default=60,
                        help="fixed reference subset size for each eval cell")

    args = parser.parse_args()
    main(args)
