#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SigFormer V6 training script.

Design goals for V6:
1. Keep the V5 core data simulator and transformer logic recognizable.
2. Remove trainable entmax alpha. Use an explicit simplex schedule instead.
3. Let reconstruction loss really backpropagate, but only after warmup and with an FP penalty.
4. Evaluate every 5 epochs by default, not every epoch.
5. Save per-batch loss curves for each epoch.
6. Add a simple OOD holdout check based on residual / novelty diagnostics.
7. Stretch the default training to 300 epochs, with the last 100 epochs balanced and sparse.
"""

import argparse
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from YZ_vis_sig import *  # get_COSMIC

from s01_SigFormer_Core import SigFormerCore
from s02_Data_Generation import (
    _sample_depth_for_profile,
    build_denovo_signatures,
    sample_active_signatures_and_profile,
    sample_noisy_counts_from_profile,
    summarize_cosmic_signatures,
)
from s03_Train_Utils import (
    bind_log_file,
    build_confidence_target,
    composition_cosine_loss,
    composition_mse_loss,
    ensure_dir,
    eval_and_plot_grid,
    false_positive_weak_loss,
    plot_epoch_batch_losses,
    plot_global_loss_grad_lr,
    print_log,
)

# ============================================================
# Global defaults
# ============================================================


COMP_ALPHA_BINS: List[Tuple[float, float]] = [
    (0.05, 0.2),
    (0.2, 0.8),
    (0.8, 2.0),
    (2.0, 5.0),
    (5.0, 20.0),
]

PROFILE_NOISE_LEVELS: List[float] = [40.0, 120.0, 280.0, 2300.0]
DEPTH_MODES: List[str] = ["low", "medium", "high"]
MAX_ACTIVE_BY_DEPTH = {"low": 4, "medium": 10, "high": 16, "mixed": 16}
REF_BUCKETS = [(2, 15), (16, 40), (41, 80), (81, 200)]
REF_BUCKET_WEIGHTS = [0.18, 0.22, 0.30, 0.30]


# ============================================================
# Seeds and schedules
# ============================================================


def make_run_seed() -> int:
    return int(time.time() * 1000.0) % (2**32 - 1)



def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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
        return max(1e-8, epoch_pos / float(max(1.0, warmup_epochs)))
    if epoch_pos < hold_until_epoch or hold_until_epoch >= n_epochs:
        return 1.0
    t = (epoch_pos - hold_until_epoch) / float(max(1.0, n_epochs - hold_until_epoch))
    return max(min_factor, 1.0 - (1.0 - min_factor) * t)



def build_default_curriculum(n_epochs: int) -> pd.DataFrame:
    """
    Stage 1: softmax warm start, no recon gradient yet.
    Stage 2: entmax15, full IID difficulty.
    Stage 3: entmax15 + bucketed reference sizes + stronger FP control.
    Stage 4: final 100 epochs, sparsemax + balanced tail.
    """
    n_epochs = max(1, int(n_epochs))
    tail_epochs = min(100, max(30, n_epochs // 3))
    stage1_end = max(20, int(round(0.2 * (n_epochs - tail_epochs))))
    stage2_end = max(stage1_end + 40, int(round(0.6 * (n_epochs - tail_epochs))))
    stage3_end = n_epochs - tail_epochs
    stage3_end = max(stage2_end + 20, stage3_end)
    stage3_end = min(stage3_end, n_epochs - 1)

    rows = [
        dict(
            epoch_stt=1,
            epoch_end=stage1_end,
            simplex="softmax",
            subset_mode="uniform",
            depth_low=0.2,
            depth_mid=0.4,
            depth_hig=0.4,
            norm_frac=0.35,
            n_active_min=3,
            n_active_max=10,
            train_k_min=20,
            train_k_max=90,
            lambda_recon=0.00,
            lambda_fp_weak=0.00,
            lambda_comp_cos=0.00,
        ),
        dict(
            epoch_stt=stage1_end + 1,
            epoch_end=stage2_end,
            simplex="entmax15",
            subset_mode="uniform",
            depth_low=0.30,
            depth_mid=0.40,
            depth_hig=0.30,
            norm_frac=0.25,
            n_active_min=2,
            n_active_max=16,
            train_k_min=10,
            train_k_max=120,
            lambda_recon=0.03,
            lambda_fp_weak=0.015,
            lambda_comp_cos=0.00,
        ),
        dict(
            epoch_stt=stage2_end + 1,
            epoch_end=stage3_end,
            simplex="entmax15",
            subset_mode="bucketed",
            depth_low=0.34,
            depth_mid=0.33,
            depth_hig=0.33,
            norm_frac=0.25,
            n_active_min=2,
            n_active_max=16,
            train_k_min=4,
            train_k_max=180,
            lambda_recon=0.05,
            lambda_fp_weak=0.035,
            lambda_comp_cos=0.01,
        ),
        dict(
            epoch_stt=stage3_end + 1,
            epoch_end=n_epochs,
            simplex="sparsemax",
            subset_mode="bucketed",
            depth_low=0.34,
            depth_mid=0.33,
            depth_hig=0.33,
            norm_frac=0.25,
            n_active_min=1,
            n_active_max=16,
            train_k_min=2,
            train_k_max=200,
            lambda_recon=0.05,
            lambda_fp_weak=0.060,
            lambda_comp_cos=0.02,
        ),
    ]
    return pd.DataFrame(rows)



def load_curriculum(args: argparse.Namespace, out_dir: str) -> pd.DataFrame:
    if args.curriculum is not None and os.path.isfile(args.curriculum):
        df = pd.read_csv(args.curriculum, sep="\t")
        print_log(f"[CURRICULUM] Loaded curriculum from {args.curriculum}", session="INIT")
    else:
        df = build_default_curriculum(args.n_epochs)
        cur_path = os.path.join(out_dir, "curriculum.tsv")
        df.to_csv(cur_path, sep="\t", index=False)
        print_log(f"[CURRICULUM] Wrote default curriculum to {cur_path}", session="INIT")
    return df



def get_curriculum_row_for_epoch(curriculum_df: pd.DataFrame, epoch_idx: int) -> pd.Series:
    epoch_num = epoch_idx + 1
    mask = (curriculum_df["epoch_stt"] <= epoch_num) & (curriculum_df["epoch_end"] >= epoch_num)
    if not mask.any():
        return curriculum_df.iloc[-1]
    return curriculum_df[mask].iloc[0]


# ============================================================
# Reference bank
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
    print_log(
        f"[INFO] df_COSMIC: {df_COSMIC.shape}, df_denovo: {df_denovo.shape}, df_refsig: {df_refsig.shape}",
        session="REFBANK",
    )
    return df_COSMIC, df_denovo, df_refsig, is_cosmic, is_denovo


# ============================================================
# Batch simulation
# ============================================================


def _sample_index(rng: np.random.Generator, n: int, weights: Optional[np.ndarray] = None) -> int:
    if weights is None:
        return int(rng.integers(n))
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != n or np.all(weights <= 0):
        return int(rng.integers(n))
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    return int(rng.choice(n, p=weights))



def sample_ref_subset_indices_uniform(
    rng: np.random.Generator,
    n_ref: int,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    k_min: int,
    k_max: int,
) -> np.ndarray:
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
    min_cos = min(int(math.ceil(0.3 * k)), len(idx_cos_all))
    max_cos = min(k, len(idx_cos_all))
    if max_cos < min_cos:
        max_cos = min_cos
    n_cos = int(rng.integers(min_cos, max_cos + 1)) if max_cos > 0 else 0
    n_den = min(k - n_cos, len(idx_den_all))
    if n_den < k - n_cos:
        n_cos = k - n_den
    idx_cos = rng.choice(idx_cos_all, size=n_cos, replace=False)
    idx_den = rng.choice(idx_den_all, size=n_den, replace=False)
    subset_idx = np.concatenate([idx_cos, idx_den])
    rng.shuffle(subset_idx)
    return subset_idx



def sample_ref_subset_indices_bucketed(
    rng: np.random.Generator,
    n_ref: int,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    buckets: List[Tuple[int, int]] = REF_BUCKETS,
    weights: List[float] = REF_BUCKET_WEIGHTS,
) -> np.ndarray:
    if n_ref <= 0:
        return np.array([], dtype=int)
    bucket_idx = int(rng.choice(len(buckets), p=np.asarray(weights) / np.sum(weights)))
    k_min, k_max = buckets[bucket_idx]
    k_min = max(1, int(k_min))
    k_max = min(max(k_min, int(k_max)), n_ref)
    if n_ref <= k_min:
        return np.arange(n_ref, dtype=int)
    k = int(rng.integers(k_min, k_max + 1))
    if k >= n_ref:
        return np.arange(n_ref, dtype=int)
    idx_cos_all = np.where(is_cosmic)[0]
    idx_den_all = np.where(is_denovo)[0]
    if (len(idx_cos_all) == 0) or (len(idx_den_all) == 0):
        return rng.choice(n_ref, size=k, replace=False)
    n_cos = int(rng.integers(1, min(k, len(idx_cos_all)) + 1))
    n_den = min(k - n_cos, len(idx_den_all))
    if n_den < k - n_cos:
        n_cos = k - n_den
    idx_cos = rng.choice(idx_cos_all, size=n_cos, replace=False)
    idx_den = rng.choice(idx_den_all, size=n_den, replace=False)
    subset_idx = np.concatenate([idx_cos, idx_den])
    rng.shuffle(subset_idx)
    return subset_idx



def choose_train_sim_params(cur_row: pd.Series) -> Dict[str, Any]:
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
    return {
        "comp_alpha_ranges": COMP_ALPHA_BINS,
        "profile_concs": PROFILE_NOISE_LEVELS,
        "depth_modes": DEPTH_MODES,
        "n_active_range": (int(cur_row.get("n_active_min", 2)), int(cur_row.get("n_active_max", 16))),
        "depth_mode_weights": depth_weights,
        "norm_frac": float(cur_row.get("norm_frac", 0.25)),
        "subset_mode": str(cur_row.get("subset_mode", "uniform")).lower(),
        "train_k_min": int(cur_row.get("train_k_min", 10)),
        "train_k_max": int(cur_row.get("train_k_max", 120)),
    }



def simulate_batch_train(
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    sim_params: Dict[str, Any],
) -> Dict[str, Any]:
    n_ref_total = df_refsig.shape[0]
    n_ctx = df_refsig.shape[1]

    if sim_params["subset_mode"] == "bucketed":
        subset_idx = sample_ref_subset_indices_bucketed(rng, n_ref_total, is_cosmic, is_denovo)
    else:
        subset_idx = sample_ref_subset_indices_uniform(
            rng,
            n_ref_total,
            is_cosmic,
            is_denovo,
            k_min=sim_params["train_k_min"],
            k_max=sim_params["train_k_max"],
        )

    df_ref_sub = df_refsig.iloc[subset_idx].reset_index(drop=True)
    k_sub = df_ref_sub.shape[0]
    ref_profiles_sub = df_ref_sub.values.astype(np.float32)

    x_in = np.zeros((batch_size, n_ctx), dtype=np.float32)
    comp_full = np.zeros((batch_size, k_sub), dtype=np.float32)
    profile_noisy = np.zeros((batch_size, n_ctx), dtype=np.float32)
    counts = np.zeros((batch_size, n_ctx), dtype=np.int32)
    depths = np.zeros(batch_size, dtype=np.float32)

    comp_alpha_ranges = sim_params["comp_alpha_ranges"]
    profile_concs = sim_params["profile_concs"]
    depth_modes = sim_params["depth_modes"]
    n_active_range = sim_params["n_active_range"]
    depth_weights = sim_params.get("depth_mode_weights", None)
    norm_frac = float(sim_params.get("norm_frac", 0.25))

    for i in range(batch_size):
        comp_range = comp_alpha_ranges[_sample_index(rng, len(comp_alpha_ranges), None)]
        prof_conc = float(profile_concs[_sample_index(rng, len(profile_concs), None)])
        depth_mode = depth_modes[_sample_index(rng, len(depth_modes), depth_weights)]

        max_active = min(n_active_range[1], MAX_ACTIVE_BY_DEPTH.get(depth_mode, n_active_range[1]), k_sub)
        min_active = min(max(1, n_active_range[0]), max_active)
        n_active = int(rng.integers(min_active, max_active + 1))
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
        depths[i] = depth
        x_in[i] = profile_noisy_i if rng.random() < norm_frac else counts_i.astype(np.float32)

    perm = rng.permutation(k_sub)
    ref_profiles_sub = ref_profiles_sub[perm]
    comp_full = comp_full[:, perm]
    ref_profiles_batch = np.tile(ref_profiles_sub[None, :, :], (batch_size, 1, 1))

    return {
        "sample_profile_in": x_in,
        "comp_full": comp_full,
        "profile_noisy": profile_noisy,
        "counts": counts,
        "depths": depths,
        "ref_profiles": ref_profiles_batch,
        "ref_profiles_sub": ref_profiles_sub,
        "is_cosmic_sub": is_cosmic[subset_idx][perm],
        "is_denovo_sub": is_denovo[subset_idx][perm],
        "k_sub": int(k_sub),
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
    n_ref_total = df_refsig.shape[0]
    subset_idx = sample_ref_subset_indices_uniform(
        rng,
        n_ref_total,
        is_cosmic,
        is_denovo,
        k_min=k_fixed,
        k_max=k_fixed,
    )
    df_ref_sub = df_refsig.iloc[subset_idx].reset_index(drop=True)
    k_sub = df_ref_sub.shape[0]
    ref_profiles_sub = df_ref_sub.values.astype(np.float32)
    n_ctx = ref_profiles_sub.shape[1]

    x_in = np.zeros((batch_size, n_ctx), dtype=np.float32)
    comp_full = np.zeros((batch_size, k_sub), dtype=np.float32)
    profile_noisy = np.zeros((batch_size, n_ctx), dtype=np.float32)
    counts = np.zeros((batch_size, n_ctx), dtype=np.int32)
    depths = np.zeros(batch_size, dtype=np.float32)

    max_active = min(MAX_ACTIVE_BY_DEPTH.get(depth_mode, 16), k_sub)
    for i in range(batch_size):
        n_active = int(rng.integers(1, max_active + 1))
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
        depths[i] = depth
        x_in[i] = profile_noisy_i if input_mode == "normalized" else counts_i.astype(np.float32)

    perm = rng.permutation(k_sub)
    ref_profiles_sub = ref_profiles_sub[perm]
    comp_full = comp_full[:, perm]
    ref_profiles_batch = np.tile(ref_profiles_sub[None, :, :], (batch_size, 1, 1))
    return {
        "sample_profile_in": x_in,
        "comp_full": comp_full,
        "profile_noisy": profile_noisy,
        "counts": counts,
        "depths": depths,
        "ref_profiles": ref_profiles_batch,
        "ref_profiles_sub": ref_profiles_sub,
        "is_cosmic_sub": is_cosmic[subset_idx][perm],
        "is_denovo_sub": is_denovo[subset_idx][perm],
    }


# ============================================================
# Training loop
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
    loss_fp_trace_values: List[float],
    grad_trace_steps: List[int],
    grad_trace_values: List[float],
    lr_trace_steps: List[int],
    lr_trace_values: List[float],
    cur_row: pd.Series,
    simplex_for_model: str,
    log_fh,
) -> Tuple[int, int, pd.DataFrame]:
    model.train()
    n_batches = args.n_batches
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    epoch_seed = (args.base_seed + (epoch_idx + 1) * 10007) % (2**32 - 1)
    set_global_seeds(epoch_seed)
    rng = np.random.default_rng(epoch_seed)
    sim_params = choose_train_sim_params(cur_row)

    lambda_recon_stage = float(cur_row.get("lambda_recon", args.lambda_recon))
    lambda_fp_stage = float(cur_row.get("lambda_fp_weak", args.lambda_fp_weak))
    lambda_comp_cos_stage = float(cur_row.get("lambda_comp_cos", args.lambda_comp_cos))

    run_loss_total = 0.0
    run_loss_compo = 0.0
    run_loss_recon = 0.0
    run_loss_conf = 0.0
    run_loss_fp = 0.0
    counts_per_run = 0
    current_lr = args.lr_base
    batch_records: List[Dict[str, float]] = []

    for batch_idx in range(n_batches):
        epoch_pos = epoch_idx + batch_idx / max(1, n_batches)
        lr_factor = compute_lr_factor_for_epoch(
            epoch_pos=epoch_pos,
            n_epochs=n_epochs,
            warmup_epochs=args.lr_warmup_epochs,
            hold_until_epoch=args.lr_hold_until_epoch,
            min_factor=args.lr_min_factor,
        )
        current_lr = args.lr_base * lr_factor
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        batch = simulate_batch_train(
            df_refsig=df_refsig,
            is_cosmic=is_cosmic,
            is_denovo=is_denovo,
            batch_size=batch_size,
            rng=rng,
            sim_params=sim_params,
        )

        x = torch.tensor(batch["sample_profile_in"], dtype=torch.float32, device=device)
        comp_true = torch.tensor(batch["comp_full"], dtype=torch.float32, device=device)
        profile_noisy = torch.tensor(batch["profile_noisy"], dtype=torch.float32, device=device)
        depths = torch.tensor(batch["depths"], dtype=torch.float32, device=device)
        ref_profiles_batch = torch.tensor(batch["ref_profiles"], dtype=torch.float32, device=device)

        B = x.size(0)
        optimizer.zero_grad(set_to_none=True)
        compo_pred, confi_pred, aux = model(x, ref_profiles_batch, simplex=simplex_for_model, return_aux=True)

        loss_compo = composition_mse_loss(compo_pred, comp_true)
        loss_comp_cos = torch.tensor(0.0, device=device)
        if lambda_comp_cos_stage > 0:
            loss_comp_cos = composition_cosine_loss(compo_pred, comp_true)

        pred_profile = aux["recon_profile"]
        diff = pred_profile - profile_noisy
        per_sample_recon = (diff ** 2).mean(dim=1)
        per_sample_recon = per_sample_recon / (depths + 1e-6)
        loss_recon = per_sample_recon.mean()

        with torch.no_grad():
            conf_target = build_confidence_target(
                compo_pred_detach=compo_pred.detach(),
                comp_true=comp_true,
                conf_scale=args.conf_scale,
                thr_true=args.fp_thr_act,
                fp_scale=args.conf_fp_scale,
                miss_scale=args.conf_miss_scale,
            )
        loss_conf = torch.nn.functional.mse_loss(confi_pred, conf_target, reduction="mean")
        loss_fp = false_positive_weak_loss(compo_pred, comp_true, thr_act=args.fp_thr_act, power=args.fp_power)

        loss_total = (
            args.lambda_comp * loss_compo
            + lambda_comp_cos_stage * loss_comp_cos
            + lambda_recon_stage * loss_recon
            + args.lambda_conf * loss_conf
            + lambda_fp_stage * loss_fp
        )
        loss_total.backward()

        grad_norm_raw = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_value = float(grad_norm_raw.detach().cpu().item()) if isinstance(grad_norm_raw, torch.Tensor) else float(grad_norm_raw)
        optimizer.step()

        global_step += 1
        sample_seen += B
        run_loss_total += float(loss_total.item())
        run_loss_compo += float(loss_compo.item())
        run_loss_recon += float(loss_recon.item())
        run_loss_conf += float(loss_conf.item())
        run_loss_fp += float(loss_fp.item())
        counts_per_run += 1

        batch_records.append(
            {
                "epoch": epoch_idx + 1,
                "batch": batch_idx + 1,
                "lr": current_lr,
                "k_sub": float(batch["k_sub"]),
                "loss_total": float(loss_total.item()),
                "loss_comp": float(loss_compo.item()),
                "loss_comp_cos": float(loss_comp_cos.item()),
                "loss_recon": float(loss_recon.item()),
                "loss_conf": float(loss_conf.item()),
                "loss_fp": float(loss_fp.item()),
                "grad_norm": grad_value,
            }
        )

        if ((batch_idx + 1) % args.log_every == 0) or (batch_idx + 1 == n_batches):
            run_loss_total /= max(1, counts_per_run)
            run_loss_compo /= max(1, counts_per_run)
            run_loss_recon /= max(1, counts_per_run)
            run_loss_conf /= max(1, counts_per_run)
            run_loss_fp /= max(1, counts_per_run)
            msg = (
                f"[EPOCH {epoch_idx + 1:03d}/{n_epochs:03d}] "
                f"batch {batch_idx + 1:4d}/{n_batches:04d}, "
                f"samples seen:{sample_seen:.3e}, lr={current_lr:.2e}, "
                f"loss total={run_loss_total:.3e}, comp={run_loss_compo:.3e}, "
                f"recon={run_loss_recon:.3e}, conf={run_loss_conf:.3e}, fp={run_loss_fp:.3e}"
            )
            print_log(msg, session="TRAIN", log_fh=log_fh)
            run_loss_total = run_loss_compo = run_loss_recon = run_loss_conf = run_loss_fp = 0.0
            counts_per_run = 0

        LOSS_record_Xs.append(sample_seen)
        loss_trace_values.append(float(loss_total.item()))
        loss_comp_trace_values.append(float((args.lambda_comp * loss_compo).item()))
        loss_recon_trace_values.append(float((lambda_recon_stage * loss_recon).item()))
        loss_conf_trace_values.append(float((args.lambda_conf * loss_conf).item()))
        loss_fp_trace_values.append(float((lambda_fp_stage * loss_fp).item()))
        grad_trace_steps.append(sample_seen)
        grad_trace_values.append(grad_value)
        lr_trace_steps.append(sample_seen)
        lr_trace_values.append(current_lr)

    return global_step, sample_seen, pd.DataFrame(batch_records)


# ============================================================
# Eval dataset builder
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
        print_log(f"[EVAL-DATA] Building eval cells for depth_category={depth_category}", session="EVALDATA")
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



def run_ood_holdout_eval(
    model: SigFormerCore,
    device: torch.device,
    df_refsig: pd.DataFrame,
    is_cosmic: np.ndarray,
    is_denovo: np.ndarray,
    out_dir: str,
    simplex_for_model: str,
    seed: int = 9527,
    n_samples: int = 768,
    k_visible: int = 80,
) -> Dict[str, float]:
    """
    Hold out one active signature from the visible reference set and ask whether
    residual-based novelty rises with hidden mass.
    """
    rng = np.random.default_rng(seed)
    subset_idx = sample_ref_subset_indices_uniform(rng, df_refsig.shape[0], is_cosmic, is_denovo, k_visible, k_visible)
    visible_set = set(subset_idx.tolist())
    hidden_pool = np.array([i for i in range(df_refsig.shape[0]) if i not in visible_set], dtype=int)
    if hidden_pool.size == 0:
        return {"ood_corr": float("nan"), "ood_mean_score": float("nan")}

    df_visible = df_refsig.iloc[subset_idx].reset_index(drop=True)
    ref_profiles = df_visible.values.astype(np.float32)
    n_ctx = ref_profiles.shape[1]

    x_in = np.zeros((n_samples, n_ctx), dtype=np.float32)
    hidden_mass = np.zeros(n_samples, dtype=np.float32)
    ref_batch = np.tile(ref_profiles[None, :, :], (n_samples, 1, 1))

    for i in range(n_samples):
        n_active_vis = int(rng.integers(1, min(8, df_visible.shape[0]) + 1))
        comp_alpha = float(rng.uniform(0.2, 5.0))
        a_out = sample_active_signatures_and_profile(
            df_refsig=df_visible,
            n_active=n_active_vis,
            comp_dirichlet_alpha=comp_alpha,
            min_composition=0.01,
            rng=rng,
        )
        profile_clean = a_out["profile_clean"]

        use_hidden = rng.random() < 0.6
        if use_hidden:
            hidden_idx = int(rng.choice(hidden_pool))
            hidden_sig = df_refsig.values[hidden_idx].astype(float)
            hidden_sig = hidden_sig / (hidden_sig.sum() + 1e-12)
            mass = float(rng.uniform(0.05, 0.30))
            profile_clean = (1.0 - mass) * profile_clean + mass * hidden_sig
            hidden_mass[i] = mass

        depth_mode = rng.choice(["low", "medium", "high"])
        depth = _sample_depth_for_profile(profile_clean, depth_mode, rng)
        profile_conc = float(rng.choice(PROFILE_NOISE_LEVELS))
        b_out = sample_noisy_counts_from_profile(profile_clean, depth=int(depth), profile_dirichlet_conc=profile_conc, rng=rng)
        x_in[i] = b_out["counts"].astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(x_in, dtype=torch.float32, device=device)
        ref = torch.tensor(ref_batch, dtype=torch.float32, device=device)
        _, _, aux = model(x, ref, simplex=simplex_for_model, return_aux=True)
        novelty_score = aux["residual_cosine"].detach().cpu().numpy()

    corr = np.corrcoef(hidden_mass, novelty_score)[0, 1] if np.std(hidden_mass) > 0 and np.std(novelty_score) > 0 else float("nan")
    mean_score_ood = float(np.mean(novelty_score[hidden_mass > 0])) if np.any(hidden_mass > 0) else float("nan")
    mean_score_iid = float(np.mean(novelty_score[hidden_mass == 0])) if np.any(hidden_mass == 0) else float("nan")

    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(hidden_mass, novelty_score, s=6, alpha=0.35)
    ax.set_xlabel("hidden signature mass")
    ax.set_ylabel("novelty score (residual cosine)")
    ax.set_title(f"OOD holdout check, corr={corr:.3f}")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "ood_holdout_scatter.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print_log(f"[PLOT] Saved OOD holdout scatter to {fig_path}", session="EVAL")
    return {"ood_corr": float(corr), "ood_mean_score": mean_score_ood, "iid_mean_score": mean_score_iid}


# ============================================================
# Main
# ============================================================


def main(args):
    print_log("initializing", session="INIT")
    out_dir = args.dir
    ensure_dir(out_dir)
    DIR_eval_figs = os.path.join(out_dir, "2_eval_figs")
    DIR_model_wts = os.path.join(out_dir, "3_model_wts")
    DIR_batch_loss = os.path.join(out_dir, "4_batch_loss")
    ensure_dir(DIR_eval_figs)
    ensure_dir(DIR_model_wts)
    ensure_dir(DIR_batch_loss)

    if args.base_seed is None:
        args.base_seed = make_run_seed()
    args.base_seed = int(args.base_seed % (2**32 - 1))
    set_global_seeds(args.base_seed)
    print_log(f"[SEED] base_seed={args.base_seed}", session="INIT")

    if args.device == "cuda" and not torch.cuda.is_available():
        print_log("[WARN] CUDA not available, falling back to CPU", session="INIT")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log_path = os.path.join(out_dir, "train_log.txt")
    log_fh = open(log_path, "w", buffering=1)
    bind_log_file(log_fh)
    log_fh.write(f"[INFO] Run dir: {out_dir}\n")
    log_fh.write(f"[INFO] base_seed: {args.base_seed}\n")
    log_fh.flush()

    curriculum_df = load_curriculum(args, out_dir=out_dir)
    print_log(f"[CURRICULUM] Using curriculum with {len(curriculum_df)} rows", session="INIT")

    df_COSMIC, df_denovo, df_refsig, is_cosmic, is_denovo = build_reference_bank()
    n_chann = df_refsig.shape[1]

    print_log("build model", session="MODEL")
    model = SigFormerCore(
        n_chann=n_chann,
        d_model=args.model_d_model,
        n_heads=args.model_n_heads,
        n_L_smp=args.model_smp_n_lyr,
        n_L_ref=args.model_ref_n_lyr,
        dropout=args.model_dropout,
        confidence_detach_backbone=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_base, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        sample_seen = int(ckpt.get("sample_seen", 0))
        print_log(f"[RESUME] Loaded checkpoint from {args.resume_ckpt}", session="MODEL")
        print_log(f"[RESUME] start_epoch={start_epoch + 1}, global_step={global_step}, sample_seen={sample_seen}", session="MODEL")

    print_log("build eval cells", session="EVALDATA")
    eval_cells_all = build_all_eval_cells(
        df_refsig=df_refsig,
        is_cosmic=is_cosmic,
        is_denovo=is_denovo,
        n_per_combo=args.eval_n_per_combo,
        base_seed=args.base_seed + 2025,
        k_eval=args.eval_k,
    )

    global_step = 0
    sample_seen = 0
    LOSS_record_Xs: List[int] = []
    loss_trace_values: List[float] = []
    loss_comp_trace_values: List[float] = []
    loss_recon_trace_values: List[float] = []
    loss_conf_trace_values: List[float] = []
    loss_fp_trace_values: List[float] = []
    grad_trace_steps: List[int] = []
    grad_trace_values: List[float] = []
    lr_trace_steps: List[int] = []
    lr_trace_values: List[float] = []

    summary_tsv_path = os.path.join(out_dir, "summary_epochs.tsv")
    if not os.path.exists(summary_tsv_path):
        with open(summary_tsv_path, "w") as fh:
            fh.write(
                "epoch\ttrain_loss_avg\ttrain_loss_last\ttrain_conf_loss_avg\ttrain_conf_loss_last\t"
                "r2_cosmic_norm\tr2_cosmic_low\tr2_cosmic_sufficient\tr2_cosmic_deep\t"
                "r2_denovo_norm\tr2_denovo_low\tr2_denovo_sufficient\tr2_denovo_deep\t"
                "f1_mask_norm\tf1_mask_low\tf1_mask_sufficient\tf1_mask_deep\t"
                "confidence_corr_norm\tconfidence_corr_low\tconfidence_corr_sufficient\tconfidence_corr_deep\t"
                "ood_corr\tood_mean_score\tiid_mean_score\n"
            )

    print_log("start training", session="TRAIN")
    for epoch_idx in range(start_epoch, args.n_epochs):
        print_log(f"[EPOCH] ===== Epoch {epoch_idx + 1}/{args.n_epochs} =====", session="TRAIN")
        cur_row = get_curriculum_row_for_epoch(curriculum_df, epoch_idx)
        simplex_for_model = str(cur_row.get("simplex", "entmax15")).lower()

        loss_before = len(loss_trace_values)
        conf_before = len(loss_conf_trace_values)

        global_step, sample_seen, batch_df = train_one_epoch(
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
            loss_fp_trace_values=loss_fp_trace_values,
            grad_trace_steps=grad_trace_steps,
            grad_trace_values=grad_trace_values,
            lr_trace_steps=lr_trace_steps,
            lr_trace_values=lr_trace_values,
            cur_row=cur_row,
            simplex_for_model=simplex_for_model,
            log_fh=log_fh,
        )

        plot_epoch_batch_losses(DIR_batch_loss, epoch_idx, batch_df)

        loss_after = len(loss_trace_values)
        conf_after = len(loss_conf_trace_values)
        train_loss_epoch = float(np.mean(loss_trace_values[loss_before:loss_after])) if loss_after > loss_before else float("nan")
        train_loss_last = float(loss_trace_values[-1]) if loss_after > 0 else float("nan")
        train_loss_conf_epoch = float(np.mean(loss_conf_trace_values[conf_before:conf_after])) if conf_after > conf_before else float("nan")
        train_loss_conf_last = float(loss_conf_trace_values[-1]) if conf_after > 0 else float("nan")

        ckpt_path = os.path.join(DIR_model_wts, f"SigFormer_v6_epoch{epoch_idx + 1:03d}.pt")
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

        need_eval = ((epoch_idx + 1) % args.eval_every == 0) or (epoch_idx + 1 == args.n_epochs)
        r2_summary = {}
        ood_summary = {"ood_corr": float("nan"), "ood_mean_score": float("nan"), "iid_mean_score": float("nan")}

        if need_eval:
            print_log("[EVAL] start eval", session="EVAL")
            for depth_category in ["normalized", "low", "sufficient", "deep"]:
                fig_path = os.path.join(DIR_eval_figs, f"epoch{epoch_idx + 1:03d}_eval_{depth_category}.png")
                out = eval_and_plot_grid(
                    model=model,
                    device=device,
                    depth_category=depth_category,
                    cells=eval_cells_all[depth_category],
                    comp_bins=COMP_ALPHA_BINS,
                    profile_concs=PROFILE_NOISE_LEVELS,
                    fig_out_path=fig_path,
                    simplex_for_model=simplex_for_model,
                )
                r2_summary[(depth_category, "cosmic")] = float(out["r2_cosmic"].values.mean())
                r2_summary[(depth_category, "denovo")] = float(out["r2_denovo"].values.mean())
                r2_summary[(depth_category, "f1_masked")] = float(out["f1_masked"].values.mean())
                r2_summary[(depth_category, "confidence_corr")] = float(out["confidence_corr"].values.mean())
                print_log(
                    f"[EVAL] epoch={epoch_idx + 1:03d}, depth={depth_category}, "
                    f"R2(COSMIC)={r2_summary[(depth_category, 'cosmic')]:.4f}, "
                    f"R2(denovo)={r2_summary[(depth_category, 'denovo')]:.4f}, "
                    f"F1(masked)={r2_summary[(depth_category, 'f1_masked')]:.4f}",
                    session="EVAL",
                )

            ood_summary = run_ood_holdout_eval(
                model=model,
                device=device,
                df_refsig=df_refsig,
                is_cosmic=is_cosmic,
                is_denovo=is_denovo,
                out_dir=DIR_eval_figs,
                simplex_for_model=simplex_for_model,
                seed=args.base_seed + epoch_idx + 9000,
                n_samples=args.ood_eval_n_samples,
                k_visible=args.ood_eval_k_visible,
            )
            print_log(
                f"[EVAL] OOD holdout corr={ood_summary['ood_corr']:.4f}, "
                f"OOD mean score={ood_summary['ood_mean_score']:.4f}, "
                f"IID mean score={ood_summary['iid_mean_score']:.4f}",
                session="EVAL",
            )

        plot_global_loss_grad_lr(
            out_dir=out_dir,
            epoch_idx=epoch_idx,
            args=args,
            LOSS_record_Xs=LOSS_record_Xs,
            loss_trace_values=loss_trace_values,
            loss_comp_trace_values=loss_comp_trace_values,
            loss_recon_trace_values=loss_recon_trace_values,
            loss_conf_trace_values=loss_conf_trace_values,
            loss_fp_trace_values=loss_fp_trace_values,
            grad_trace_steps=grad_trace_steps,
            grad_trace_values=grad_trace_values,
            lr_trace_steps=lr_trace_steps,
            lr_trace_values=lr_trace_values,
        )

        with open(summary_tsv_path, "a") as fh:
            fh.write(
                f"{epoch_idx + 1}\t{train_loss_epoch:.6e}\t{train_loss_last:.6e}\t"
                f"{train_loss_conf_epoch:.6e}\t{train_loss_conf_last:.6e}\t"
                f"{r2_summary.get(('normalized', 'cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low', 'cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient', 'cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep', 'cosmic'), float('nan')):.6f}\t"
                f"{r2_summary.get(('normalized', 'denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low', 'denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient', 'denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep', 'denovo'), float('nan')):.6f}\t"
                f"{r2_summary.get(('normalized', 'f1_masked'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low', 'f1_masked'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient', 'f1_masked'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep', 'f1_masked'), float('nan')):.6f}\t"
                f"{r2_summary.get(('normalized', 'confidence_corr'), float('nan')):.6f}\t"
                f"{r2_summary.get(('low', 'confidence_corr'), float('nan')):.6f}\t"
                f"{r2_summary.get(('sufficient', 'confidence_corr'), float('nan')):.6f}\t"
                f"{r2_summary.get(('deep', 'confidence_corr'), float('nan')):.6f}\t"
                f"{ood_summary.get('ood_corr', float('nan')):.6f}\t"
                f"{ood_summary.get('ood_mean_score', float('nan')):.6f}\t"
                f"{ood_summary.get('iid_mean_score', float('nan')):.6f}\n"
            )
        print_log(f"[SUMMARY] epoch={epoch_idx + 1:03d} summary written to {summary_tsv_path}", session="EVAL")

    log_fh.close()
    print("training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigFormer V6 training")

    g_basic = parser.add_argument_group("basic")
    g_basic.add_argument("--dir", type=str, required=True, help="output directory for this run")
    g_basic.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    g_basic.add_argument("--base_seed", type=int, default=None)
    g_basic.add_argument("--curriculum", type=str, default=None, help="optional curriculum TSV path")
    g_basic.add_argument("--resume_ckpt", type=str, default=None, help="resume from an existing checkpoint")

    g_model = parser.add_argument_group("model")
    g_model.add_argument("--model_d_model", type=int, default=256)
    g_model.add_argument("--model_n_heads", type=int, default=8)
    g_model.add_argument("--model_smp_n_lyr", type=int, default=2)
    g_model.add_argument("--model_ref_n_lyr", type=int, default=4)
    g_model.add_argument("--model_dropout", type=float, default=0.1)

    g_train = parser.add_argument_group("train")
    g_train.add_argument("--n_epochs", type=int, default=300)
    g_train.add_argument("--n_batches", type=int, default=4000)
    g_train.add_argument("--batch_size", type=int, default=64)
    g_train.add_argument("--lr_base", type=float, default=3e-4)
    g_train.add_argument("--weight_decay", type=float, default=1e-4)
    g_train.add_argument("--grad_clip", type=float, default=1.0)
    g_train.add_argument("--lr_warmup_epochs", type=int, default=5)
    g_train.add_argument("--lr_hold_until_epoch", type=int, default=80)
    g_train.add_argument("--lr_min_factor", type=float, default=0.05)
    g_train.add_argument("--log_every", type=int, default=100)

    g_loss = parser.add_argument_group("loss")
    g_loss.add_argument("--lambda_comp", type=float, default=1.0)
    g_loss.add_argument("--lambda_recon", type=float, default=0.05, help="fallback value if curriculum row does not override")
    g_loss.add_argument("--lambda_conf", type=float, default=0.002)
    g_loss.add_argument("--lambda_fp_weak", type=float, default=0.05, help="fallback value if curriculum row does not override")
    g_loss.add_argument("--lambda_comp_cos", type=float, default=0.0, help="fallback value if curriculum row does not override")
    g_loss.add_argument("--fp_thr_act", type=float, default=0.02)
    g_loss.add_argument("--fp_power", type=float, default=1.0)
    g_loss.add_argument("--conf_scale", type=float, default=4.0)
    g_loss.add_argument("--conf_fp_scale", type=float, default=12.0)
    g_loss.add_argument("--conf_miss_scale", type=float, default=8.0)

    g_eval = parser.add_argument_group("eval")
    g_eval.add_argument("--eval_every", type=int, default=5)
    g_eval.add_argument("--eval_n_per_combo", type=int, default=1200)
    g_eval.add_argument("--eval_k", type=int, default=60)
    g_eval.add_argument("--ood_eval_n_samples", type=int, default=768)
    g_eval.add_argument("--ood_eval_k_visible", type=int, default=80)

    args = parser.parse_args()
    main(args)
