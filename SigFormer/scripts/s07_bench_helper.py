#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark helpers for SigFormer.

Design rules: one benchmark bank, one per-batch pool, explicit model calls, deterministic
folder layout, deterministic reading. Method wrappers live in ``s06_wrapper``.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import re
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, eye, save_npz
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


from . import s02_Data as data_v9
from . import s04_Util_apply as SgF_util
from .s06_wrapper import (
    R_env,
    CLASS_wrapper_MuSiCal,
    CLASS_wrapper_SigFormer,
    CLASS_wrapper_SigProfilerAssignment,
    CLASS_wrapper_sigfit,
    CLASS_wrapper_SigLASSO,
    CLASS_wrapper_sig_tool_lib,
)

ensure_dir = SgF_util.ensure_dir


# ============================================================================
# Benchmark configuration and deterministic path helpers
# ============================================================================

METHOD_ORDER = ["SgF_raw", "SgF", "Mus", "SPA", "sft", "sLS", "stl"]
METHOD_LABELS = {
    "SgF_raw": "SigFormer raw",
    "SgF": "SigFormer",
    "Mus": "MuSiCal",
    "SPA": "SigProfilerAssignment",
    "sft": "sigfit",
    "sLS": "sigLASSO",
    "stl": "signature.tools.lib",
}
OOD_PRED_METHODS = ["SgF_raw", "SgF", "sLS", "stl"]
BENCHMARK_MODES = ["no_ood", "random", "cosmic_titration", "ood_titration"]
MODE_ALIASES = {"denovo_titration": "ood_titration", "no_OOC": "no_ood", "random_OOC": "random", "titration_COSMIC": "cosmic_titration", "titration_OOC": "ood_titration"}


def log_msg(message, verbose=True):
    if verbose:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

def normalize_mode(mode: str) -> str:
    mode = str(mode)
    return MODE_ALIASES.get(mode, mode)


def stable_seed(base_seed=0, *parts) -> int:
    txt = "|".join([str(base_seed)] + [str(x) for x in parts])
    return int(hashlib.md5(txt.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


_sample_seed = stable_seed


def parse_ref_size(text_or_dict) -> Dict[str, int]:
    if isinstance(text_or_dict, dict):
        return {str(k).upper(): int(v) for k, v in text_or_dict.items()}
    out = {"COSMIC": 80, "MOCK": 20}
    if text_or_dict:
        out = {}
        for part in str(text_or_dict).split(","):
            if part.strip():
                k, v = part.split(":", 1)
                out[k.strip().upper()] = int(v)
    return out


def parse_steps(text_or_steps) -> List[float]:
    if isinstance(text_or_steps, (list, tuple, np.ndarray)):
        return [float(x) for x in text_or_steps]
    return [float(x) for x in str(text_or_steps).replace(" ", "").split(",") if x != ""]


def parse_signatures(text) -> List[str]:
    if text is None:
        return ["auto"]
    text = str(text).strip()
    if text in {"", "auto", "all_cosmic"}:
        return [text or "auto"]
    return [x for x in text.replace(";", ",").replace(" ", ",").split(",") if x]


def cosmic_version_label(cosmic_version="v3.4") -> str:
    s = str(cosmic_version)
    return s if s.startswith("v") else f"v{s}"


def benchmark_root(date_tag=None, cosmic_version="v3.4", base_dir=".") -> Path:
    day = date_tag or _dt.datetime.now().strftime("%Y%m%d")
    return Path(base_dir) / f"benchmark_{day}_COSMIC_{cosmic_version_label(cosmic_version)}"


def condition_slug(noise_bin, depth_bin, active_bin, compo_bin) -> str:
    return f"condition_noise-{noise_bin}_depth-{depth_bin}_active-{active_bin}_compo-{compo_bin}"


def condition_path(root, mode, noise_bin, depth_bin, active_bin, compo_bin, signature=None) -> Path:
    mode = normalize_mode(mode)
    active = active_bin if mode in {"no_ood", "random"} else "NA"
    path = Path(root) / f"mode_{mode}" / condition_slug(noise_bin, depth_bin, active, compo_bin)
    if mode == "cosmic_titration" and signature not in {None, "", "auto", "all_cosmic"}:
        path = path / str(signature)
    return path


def MAKE_config_batch(mode="no_ood", BSize=64, depth_bin="401-7000", noise_bin="0.90-0.95", compo_bin=1.0,
                      active_bin="1-3", REF_size=None, pcNOM=0.30, random_ood_frac=0.30,
                      random_ood_max_compo=0.60, titer_steps=None, titer_std=0.05, titer_nsmps=10,
                      **kwargs):
    mode = normalize_mode(mode)
    cfg = copy.deepcopy(data_v9.DEFAULT_CONFIG_SINGLE)
    cfg.update(dict(mode=mode, BSize=int(BSize), pcNOM=float(pcNOM), pcOOD=0.0,
                    random_ood_frac=float(random_ood_frac), random_ood_max_compo=float(random_ood_max_compo),
                    titer_std=float(titer_std), titer_nsmps=int(titer_nsmps)))
    cfg["REF_size"] = parse_ref_size(REF_size or cfg.get("REF_size", {"COSMIC": 50, "MOCK": 50}))
    cfg["DEPTH"] = {str(depth_bin): 1}
    cfg["ACTVE"] = {str(active_bin): 1}
    cfg["COMPO"] = {float(compo_bin): 1}
    default_noise = data_v9.DEFAULT_CONFIG_SINGLE.get("NOISE", {})
    cfg["NOISE"] = {str(noise_bin): float(default_noise.get(str(noise_bin), kwargs.get("noise_alpha", 500.0)))}
    cfg["titer_steps"] = parse_steps(titer_steps if titer_steps is not None else [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    cfg["ood_min_compo"] = float(kwargs.get("ood_min_compo", 0.03))
    cfg["titration_bg_active"] = kwargs.get("titration_bg_active", "1-10")
    cfg["denovo_source"] = kwargs.get("denovo_source", "leaveout_mock")
    cfg["titer_signature"] = kwargs.get("titer_signature", None)
    cfg["noise_in_band_min"] = float(kwargs.get("noise_in_band_min", 0.90))
    cfg["ref_mask_policy"] = "full_reference_except_ood"
    cfg["cosmic_version"] = kwargs.get("cosmic_version", "v3.4")
    return cfg


# ============================================================================
# Reference-bank construction and benchmark data generation
# ============================================================================


def load_or_build_bank(cosmic_path=None, cosmic_version="v3.4", n_mock=1000, random_state=2026,
                       cache_path=None, verbose=True, **mock_kwargs):
    cache = None if cache_path is None else Path(cache_path)
    if cache is not None and cache.exists():
        log_msg(f"Loading reference bank cache: {cache}", verbose)
        with open(cache, "rb") as f:
            bank, M_bank = pickle.load(f)
        log_msg(f"Loaded bank: n_ref={bank.shape[0]}, n_context={bank.shape[1]}", verbose)
        return bank, M_bank
    log_msg(f"Building reference bank: COSMIC={cosmic_version_label(cosmic_version)}, n_mock={int(n_mock)}, seed={random_state}", verbose)
    df_COSMIC = data_v9.get_COSMIC(version=cosmic_version_label(cosmic_version), path=cosmic_path)
    bank, M_bank = data_v9.build_grand_ref_pool(df_COSMIC, n_mock=int(n_mock), random_state=random_state,
                                                verbose=verbose, **mock_kwargs)
    log_msg(f"Built bank: n_ref={bank.shape[0]}, n_context={bank.shape[1]}", verbose)
    if cache is not None:
        ensure_dir(cache.parent)
        with open(cache, "wb") as f:
            pickle.dump((bank, M_bank), f)
        log_msg(f"Saved reference bank cache: {cache}", verbose)
    return bank, M_bank


def downsample_pool(bank, M_bank, cfg, rng):
    return data_v9.downsample_batch_refpool(bank, M_bank, rng=rng,
                                            n_cosmic=int(cfg["REF_size"].get("COSMIC", 0)),
                                            n_mock=int(cfg["REF_size"].get("MOCK", 0)))


def force_signature_into_pool(pool, M_pool, bank, M_bank, signature, rng):
    if signature is None or signature in set(pool.index):
        return pool, M_pool
    if signature not in set(bank.index):
        raise KeyError(f"requested signature not found in bank: {signature}")
    sig_meta = M_bank.loc[signature] if M_bank is not None and signature in set(M_bank.index) else pd.Series(dtype=object)
    want_cos = bool(sig_meta.get("is_cosmic", not str(signature).startswith("MOK")))
    cand = M_pool.index[M_pool.get("is_cosmic", pd.Series(False, index=M_pool.index)).astype(bool)] if want_cos else M_pool.index[M_pool.get("is_mock", pd.Series(False, index=M_pool.index)).astype(bool)]
    cand = [x for x in cand if x != signature]
    replace = cand[int(rng.integers(0, len(cand)))] if cand else pool.index[-1]
    pool = pd.concat([pool.drop(index=replace, errors="ignore"), bank.loc[[signature]]], axis=0)
    M_pool = pd.concat([M_pool.drop(index=replace, errors="ignore"), M_bank.loc[[signature]]], axis=0).reindex(index=pool.index)
    return pool, M_pool


def in_band_ratio(batch) -> float:
    return float(batch["M_sampl_meta"]["noise_status"].astype(str).eq("in_band").mean())


def generate_single_mode_batch(bank, M_bank, cfg, mode, rng, batch_id):
    best_batch = None
    best_ratio = -1.0
    local = copy.deepcopy(cfg)
    local["pcOOD"] = 0.0
    for trial in range(int(cfg.get("noise_qc_max_trials", 20))):
        pool, M_pool = downsample_pool(bank, M_bank, local, rng)
        seed = int(rng.integers(1, 2**31 - 1))
        batch = data_v9.generate_single_batch(pool, M_pool, config=local, random_state=seed, batch_id=batch_id)
        batch["Y_prior_mask"] = pd.DataFrame(True, index=batch["Y_prior_mask"].index, columns=batch["Y_prior_mask"].columns)
        batch["Y_compo_mask"] = pd.DataFrame(True, index=batch["Y_compo_mask"].index, columns=batch["Y_compo_mask"].columns)
        batch["Y__OOD__mask"] = pd.DataFrame(False, index=batch["Y__OOD__mask"].index, columns=batch["Y__OOD__mask"].columns)
        ratio = in_band_ratio(batch)
        if ratio > best_ratio:
            best_batch, best_ratio = batch, ratio
        if ratio >= float(cfg.get("noise_in_band_min", 0.90)):
            break
    if mode == "random":
        apply_random_ood(best_batch, cfg, rng)
    best_batch["M_sampl_meta"]["benchmark_mode"] = mode
    best_batch["M_sampl_meta"]["noise_in_band_ratio"] = best_ratio
    return best_batch


def apply_random_ood(batch, cfg, rng):
    y = batch["Y_compo_true"].astype(float)
    min_compo = float(cfg.get("ood_min_compo", 0.03))
    max_compo = float(cfg.get("random_ood_max_compo", cfg.get("ood_max_compo", 0.60)))
    eligible = (y >= min_compo) & (y <= max_compo)
    rows = eligible.index[eligible.any(axis=1)].to_numpy()
    frac = float(cfg.get("random_ood_frac", cfg.get("pcOOD", 0.30)))
    n_pick = min(len(rows), int(math.ceil(frac * y.shape[0])))
    chosen = rows if len(rows) <= n_pick else rng.choice(rows, size=n_pick, replace=False)
    for sid in chosen:
        sigs = eligible.columns[eligible.loc[sid].to_numpy(bool)].to_numpy()
        sig = str(rng.choice(sigs))
        batch["Y__OOD__mask"].loc[sid, sig] = True
        batch["Y_prior_mask"].loc[sid, sig] = False
        batch["Y_compo_mask"].loc[sid, sig] = False
        batch["M_sampl_meta"].loc[sid, "n_ood"] = int(batch["Y__OOD__mask"].loc[sid].sum())
        batch["M_sampl_meta"].loc[sid, "ood_refs"] = json.dumps([sig])
        batch["M_sampl_meta"].loc[sid, "benchmark_ood_source"] = "random_leaveout"
        batch["M_sampl_meta"].loc[sid, "random_ood_compo"] = float(y.loc[sid, sig])
    not_chosen = batch["M_sampl_meta"].index.difference(pd.Index(chosen))
    batch["M_sampl_meta"].loc[not_chosen, "benchmark_ood_source"] = "no_ood"


def choose_titration_signature(pool, M_pool, bank, M_bank, cfg, mode, rng):
    requested = cfg.get("titer_signature", None)
    if requested not in {None, "", "auto"}:
        sig = str(requested)
    elif mode == "cosmic_titration":
        sig = str(M_bank.index[M_bank["is_cosmic"].astype(bool)][0])
    else:
        mock_pool = M_bank.index[M_bank["is_mock"].astype(bool)].to_numpy()
        sig = str(rng.choice(mock_pool))
    pool, M_pool = force_signature_into_pool(pool, M_pool, bank, M_bank, sig, rng)
    return sig, pool, M_pool


def sample_background_composition(n_bg, bg_mass, alpha, min_frac, rng):
    if n_bg <= 0 or bg_mass <= 0:
        return np.array([], dtype=float)
    floor = min(float(min_frac), 0.80 * bg_mass / max(n_bg, 1))
    for _ in range(200):
        comp = rng.dirichlet(np.full(n_bg, float(alpha))) * bg_mass
        if np.all(comp >= floor):
            return comp
    comp = rng.dirichlet(np.full(n_bg, float(alpha))) * bg_mass
    return comp


def make_titration_row(sid, pool, cfg, mode, titer_signature, step, rng, rep):
    ood_idx = int(pool.index.get_loc(titer_signature))
    target = 0.0 if float(step) <= 0 else float(np.clip(rng.normal(float(step), float(cfg.get("titer_std", 0.05))), 0.0, 1.0))
    bg_lo, bg_hi = data_v9.parse_range_key(cfg.get("titration_bg_active", "1-10"))
    n_bg = 0 if target >= 0.999 else int(rng.integers(bg_lo, bg_hi + 1))
    n_bg = min(n_bg, max(0, pool.shape[0] - 1))
    bg_idx = data_v9.sample_active_indices(pool.shape[0], n_bg, rng, exclude=[ood_idx]) if n_bg > 0 else np.array([], dtype=int)
    depth, depth_band = data_v9.sample_depth(cfg, max(1, n_bg + int(target > 0)), rng)
    alpha = float(data_v9.weighted_choice(cfg["COMPO"], rng))
    bg_mass = max(0.0, 1.0 - target)
    bg_comp = sample_background_composition(n_bg, bg_mass, alpha, cfg.get("ood_min_compo", 0.03), rng)
    active_idx = np.r_[([ood_idx] if target > 0 else []), bg_idx].astype(int)
    comp = np.r_[([target] if target > 0 else []), bg_comp].astype(float)
    if active_idx.size == 0:
        bg_idx = data_v9.sample_active_indices(pool.shape[0], 1, rng, exclude=[ood_idx])
        active_idx = bg_idx.astype(int)
        comp = np.array([1.0], dtype=float)
    comp = data_v9.normalize_vec(comp)
    x_true = data_v9.profile_from_composition(pool, active_idx, comp)
    x_noisy, noise_band, noise_alpha, noise_status, noise_cosine = data_v9.perturb_profile(x_true, cfg, rng)
    depth, depth_band = data_v9.sample_depth(cfg, len(active_idx), rng)
    counts = rng.multinomial(depth, x_noisy)
    x_data = counts / max(int(counts.sum()), 1) if rng.random() < float(cfg.get("pcNOM", 0.0)) else counts.astype(float)
    y = np.zeros(pool.shape[0], dtype=float)
    y[active_idx] = comp
    y_ood = np.zeros(pool.shape[0], dtype=bool)
    if mode == "ood_titration" and target > 0:
        y_ood[ood_idx] = True
    y_prior = np.ones(pool.shape[0], dtype=bool)
    y_mask = y_prior & ~y_ood
    ood_refs = [str(titer_signature)] if bool(y_ood[ood_idx]) else []
    active_refs = pool.index[active_idx].tolist()
    return {"meta": {"sample_id": sid, "scheme": "titration", "benchmark_mode": mode,
                      "titer_step": float(step), "titer_actual": float(target),
                      "titer_signature": str(titer_signature),
                      "titer_source": "COSMIC" if mode == "cosmic_titration" else str(cfg.get("denovo_source", "leaveout_mock")),
                      "titer_rep": int(rep), "n_active": int(len(active_idx)), "depth": int(depth),
                      "depth_band": depth_band, "comp_alpha": alpha, "n_pruned_active": 0,
                      "noise_band": noise_band, "noise_alpha": noise_alpha, "noise_status": noise_status,
                      "noise_cosine": noise_cosine, "normalized_input": bool(np.isclose(x_data.sum(), 1.0)),
                      "prior_target_size": int(pool.shape[0]), "prior_actual_size": int(y_prior.sum()),
                      "n_ood": int(y_ood.sum()), "active_refs": json.dumps(active_refs),
                      "ood_refs": json.dumps(ood_refs), "profile_entropy": data_v9.shannon_entropy(x_true),
                      "profile_gini": data_v9.gini_coef(x_true), "input_total": float(x_data.sum())},
            "y_compo": y, "y_count": y * depth, "y_active": y > 0, "y_prior": y_prior,
            "y_ood": y_ood, "y_mask": y_mask, "x_true": x_true, "x_noisy": x_noisy,
            "x_count": counts, "x_data": x_data}


def generate_titration_batch(bank, M_bank, cfg, mode, rng, batch_id):
    pool, M_pool = downsample_pool(bank, M_bank, cfg, rng)
    titer_signature, pool, M_pool = choose_titration_signature(pool, M_pool, bank, M_bank, cfg, mode, rng)
    rows = []
    for step in cfg.get("titer_steps", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        for rep in range(int(cfg.get("titer_nsmps", 12))):
            sid = f"T{len(rows):06d}"
            rows.append(make_titration_row(sid, pool, cfg, mode, titer_signature, step, rng, rep))
    batch = data_v9.assemble_batch(rows, pool, M_pool, batch_kind="titration", batch_id=batch_id)
    batch["M_sampl_meta"]["noise_in_band_ratio"] = in_band_ratio(batch)
    return batch


def add_observed_cosine(batch):
    obs = []
    for sid in batch["X_count_data"].index:
        p = data_v9.normalize_vec(batch["X_count_data"].loc[sid].to_numpy(float))
        t = batch["X_profl_true"].loc[sid].to_numpy(float)
        obs.append(float(data_v9.cosine_vec_mat(p, t.reshape(1, -1))[0]))
    batch["M_sampl_meta"]["observed_cosine"] = obs
    return batch


def generate_benchmark_batch(bank, M_bank, cfg=None, mode="random", random_state=None, batch_id="benchmark_v7_3"):
    mode = normalize_mode(mode)
    rng = np.random.default_rng(random_state)
    cfg = MAKE_config_batch(mode=mode) if cfg is None else copy.deepcopy(cfg)
    if mode in {"no_ood", "random"}:
        batch = generate_single_mode_batch(bank, M_bank, cfg, mode, rng, batch_id)
    elif mode in {"cosmic_titration", "ood_titration"}:
        batch = generate_titration_batch(bank, M_bank, cfg, mode, rng, batch_id)
    else:
        raise ValueError(f"unknown benchmark mode: {mode}")
    add_observed_cosine(batch)
    batch["batch_info"].update({"benchmark_mode": mode, "ref_mask_policy": cfg.get("ref_mask_policy")})
    return batch


def ref_mask_from_benchmark(batch, full_reference=True):
    mask = pd.DataFrame(True, index=batch["X_profl_data"].index, columns=batch["R_refsigpool"].index)
    if not full_reference and "Y_compo_mask" in batch:
        mask = batch["Y_compo_mask"].astype(bool).copy()
    if "Y__OOD__mask" in batch:
        mask &= ~batch["Y__OOD__mask"].astype(bool)
    return mask


# ============================================================================
# Method execution and standardized outputs
# ============================================================================


def r_seconds(runner):
    path = getattr(runner, "output_dir", None)
    name = getattr(runner, "method_name", None)
    if path is None or name is None:
        return None
    for suffix in ["__runtime.tsv", "__python_wall_runtime.tsv"]:
        f = Path(path) / f"{name}{suffix}"
        if f.exists():
            df = pd.read_csv(f, sep="\t")
            col = "seconds" if "seconds" in df.columns else df.columns[-1]
            return float(df[col].iloc[0])
    return None


def standardize_method_result(out, sample_index=None, ref_index=None, context_index=None):
    """Return one explicit result dictionary: composition, reconstruction and OOD mass."""
    if isinstance(out, dict):
        comp = out.get("compo", out.get("composition"))
        recon, ood = out.get("recon"), out.get("ood")
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        comp, recon = out[:2]
        ood = out[2] if len(out) > 2 else None
    else:
        raise ValueError(f"Unsupported method result: {type(out)}")
    if comp is None or recon is None:
        raise ValueError("A method result must contain composition and reconstruction")
    comp, recon = comp.copy(), recon.copy()
    if sample_index is not None:
        comp = comp.reindex(index=sample_index, columns=ref_index, fill_value=0.0)
        recon = recon.reindex(index=sample_index, columns=context_index, fill_value=0.0)
    if ood is None:
        ood = pd.DataFrame(0.0, index=comp.index, columns=["OOD"])
    else:
        ood = ood.copy().reindex(index=comp.index, columns=["OOD"], fill_value=0.0)
    return {"compo": comp, "recon": recon, "ood": ood}


def RUN_method_timed(key, runner, batch, ref_mask, method_results=None, timing_rows=None, label=None,
                     unit_label=None, verbose=True):
    """Run one method and store a standardized, visibly named result dictionary."""
    label = label or METHOD_LABELS.get(key, key)
    n = int(batch["X_profl_data"].shape[0])
    prefix = f"{unit_label} | " if unit_label else ""
    log_msg(f"{prefix}{label} start: n_samples={n}", verbose)
    t0 = time.perf_counter()
    out = standardize_method_result(runner(batch["X_profl_data"], batch["R_refsigpool"], ref_mask),
                                    batch["X_profl_data"].index, batch["R_refsigpool"].index,
                                    batch["X_profl_data"].columns)
    wall = time.perf_counter() - t0
    rt = r_seconds(runner)
    sec = rt if rt is not None else wall
    note = "method internal runtime" if rt is not None else "Python wall runtime"
    row = {"method": key, "method_label": label, "wall_seconds": wall,
           "inference_seconds": sec, "sec_per_1000_samples": sec / max(n, 1) * 1000,
           "timer_note": note}
    log_msg(f"{prefix}{label} done: {sec:.2f}s, {row['sec_per_1000_samples']:.2f}s / 1000 samples", verbose)
    if method_results is not None:
        method_results[key] = out
    if timing_rows is not None:
        timing_rows.append(row)
    return out

def prefix_batch_sample_ids(batch, unit_label):
    out = copy.deepcopy(batch)
    old = list(out["X_profl_data"].index)
    new = [f"{unit_label}__{x}" for x in old]
    mapper = dict(zip(old, new))
    for key, val in out.items():
        if isinstance(val, pd.DataFrame) and list(val.index) == old:
            val.index = new
    out["M_sampl_meta"].index = new
    out["M_sampl_meta"].loc[:, "unit_label"] = unit_label
    out["batch_info"]["unit_label"] = unit_label
    return out, mapper


def rename_method_result_samples(method_results, mapper):
    renamed = {}
    for key, out in method_results.items():
        out = standardize_method_result(out)
        renamed[key] = {name: frame.rename(index=mapper) for name, frame in out.items()}
    return renamed


def combine_benchmark_units(unit_records):
    batches, method_pieces, timing = [], {}, []
    for rec in unit_records:
        batch, mapper = prefix_batch_sample_ids(rec["batch"], rec["unit_label"])
        batches.append(batch)
        for key, out in rename_method_result_samples(rec["method_results"], mapper).items():
            method_pieces.setdefault(key, []).append(out)
        if "timing" in rec and rec["timing"] is not None and not rec["timing"].empty:
            t = rec["timing"].copy()
            t["unit_label"] = rec["unit_label"]
            timing.append(t)
    ref_names = pd.Index([])
    for batch in batches:
        ref_names = ref_names.union(batch["R_refsigpool"].index)
    combined = {"R_refsigpool": pd.concat([b["R_refsigpool"] for b in batches]).loc[lambda x: ~x.index.duplicated()].reindex(ref_names),
                "M_refsigpool": pd.concat([b["M_refsigpool"] for b in batches]).loc[lambda x: ~x.index.duplicated()].reindex(ref_names),
                "M_context_meta": batches[0]["M_context_meta"]}
    for key in ["M_sampl_meta", "X_profl_true", "X_profl_noisy", "X_count_data", "X_profl_data"]:
        combined[key] = pd.concat([b[key] for b in batches])
    for key in ["Y_compo_true", "Y_count_true", "Y_active_mask", "Y_prior_mask", "Y__OOD__mask", "Y_compo_mask"]:
        combined[key] = pd.concat([b[key].reindex(columns=ref_names, fill_value=0) for b in batches])
    combined["batch_info"] = {"batch_id": "combined", "batch_kind": "combined",
        "n_samples": len(combined["X_profl_data"]), "n_ref": len(ref_names),
        "n_context": combined["X_profl_data"].shape[1], "n_units": len(batches),
        "benchmark_mode": batches[0]["batch_info"].get("benchmark_mode")}
    results = {}
    for key, pieces in method_pieces.items():
        results[key] = {"compo": pd.concat([x["compo"].reindex(columns=ref_names, fill_value=0.0) for x in pieces]),
                        "recon": pd.concat([x["recon"] for x in pieces]),
                        "ood": pd.concat([x["ood"] for x in pieces])}
    timing_df = pd.concat(timing, ignore_index=True) if timing else pd.DataFrame()
    return combined, results, timing_df

# ============================================================================
# Benchmark metrics and visualization
# ============================================================================


def metric_mse(a, b):
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def metric_cos(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    if d < 1e-12:
        return 1.0 if np.linalg.norm(a - b) < 1e-12 else 0.0
    return float(np.dot(a, b) / d)


def metric_r2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ss = float(np.sum((a - b) ** 2))
    tt = float(np.sum((a - np.mean(a)) ** 2))
    if tt < 1e-12:
        return 1.0 if ss < 1e-12 else 0.0
    return float(1.0 - ss / tt)


def active_metrics(y_true, y_pred, threshold=0.01):
    yt = np.asarray(y_true, dtype=float).ravel() > threshold
    yp = np.asarray(y_pred, dtype=float).ravel() > threshold
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    tn = float(np.sum(~yt & ~yp))
    sens = tp / max(tp + fn, 1e-12)
    spec = tn / max(tn + fp, 1e-12)
    prec = tp / max(tp + fp, 1e-12)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return sens, spec, f1


def compute_benchmark_metrics(batch, method_results):
    smp = batch["Y_compo_true"].index
    ref = batch["R_refsigpool"].index
    ctx = batch["R_refsigpool"].columns
    results = {key: standardize_method_result(out, smp, ref, ctx) for key, out in method_results.items()}
    compo = {key: out["compo"] for key, out in results.items()}
    recon = {key: out["recon"] for key, out in results.items()}
    ood = {key: out["ood"] for key, out in results.items()}
    true_all = batch["Y_compo_true"].reindex(index=smp, columns=ref, fill_value=0.0).astype(float)
    true_ood_mask = batch["Y__OOD__mask"].reindex(index=smp, columns=ref, fill_value=False).astype(bool)
    true_known = true_all.mask(true_ood_mask, 0.0)
    true_ood_mass = true_all.where(true_ood_mask, 0.0).sum(1)
    x_true = batch["X_profl_true"].astype(float)
    rows, residuals = [], {}
    R = batch["R_refsigpool"].to_numpy(float)
    for key in compo:
        c = compo[key].clip(lower=0).fillna(0.0)
        r = recon[key].clip(lower=0).fillna(0.0)
        oo = ood[key]["OOD"].clip(lower=0).fillna(0.0)
        pos_resid = (x_true - r).clip(lower=0.0)
        residuals[key] = pos_resid
        sens, spec, f1 = active_metrics(true_known.to_numpy(float), c.to_numpy(float))
        for sid in smp:
            yt = true_known.loc[sid].to_numpy(float)
            yp = c.loc[sid].to_numpy(float)
            rt = x_true.loc[sid].to_numpy(float)
            rp = r.loc[sid].to_numpy(float)
            pr = pos_resid.loc[sid].to_numpy(float)
            ood_vec = true_all.loc[sid].where(true_ood_mask.loc[sid], 0.0).to_numpy(float) @ R
            rows.append({"sample_id": sid, "method": key, "method_label": METHOD_LABELS.get(key, key),
                         "composition_mse": metric_mse(yt, yp), "composition_r2": metric_r2(yt, yp),
                         "composition_r2_clip0": max(0.0, metric_r2(yt, yp)), "composition_cos": metric_cos(yt, yp),
                         "recon_mse": metric_mse(rt, rp), "recon_r2": metric_r2(rt, rp),
                         "recon_r2_clip0": max(0.0, metric_r2(rt, rp)), "recon_cos": metric_cos(rt, rp),
                         "true_ood_mass": float(true_ood_mass.loc[sid]), "pred_ood_mass": float(oo.loc[sid]),
                         "positive_residual_mass": float(pr.sum()), "residual_ood_profile_mse": metric_mse(ood_vec, pr),
                         "residual_ood_profile_r2": metric_r2(ood_vec, pr),
                         "residual_ood_profile_cos": metric_cos(ood_vec, pr),
                         "active_sensitivity": sens, "active_specificity": spec, "active_F1": f1})
    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)
    return metrics, summary, compo, recon, ood, residuals


def summarize_metrics(metrics):
    if metrics.empty:
        return pd.DataFrame()
    cols = [c for c in metrics.columns if c not in {"sample_id", "method", "method_label"}]
    summary = metrics.groupby(["method", "method_label"], as_index=False)[cols].agg(["mean", "median", "std"])
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary["n_samples"] = metrics.groupby(["method", "method_label"]).size().to_numpy()
    return summary


def plot_profile_noise_qc(batch, span=0.005, ax=None, title="Profile-noise QC"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    for x in [0.85, 0.90, 0.95, 0.98, 1.00]:
        ax.axvline(x, ls="--", lw=1, color="#616161")
    bins = np.arange(0.75 - span / 2, 1 + span, span)
    meta = batch["M_sampl_meta"]
    ax.hist(meta["noise_cosine"].astype(float), bins=bins, alpha=0.55, label="expected perturbed profile")
    ax.hist(meta["observed_cosine"].astype(float), bins=bins, alpha=0.45, label="observed count profile")
    ok = meta["noise_status"].astype(str).eq("in_band").mean()
    ax.set(xlabel="cosine to clean profile", ylabel="samples", title=f"{title} | in-bin ratio={ok:.1%}")
    ax.legend(frameon=False)
    return fig, ax


def plot_compo_scatter_grid(true_compo, pred_compo, methods=None, title_prefix="", max_points=8000):
    methods = methods or [m for m in METHOD_ORDER if m in pred_compo]
    n = max(1, len(methods))
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, key in zip(axes, methods):
        SgF_util.plot_scatter_compo(true_compo, pred_compo[key].reindex_like(true_compo).fillna(0.0),
                                    max_points=max_points, title=f"{title_prefix}{METHOD_LABELS.get(key, key)}", fig=fig, ax=ax)
    for ax in axes[len(methods):]:
        ax.axis("off")
    return fig, axes


def plot_metric_hist_grid(metrics, metric_cols=None, methods=None, title="Per-sample metric distributions"):
    metric_cols = metric_cols or ["composition_mse", "composition_r2", "composition_cos", "recon_mse", "recon_r2", "recon_cos"]
    methods = methods or [m for m in METHOD_ORDER if m in set(metrics["method"])]
    fig, axes = plt.subplots(max(1, len(methods)), len(metric_cols), figsize=(3.0 * len(metric_cols), 2.0 * max(1, len(methods))),
                             sharex="col", sharey="col", constrained_layout=True)
    axes = np.atleast_2d(axes)
    bins_by = {}
    for col in metric_cols:
        vals = metrics[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if col.endswith("r2") or col.endswith("cos"):
            bins_by[col] = np.r_[-0.015, -0.005, np.arange(0.005, 1.015, 0.01)]
        else:
            hi = float(np.nanquantile(vals, 0.99)) if vals.size else 1.0
            hi = max(hi, float(np.nanmax(vals)) if vals.size else 1.0, 1e-12)
            bw = hi / 40.0
            bins_by[col] = np.arange(-bw / 2, hi + bw, bw)
    for i, method in enumerate(methods):
        dat = metrics.loc[metrics["method"].eq(method)]
        for j, col in enumerate(metric_cols):
            ax = axes[i, j]
            v = dat[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if col.endswith("r2") or col.endswith("cos"):
                ax.hist(np.where(v < 0, -0.01, np.clip(v, 0, 1)), bins=bins_by[col], alpha=0.85)
            else:
                ax.hist(v, bins=bins_by[col], alpha=0.85)
            if i == 0:
                ax.set_title(col.replace("_", "\n"), fontsize=9)
            if j == 0:
                ax.set_ylabel(METHOD_LABELS.get(method, method))
    fig.suptitle(title)
    return fig, axes


def plot_ood_scatter_grid(true_ood_mass, pred_ood, methods=None, title_prefix="OOD mass | "):
    methods = methods or [m for m in OOD_PRED_METHODS if m in pred_ood]
    fig, axes = plt.subplots(1, max(1, len(methods)), figsize=(4.6 * max(1, len(methods)), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        y = pred_ood[method].reindex(index=true_ood_mass.index, columns=["OOD"], fill_value=0.0)["OOD"]
        ax.scatter(true_ood_mass, y, s=16, alpha=0.70)
        ax.plot([-0.02, 1.02], [-0.02, 1.02], "k--", lw=1)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="true OOD mass", ylabel="predicted OOD mass",
               title=f"{title_prefix}{METHOD_LABELS.get(method, method)}")
    return fig, axes


def plot_metric_heatmap(summary_all, ax, x_col, y_col, method, metric="composition_r2_mean", x_order=None, y_order=None,
                        vlim=(0, 1), cmap="viridis", filters=None, annotate=False):
    df = summary_all.copy()
    df = df[df["method_label"].eq(method) | df["method"].eq(method)]
    if filters:
        for key, val in filters.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            df = df[df[key].isin(vals)]
    x_order = list(x_order or sorted(df[x_col].dropna().unique()))
    y_order = list(y_order or sorted(df[y_col].dropna().unique()))
    mat = df.pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean").reindex(index=y_order, columns=x_order)
    im = ax.imshow(mat.to_numpy(float), vmin=None if vlim is None else vlim[0], vmax=None if vlim is None else vlim[1], cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(x_order)), x_order, rotation=45, ha="right")
    ax.set_yticks(range(len(y_order)), y_order)
    ax.set_title(f"{METHOD_LABELS.get(method, method)}: {metric}")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        annotate_matrix(ax, mat)
    return mat


def plot_metric_diff_heatmap(summary_all, ax, x_col, y_col, method_1, method_2, metric="composition_r2_mean",
                             x_order=None, y_order=None, vlim=None, cmap="bwr", filters=None, annotate=False):
    df = summary_all.copy()
    if filters:
        for key, val in filters.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            df = df[df[key].isin(vals)]
    df = df[df["method_label"].isin([method_1, method_2]) | df["method"].isin([method_1, method_2])]
    x_order = list(x_order or sorted(df[x_col].dropna().unique()))
    y_order = list(y_order or sorted(df[y_col].dropna().unique()))
    a = df[df["method_label"].eq(method_1) | df["method"].eq(method_1)].pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean")
    b = df[df["method_label"].eq(method_2) | df["method"].eq(method_2)].pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean")
    mat = (a - b).reindex(index=y_order, columns=x_order)
    lim = np.nanmax(np.abs(mat.to_numpy(float))) if vlim is None and np.isfinite(mat.to_numpy(float)).any() else None
    low, high = ((-lim, lim) if vlim is None else vlim)
    im = ax.imshow(mat.to_numpy(float), vmin=low, vmax=high, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(x_order)), x_order, rotation=45, ha="right")
    ax.set_yticks(range(len(y_order)), y_order)
    ax.set_title(f"{METHOD_LABELS.get(method_1, method_1)} - {METHOD_LABELS.get(method_2, method_2)}: {metric}")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        annotate_matrix(ax, mat)
    return mat


def annotate_matrix(ax, mat):
    vals = mat.to_numpy(float)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)


def plot_grouped_method_boxplot(summary_all, ax, condition_col, metric="composition_r2_mean", condition_order=None,
                                method_order=None, filters=None, showfliers=False):
    df = summary_all.copy()
    if filters:
        for key, val in filters.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            df = df[df[key].isin(vals)]
    condition_order = list(condition_order or sorted(df[condition_col].dropna().unique()))
    method_order = list(method_order or [m for m in METHOD_ORDER if m in set(df["method"])])
    positions, data, labels = [], [], []
    pos = 0.0
    for cond in condition_order:
        for method in method_order:
            vals = df.loc[df[condition_col].eq(cond) & df["method"].eq(method), metric].dropna().to_numpy(float)
            if vals.size:
                data.append(vals)
                positions.append(pos)
                labels.append(METHOD_LABELS.get(method, method))
            pos += 1.0
        pos += 0.8
    ax.boxplot(data, positions=positions, widths=0.7, showfliers=showfliers)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {condition_col}")
    return {"positions": positions, "labels": labels, "data": data}


# ============================================================================
# Result persistence and batch-result loading
# ============================================================================


def save_benchmark_outputs(batch, method_results, condition_dir, timing=None, make_plots=True, prefix="benchmark"):
    condition_dir = ensure_dir(condition_dir)
    with open(condition_dir / f"{prefix}__batch.pkl", "wb") as f:
        pickle.dump(batch, f)
    for key, val in batch.items():
        if isinstance(val, pd.DataFrame):
            data_v9.dataframe_for_tsv(val).to_csv(condition_dir / f"{prefix}__{key}.tsv", sep="\t")
    metrics, summary, compo, recon, ood, residuals = compute_benchmark_metrics(batch, method_results)
    metrics.to_csv(condition_dir / "per_sample_metrics.tsv", sep="\t", index=False)
    summary.to_csv(condition_dir / "summary.tsv", sep="\t", index=False)
    if timing is not None:
        timing.to_csv(condition_dir / "method_runtime.tsv", sep="\t", index=False)
    for key in compo:
        compo[key].to_csv(condition_dir / f"{key}__pred_compo.tsv", sep="\t")
        recon[key].to_csv(condition_dir / f"{key}__pred_recon.tsv", sep="\t")
        ood[key].to_csv(condition_dir / f"{key}__pred_ood.tsv", sep="\t")
        residuals[key].to_csv(condition_dir / f"{key}__positive_residual.tsv", sep="\t")
    if make_plots:
        save_standard_plots(batch, metrics, compo, ood, condition_dir)
    return metrics, summary


def save_standard_plots(batch, metrics, compo, ood, condition_dir):
    figs = {}
    figs["profile_noise_qc"] = plot_profile_noise_qc(batch)[0]
    true_known = batch["Y_compo_true"].mask(batch["Y__OOD__mask"].astype(bool), 0.0)
    figs["composition_scatter_grid"] = plot_compo_scatter_grid(true_known, compo)[0]
    figs["metric_hist_grid"] = plot_metric_hist_grid(metrics)[0]
    true_ood = batch["Y_compo_true"].where(batch["Y__OOD__mask"].astype(bool), 0.0).sum(1)
    figs["ood_scatter_grid"] = plot_ood_scatter_grid(true_ood, ood)[0]
    for name, fig in figs.items():
        fig.savefig(Path(condition_dir) / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(Path(condition_dir) / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def _normalise_methods_spec(methods):
    """Return a clean list of method names from list-like, comma text, JSON text, or repr text."""
    if methods is None:
        return list(METHOD_ORDER)
    if isinstance(methods, float) and np.isnan(methods):
        return list(METHOD_ORDER)
    if isinstance(methods, np.ndarray):
        methods = methods.tolist()
    if isinstance(methods, str):
        text = methods.strip()
        if text in {"", "auto", "None", "nan"}:
            return list(METHOD_ORDER)
        parsed = None
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                import ast
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple, set, np.ndarray)):
            methods = parsed
        else:
            methods = [x for x in text.replace(";", ",").replace(" ", ",").split(",") if x]
    if isinstance(methods, (set, tuple)):
        methods = list(methods)
    if not isinstance(methods, list):
        try:
            methods = list(methods)
        except TypeError:
            methods = [methods]
    return [str(x) for x in methods]


def expected_files(condition_dir, methods):
    methods = _normalise_methods_spec(methods)
    base = ["benchmark__batch.pkl", "per_sample_metrics.tsv", "summary.tsv", "method_runtime.tsv"]
    method_files = []
    for method in methods:
        method_files.extend([f"{method}__pred_compo.tsv", f"{method}__pred_recon.tsv", f"{method}__pred_ood.tsv"])
    return [Path(condition_dir) / x for x in base + method_files]


def read_condition_outputs(condition_dir, methods=None):
    condition_dir = Path(condition_dir)
    methods = _normalise_methods_spec(methods or METHOD_ORDER)
    files = expected_files(condition_dir, methods)
    exists = {p.name: p.exists() for p in files}
    complete = all(exists.values())
    summary = pd.read_csv(condition_dir / "summary.tsv", sep="\t") if (condition_dir / "summary.tsv").exists() else pd.DataFrame()
    metrics = pd.read_csv(condition_dir / "per_sample_metrics.tsv", sep="\t") if (condition_dir / "per_sample_metrics.tsv").exists() else pd.DataFrame()
    timing = pd.read_csv(condition_dir / "method_runtime.tsv", sep="\t") if (condition_dir / "method_runtime.tsv").exists() else pd.DataFrame()
    status = pd.DataFrame([{"condition_dir": str(condition_dir), "complete": complete,
                            "n_expected_files": len(files), "n_done_files": int(sum(exists.values())),
                            "missing_files": json.dumps([k for k, v in exists.items() if not v])}])
    return status, summary, metrics, timing


def _config_scalar_for_column(value):
    """Make a job-table metadata value safe for assignment to a DataFrame column."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return value


def attach_config_columns(df, config_row):
    if df.empty:
        return df
    out = df.copy()
    for key, value in config_row.items():
        if key not in out.columns:
            out[key] = _config_scalar_for_column(value)
    return out


def read_submitted_results(job_table, methods=None):
    statuses, summaries, metrics, timings = [], [], [], []
    for _, row in job_table.iterrows():
        row_methods = methods if methods is not None else row.get("methods", METHOD_ORDER)
        status, summary, metric, timing = read_condition_outputs(row["condition_dir"], methods=row_methods)
        row_dict = row.to_dict()
        statuses.append(attach_config_columns(status, row_dict))
        summaries.append(attach_config_columns(summary, row_dict))
        metrics.append(attach_config_columns(metric, row_dict))
        timings.append(attach_config_columns(timing, row_dict))
    return pd.concat(statuses, ignore_index=True), pd.concat(summaries, ignore_index=True), pd.concat(metrics, ignore_index=True), pd.concat(timings, ignore_index=True)

# ============================================================================
# Batch-job bookkeeping and programmatic benchmark entry points
# ============================================================================


def make_job_table(root, modes, depth_bins, noise_bins, compo_bins, active_bins, cosmic_signatures=None, methods_by_mode=None):
    rows = []
    for mode in modes:
        mode = normalize_mode(mode)
        active_values = active_bins if mode in {"no_ood", "random"} else ["NA"]
        signatures = cosmic_signatures if mode == "cosmic_titration" and cosmic_signatures else [None]
        for depth in depth_bins:
            for noise in noise_bins:
                for compo in compo_bins:
                    for active in active_values:
                        for sig in signatures:
                            cdir = condition_path(root, mode, noise, depth, active, compo, signature=sig)
                            rows.append({"mode": mode, "depth": depth, "noise": noise, "compo": str(compo),
                                         "active": active, "signature": sig, "condition_dir": str(cdir),
                                         "methods": (methods_by_mode or {}).get(mode, METHOD_ORDER)})
    return pd.DataFrame(rows)

def batch_progress_summary(batch):
    meta = batch["M_sampl_meta"]
    n_sample = int(batch["X_profl_data"].shape[0])
    n_ref = int(batch["R_refsigpool"].shape[0])
    n_ood = int(batch["Y__OOD__mask"].astype(bool).any(axis=1).sum()) if "Y__OOD__mask" in batch else 0
    ratio = float(meta["noise_status"].astype(str).eq("in_band").mean()) if "noise_status" in meta else float("nan")
    sig = meta["titer_signature"].iloc[0] if "titer_signature" in meta and len(meta) else None
    target = f", titer_signature={sig}" if sig not in {None, "", "nan"} else ""
    return f"n_samples={n_sample}, n_ref_pool={n_ref}, n_ood_samples={n_ood}, noise_in_band={ratio:.3f}{target}"


def run_benchmark_unit(bank, M_bank, cfg, mode, unit_label, runners, method_keys=None, seed=0,
                       save_dir=None, make_plots=True, verbose=True):
    mode = normalize_mode(mode)
    method_keys = method_keys or list(runners.keys())
    log_msg(f"{unit_label} | generate batch start: mode={mode}, seed={seed}", verbose)
    t0 = time.perf_counter()
    batch = generate_benchmark_batch(bank, M_bank, cfg=cfg, mode=mode, random_state=seed, batch_id=f"{mode}_{unit_label}")
    log_msg(f"{unit_label} | generate batch done in {time.perf_counter() - t0:.2f}s: {batch_progress_summary(batch)}", verbose)
    ref_mask = ref_mask_from_benchmark(batch, full_reference=True)
    method_results, timing_rows = {}, []
    for j, key in enumerate(method_keys, 1):
        log_msg(f"{unit_label} | method {j}/{len(method_keys)} queued: {METHOD_LABELS.get(key, key)}", verbose)
        RUN_method_timed(key, runners[key], batch, ref_mask, method_results, timing_rows, unit_label=unit_label, verbose=verbose)
    timing = pd.DataFrame(timing_rows)
    record = {"unit_label": unit_label, "batch": batch, "method_results": method_results, "timing": timing}
    if save_dir is not None:
        save_dir = ensure_dir(save_dir)
        ref_mask.astype(int).to_csv(save_dir / "ref_mask.tsv", sep="\t")
        log_msg(f"{unit_label} | saving unit outputs: {save_dir}", verbose)
        save_benchmark_outputs(batch, method_results, save_dir, timing=timing, make_plots=make_plots)
    return record


def run_condition_batches(bank, M_bank, cfg, mode, runners, condition_dir, n_batches=2,
                          base_seed=19970717, method_keys=None, make_plots=True, verbose=True):
    records = []
    mode = normalize_mode(mode)
    n_batches = int(n_batches)
    log_msg(f"condition start: mode={mode}, n_batches={n_batches}, methods={list(method_keys or runners.keys())}", verbose)
    for i in range(n_batches):
        label = f"bch{i:03d}"
        seed = stable_seed(base_seed, mode, label, cfg.get("titer_signature"))
        log_msg(f"batch {i + 1}/{n_batches} start: {label}", verbose)
        records.append(run_benchmark_unit(bank, M_bank, cfg, mode, label, runners,
                                          method_keys=method_keys, seed=seed, save_dir=None,
                                          make_plots=False, verbose=verbose))
        log_msg(f"batch {i + 1}/{n_batches} done: {label}", verbose)
    log_msg(f"combining {len(records)} batch result(s)", verbose)
    combined_batch, combined_results, combined_timing = combine_benchmark_units(records)
    log_msg(f"saving combined condition outputs: {condition_dir}", verbose)
    save_benchmark_outputs(combined_batch, combined_results, condition_dir, timing=combined_timing, make_plots=make_plots)
    log_msg(f"condition done: {condition_dir}", verbose)
    return {"batch": combined_batch, "method_results": combined_results, "timing": combined_timing,
            "condition_dir": Path(condition_dir)}


def add_condition_columns(df, mode=None, depth_bin=None, noise_bin=None, active_bin=None, compo_bin=None, signature=None):
    if df.empty:
        return df
    out = df.copy()
    out["mode"],out["depth"],out["noise"],out["active"],out["compo"],out["signature"] = mode,depth_bin,noise_bin,active_bin,str(compo_bin),signature
    return out
