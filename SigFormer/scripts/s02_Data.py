"""

Synthetic SBS96 data generation for SigFormer training and benchmarking.

Core output keys
----------------
R_refsigpool    : rows = reference signatures, cols = SBS96 trinucleotide contexts
M_refsigpool    : rows = reference signatures, cols = reference metadata/features
M_context_meta  : rows = SBS96 contexts, cols = readable mutation/context fields
M_sampl_meta    : rows = samples, cols = sample-level generation metadata
Y_compo_true    : rows = samples, cols = references, true compositions
Y_count_true    : rows = samples, cols = references, expected component mutation counts
Y_active_mask   : rows = samples, cols = references, true active indicators
Y_prior_mask    : rows = samples, cols = references, signatures visible as prior knowledge
Y__OOD__mask    : rows = samples, cols = references, active signatures intentionally left out
Y_compo_mask    : rows = samples, cols = references, signatures allowed for inference/input
X_profl_true    : rows = samples, cols = SBS96, clean mixture profile before profile noise
X_profl_noisy   : rows = samples, cols = SBS96, Dirichlet-perturbed profile before count draw
X_count_data    : rows = samples, cols = SBS96, observed mutation counts
X_profl_data    : rows = samples, cols = SBS96, model input; counts or ratios per pcNOM

The module intentionally has no CLI section. Import it and call its functions.
"""

import os, json, math, pickle, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


# =========================================================
# 0. SBS96 signature coordinate system
# =========================================================

ntcomp = {'T': 'A', 'G': 'C', 'C': 'G', 'A': 'T', 'N': 'M'}
VEC_substit = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
VEC_context = [f"{x}-{y}" for x in "ACGT" for y in "ACGT"]
VEC_sub_ctx = [f"{sub},{ctx}" for sub in VEC_substit for ctx in VEC_context]
XTIC_3ntctx = [f"{ctx[4]}{ctx[0]}{ctx[6]}" for ctx in VEC_sub_ctx]

DEFAULT_CONFIG_SINGLE = {
    "BSize": 64,
    "REF_size": {"COSMIC":50, "MOCK":50},
    "ACTVE": {"1-3": 1, "4-6": 2, "7-10": 1, "11-20": 1},
    "PRIOR": {"4-6": 1, "7-10": 1, "11-20": 1, "21-40": 1, "41-60": 1, "61-120": 1, "130": 1},
    "pcOOD": 0.30,
    "ood_min_compo": 0.05,
    "NOISE": {"0.85-0.90": 140, "0.90-0.95": 240, "0.95-1.00": 1800},
    "DEPTH": {"100-400": 1, "401-7000": 1, "7000-100000": 1},
    "pcNOM": 0.30,
    "COMPO": {0.1: 1, 1.0: 2, 10.0: 1},
}


# =========================================================
# 1. Small numeric utilities
# =========================================================

def normalize_rows(arr: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows or columns to sum to 1."""
    arr = np.asarray(arr, dtype=float)
    s = arr.sum(axis=axis, keepdims=True)
    s = np.where(s < eps, eps, s)
    return arr / s

def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize one vector to sum to 1."""
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, None)
    s = float(v.sum())
    return v / max(s, eps)

def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Natural-log Shannon entropy of a probability vector."""
    p = normalize_vec(p, eps=eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def gini_coef(p: np.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient for a non-negative vector."""
    p = normalize_vec(p, eps=eps)
    ps = np.sort(p)
    n = ps.size
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * ps) / (n * np.sum(ps) + eps))


def cosine_vec_mat(v: np.ndarray, M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity between vector v and each row of M."""
    v = np.asarray(v, dtype=float)
    M = np.asarray(M, dtype=float)
    vn = max(float(np.linalg.norm(v)), eps)
    Mn = np.maximum(np.linalg.norm(M, axis=1), eps)
    return (M @ v) / (Mn * vn)


def cosine_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine similarity for row-vectors."""
    X = np.asarray(X, dtype=float)
    nrm = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    Xn = X / nrm
    return Xn @ Xn.T


def parse_range_key(x: Any, default_hi: Optional[int] = None) -> Tuple[int, int]:
    """Parse keys like '2-5', '130', 3, or (2, 5) into inclusive integer bounds."""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, (int, np.integer)):
        return int(x), int(x)
    sx = str(x).strip().lower()
    if sx in {"full", "all"}:
        hi = 999999 if default_hi is None else int(default_hi)
        return hi, hi
    if "-" in sx:
        lo, hi = sx.split("-", 1)
        return int(float(lo)), int(float(hi))
    return int(float(sx)), int(float(sx))


def weighted_choice(mapping: Dict[Any, float], rng: np.random.Generator) -> Any:
    """Sample one key from a weight dictionary. Zero and negative weights are ignored."""
    items = [(k, float(v)) for k, v in mapping.items() if float(v) > 0]
    if len(items) == 0:
        raise ValueError("weight mapping has no positive weights")
    keys, weights = zip(*items)
    probs = np.asarray(weights, dtype=float) / np.sum(weights)
    return keys[int(rng.choice(len(keys), p=probs))]


def sample_int_from_weighted_ranges(mapping: Dict[Any, float], rng: np.random.Generator, default_hi: Optional[int] = None) -> int:
    """Choose a weighted range key, then sample an integer uniformly inside it."""
    key = weighted_choice(mapping, rng)
    lo, hi = parse_range_key(key, default_hi=default_hi)
    return int(rng.integers(lo, hi + 1))


# =========================================================
# 2. COSMIC reader and reference metadata
# =========================================================

def get_COSMIC(genome: str = "GRCh38", version: str = "v3.4", path: Optional[str] = None) -> pd.DataFrame:
    """Read COSMIC SBS signatures and return rows=signatures, cols=VEC_sub_ctx."""
    
    if path is None:
        resource_dir = Path(__file__).resolve().parents[1] / "resource"
        path = resource_dir / f"COSMIC_{version}_SBS_{genome}.txt"
    df_refsig = pd.read_csv(Path(path).expanduser(), sep="\t", index_col=0).T
    df_refsig = df_refsig.loc[~df_refsig.index.isna(), :]
    df_refsig.columns = [f"{i[2]}>{i[4]},{i[0]}-{i[6]}" for i in df_refsig.columns]
    return df_refsig.loc[:, VEC_sub_ctx].copy()


def standardize_refsig(df_refsig: pd.DataFrame, columns: Sequence[str] = VEC_sub_ctx) -> pd.DataFrame:
    """Reorder SBS96 columns and row-normalize signatures."""
    missing = [c for c in columns if c not in df_refsig.columns]
    if len(missing) > 0:
        raise ValueError(f"df_refsig is missing {len(missing)} SBS96 columns; first missing={missing[:5]}")
    df = df_refsig.loc[:, list(columns)].copy().astype(float)
    df[df < 0] = 0.0
    df.loc[:, :] = normalize_rows(df.values, axis=1)
    return df


def build_context_meta() -> pd.DataFrame:
    """Readable metadata for the SBS96 coordinate system."""
    rows = []
    for sub_ctx, xtic in zip(VEC_sub_ctx, XTIC_3ntctx):
        sub, ctx = sub_ctx.split(",")
        left, right = ctx.split("-")
        rows.append({"substitution": sub, "context": ctx, "left": left, "right": right, "xtic_3ntctx": xtic})
    return pd.DataFrame(rows, index=VEC_sub_ctx)


def summarize_refsig(df_refsig: pd.DataFrame, sparsity_threshold: float = 1e-6) -> pd.DataFrame:
    """Compute entropy, gini, sparsity, peak count, and cosine-neighbor summary per signature."""
    arr = normalize_rows(df_refsig.values, axis=1)
    cos = cosine_matrix(arr)
    n = arr.shape[0]
    meta = pd.DataFrame(index=df_refsig.index)
    meta["entropy"] = [shannon_entropy(arr[i]) for i in range(n)]
    meta["gini"] = [gini_coef(arr[i]) for i in range(n)]
    meta["sparsity"] = np.mean(arr <= sparsity_threshold, axis=1)
    meta["n_zero"] = np.sum(arr == 0, axis=1)
    meta["n_gt_0p03"] = np.sum(arr > 0.03, axis=1)
    meta["max_prob"] = np.max(arr, axis=1)
    meta["mean_prob"] = np.mean(arr, axis=1)
    meta["max_cos_to_other"] = [float(np.max(np.delete(cos[i], i))) if n > 1 else 0.0 for i in range(n)]
    meta["mean_cos_to_other"] = [float(np.mean(np.delete(cos[i], i))) if n > 1 else 0.0 for i in range(n)]
    return meta


# =========================================================
# 3. Mock de novo signature generation
# =========================================================

def sample_cosmic_combo_bank(cosmic_arr: np.ndarray, n_combo: int = 2048, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Approximate the COSMIC convex-combination space by random mixtures."""
    rng = np.random.default_rng() if rng is None else rng
    cosmic_arr = normalize_rows(cosmic_arr, axis=1)
    bank = np.zeros((n_combo, cosmic_arr.shape[1]), dtype=float)
    for i in range(n_combo):
        k = int(rng.integers(2, min(8, cosmic_arr.shape[0]) + 1))
        idx = rng.choice(cosmic_arr.shape[0], size=k, replace=False)
        w = rng.dirichlet(np.full(k, rng.choice([0.1, 1.0, 10.0])))
        bank[i] = normalize_vec(w @ cosmic_arr[idx])
    return bank


def empirical_stat_bounds(df_refsig: pd.DataFrame, sparsity_threshold: float = 1e-6, q: Tuple[float, float] = (0.01, 0.99)) -> Dict[str, Tuple[float, float]]:
    """COSMIC empirical distribution bounds used to filter mock signatures."""
    meta = summarize_refsig(df_refsig, sparsity_threshold=sparsity_threshold)
    return {c: tuple(np.quantile(meta[c].values, q)) for c in ["entropy", "gini", "sparsity", "max_prob"]}


def make_mock_candidate(cosmic_arr: np.ndarray, bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator,
                        min_zero: int = 5, min_peak: int = 5) -> np.ndarray:
    """Generate one sparse-but-COSMIC-like candidate on the SBS96 simplex."""
    n_ctx = cosmic_arr.shape[1]
    template = cosmic_arr[int(rng.integers(0, cosmic_arr.shape[0]))]
    mode = rng.choice(["shuffle_template", "spike_slab", "hybrid"], p=[0.35, 0.40, 0.25])

    if mode == "shuffle_template":
        base = template[rng.permutation(n_ctx)] + rng.uniform(1e-5, 1e-3, size=n_ctx)
        alpha = normalize_vec(base) * float(rng.uniform(60, 450))
        cand = rng.dirichlet(np.maximum(alpha, 1e-7))
    elif mode == "hybrid":
        donor = cosmic_arr[int(rng.integers(0, cosmic_arr.shape[0]))]
        base = normalize_vec(0.45 * template[rng.permutation(n_ctx)] + 0.55 * donor[rng.permutation(n_ctx)])
        cand = rng.dirichlet(np.maximum(base * float(rng.uniform(40, 300)), 1e-7))
    else:
        cand = rng.dirichlet(np.full(n_ctx, float(rng.uniform(0.04, 0.40))))

    n_zero = int(rng.integers(min_zero, max(min_zero + 1, min(42, n_ctx // 2)) + 1))
    zero_idx = rng.choice(n_ctx, size=n_zero, replace=False)
    cand[zero_idx] = 0.0
    cand = normalize_vec(cand)

    if np.sum(cand > 0.03) < min_peak:
        nonzero = np.where(cand > 0)[0]
        peak_idx = rng.choice(nonzero, size=min(min_peak, len(nonzero)), replace=False)
        add_mass = min(float(rng.uniform(0.20, 0.45)), 0.03 * len(peak_idx) + 0.25)
        cand *= 1.0 - add_mass
        cand[peak_idx] += add_mass / len(peak_idx)
        cand[zero_idx] = 0.0
        cand = normalize_vec(cand)

    return cand


def mock_candidate_passes(cand: np.ndarray, ref_arr: np.ndarray, mock_arr: Optional[np.ndarray], combo_bank: Optional[np.ndarray],
                          bounds: Dict[str, Tuple[float, float]], cosine_max: float = 0.8,
                          combo_cosine_max: float = 0.88) -> Tuple[bool, Dict[str, float]]:
    """Check mock constraints and return diagnostics."""
    cand = normalize_vec(cand)
    stats = {
        "entropy": shannon_entropy(cand), "gini": gini_coef(cand), "sparsity": float(np.mean(cand <= 1e-6)),
        "max_prob": float(np.max(cand)), "n_zero": int(np.sum(cand == 0)), "n_gt_0p03": int(np.sum(cand > 0.03)),
        "max_cos_to_cosmic": float(np.max(cosine_vec_mat(cand, ref_arr))), "max_cos_to_mock": 0.0,
        "max_cos_to_combo": 0.0,
    }
    if stats["n_zero"] < 5 or stats["n_gt_0p03"] < 5:
        return False, stats
    for key in ["entropy", "gini", "sparsity", "max_prob"]:
        lo, hi = bounds[key]
        pad = 0.10 * max(hi - lo, 1e-8)
        if not (lo - pad <= stats[key] <= hi + pad):
            return False, stats
    if stats["max_cos_to_cosmic"] >= cosine_max:
        return False, stats
    if mock_arr is not None and mock_arr.shape[0] > 0:
        stats["max_cos_to_mock"] = float(np.max(cosine_vec_mat(cand, mock_arr)))
        if stats["max_cos_to_mock"] >= cosine_max:
            return False, stats
    if combo_bank is not None and combo_bank.shape[0] > 0:
        stats["max_cos_to_combo"] = float(np.max(cosine_vec_mat(cand, combo_bank)))
        if stats["max_cos_to_combo"] >= combo_cosine_max:
            return False, stats
    return True, stats


def build_mock_denovo_signatures(df_COSMIC: pd.DataFrame, n_mock: int = 1000, cosine_max: float = 0.8,
                                 combo_cosine_max: float = 0.88, max_trials: int = 250000,
                                 combo_bank_size: int = 2048, random_state: Optional[int] = 2026,
                                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build MOK000.. mock de novo signatures with COSMIC-like statistics but expanded space."""
    rng = np.random.default_rng(random_state)
    df_COSMIC = standardize_refsig(df_COSMIC)
    cosmic_arr = df_COSMIC.values
    bounds = empirical_stat_bounds(df_COSMIC, sparsity_threshold=1e-6, q=(0.01, 0.99))
    bounds["sparsity"] = (min(bounds["sparsity"][0], 5 / len(VEC_sub_ctx)), max(bounds["sparsity"][1], 0.45))
    combo_bank = sample_cosmic_combo_bank(cosmic_arr, n_combo=combo_bank_size, rng=rng)

    mocks, records, trials = [], [], 0
    while len(mocks) < n_mock and trials < max_trials:
        trials += 1
        cand = make_mock_candidate(cosmic_arr, bounds, rng)
        mock_arr = np.vstack(mocks) if len(mocks) > 0 else None
        ok, stats = mock_candidate_passes(cand, cosmic_arr, mock_arr, combo_bank, bounds, cosine_max, combo_cosine_max)
        if not ok:
            continue
        name = f"MOK{len(mocks)+1:03d}"
        mocks.append(cand)
        records.append({"signature": name, **stats})
        if verbose and len(mocks) % 100 == 0:
            print(f"[mock] accepted {len(mocks)}/{n_mock} after {trials} trials")

    if len(mocks) < n_mock:
        raise RuntimeError(f"Only generated {len(mocks)}/{n_mock} mocks after {trials} trials. Relax combo_cosine_max or max_trials.")
    df_mock = pd.DataFrame(np.vstack(mocks), index=[f"MOK{i+1:03d}" for i in range(n_mock)], columns=VEC_sub_ctx)
    M_mock = pd.DataFrame(records).set_index("signature")
    M_mock["ref_type"] = "mock_denovo"
    M_mock["is_mock"] = True
    M_mock["is_cosmic"] = False
    return df_mock, M_mock


def build_grand_ref_pool(df_COSMIC: pd.DataFrame, n_mock: int = 1000, random_state: Optional[int] = 2026,
                         verbose: bool = True, **mock_kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine standardized COSMIC and mock de novo signatures into one grand reference pool."""
    df_COSMIC = standardize_refsig(df_COSMIC)
    df_mock, M_mock = build_mock_denovo_signatures(df_COSMIC, n_mock=n_mock, random_state=random_state, verbose=verbose, **mock_kwargs)
    R = pd.concat([df_COSMIC, df_mock], axis=0)
    M_cos = summarize_refsig(df_COSMIC)
    M_cos["ref_type"], M_cos["is_cosmic"], M_cos["is_mock"] = "COSMIC", True, False
    M = pd.concat([M_cos, M_mock], axis=0).loc[R.index]
    M["ref_index"] = np.arange(R.shape[0])
    return R, M


# =========================================================
# 4. Batch reference set and per-sample primitives
# =========================================================

def downsample_batch_refpool(R_grand: pd.DataFrame, M_grand: Optional[pd.DataFrame] = None, n_cosmic: int = 65,
                             n_mock: int = 65, rng: Optional[np.random.Generator] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Down-sample 65 COSMIC + 65 mock signatures for one batch, or fewer if unavailable."""
    rng = np.random.default_rng() if rng is None else rng
    if M_grand is None:
        M_grand = pd.DataFrame(index=R_grand.index)
        M_grand["ref_type"] = ["mock_denovo" if str(x).startswith("MOK") else "COSMIC" for x in R_grand.index]
        M_grand["is_cosmic"] = M_grand["ref_type"].eq("COSMIC")
        M_grand["is_mock"] = ~M_grand["is_cosmic"]
    if "is_cosmic" not in M_grand.columns:
        M_grand["is_cosmic"] = [not str(x).startswith("MOK") for x in M_grand.index]
    if "is_mock" not in M_grand.columns:
        M_grand["is_mock"] = [str(x).startswith("MOK") for x in M_grand.index]
    cosmic_names = M_grand.index[M_grand["is_cosmic"].astype(bool)].to_numpy()
    mock_names = M_grand.index[M_grand["is_mock"].astype(bool)].to_numpy()
    take_cos = rng.choice(cosmic_names, size=min(n_cosmic, len(cosmic_names)), replace=False) if len(cosmic_names) else []
    take_mok = rng.choice(mock_names, size=min(n_mock, len(mock_names)), replace=False) if len(mock_names) else []
    names = list(take_cos) + list(take_mok)
    rng.shuffle(names)
    R = standardize_refsig(R_grand.loc[names])
    M = M_grand.reindex(names).copy()
    M["batch_ref_index"] = np.arange(len(names))
    return R, M


def sample_depth(config: Dict[str, Any], n_active: int, rng: np.random.Generator) -> Tuple[int, str]:
    """Sample depth from configured bands, with lower bound >= n_active * 30."""
    key = weighted_choice(config["DEPTH"], rng)
    lo, hi = parse_range_key(key)
    lo = max(lo, int(n_active) * 30)
    if lo > hi:
        hi = lo
    if hi / max(lo, 1) > 20:
        depth = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    else:
        depth = int(rng.integers(lo, hi + 1))
    return depth, str(key)


def sample_active_indices(n_ref: int, n_active: int, rng: np.random.Generator, exclude: Optional[Iterable[int]] = None) -> np.ndarray:
    """Sample active reference indices without replacement."""
    excluded = set([] if exclude is None else [int(x) for x in exclude])
    pool = np.array([i for i in range(n_ref) if i not in excluded], dtype=int)
    if n_active > len(pool):
        n_active = len(pool)
    return rng.choice(pool, size=n_active, replace=False)


def sample_composition(n_active: int, alpha: float, depth: int, rng: np.random.Generator, min_frac: float = 0.02,
                       min_count: int = 10, force_pos: Optional[int] = None, force_min: float = 0.10,
                       max_trials: int = 5) -> Tuple[np.ndarray, bool]:
    """Sample composition; if impossible after trials, prune one active reference upstream."""
    cutoff = max(float(min_frac), float(min_count) / max(int(depth), 1))
    if cutoff * n_active > 0.95:
        cutoff = 0.95 / max(n_active, 1)
    avec = np.full(n_active, float(alpha), dtype=float)
    for _ in range(max_trials):
        comp = rng.dirichlet(avec)
        ok = bool(np.all(comp >= cutoff))
        if force_pos is not None:
            ok = ok and bool(comp[int(force_pos)] >= force_min)
        if ok:
            return comp, True
    return rng.dirichlet(avec), False


def sample_composition_with_pruning(active_idx: Sequence[int], alpha: float, depth: int, rng: np.random.Generator,
                                    force_ref: Optional[int] = None, force_min: float = 0.10) -> Tuple[np.ndarray, np.ndarray, int]:
    """Try composition constraints; prune random non-forced active signatures after repeated failures."""
    active = list(map(int, active_idx))
    pruned = 0
    while len(active) > 0:
        force_pos = active.index(int(force_ref)) if force_ref is not None and int(force_ref) in active else None
        comp, ok = sample_composition(len(active), alpha, depth, rng, force_pos=force_pos, force_min=force_min, max_trials=5)
        if ok or len(active) == 1:
            comp = np.maximum(comp, 0.0)
            if force_pos is not None and comp[force_pos] < force_min and len(comp) > 1:
                deficit = force_min - comp[force_pos]
                donor = np.array([i for i in range(len(comp)) if i != force_pos])
                comp[donor] *= max(0.0, 1.0 - force_min) / max(float(comp[donor].sum()), 1e-12)
                comp[force_pos] = force_min
            return np.asarray(active, dtype=int), normalize_vec(comp), pruned
        candidates = [x for x in active if force_ref is None or x != int(force_ref)]
        if len(candidates) == 0:
            return np.asarray(active, dtype=int), normalize_vec(comp), pruned
        active.remove(int(rng.choice(candidates)))
        pruned += 1
    raise RuntimeError("all active references were pruned; check depth/active/composition config")


def profile_from_composition(R_refsigpool: pd.DataFrame, active_idx: Sequence[int], comp: np.ndarray) -> np.ndarray:
    """Clean SBS96 mixture profile from active signatures and composition."""
    sig = R_refsigpool.values[np.asarray(active_idx, dtype=int)]
    sig = normalize_rows(sig, axis=1)
    return normalize_vec(np.asarray(comp, dtype=float) @ sig)


def perturb_profile(profile: np.ndarray, config: Dict[str, Any], rng: np.random.Generator,
                    max_trials: int = 500) -> Tuple[np.ndarray, str, float, str, float]:
    """Dirichlet perturb clean profile; enforce target cosine band if possible."""
    band_key = weighted_choice(config["NOISE"], rng)
    conc = float(config["NOISE"][band_key])
    lo, hi = parse_range_key(str(band_key).replace("0.", "")) if False else tuple(map(float, str(band_key).split("-")))
    profile = normalize_vec(profile)
    best_high, best_any = None, None
    for _ in range(max_trials):
        p = rng.dirichlet(np.maximum(profile * conc, 1e-8))
        cos = float(cosine_vec_mat(p, profile.reshape(1, -1))[0])
        if lo <= cos <= hi:
            return p, str(band_key), conc, "in_band", cos
        if cos > hi and best_high is None:
            best_high = (p, cos)
        if best_any is None or abs(cos - 0.5 * (lo + hi)) < abs(best_any[1] - 0.5 * (lo + hi)):
            best_any = (p, cos)
    if best_high is not None:
        return best_high[0], str(band_key), conc, "upsampled_high_cos", best_high[1]
    return best_any[0], str(band_key), conc, "closest_after_trials", best_any[1]


def build_prior_mask(active_idx: Sequence[int], ood_idx: Sequence[int], n_ref: int, target_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Prior mask = active non-OOD plus random inactive signatures, upsampled to target_size."""
    mask = np.zeros(n_ref, dtype=bool)
    active = set(map(int, active_idx))
    ood = set(map(int, ood_idx))
    visible_active = np.array(sorted(active - ood), dtype=int)
    mask[visible_active] = True
    target = int(min(max(target_size, len(visible_active)), n_ref))
    blocked = active | ood
    fillers = np.array([i for i in range(n_ref) if i not in blocked], dtype=int)
    n_fill = max(0, target - int(mask.sum()))
    if n_fill > 0 and len(fillers) > 0:
        mask[rng.choice(fillers, size=min(n_fill, len(fillers)), replace=False)] = True
    return mask


def choose_sample_ood(active_idx: Sequence[int], comp: np.ndarray, pcOOD: float, rng: np.random.Generator) -> np.ndarray:
    """With pcOOD, choose one active component below 50 percent as leave-out; singleton may be chosen."""
    if rng.random() >= float(pcOOD) or len(active_idx) == 0:
        return np.array([], dtype=int)
    active_idx = np.asarray(active_idx, dtype=int)
    comp = np.asarray(comp, dtype=float)
    cand = np.where(comp < 0.50)[0]
    if len(cand) == 0:
        cand = np.arange(len(active_idx))
    return np.array([int(active_idx[int(rng.choice(cand))])], dtype=int)


def finalize_sample_row(sample_id: str, R: pd.DataFrame, config: Dict[str, Any], rng: np.random.Generator,
                        active_idx: Sequence[int], cohort_meta: Optional[Dict[str, Any]] = None,
                        force_ood_idx: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Generate all arrays and metadata for one sample."""
    n_ref = R.shape[0]
    depth0, depth_band = sample_depth(config, len(active_idx), rng)
    comp_alpha = float(weighted_choice(config["COMPO"], rng))
    force_ref = None if not force_ood_idx else int(list(force_ood_idx)[0])
    active_idx, comp, n_pruned = sample_composition_with_pruning(active_idx, comp_alpha, depth0, rng, force_ref=force_ref,
                                                                 force_min=float(config.get("ood_min_compo", 0.05)))
    depth, depth_band = sample_depth(config, len(active_idx), rng)
    profile_true = profile_from_composition(R, active_idx, comp)
    profile_noisy, noise_band, noise_alpha, noise_status, noise_cosine = perturb_profile(profile_true, config, rng)
    counts = rng.multinomial(depth, profile_noisy)
    normalized_input = bool(rng.random() < float(config.get("pcNOM", 0.0)))
    x_data = counts / max(int(counts.sum()), 1) if normalized_input else counts.astype(float)

    if force_ood_idx is None:
        ood_idx = choose_sample_ood(active_idx, comp, float(config.get("pcOOD", 0.0)), rng)
    else:
        ood_idx = np.array([int(x) for x in force_ood_idx if int(x) in set(active_idx)], dtype=int)
    target_prior = sample_int_from_weighted_ranges(config["PRIOR"], rng, default_hi=n_ref)
    target_prior = min(target_prior, n_ref)
    prior_mask = build_prior_mask(active_idx, ood_idx, n_ref, target_prior, rng)

    y_compo = np.zeros(n_ref, dtype=float)
    y_compo[np.asarray(active_idx, dtype=int)] = comp
    y_active = y_compo > 0
    y_ood = np.zeros(n_ref, dtype=bool)
    y_ood[ood_idx] = True
    y_prior = prior_mask.astype(bool)
    y_mask = (y_prior & ~y_ood).astype(bool)
    y_count = y_compo * depth

    meta = {
        "sample_id": sample_id, "scheme": "single", "n_active": int(len(active_idx)), "depth": int(depth),
        "depth_band": depth_band, "comp_alpha": comp_alpha, "n_pruned_active": int(n_pruned),
        "noise_band": noise_band, "noise_alpha": noise_alpha, "noise_status": noise_status, "noise_cosine": noise_cosine,
        "normalized_input": normalized_input, "prior_target_size": int(target_prior), "prior_actual_size": int(y_prior.sum()),
        "n_ood": int(y_ood.sum()), "active_refs": json.dumps(R.index[active_idx].tolist()),
        "ood_refs": json.dumps(R.index[ood_idx].tolist()), "profile_entropy": shannon_entropy(profile_true),
        "profile_gini": gini_coef(profile_true), "input_total": float(x_data.sum()),
    }
    if cohort_meta is not None:
        meta.update(cohort_meta)
    return {"meta": meta, "y_compo": y_compo, "y_count": y_count, "y_active": y_active, "y_prior": y_prior,
            "y_ood": y_ood, "y_mask": y_mask, "x_true": profile_true, "x_noisy": profile_noisy,
            "x_count": counts, "x_data": x_data}


def assemble_batch(rows: List[Dict[str, Any]], R: pd.DataFrame, M_ref: pd.DataFrame, batch_kind: str,
                   batch_id: str = "batch") -> Dict[str, Any]:
    """Convert row dictionaries into the standard batch data structure."""
    sample_ids = [r["meta"]["sample_id"] for r in rows]
    refs, ctx = R.index.tolist(), R.columns.tolist()
    meta = pd.DataFrame([r["meta"] for r in rows]).set_index("sample_id")
    meta["batch_kind"], meta["batch_id"] = batch_kind, batch_id
    out = {
        "R_refsigpool": R.copy(), "M_refsigpool": M_ref.copy(), "M_context_meta": build_context_meta(), "M_sampl_meta": meta,
        "Y_compo_true": pd.DataFrame(np.vstack([r["y_compo"] for r in rows]), index=sample_ids, columns=refs),
        "Y_count_true": pd.DataFrame(np.vstack([r["y_count"] for r in rows]), index=sample_ids, columns=refs),
        "Y_active_mask": pd.DataFrame(np.vstack([r["y_active"] for r in rows]).astype(int), index=sample_ids, columns=refs),
        "Y_prior_mask": pd.DataFrame(np.vstack([r["y_prior"] for r in rows]).astype(int), index=sample_ids, columns=refs),
        "Y__OOD__mask": pd.DataFrame(np.vstack([r["y_ood"] for r in rows]).astype(int), index=sample_ids, columns=refs),
        "Y_compo_mask": pd.DataFrame(np.vstack([r["y_mask"] for r in rows]).astype(int), index=sample_ids, columns=refs),
        "X_profl_true": pd.DataFrame(np.vstack([r["x_true"] for r in rows]), index=sample_ids, columns=ctx),
        "X_profl_noisy": pd.DataFrame(np.vstack([r["x_noisy"] for r in rows]), index=sample_ids, columns=ctx),
        "X_count_data": pd.DataFrame(np.vstack([r["x_count"] for r in rows]), index=sample_ids, columns=ctx),
        "X_profl_data": pd.DataFrame(np.vstack([r["x_data"] for r in rows]), index=sample_ids, columns=ctx),
    }
    out["batch_info"] = {"batch_id": batch_id, "batch_kind": batch_kind, "n_samples": len(rows), "n_ref": R.shape[0], "n_context": R.shape[1]}
    return out


# =========================================================
# 5. Single-sample batch generation
# =========================================================

def generate_single_batch(R_grand: pd.DataFrame, M_grand: Optional[pd.DataFrame] = None, config: Optional[Dict[str, Any]] = None,
                          random_state: Optional[int] = None, batch_id: str = "single_batch") -> Dict[str, Any]:
    """Generate a batch of independent single samples."""
    rng = np.random.default_rng(random_state)
    cfg = {**DEFAULT_CONFIG_SINGLE, **({} if config is None else config)}
    R, M = downsample_batch_refpool(R_grand, M_grand, n_cosmic=cfg["REF_size"]["COSMIC"], n_mock=cfg["REF_size"]["MOCK"], rng=rng)
    rows = []
    for i in range(int(cfg["BSize"])):
        n_active = sample_int_from_weighted_ranges(cfg["ACTVE"], rng, default_hi=R.shape[0])
        n_active = min(n_active, R.shape[0])
        active_idx = sample_active_indices(R.shape[0], n_active, rng)
        row = finalize_sample_row(f"S{i:06d}", R, cfg, rng, active_idx)
        rows.append(row)
    return assemble_batch(rows, R, M, batch_kind="single", batch_id=batch_id)


# =========================================================
# 6. Cohort batch generation
# =========================================================

def dataframe_for_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame TSV-safe without destroying the in-memory object."""
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda x: isinstance(x, (list, tuple, dict, set))).any():
            out[c] = out[c].map(lambda x: json.dumps(list(x)) if isinstance(x, set) else json.dumps(x))
    return out


def export_batch(batch: Dict[str, Any], out_dir: str, prefix: str = "batch", formats: Sequence[str] = ("pkl", "tsv"),
                 sigprofiler: bool = True) -> Dict[str, str]:
    """Export a simulated batch to pkl/tsv and optionally SigProfiler-like count matrix."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = {}
    if "pkl" in formats:
        p = out_path / f"{prefix}.pkl"
        with open(p, "wb") as f:
            pickle.dump(batch, f)
        written["pkl"] = str(p)
    if "tsv" in formats:
        for key, val in batch.items():
            if isinstance(val, pd.DataFrame):
                p = out_path / f"{prefix}__{key}.tsv"
                dataframe_for_tsv(val).to_csv(p, sep="\t")
                written[f"tsv:{key}"] = str(p)
    if sigprofiler and "X_count_data" in batch:
        mat = batch["X_count_data"].T.copy()
        mat.index = XTIC_3ntctx
        p = out_path / f"{prefix}__sigprofiler_SBS96_matrix.tsv"
        mat.to_csv(p, sep="\t")
        written["sigprofiler"] = str(p)
    return written
