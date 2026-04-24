import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Core math helpers
# =========================================================


def _safe_normalize_rows(arr: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    denom = arr.sum(axis=axis, keepdims=True)
    denom = np.where(denom < eps, eps, denom)
    return arr / denom


def _shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _gini(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + eps)
    p_sorted = np.sort(p)
    n = p_sorted.size
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * p_sorted) / (n * np.sum(p_sorted) + eps))


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm <= 0, 1e-12, norm)
    X_norm = X / norm
    return X_norm @ X_norm.T


def _cosine_similarity_vec_mat(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    M = np.asarray(M, dtype=float)
    v_norm = np.linalg.norm(v)
    m_norm = np.linalg.norm(M, axis=1)
    v_norm = 1e-12 if v_norm <= 0 else v_norm
    m_norm = np.where(m_norm <= 0, 1e-12, m_norm)
    return (M @ v) / (m_norm * v_norm)


# =========================================================
# COSMIC summary and de novo bank
# =========================================================


def summarize_cosmic_signatures(
    df_COSMIC: pd.DataFrame,
    sparsity_threshold: float = 1e-4,
) -> pd.DataFrame:
    arr = _safe_normalize_rows(df_COSMIC.values, axis=1)
    n_sig, _ = arr.shape

    entropy_list = []
    sparsity_list = []
    gini_list = []
    for i in range(n_sig):
        vec = arr[i]
        entropy_list.append(_shannon_entropy(vec))
        sparsity_list.append(float(np.mean(vec < sparsity_threshold)))
        gini_list.append(_gini(vec))

    cosine_mat = _cosine_similarity_matrix(arr)
    mean_other = []
    max_other = []
    for i in range(n_sig):
        others = np.delete(cosine_mat[i], i)
        mean_other.append(float(np.mean(others)))
        max_other.append(float(np.max(others)))

    return pd.DataFrame(
        {
            "signature": df_COSMIC.index.to_list(),
            "entropy": entropy_list,
            f"sparsity(<{sparsity_threshold:.0e})": sparsity_list,
            "gini": gini_list,
            "mean_cosine_to_others": mean_other,
            "max_cosine_to_others": max_other,
        }
    ).set_index("signature")


def build_denovo_signatures(
    df_COSMIC: pd.DataFrame,
    n_denovo: int = 500,
    target_cosine_max: float = 0.8,
    alpha_scale_range: Tuple[float, float] = (150.0, 600.0),
    max_total_trials: int = 50000,
    random_state: Optional[int] = 2025,
) -> pd.DataFrame:
    """
    Build a de novo bank with both COSMIC-near and COSMIC-far signatures.
    The logic is intentionally close to your V5 version.
    """
    rng = np.random.default_rng(random_state)
    cosmic_arr = _safe_normalize_rows(df_COSMIC.values, axis=1)
    n_cosmic, n_ctx = cosmic_arr.shape

    cosmic_stat = summarize_cosmic_signatures(df_COSMIC, sparsity_threshold=1e-4)
    sparsity_col = [c for c in cosmic_stat.columns if c.startswith("sparsity(")][0]
    cosmic_entropy = cosmic_stat["entropy"].values
    cosmic_sparsity = cosmic_stat[sparsity_col].values
    ent_lo, ent_hi = np.quantile(cosmic_entropy, [0.05, 0.95])
    sps_lo, sps_hi = np.quantile(cosmic_sparsity, [0.05, 0.95])

    _, _, Vt = np.linalg.svd(cosmic_arr, full_matrices=False)
    r_subspace = min(30, Vt.shape[0], Vt.shape[1])
    subspace_basis = Vt[:r_subspace].T

    def _subspace_R2(vec: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
        vec = np.asarray(vec, dtype=float)
        vec_norm2 = float(np.sum(vec * vec)) + eps
        proj_coef = basis.T @ vec
        proj_norm2 = float(np.sum(proj_coef * proj_coef))
        return proj_norm2 / vec_norm2

    frac_far = 0.3
    target_n_far = int(round(n_denovo * frac_far))
    target_n_near = n_denovo - target_n_far
    far_cosine_max_to_cosmic = min(0.6, target_cosine_max - 0.05)
    far_subspace_R2_max = 0.85

    denovo_list = []
    denovo_names = []
    n_near = 0
    n_far = 0
    trials = 0

    while len(denovo_list) < n_denovo and trials < max_total_trials:
        trials += 1

        if n_far < target_n_far and n_near < target_n_near:
            mode = "far" if rng.random() < frac_far else "near"
        elif n_far < target_n_far:
            mode = "far"
        else:
            mode = "near"

        if mode == "near":
            idx = rng.integers(0, n_cosmic)
            template = cosmic_arr[idx]
            scale = rng.uniform(alpha_scale_range[0], alpha_scale_range[1])
            alpha = np.where(template * scale <= 0, 1e-6, template * scale)
            cand = rng.dirichlet(alpha)

            if np.max(_cosine_similarity_vec_mat(cand, cosmic_arr)) >= target_cosine_max:
                continue
            if denovo_list:
                denovo_arr = np.vstack(denovo_list)
                if np.max(_cosine_similarity_vec_mat(cand, denovo_arr)) >= target_cosine_max:
                    continue

            denovo_list.append(cand)
            denovo_names.append(f"DeNovo_near_{len(denovo_list):04d}")
            n_near += 1
            continue

        alpha0 = rng.uniform(0.2, 40.0)
        alpha_vec = np.full(n_ctx, alpha0 / n_ctx, dtype=float)
        cand = rng.dirichlet(alpha_vec)
        cand_entropy = _shannon_entropy(cand)
        cand_sparsity = float(np.mean(cand < 1e-4))

        if not (ent_lo <= cand_entropy <= ent_hi):
            continue
        if not (sps_lo <= cand_sparsity <= sps_hi):
            continue
        if np.max(_cosine_similarity_vec_mat(cand, cosmic_arr)) >= far_cosine_max_to_cosmic:
            continue
        if denovo_list:
            denovo_arr = np.vstack(denovo_list)
            if np.max(_cosine_similarity_vec_mat(cand, denovo_arr)) >= target_cosine_max:
                continue
        if _subspace_R2(cand, subspace_basis) >= far_subspace_R2_max:
            continue

        denovo_list.append(cand)
        denovo_names.append(f"DeNovo_far_{len(denovo_list):04d}")
        n_far += 1

    if not denovo_list:
        raise RuntimeError("Failed to generate any de novo signatures.")

    denovo_arr = np.vstack(denovo_list)
    print(
        f"[INFO] Requested {n_denovo} de novo signatures, "
        f"got {len(denovo_list)} (near={n_near}, far={n_far}) after {trials} trials."
    )
    return pd.DataFrame(denovo_arr, index=denovo_names, columns=df_COSMIC.columns)


# =========================================================
# Profile and count simulation
# =========================================================


def sample_active_signatures_and_profile(
    df_refsig: pd.DataFrame,
    n_active: int,
    comp_dirichlet_alpha: float,
    min_composition: float = 0.005,
    max_trials: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    n_sig_total = df_refsig.shape[0]
    if n_active > n_sig_total:
        raise ValueError(f"n_active={n_active} > total signatures={n_sig_total}")

    active_idx = rng.choice(n_sig_total, size=n_active, replace=False)
    active_names = df_refsig.index[active_idx].tolist()
    alpha_vec = np.full(n_active, comp_dirichlet_alpha, dtype=float)

    composition = None
    for _ in range(max_trials):
        cand = rng.dirichlet(alpha_vec)
        if np.all(cand >= min_composition):
            composition = cand
            break

    if composition is None:
        cand = rng.dirichlet(alpha_vec)
        cand = np.asarray(cand, dtype=float)
        cand[cand < min_composition] = min_composition
        cand = cand / cand.sum()
        composition = cand

    sig_matrix = df_refsig.values[active_idx]
    sig_matrix = _safe_normalize_rows(sig_matrix, axis=1)
    profile_clean = composition @ sig_matrix

    return {
        "active_names": active_names,
        "active_idx": active_idx,
        "composition": composition,
        "profile_clean": profile_clean,
    }


def sample_noisy_counts_from_profile(
    profile_clean: np.ndarray,
    depth: int,
    profile_dirichlet_conc: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    if depth <= 0:
        raise ValueError("depth must be positive")

    profile_clean = np.asarray(profile_clean, dtype=float)
    profile_clean = profile_clean / (profile_clean.sum() + 1e-12)
    alpha = np.where(profile_clean * profile_dirichlet_conc <= 0, 1e-6, profile_clean * profile_dirichlet_conc)
    profile_noisy = rng.dirichlet(alpha)
    counts = rng.multinomial(depth, profile_noisy)
    profile_noisy = counts / counts.sum()
    return {"counts": counts, "profile_noisy": profile_noisy}


# =========================================================
# Depth model
# =========================================================


DEPTH_BANDS = {
    "low": (100, 300),
    "medium": (1000, 5000),
    "high": (15000, 50000),
}


@dataclass
class SimulationConfig:
    name: str
    n_samples: int
    n_active_range: Tuple[int, int]
    comp_alpha_range: Tuple[float, float]
    min_composition: float
    profile_dirichlet_conc: float
    depth_mode: str
    random_state: Optional[int] = None


def _profile_complexity(profile_clean: np.ndarray) -> Tuple[float, float, float]:
    profile_clean = np.asarray(profile_clean, dtype=float)
    profile_clean = profile_clean / (profile_clean.sum() + 1e-12)
    entropy = _shannon_entropy(profile_clean)
    gini = _gini(profile_clean)
    h_max = math.log(profile_clean.size) if profile_clean.size > 1 else 1.0
    h_norm = entropy / h_max
    complexity = 0.5 * h_norm + 0.5 * (1.0 - gini)
    complexity = float(max(0.0, min(1.0, complexity)))
    return entropy, gini, complexity


def _sample_depth_for_profile(
    profile_clean: np.ndarray,
    depth_mode: str,
    rng: np.random.Generator,
) -> int:
    _, _, complexity = _profile_complexity(profile_clean)

    if depth_mode == "mixed":
        mode = rng.choice(["low", "medium", "high"])
    else:
        if depth_mode not in DEPTH_BANDS:
            raise ValueError(f"Unknown depth_mode: {depth_mode}")
        mode = depth_mode

    d_min, d_max = DEPTH_BANDS[mode]
    depth = int(round(d_min + complexity * (d_max - d_min)))
    depth = max(100, min(depth, 50000))
    return depth


def simulate_dataset(config: SimulationConfig, df_refsig: pd.DataFrame) -> Dict[str, Any]:
    rng = np.random.default_rng(config.random_state)
    n_samples = config.n_samples
    n_ref = df_refsig.shape[0]
    ctx_cols = df_refsig.columns.to_list()
    ref_names = df_refsig.index.to_list()

    comp_mat = np.zeros((n_samples, n_ref), dtype=float)
    profile_clean_mat = np.zeros((n_samples, len(ctx_cols)), dtype=float)
    counts_mat = np.zeros((n_samples, len(ctx_cols)), dtype=int)
    meta_records = []

    for i in range(n_samples):
        n_active = rng.integers(config.n_active_range[0], config.n_active_range[1] + 1)
        alpha_c = rng.uniform(config.comp_alpha_range[0], config.comp_alpha_range[1])

        a_out = sample_active_signatures_and_profile(
            df_refsig=df_refsig,
            n_active=n_active,
            comp_dirichlet_alpha=alpha_c,
            min_composition=config.min_composition,
            rng=rng,
        )
        active_idx = a_out["active_idx"]
        comp_vec = a_out["composition"]
        profile_clean = a_out["profile_clean"]
        comp_mat[i, active_idx] = comp_vec
        profile_clean_mat[i] = profile_clean

        entropy, gini, complexity = _profile_complexity(profile_clean)
        depth = _sample_depth_for_profile(profile_clean, config.depth_mode, rng)
        b_out = sample_noisy_counts_from_profile(
            profile_clean=profile_clean,
            depth=depth,
            profile_dirichlet_conc=config.profile_dirichlet_conc,
            rng=rng,
        )
        counts_mat[i] = b_out["counts"]

        meta_records.append(
            {
                "sample_id": f"S{i:06d}",
                "n_active": int(n_active),
                "comp_alpha": float(alpha_c),
                "profile_dirichlet_conc": float(config.profile_dirichlet_conc),
                "depth": int(depth),
                "depth_mode_used": config.depth_mode,
                "profile_entropy": entropy,
                "profile_gini": gini,
                "profile_complexity": complexity,
            }
        )

    sample_ids = [f"S{i:06d}" for i in range(n_samples)]
    return {
        "df_composition": pd.DataFrame(comp_mat, index=sample_ids, columns=ref_names),
        "df_profile_clean": pd.DataFrame(profile_clean_mat, index=sample_ids, columns=ctx_cols),
        "df_counts": pd.DataFrame(counts_mat, index=sample_ids, columns=ctx_cols),
        "df_meta": pd.DataFrame(meta_records).set_index("sample_id"),
    }
