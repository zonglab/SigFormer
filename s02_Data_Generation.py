import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# =========================================================
# 0. utils
# =========================================================

def _safe_normalize_rows(arr: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize rows to sum=1 (for probability vectors).
    """
    arr = np.asarray(arr, dtype=float)
    s = arr.sum(axis=axis, keepdims=True)
    s = np.where(s < eps, eps, s)
    return arr / s


def _shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy for a probability vector (natural log).
    """
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + eps)
    p_clip = np.clip(p, eps, 1.0)
    return float(-np.sum(p_clip * np.log(p_clip)))


def _gini(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gini coefficient for a probability vector.
    """
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + eps)
    p_sorted = np.sort(p)
    n = p_sorted.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * p_sorted)) / (n * np.sum(p_sorted) + eps))


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix for row-vectors.
    """
    X = np.asarray(X, dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    Xn = X / norm
    return Xn @ Xn.T


def _cosine_similarity_vec_mat(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between vector v and each row in matrix M.
    """
    v = np.asarray(v, dtype=float)
    M = np.asarray(M, dtype=float)
    v_norm = np.linalg.norm(v)
    M_norm = np.linalg.norm(M, axis=1)
    v_norm = 1e-12 if v_norm == 0 else v_norm
    M_norm = np.where(M_norm == 0, 1e-12, M_norm)
    return (M @ v) / (M_norm * v_norm)


# =========================================================
# 1. read COSMIC and get stats
# =========================================================

def get_COSMIC(path):
    df_refsig = pd.read_csv(path, sep="\t", index_col=0).T
    df_refsig = df_refsig.loc[~df_refsig.index.isna(),]
    df_refsig.columns = [f"{i[2]}>{i[4]},{i[0]}-{i[6]}" for i in df_refsig.columns]
    return df_refsig.loc[:,VEC_sub_ctx].copy()


def summarize_cosmic_signatures(
    df_COSMIC: pd.DataFrame,
    sparsity_threshold: float = 1e-4
) -> pd.DataFrame:
    """
    Compute summary statistics for a COSMIC-style signature matrix.
    
    Input
    -----
    df_COSMIC
        Rows = signatures, columns = contexts (typically 96 trinucleotides).
        Rows are normalized to sum to 1 before statistics are computed.
    
    Computed metrics
    ----------------
    - entropy: Shannon entropy (natural log)
    - sparsity: fraction of entries below `sparsity_threshold`
    - gini: Gini coefficient
    - mean/max cosine similarity to other signatures (excluding self)
    
    Returns
    -------
    pd.DataFrame
        Indexed by signature name.
    """
    arr = _safe_normalize_rows(df_COSMIC.values, axis=1)
    K, M = arr.shape
    entropies = []
    sparsities = []
    ginis = []

    for i in range(K):
        p = arr[i]
        entropies.append(_shannon_entropy(p))
        sparsities.append(float(np.mean(p < sparsity_threshold)))
        ginis.append(_gini(p))

    # pairwise cosine
    cos_mat = _cosine_similarity_matrix(arr)
    cos_mat_no_diag = cos_mat - np.eye(K)

    mean_cos_to_others = []
    max_cos_to_others = []
    for i in range(K):
        others = np.delete(cos_mat[i], i)
        mean_cos_to_others.append(float(np.mean(others)))
        max_cos_to_others.append(float(np.max(others)))

    df_stat = pd.DataFrame({
        "row_name": df_COSMIC.index.to_list(),
        "entropy": entropies,
        "sparsity(<{:.0e})".format(sparsity_threshold): sparsities,
        "gini": ginis,
        "mean_cosine_to_others": mean_cos_to_others,
        "max_cosine_to_others": max_cos_to_others,
    }).set_index("row_name")

    return df_stat


# =========================================================
# 2. Build a de novo signature space
# =========================================================

def build_denovo_signatures(
    df_COSMIC: pd.DataFrame,
    n_denovo: int = 500,
    target_cosine_max: float = 0.8,
    alpha_scale_range: Tuple[float, float] = (150.0, 600.0),
    max_total_trials: int = 50000,
    random_state: Optional[int] = 2025
) -> pd.DataFrame:
    """
    Build a de novo signature bank containing two complementary modes:
    
    1) COSMIC-near: pick a single COSMIC signature as a template and apply Dirichlet
       jitter, keeping the overall shape similar but not *too* similar.
    2) COSMIC-far: sample directly on the 96D simplex, while enforcing:
       - sufficiently low cosine similarity to every COSMIC signature
       - not well reconstructed by a low-rank COSMIC subspace (to avoid being a mere
         linear combination of COSMIC signatures)
       - entropy and sparsity within the empirical COSMIC range (to avoid extreme
         1-hot spikes or overly flat distributions)
    
    This yields coverage both around known COSMIC patterns and in regions of the
    signature space that COSMIC does not span well (i.e., truly de novo signatures).
    
    Parameters
    ----------
    df_COSMIC
        Rows are COSMIC signatures, columns are 96 trinucleotide contexts.
    n_denovo
        Target number of de novo signatures to generate.
    target_cosine_max
        Upper bound on cosine similarity (used for de novo-vs-de novo and the
        COSMIC-near constraint against COSMIC).
    alpha_scale_range
        Scaling range for the Dirichlet concentration in "near" mode.
    max_total_trials
        Max total sampling attempts (prevents infinite loops).
    random_state
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        de novo signatures with rows = de novo names, columns = 96 contexts.
    """
    rng = np.random.default_rng(random_state)

    ### normalize to probability mass function
    cosmic_arr = _safe_normalize_rows(df_COSMIC.values, axis=1)
    n_cosmic, n_ctx = cosmic_arr.shape

    ### get entropy/sparsity distribution
    df_cosmic_stat = summarize_cosmic_signatures(df_COSMIC, sparsity_threshold=1e-4)
    sparsity_col = [c for c in df_cosmic_stat.columns if c.startswith("sparsity(")][0]
    cosmic_entropy = df_cosmic_stat["entropy"].values
    cosmic_sparsity = df_cosmic_stat[sparsity_col].values

    ent_lo, ent_hi = np.quantile(cosmic_entropy, [0.05, 0.95])
    sps_lo, sps_hi = np.quantile(cosmic_sparsity, [0.05, 0.95])

    ### Approximate COSMIC space with SVD
    # cosmic_arr: (n_cosmic, n_ctx)
    U_svd, S_svd, Vt_svd = np.linalg.svd(cosmic_arr, full_matrices=False)
    # Vt_svd.shape = (r_max, n_ctx)
    r_subspace = min(30, Vt_svd.shape[0], Vt_svd.shape[1])
    subspace_basis = Vt_svd[:r_subspace].T  # (n_ctx, r_subspace), columns are orthonormal

    def _subspace_R2(vec: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
        v = np.asarray(vec, dtype=float)
        v_norm2 = float(np.sum(v * v)) + eps
        proj_coef = basis.T @ v              # (r,)
        proj_norm2 = float(np.sum(proj_coef * proj_coef))
        return proj_norm2 / v_norm2

    ### fixed ratio of "near" and "far" de novos
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
            # --------------- COSMIC-near -------------------------------------
            idx = rng.integers(0, n_cosmic)
            template = cosmic_arr[idx]

            scale = rng.uniform(alpha_scale_range[0], alpha_scale_range[1])
            alpha = template * scale
            alpha = np.where(alpha <= 0, 1e-6, alpha)

            cand = rng.dirichlet(alpha)

            # cosine similarity to COSMIC signatures
            cos_cosmic = _cosine_similarity_vec_mat(cand, cosmic_arr)
            if np.max(cos_cosmic) >= target_cosine_max:
                continue

            # cosine similarity to already accepted de novo signatures
            if len(denovo_list) > 0:
                denovo_arr = np.vstack(denovo_list)
                cos_denovo = _cosine_similarity_vec_mat(cand, denovo_arr)
                if np.max(cos_denovo) >= target_cosine_max:
                    continue

            denovo_list.append(cand)
            denovo_names.append(f"DeNovo_near_{len(denovo_list):04d}")
            n_near += 1

        else:
            # --------------- COSMIC-far --------------------------------------
            # control sparsity with Dirichlet first
            alpha0 = rng.uniform(0.2, 40.0)
            alpha_vec = np.full(n_ctx, alpha0 / n_ctx, dtype=float)
            cand = rng.dirichlet(alpha_vec)

            # restrict sparsity/entropy to avoid extreme case
            cand_ent = _shannon_entropy(cand)
            cand_sps = float(np.mean(cand < 1e-4))

            if not (ent_lo <= cand_ent <= ent_hi): continue
            if not (sps_lo <= cand_sps <= sps_hi): continue

            # far enough from COSMIC
            cos_cosmic = _cosine_similarity_vec_mat(cand, cosmic_arr)
            if np.max(cos_cosmic) >= far_cosine_max_to_cosmic:
                continue

            # avoid generating too similar de novos
            if len(denovo_list) > 0:
                denovo_arr = np.vstack(denovo_list)
                cos_denovo = _cosine_similarity_vec_mat(cand, denovo_arr)
                if np.max(cos_denovo) >= target_cosine_max:
                    continue

            # avoid being linear combination of COSMICS
            R2_sub = _subspace_R2(cand, subspace_basis)
            if R2_sub >= far_subspace_R2_max:
                continue

            denovo_list.append(cand)
            denovo_names.append(f"DeNovo_far_{len(denovo_list):04d}")
            n_far += 1

    if len(denovo_list) == 0:
        raise RuntimeError("Failed to generate any de novo signatures under current constraints.")

    denovo_arr = np.vstack(denovo_list)
    df_denovo = pd.DataFrame(denovo_arr, index=denovo_names, columns=df_COSMIC.columns)

    print(
        f"[INFO] Requested {n_denovo} de novo signatures, "
        f"got {len(denovo_list)} (near={n_near}, far={n_far}) after {trials} trials."
    )
    return df_denovo


# =========================================================
# 3. draw signature + composition
# =========================================================

def sample_active_signatures_and_profile(
    df_refsig: pd.DataFrame,
    n_active: int,
    comp_dirichlet_alpha: float,
    min_composition: float = 0.005,   # 0.5%
    max_trials: int = 64,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Randomly select a subset of signatures and sample a mixture profile.
    
    Steps
    -----
    1) Choose `n_active` distinct signatures from `df_refsig` (without replacement).
    2) Sample a composition vector from Dirichlet(`comp_dirichlet_alpha`), enforcing
       `min_composition` for each active signature. If the constraint is not met
       within `max_trials`, a fallback clamps small entries then renormalizes.
    3) Construct the clean 96-context profile as a convex combination of the chosen
       signatures.
    
    Returns
    -------
    dict
        active_names: list[str]
        active_idx: np.ndarray (n_active,)
        composition: np.ndarray (n_active,)
        profile_clean: np.ndarray (96,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sig_total = df_refsig.shape[0]
    if n_active > n_sig_total:
        raise ValueError(f"n_active={n_active} > total signatures={n_sig_total}")

    active_idx = rng.choice(n_sig_total, size=n_active, replace=False)
    active_names = df_refsig.index[active_idx].tolist()

    alpha_vec = np.full(n_active, comp_dirichlet_alpha, dtype=float)

    comp = None
    for _ in range(max_trials):
        cand = rng.dirichlet(alpha_vec)
        if np.all(cand >= min_composition):
            comp = cand
            break

    if comp is None:
        cand = rng.dirichlet(alpha_vec)
        cand = np.asarray(cand, dtype=float)
        too_small = cand < min_composition
        if np.any(too_small):
            cand[too_small] = min_composition
        cand = cand / cand.sum()
        comp = cand

    sig_matrix = df_refsig.values[active_idx]   # (n_active, 96)
    sig_matrix = _safe_normalize_rows(sig_matrix, axis=1)
    profile_clean = comp @ sig_matrix          # (96,)

    return {"active_names": active_names,
            "active_idx": active_idx,
            "composition": comp,
            "profile_clean": profile_clean,}


# =========================================================
# 4. Dirichlet-Multinomial + depth
# =========================================================

def sample_noisy_counts_from_profile(
    profile_clean: np.ndarray,
    depth: int,
    profile_dirichlet_conc: float,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Generate noisy counts from a clean 96-context profile using a Dirichlet-Multinomial.
    
    Model
    -----
    1) p_noisy ~ Dirichlet(profile_dirichlet_conc * profile_clean)
    2) counts  ~ Multinomial(depth, p_noisy)
    
    Parameters
    ----------
    profile_clean
        Noiseless 96-dim probability vector.
    depth
        Total mutation count to draw.
    profile_dirichlet_conc
        Scalar concentration; larger => less noise (p_noisy closer to profile_clean).
    
    Returns
    -------
    dict
        counts: np.ndarray (96,)
        profile_noisy: np.ndarray (96,)  # normalized counts
    """
    if rng is None:
        rng = np.random.default_rng()

    if depth <= 0:
        raise ValueError("depth must be positive")

    p = np.asarray(profile_clean, dtype=float)
    p = p / (p.sum() + 1e-12)

    alpha = p * profile_dirichlet_conc
    alpha = np.where(alpha <= 0, 1e-6, alpha)

    p_noisy = rng.dirichlet(alpha)
    counts = rng.multinomial(depth, p_noisy)
    profile_noisy = counts / counts.sum()

    return {
        "counts": counts,
        "profile_noisy": profile_noisy,
    }


# =========================================================
# 5. Scheduler
# =========================================================

DEPTH_BANDS = {"low":    (100, 300),
               "medium": (1000, 5000),
               "high":   (15000, 50000),
}


@dataclass
class SimulationConfig:
    name: str
    n_samples: int
    n_active_range: Tuple[int, int]       # (min_n_sig, max_n_sig)
    comp_alpha_range: Tuple[float, float] # Dirichlet alpha for composition
    min_composition: float                # 0.005 -> 0.5%
    profile_dirichlet_conc: float         # noise level: 40, 120, 280, 2300 ...
    depth_mode: str                       # "low", "medium", "high", or "mixed"
    random_state: Optional[int] = None


def _profile_complexity(profile_clean: np.ndarray) -> Tuple[float, float, float]:
    """
    Given a 96-dim noiseless profile, calculate:
    - entropy: Shannon entropy
    - gini: Gini index
    - complexity: [0-1]：
          complexity = 0.5 * (H / H_max) + 0.5 * (1 - gini)
    """
    p = np.asarray(profile_clean, dtype=float)
    p = p / (p.sum() + 1e-12)

    H = _shannon_entropy(p)
    K = p.size
    H_max = math.log(K) if K > 1 else 1.0
    H_norm = H / H_max

    g = _gini(p)
    complexity = 0.5 * H_norm + 0.5 * (1.0 - g)
    # clamp to [0, 1]
    complexity = float(max(0.0, min(1.0, complexity)))
    return H, g, complexity


def _sample_depth_for_profile(
    profile_clean: np.ndarray,
    depth_mode: str,
    rng: np.random.Generator
) -> int:
    """
    Sample a sequencing depth for a given clean (noiseless) profile using
    entropy/Gini-derived complexity, then map it into a depth band.
    
    depth_mode
      - "low"    : map into DEPTH_BANDS["low"]
      - "medium" : map into DEPTH_BANDS["medium"]
      - "high"   : map into DEPTH_BANDS["high"]
      - "mixed"  : uniformly pick one band from low/medium/high each time
    
    Heuristic
      * flatter profiles (higher entropy, lower Gini) => higher complexity => larger depth
      * sparser profiles (lower entropy, higher Gini) => lower complexity => smaller depth
    
    Returns an integer depth in [100, 50000].
    """
    H, g, complexity = _profile_complexity(profile_clean)

    # Pick a depth band based on the mode
    if depth_mode == "mixed":
        mode = rng.choice(["low", "medium", "high"])
    else:
        if depth_mode not in DEPTH_BANDS:
            raise ValueError(f"Unknown depth_mode: {depth_mode}")
        mode = depth_mode

    d_min, d_max = DEPTH_BANDS[mode]

    # Map complexity in [0, 1] into the chosen band
    base_depth = d_min + complexity * (d_max - d_min)
    depth = int(np.round(base_depth))

    # Sanity clamp
    depth = max(100, min(depth, 50000))
    return depth


def simulate_dataset(
    config: SimulationConfig,
    df_refsig: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate one full simulated dataset given a SimulationConfig and a reference
    signature bank.
    
    Outputs
    -------
    df_composition
        (n_samples, n_refsig) composition matrix over reference signatures.
    df_profile_clean
        (n_samples, 96) noiseless mutation profiles.
    df_counts
        (n_samples, 96) noisy mutation counts sampled from Dirichlet-Multinomial.
    df_meta
        (n_samples, meta columns) per-sample metadata (active signatures, depth, etc.).
    """
    rng = np.random.default_rng(config.random_state)

    n_samples = config.n_samples
    n_ref = df_refsig.shape[0]
    ctx_cols = df_refsig.columns.to_list()
    ref_names = df_refsig.index.to_list()

    # Preallocate outputs
    comp_mat = np.zeros((n_samples, n_ref), dtype=float)
    profile_clean_mat = np.zeros((n_samples, len(ctx_cols)), dtype=float)
    counts_mat = np.zeros((n_samples, len(ctx_cols)), dtype=int)

    meta_records = []

    for i in range(n_samples):
        # 1) Randomly choose the number of active signatures
        min_n, max_n = config.n_active_range
        n_active = rng.integers(min_n, max_n + 1)

        # 2) Draw a Dirichlet alpha for the composition
        alpha_c = rng.uniform(config.comp_alpha_range[0], config.comp_alpha_range[1])

        # 3) Sample active signatures + composition + noiseless profile
        a_out = sample_active_signatures_and_profile(
            df_refsig=df_refsig,
            n_active=n_active,
            comp_dirichlet_alpha=alpha_c,
            min_composition=config.min_composition,
            rng=rng
        )

        active_idx = a_out["active_idx"]
        comp_vec = a_out["composition"]
        profile_clean = a_out["profile_clean"]

        # Write the composition into the full (n_ref) matrix
        comp_mat[i, active_idx] = comp_vec
        profile_clean_mat[i] = profile_clean

        # 4) Decide depth from profile entropy/Gini/complexity + depth_mode
        H_prof, g_prof, complexity_prof = _profile_complexity(profile_clean)
        depth = _sample_depth_for_profile(profile_clean, config.depth_mode, rng)

        # 5) Sample noisy counts via Dirichlet-Multinomial
        b_out = sample_noisy_counts_from_profile(
            profile_clean=profile_clean,
            depth=depth,
            profile_dirichlet_conc=config.profile_dirichlet_conc,
            rng=rng
        )
        counts = b_out["counts"]

        counts_mat[i] = counts

        meta_records.append({
            "sample_id": f"S{i:06d}",
            "n_active": n_active,
            "comp_alpha": alpha_c,
            "profile_dirichlet_conc": config.profile_dirichlet_conc,
            "depth": depth,
            "depth_mode_used": config.depth_mode,
            "profile_entropy": H_prof,
            "profile_gini": g_prof,
            "profile_complexity": complexity_prof,
        })

    sample_ids = [f"S{i:06d}" for i in range(n_samples)]

    df_composition = pd.DataFrame(comp_mat, index=sample_ids, columns=ref_names)
    df_profile_clean = pd.DataFrame(profile_clean_mat, index=sample_ids, columns=ctx_cols)
    df_counts = pd.DataFrame(counts_mat, index=sample_ids, columns=ctx_cols)
    df_meta = pd.DataFrame(meta_records).set_index("sample_id")

    return {
        "df_composition": df_composition,
        "df_profile_clean": df_profile_clean,
        "df_counts": df_counts,
        "df_meta": df_meta,
    }


# =========================================================
# 6. Example: combine COSMIC + de novo + a few configs
# =========================================================

### if __name__ == "__main__":
###     PATH_COSMIC = "fill in your path to COSMIC"
###     df_COSMIC = get_COSMIC(PATH_COSMIC)  # rows = COSMIC names, cols = 96 contexts
###
###     print("[STEP] Summarizing COSMIC signatures...")
###     df_cosmic_stat = summarize_cosmic_signatures(df_COSMIC, sparsity_threshold=1e-4)
###     # Print a few rows as a quick sanity check
###     print(df_cosmic_stat.head())
###
###     print("\n[STEP] Building de novo signature bank...")
###     df_denovo = build_denovo_signatures(
###         df_COSMIC,
###         n_denovo=500,
###         target_cosine_max=0.8,
###         alpha_scale_range=(150.0, 600.0),
###         max_total_trials=50000,
###         random_state=2025
###     )
###
###     # Merge into a single reference signature bank
###     df_refsig = pd.concat([df_COSMIC, df_denovo], axis=0)
###     print(f"[INFO] df_refsig shape: {df_refsig.shape}")
###
###     # Define a few simulation schemes
###     cfg_sparse_low_depth_high_noise = SimulationConfig(
###         name="sparse_lowDepth_highNoise",
###         n_samples=2000,
###         n_active_range=(2, 5),
###         comp_alpha_range=(0.05, 0.2),     # a few signatures dominate
###         min_composition=0.005,            # 0.5%
###         profile_dirichlet_conc=40.0,      # high noise, cosine ~ < 0.8
###         depth_mode="low",
###         random_state=1001,
###     )
###
###     cfg_mixed_mediumDepth_mediumNoise = SimulationConfig(
###         name="mixed_mediumDepth_mediumNoise",
###         n_samples=2000,
###         n_active_range=(3, 10),
###         comp_alpha_range=(0.2, 0.8),      # still relatively sparse
###         min_composition=0.005,
###         profile_dirichlet_conc=120.0,     # medium noise, cosine ~ 0.85
###         depth_mode="medium",
###         random_state=1002,
###     )
###
###     cfg_even_highDepth_lowNoise = SimulationConfig(
###         name="even_highDepth_lowNoise",
###         n_samples=2000,
###         n_active_range=(8, 16),
###         comp_alpha_range=(2.0, 10.0),     # closer to even composition
###         min_composition=0.005,
###         profile_dirichlet_conc=2300.0,    # low noise, cosine ~ 0.95
###         depth_mode="high",
###         random_state=1003,
###     )
###
###     # Choose one or more configs to generate data; this example uses one
###     cfg = cfg_mixed_mediumDepth_mediumNoise
###     print(f"\n[STEP] Simulating dataset with config: {cfg.name}")
###     out = simulate_dataset(cfg, df_refsig=df_refsig)
###
###     df_compo = out["df_composition"]
###     df_prof = out["df_profile_clean"]
###     df_counts = out["df_counts"]
###     df_meta = out["df_meta"]
###
###     print("[RESULT] compositions shape:", df_compo.shape)
###     print("[RESULT] profiles (clean) shape:", df_prof.shape)
###     print("[RESULT] counts shape:", df_counts.shape)
###     print("[RESULT] meta shape:", df_meta.shape)
###
