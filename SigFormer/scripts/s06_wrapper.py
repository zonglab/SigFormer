#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model wrappers for SigFormer and benchmark comparison methods.

This file contains model-facing code only. Benchmark data generation, result
saving, and plotting live in ``s07_bench_helper``.
"""

from __future__ import annotations

import os, shutil, subprocess, tempfile, time, uuid, hashlib, warnings, contextlib, shlex, math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls, lsq_linear
from sklearn.metrics.pairwise import cosine_similarity
import torch

try:
    import musical
except ImportError:
    musical = None
from .s01_Core import SigFormerCore
try:
    from SigProfilerAssignment import Analyzer as Analyze
except ImportError:
    Analyze = None
from . import s04_Util_apply as SgF_util

# None means: use the conda environment that launched Python.
R_env = None


def ensure_ref_mask(df_refmask, index, columns):
    if df_refmask is None:
        return pd.DataFrame(True, index=index, columns=columns)
    return df_refmask.reindex(index=index, columns=columns, fill_value=False).astype(bool)


def ckpt_epoch_number(path):
    return int(path.stem.rsplit("_", 1)[-1])


def reconstruct_from_compo(df_compo: pd.DataFrame, df_ref_sig: pd.DataFrame) -> pd.DataFrame:
    vals = df_compo.reindex(columns=df_ref_sig.index, fill_value=0.0).to_numpy(float) @ df_ref_sig.to_numpy(float)
    return pd.DataFrame(vals, index=df_compo.index.copy(), columns=df_ref_sig.columns.copy())


class CLASS_wrapper_MuSiCal:
    """Callable wrapper around ``musical.refit.refit``."""

    def __init__(self, method: str = "likelihood_bidirectional", thresh: float = 0.001):
        self.config = dict(method=method, thresh=thresh)

    def __call__(self, df_3nt_raw: pd.DataFrame, df_ref_sig: pd.DataFrame,
                 df_refmask: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:


        if musical is None:
            raise ImportError("musical is required for CLASS_wrapper_MuSiCal")
        df_refmask = ensure_ref_mask(df_refmask, df_3nt_raw.index, df_ref_sig.index)
        df_compo = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=df_ref_sig.index)
        groups = df_refmask.groupby(list(df_refmask.columns), sort=False).groups
        for idx_smp in groups.values():
            mask = df_refmask.loc[idx_smp].iloc[0].astype(bool)
            if mask.sum() == 0:
                mask[:] = True
            idx_ref = df_ref_sig.index[mask]
            df_i, _ = musical.refit.refit(df_3nt_raw.loc[idx_smp].T, df_ref_sig.loc[idx_ref].T, **self.config)
            df_i = SgF_util.sum_scale(df_i.T).reindex(index=idx_smp, columns=idx_ref, fill_value=0.0)
            df_compo.loc[idx_smp, idx_ref] = df_i
        return df_compo, reconstruct_from_compo(df_compo, df_ref_sig)


class CLASS_wrapper_SigFormer:
    """Checkpoint wrapper. Calling the wrapper returns refined SigFormer by default; ``predict_raw`` exposes raw output."""

    def __init__(self, PATH_model: str, epoch=None, device=None, simplex="softmax", strict=False, refine=True, refinement_kwargs=None):
        if torch is None or SigFormerCore is None:
            raise ImportError("torch and s01_Core are required for SigFormer wrapper")
        self.PATH_model = Path(PATH_model)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = torch.load(self.PATH_model, map_location="cpu") if self.PATH_model else {}
        self.args = self.ckpt.get("args", {}) if isinstance(self.ckpt, dict) else {}
        self.simplex = self.args.get("simplex", simplex)
        self.epoch = int(self.PATH_model.stem.rsplit("_", 1)[-1]) if self.PATH_model else None
        self.cfg = dict(
            n_chann=96,
            d_model=int(self.args.get("model_d_model", 96)),
            n_heads=int(self.args.get("model_n_heads", 4)),
            n_L_smp=int(self.args.get("model_smp_n_lyr", 2)),
            n_L_ref=int(self.args.get("model_ref_n_lyr", 4)),
            n_L_smp_ref=int(self.args.get("model_smp_ref_n_lyr", 1)),
            mlp_ratio=float(self.args.get("model_mlp_ratio", 4.0)),
            dropout=float(self.args.get("model_dropout", 0.0)),
            simplex=self.simplex,
            use_tok_ood=bool(self.args.get("residual_evidence", True)),
            residual_init="zero_sample_depth",
            ood_lg_bias_init=float(self.args.get("residual_logit_bias_init", -2.0)),
        )
        self.model = SigFormerCore(**self.cfg).to(self.device)
        state = self.ckpt.get("model_state", self.ckpt) if self.ckpt else {}
        state = {k.replace("module.", "", 1): v for k, v in state.items() if torch.is_tensor(v)}
        if state:
            self.model.load_state_dict(state, strict=strict)
        self.model.eval()
        self.refine = bool(refine)
        self.refinement_kwargs = dict(refinement_kwargs or {})
        self.refiner = None
        self.last_diagnostics = pd.DataFrame()
        self._raw_cache_input = None
        self._raw_cache = None
        print({"pretrained": bool(state), "ckpt": str(self.PATH_model) if self.PATH_model else None,
               "epoch": self.epoch, "simplex": self.simplex, "d_model": self.cfg["d_model"],
               "n_heads": self.cfg["n_heads"],
               "layers": (self.cfg["n_L_smp"], self.cfg["n_L_ref"], self.cfg["n_L_smp_ref"])})

    def _predict_raw(self, df_3nt_raw, df_ref_sig, df_refmask):
        comp = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=df_ref_sig.index)
        ood = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=["OOD"])
        X_all = torch.as_tensor(df_3nt_raw.to_numpy(np.float32), device=self.device).contiguous()
        groups = df_refmask.groupby(list(df_refmask.columns), sort=False).groups
        with torch.no_grad():
            for idx_smp in groups.values():
                mask = df_refmask.loc[idx_smp].iloc[0].astype(bool)
                if not mask.any():
                    mask[:] = True
                idx_ref = df_ref_sig.index[mask]
                X = X_all[df_3nt_raw.index.get_indexer(idx_smp)]
                R = torch.as_tensor(df_ref_sig.loc[idx_ref].to_numpy(np.float32), device=self.device)
                R = R.unsqueeze(0).expand(len(idx_smp), -1, -1).contiguous()
                M = torch.ones((len(idx_smp), len(idx_ref)), dtype=torch.bool, device=self.device)
                extra = self.model(X, R, simplex=self.simplex, ref_mask=M, return_extra=True)
                comp.loc[idx_smp, idx_ref] = extra["known_composition"].cpu().numpy()
                ood.loc[idx_smp, "OOD"] = extra["ood_mass"].cpu().numpy().ravel()
        return comp, ood

    def predict_raw(self, df_3nt_raw, df_ref_sig, df_refmask=None):
        current = (df_3nt_raw, df_ref_sig, df_refmask)
        if self._raw_cache_input is not None and self._raw_cache is not None and all(a is b for a, b in zip(current, self._raw_cache_input)):
            return self._raw_cache
        mask = ensure_ref_mask(df_refmask, df_3nt_raw.index, df_ref_sig.index)
        comp, ood = self._predict_raw(df_3nt_raw, df_ref_sig, mask)
        self._raw_cache_input = current
        self._raw_cache = (comp, reconstruct_from_compo(comp, df_ref_sig), ood)
        return self._raw_cache

    def __call__(self, df_3nt_raw, df_ref_sig, df_refmask=None, comp_floor=None, min_mutations=None):
        comp, recon, ood = self.predict_raw(df_3nt_raw, df_ref_sig, df_refmask)
        if not self.refine:
            return comp, recon, ood
        self.refiner = CLASS_wrapper_linear_refinement(df_ref_sig, **self.refinement_kwargs)
        comp, recon, ood = self.refiner(df_3nt_raw, comp, df_ref_sig, df_refmask)
        self.last_diagnostics = self.refiner.last_diagnostics
        return comp, recon, ood


class CLASS_wrapper_SigProfilerAssignment:
    """Callable wrapper around ``SigProfilerAssignment.Analyzer.cosmic_fit``."""

    def __init__(self, PATH_tmp=None, hg_ver="GRCh38", cos_ver=3.4, **kwargs):
        self.PATH_tmp = Path(PATH_tmp or f"./tmp_sigprofiler_assignment/SPA_{uuid.uuid4().hex[:12]}")
        self.hg_ver = hg_ver
        self.cos_ver = cos_ver
        self.SPA_3nt = [f"{m},{l}-{r}" for l in "ACGT" for m in ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G") for r in "ACGT"]
        self.kwargs = kwargs

    @staticmethod
    def _to_spa_name(x):
        mut, ctx = str(x).split(",")
        lnt, rnt = ctx.split("-")
        return f"{lnt}[{mut}]{rnt}"

    def _run_one(self, df_smp, df_ref, out_dir):


        out_smp = df_smp.rename(columns=self._to_spa_name).T
        out_ref = df_ref.rename(columns=self._to_spa_name).T
        out_smp.index.name = out_ref.index.name = "MutationType"
        out_dir.mkdir(parents=True, exist_ok=True)
        path_smp = out_dir / "input_smp.tsv"
        path_ref = out_dir / "input_ref.tsv"
        path_exp = out_dir / "Assignment_Solution" / "Activities" / "Assignment_Solution_Activities.txt"
        out_smp.to_csv(path_smp, sep="\t")
        out_ref.to_csv(path_ref, sep="\t")
        fit_kwargs = dict(samples=str(path_smp), output=str(out_dir), input_type="matrix", context_type="96",
                          collapse_to_SBS96=True, cosmic_version=self.cos_ver, exome=False, genome_build=self.hg_ver,
                          signature_database=str(path_ref), export_probabilities=False,
                          export_probabilities_per_mutation=False, make_plots=False,
                          sample_reconstruction_plots=False, verbose=False, **self.kwargs)
        with tempfile.TemporaryFile("w+", encoding="utf-8") as f, warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            warnings.simplefilter("always")
            Analyze.cosmic_fit(**fit_kwargs)
        for w in ws:
            warnings.warn_explicit(str(w.message), w.category, w.filename, w.lineno)
        df_exp = pd.read_csv(path_exp, sep="\t", index_col=0)
        return df_exp.reindex(df_smp.index, columns=df_ref.index, fill_value=0.0).astype(float)

    def __call__(self, df_3nt_raw, df_ref_sig, df_refmask=None, keep_tmp=False):
        if Analyze is None:
            raise ImportError("SigProfilerAssignment is required for CLASS_wrapper_SigProfilerAssignment")
        df_smp = df_3nt_raw.loc[:, self.SPA_3nt].copy()
        df_ref = df_ref_sig.loc[:, self.SPA_3nt].copy()
        df_refmask = ensure_ref_mask(df_refmask, df_smp.index, df_ref.index)
        self.PATH_tmp.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="SPA_run_", dir=self.PATH_tmp))
        try:
            df_compo = pd.DataFrame(0.0, index=df_smp.index, columns=df_ref.index)
            groups = df_refmask.groupby(list(df_refmask.columns), sort=False).groups
            for i, idx_smp in enumerate(groups.values()):
                mask = df_refmask.loc[idx_smp].iloc[0].astype(bool)
                if mask.sum() == 0:
                    mask[:] = True
                idx_ref = df_ref.index[mask.to_numpy()]
                df_i = self._run_one(df_smp.loc[idx_smp], df_ref.loc[idx_ref], run_dir / f"group_{i:04d}")
                df_compo.loc[idx_smp, idx_ref] = df_i.reindex(idx_smp, columns=idx_ref, fill_value=0.0)
            df_compo = SgF_util.sum_scale(df_compo, axis=1)
            return df_compo, reconstruct_from_compo(df_compo, df_ref)
        finally:
            if not keep_tmp:
                shutil.rmtree(run_dir, ignore_errors=True)


def _hash_inputs(*dfs: pd.DataFrame) -> str:

    h = hashlib.md5()
    for df in dfs:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()

def _r_bool(x: bool) -> str:
    return "TRUE" if x else "FALSE"


R_SOURCE_COMMON = 'source(file.path(dirname(normalizePath(sub("--file=", "", commandArgs(FALSE)[grep("--file=", commandArgs(FALSE))][1]))), "benchmark_common.R"))'

R_COMMON_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (!is.na(i) && i < length(args)) args[[i + 1]] else default
}

read_tsv <- function(path, row.names = 1) {
  read.table(path, sep = "\t", header = TRUE, row.names = row.names,
             check.names = FALSE, quote = "", comment.char = "")
}

save_tsv <- function(x, path) {
  write.table(x, file = path, sep = "\t", quote = FALSE, col.names = NA)
}

finite_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "numeric"
  x[!is.finite(x) | x < 0] <- 0
  x
}

row_norm <- function(x) {
  x <- finite_matrix(x)
  s <- rowSums(x)
  s[!is.finite(s) | s <= 0] <- 1
  x / s
}

clip01 <- function(x) max(0, min(1, x))

read_runner_input <- function(input_dir) {
  list(X = finite_matrix(read_tsv(file.path(input_dir, "X_count_data.tsv"))),
       R = row_norm(read_tsv(file.path(input_dir, "R_refsigpool.tsv"))),
       mask = as.matrix(read_tsv(file.path(input_dir, "ref_mask.tsv"))) > 0)
}

init_outputs <- function(X, R, method) {
  list(pred = matrix(0, nrow(X), nrow(R), dimnames = list(rownames(X), rownames(R))),
       ood = matrix(0, nrow(X), 1, dimnames = list(rownames(X), "OOD")),
       status = data.frame(sample_id = rownames(X), method = method,
                           status = "pending", message = "", seconds = NA_real_))
}

set_status <- function(out, rows, status, message = "", seconds = NA_real_) {
  out$status[rows, c("status", "message", "seconds")] <- list(status, message, seconds)
  out
}

write_outputs <- function(out, output_dir, method, n_samples, seconds) {
  save_tsv(out$pred, file.path(output_dir, paste0(method, "__pred_compo.tsv")))
  save_tsv(out$ood, file.path(output_dir, paste0(method, "__ood_mass.tsv")))
  write.table(out$status, file = file.path(output_dir, paste0(method, "__status.tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(data.frame(method = method, n_samples = n_samples, seconds = seconds),
              file = file.path(output_dir, paste0(method, "__runtime.tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

extract_exposure <- function(obj) {
  if (is.null(obj)) return(NULL)
  if (is.matrix(obj) || is.data.frame(obj) || is.numeric(obj)) return(obj)
  if (is.list(obj)) {
    for (nm in c("exposures", "exposure", "weights", "weight", "mean", "Mean", "activities", "activity", "contribution")) {
      if (!is.null(obj[[nm]])) return(obj[[nm]])
    }
  }
  NULL
}

coerce_exposure_vector <- function(x, sample_id, sig_names) {
  mat <- as.matrix(x)
  storage.mode(mat) <- "numeric"
  rn <- rownames(mat)
  cn <- colnames(mat)
  if (!is.null(rn) && all(sig_names %in% rn)) {
    vals <- mat[sig_names, if (!is.null(cn) && sample_id %in% cn) sample_id else 1]
  } else if (!is.null(cn) && all(sig_names %in% cn)) {
    vals <- mat[if (!is.null(rn) && sample_id %in% rn) sample_id else 1, sig_names]
  } else if (length(mat) == length(sig_names)) {
    vals <- as.numeric(mat)
  } else if (nrow(mat) == length(sig_names)) {
    vals <- mat[, 1]
  } else if (ncol(mat) == length(sig_names)) {
    vals <- mat[1, ]
  } else {
    stop("Cannot coerce exposure output to signature vector")
  }
  vals <- pmax(as.numeric(vals), 0)
  vals[!is.finite(vals)] <- 0
  names(vals) <- sig_names
  vals
}

normalise_fit <- function(vals, keep_ood = FALSE) {
  raw <- sum(vals, na.rm = TRUE)
  oo <- if (keep_ood) max(0, 1 - raw) else 0
  denom <- raw + oo

  if (is.finite(denom) && denom > 0) {
    vals <- vals / denom
    oo <- oo / denom
  }

  list(vals = vals, ood = oo)
}

visible_signatures <- function(mask, R, sid) {
  visible <- colnames(mask)[as.logical(mask[sid, ])]
  intersect(visible, rownames(R))
}

run_per_mask_group <- function(input_dir, output_dir, method, fit_one, keep_ood = FALSE) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  d <- read_runner_input(input_dir)
  out <- init_outputs(d$X, d$R, method)
  keys <- apply(d$mask, 1, paste, collapse = "")
  t_all <- proc.time()[["elapsed"]]
  for (key in unique(keys)) {
    idx <- rownames(d$X)[keys == key]
    visible <- visible_signatures(d$mask, d$R, idx[1])
    if (!length(visible)) {
      out <- set_status(out, out$status$sample_id %in% idx, "skipped", "no visible signatures", 0)
      next
    }
    t0 <- proc.time()[["elapsed"]]
    fit <- tryCatch(fit_one(d$X[idx, , drop = FALSE], d$R[visible, , drop = FALSE], idx), error = function(e) e)
    dt <- proc.time()[["elapsed"]] - t0
    if (inherits(fit, "error")) {
      out <- set_status(out, out$status$sample_id %in% idx, "failed", conditionMessage(fit), dt)
      next
    }
    mat <- extract_exposure(fit)
    if (is.null(mat)) {
      out <- set_status(out, out$status$sample_id %in% idx, "failed", "no exposure matrix", dt)
      next
    }
    for (sid in idx) {
      vals <- tryCatch(coerce_exposure_vector(mat, sid, visible), error = function(e) e)
      if (inherits(vals, "error")) {
        out <- set_status(out, out$status$sample_id == sid, "failed", conditionMessage(vals), dt)
        next
      }
      fit_norm <- normalise_fit(vals, keep_ood)
      out$pred[sid, visible] <- fit_norm$vals[visible]
      out$ood[sid, "OOD"] <- fit_norm$ood
      out <- set_status(out, out$status$sample_id == sid, "ok", "", dt)
    }
  }
  write_outputs(out, output_dir, method, nrow(d$X), proc.time()[["elapsed"]] - t_all)
}

run_per_sample <- function(input_dir, output_dir, method, fit_one_sample, keep_ood = FALSE) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  d <- read_runner_input(input_dir)
  out <- init_outputs(d$X, d$R, method)
  t_all <- proc.time()[["elapsed"]]
  for (sid in rownames(d$X)) {
    visible <- visible_signatures(d$mask, d$R, sid)
    if (!length(visible)) {
      out <- set_status(out, out$status$sample_id == sid, "skipped", "no visible signatures", 0)
      next
    }
    t0 <- proc.time()[["elapsed"]]
    fit <- tryCatch(fit_one_sample(d$X[sid, , drop = FALSE], d$R[visible, , drop = FALSE], sid), error = function(e) e)
    dt <- proc.time()[["elapsed"]] - t0
    if (inherits(fit, "error")) {
      out <- set_status(out, out$status$sample_id == sid, "failed", conditionMessage(fit), dt)
      next
    }
    mat <- extract_exposure(fit)
    if (is.null(mat)) {
      out <- set_status(out, out$status$sample_id == sid, "failed", "no exposure matrix", dt)
      next
    }
    vals <- tryCatch(coerce_exposure_vector(mat, sid, visible), error = function(e) e)
    if (inherits(vals, "error")) {
      out <- set_status(out, out$status$sample_id == sid, "failed", conditionMessage(vals), dt)
      next
    }
    fit_norm <- normalise_fit(vals, keep_ood)
    out$pred[sid, visible] <- fit_norm$vals[visible]
    out$ood[sid, "OOD"] <- fit_norm$ood
    out <- set_status(out, out$status$sample_id == sid, "ok", "", dt)
  }
  write_outputs(out, output_dir, method, nrow(d$X), proc.time()[["elapsed"]] - t_all)
}'''



class CLASS_wrapper_RBase:
    method_name = "R_method"
    display_name = "R method"
    output_name = None
    script_filename = "run_method.R"
    keep_ood = False

    def __init__(self, conda_env: Optional[str] = R_env, work_dir: Optional[str] = None, fail_soft: bool = False):
        self.conda_env = conda_env or os.environ.get("CONDA_DEFAULT_ENV")
        self.work_dir = Path(work_dir or f"./tmp_{self.method_name}_{uuid.uuid4().hex[:10]}")
        self.fail_soft = fail_soft
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        self.script_dir = self.work_dir / "scripts"

    def script_body(self) -> str:
        raise NotImplementedError

    def _write_input(self, df_3nt_raw, df_ref_sig, df_refmask):
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.script_dir.mkdir(parents=True, exist_ok=True)
        df_3nt_raw.to_csv(self.input_dir / "X_count_data.tsv", sep="\t")
        df_ref_sig.to_csv(self.input_dir / "R_refsigpool.tsv", sep="\t")
        df_refmask.astype(int).to_csv(self.input_dir / "ref_mask.tsv", sep="\t")
        (self.script_dir / "benchmark_common.R").write_text(R_COMMON_SCRIPT, encoding="utf-8")
        script_path = self.script_dir / self.script_filename
        script_path.write_text(self.script_body(), encoding="utf-8")
        return script_path

    def _blank_result(self, df_3nt_raw, df_ref_sig):
        compo = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=df_ref_sig.index)
        recon = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=df_3nt_raw.columns)
        ood = pd.DataFrame(0.0, index=df_3nt_raw.index, columns=["OOD"])
        return compo, recon, ood

    def __call__(self, df_3nt_raw, df_ref_sig, df_refmask=None):
        df_refmask = ensure_ref_mask(df_refmask, df_3nt_raw.index, df_ref_sig.index)
        run_hash = _hash_inputs(df_3nt_raw, df_ref_sig, df_refmask.astype(int))[:12]
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir = self.work_dir / f"input_{run_hash}"
        self.output_dir = self.work_dir / f"output_{run_hash}"
        script_path = self._write_input(df_3nt_raw, df_ref_sig, df_refmask)

        if not self.conda_env:
            raise RuntimeError("R_env=None requires CONDA_DEFAULT_ENV to be set, or pass conda_env explicitly")
        cmd_str = (
            "source $HOME/miniconda3/etc/profile.d/conda.sh"
            f" && conda activate {shlex.quote(str(self.conda_env))}"
            f" && Rscript {shlex.quote(str(script_path))}"
            f" --input {shlex.quote(str(self.input_dir))}"
            f" --output {shlex.quote(str(self.output_dir))}"
        )
        cmd = ["bash", "-lc", cmd_str]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        wall = time.perf_counter() - t0

        # The R process, or another process using a shared temporary root, may
        # remove this directory before Python writes diagnostics. Recreate it so the
        # original failure is preserved instead of being masked by FileNotFoundError.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{self.method_name}__cmd.sh").write_text(cmd_str + "\n", encoding="utf-8")
        (self.output_dir / f"{self.method_name}__stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (self.output_dir / f"{self.method_name}__stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        pd.DataFrame([{"method": self.display_name, "python_wall_seconds": wall, "returncode": proc.returncode}]).to_csv(
            self.output_dir / f"{self.method_name}__python_runtime.tsv", sep="\t", index=False)

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-4000:]
            stdout_tail = (proc.stdout or "")[-2000:]
            msg = (
                f"{self.display_name} R wrapper failed with return code {proc.returncode}; "
                f"see {self.output_dir}. stderr_tail:\n{stderr_tail}"
            )
            if stdout_tail.strip():
                msg += f"\nstdout_tail:\n{stdout_tail}"
            if not self.fail_soft:
                raise RuntimeError(msg)
            warnings.warn(msg)
            return self._blank_result(df_3nt_raw, df_ref_sig)

        output_name = self.output_name or self.method_name
        pred_path = self.output_dir / f"{output_name}__pred_compo.tsv"
        ood_path = self.output_dir / f"{output_name}__ood_mass.tsv"
        status_path = self.output_dir / f"{output_name}__status.tsv"
        if status_path.exists():
            status = pd.read_csv(status_path, sep="\t")
            status_values = status.get("status", pd.Series(dtype=str)).astype(str)
            success_states = {"ok", "unexplained"}
            n_success = int(status_values.isin(success_states).sum())
            if n_success == 0:
                message = "; ".join(status.get("message", pd.Series(dtype=str)).dropna().astype(str).unique()[:5])
                msg = f"{self.display_name} produced no successful sample fits; see {status_path}. First messages: {message}"
                if not self.fail_soft:
                    raise RuntimeError(msg)
                warnings.warn(msg)
                return self._blank_result(df_3nt_raw, df_ref_sig)
        if not pred_path.exists() or not ood_path.exists():
            existing = ", ".join(sorted(x.name for x in self.output_dir.glob("*")))
            msg = (
                f"{self.display_name} did not write expected output files in {self.output_dir}. "
                f"Expected {pred_path.name} and {ood_path.name}; existing files: {existing or '<none>'}"
            )
            if not self.fail_soft:
                raise RuntimeError(msg)
            warnings.warn(msg)
            return self._blank_result(df_3nt_raw, df_ref_sig)
        compo = pd.read_csv(pred_path, sep="\t", index_col=0).reindex(index=df_3nt_raw.index, columns=df_ref_sig.index, fill_value=0.0)
        compo = compo.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        ood = pd.read_csv(ood_path, sep="\t", index_col=0).reindex(index=df_3nt_raw.index, columns=["OOD"], fill_value=0.0)
        ood = ood.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if not self.keep_ood:
            compo = SgF_util.sum_scale(compo, axis=1)
            ood.loc[:, "OOD"] = 0.0
        recon = reconstruct_from_compo(compo, df_ref_sig)
        runtime = pd.DataFrame([{"method": self.display_name, "n_samples": df_3nt_raw.shape[0], "seconds": time.perf_counter() - t0}])
        runtime.to_csv(self.output_dir / f"{self.method_name}__python_wall_runtime.tsv", sep="\t", index=False)
        return compo, recon, ood


class CLASS_wrapper_sigfit(CLASS_wrapper_RBase):
    method_name = "sft"
    display_name = "sigfit"
    script_filename = "run_sigfit.R"
    keep_ood = False

    def __init__(self, conda_env: Optional[str] = R_env, work_dir: Optional[str] = None, fail_soft: bool = False,
                 iter: int = 1400, warmup: Optional[int] = None, chains: int = 3,
                 cores: Optional[int] = 1, seed: int = 19970717):
        super().__init__(conda_env=conda_env, work_dir=work_dir, fail_soft=fail_soft)
        self.iter = int(iter)
        self.warmup = int(warmup) if warmup is not None else self.iter // 2
        self.chains = int(chains)
        self.cores = int(cores) if cores is not None else self.chains
        self.seed = int(seed)

    def script_body(self) -> str:
        return f'''
{R_SOURCE_COMMON}
suppressPackageStartupMessages(library(sigfit))
suppressPackageStartupMessages(library(rstan))
suppressPackageStartupMessages(library(parallel))
input_dir <- get_arg("input")
output_dir <- get_arg("output")
options(mc.cores = min({self.cores}, parallel::detectCores()))
rstan_options(auto_write = TRUE)
fit_one <- function(counts, sig, sid) {{
  fit <- sigfit::fit_signatures(counts = counts, signatures = sig,
                                model = "multinomial", refresh = 0,
                                iter = {self.iter}, warmup = {self.warmup},
                                chains = {self.chains}, seed = {self.seed})
  sigfit::retrieve_pars(fit, par = "exposures")$mean
}}
run_per_sample(input_dir, output_dir, "sft", fit_one, keep_ood = FALSE)
'''


class CLASS_wrapper_SigLASSO(CLASS_wrapper_RBase):
    method_name = "sLS"
    display_name = "sigLASSO"
    output_name = "SLS"
    script_filename = "run_siglasso.R"
    keep_ood = True

    def __init__(self, conda_env: Optional[str] = R_env, work_dir: Optional[str] = None, fail_soft: bool = False,
                 conf: float = 0.1, adaptive: bool = True, gamma: float = 1, alpha_min: float = 400,
                 iter_max=None, sd_multiplier: float = 1.0, elastic_net: bool = False, normalize: str = "none"):
        super().__init__(conda_env=conda_env, work_dir=work_dir, fail_soft=fail_soft)
        self.conf = conf
        self.adaptive = adaptive
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.iter_max = iter_max
        self.sd_multiplier = sd_multiplier
        self.elastic_net = elastic_net
        self.normalize = normalize

    def script_body(self) -> str:
        iter_max = "Inf" if self.iter_max is None else int(self.iter_max)
        return f'''
{R_SOURCE_COMMON}
suppressPackageStartupMessages(library(siglasso))
input_dir <- get_arg("input")
output_dir <- get_arg("output")
fit_one <- function(counts, sig, sid) {{
  signature <- t(sig)
  prior <- rep(1, ncol(signature))
  names(prior) <- colnames(signature)
  siglasso::siglasso(sample_spectrum = t(counts),
                     signature = signature, prior = prior,
                     conf = {float(self.conf)}, adaptive = {_r_bool(self.adaptive)}, gamma = {float(self.gamma)},
                     alpha_min = {float(self.alpha_min)}, iter_max = {iter_max}, sd_multiplier = {float(self.sd_multiplier)},
                     elastic_net = {_r_bool(self.elastic_net)}, plot = FALSE, normalize = "{self.normalize}")
}}
run_per_mask_group(input_dir, output_dir, "SLS", fit_one, keep_ood = TRUE)
'''


class CLASS_wrapper_sig_tool_lib(CLASS_wrapper_RBase):
    method_name = "stl"
    display_name = "signature.tools.lib"
    output_name = "sig_tool_lib"
    script_filename = "run_sig_tool_lib.R"
    keep_ood = True

    def __init__(self, conda_env: Optional[str] = R_env, work_dir: Optional[str] = None,
                 nmuts_threshold: int = 300, pvalue_threshold: float = 0.15,
                 pvalue_method: str = "normErrorSAD", nmuts_method: str = "residualSSD",
                 fail_soft: bool = False):
        super().__init__(conda_env=conda_env, work_dir=work_dir, fail_soft=fail_soft)
        self.nmuts_threshold = int(nmuts_threshold)
        self.pvalue_threshold = float(pvalue_threshold)
        self.pvalue_method = str(pvalue_method)
        self.nmuts_method = str(nmuts_method)

    def script_body(self) -> str:
        return f'''
{R_SOURCE_COMMON}
suppressPackageStartupMessages(library(signature.tools.lib))
input_dir <- get_arg("input")
output_dir <- get_arg("output")
quiet <- function(x) suppressMessages(suppressWarnings(x))

ood_from_info <- function(fit, sid, total) {{
  info <- fit$info_samples
  if (is.null(info) || total <= 0 || !("sample" %in% colnames(info))) return(0)

  row <- info[as.character(info$sample) == sid, , drop = FALSE]
  if (nrow(row) != 1 || !("{self.nmuts_method}" %in% colnames(row))) return(0)

  clip01(as.numeric(row[1, "{self.nmuts_method}"]) / total)
}}

status_from_info <- function(fit, sid) {{
  info <- fit$info_samples
  if (is.null(info) || !all(c("sample", "isUnexplainedSample") %in% colnames(info))) return("ok")

  row <- info[as.character(info$sample) == sid, , drop = FALSE]
  if (nrow(row) == 1 && isTRUE(as.logical(row$isUnexplainedSample[1]))) "unexplained" else "ok"
}}

scale_known <- function(vals, oo) {{
  vals <- pmax(as.numeric(vals), 0)
  vals[!is.finite(vals)] <- 0
  s <- sum(vals)
  if (s > 0) vals / s * (1 - oo) else vals
}}

run_stl <- function(input_dir, output_dir) {{
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  d <- read_runner_input(input_dir)
  out <- init_outputs(d$X, d$R, "sig_tool_lib")
  keys <- apply(d$mask, 1, paste, collapse = "")
  t_all <- proc.time()[["elapsed"]]

  for (key in unique(keys)) {{
    idx <- rownames(d$X)[keys == key]
    visible <- visible_signatures(d$mask, d$R, idx[1])

    if (!length(visible)) {{
      out <- set_status(out, out$status$sample_id %in% idx,
                        "skipped", "no visible signatures", 0)
      next
    }}

    t0 <- proc.time()[["elapsed"]]
    fit <- tryCatch(quiet(signature.tools.lib::unexplainedSamples(
      outfileRoot = NULL,
      catalogues = t(d$X[idx, , drop = FALSE]),
      sigs = t(d$R[visible, , drop = FALSE]),
      nmuts_threshold = {self.nmuts_threshold},
      pvalue_threshold = {self.pvalue_threshold},
      pvalueMethod = "{self.pvalue_method}",
      nmutsMethod = "{self.nmuts_method}"
    )), error = function(e) e)
    dt <- proc.time()[["elapsed"]] - t0

    if (inherits(fit, "error") || is.null(fit$exposures)) {{
      msg <- if (inherits(fit, "error")) conditionMessage(fit) else "no exposure matrix"
      out <- set_status(out, out$status$sample_id %in% idx, "failed", msg, dt)
      next
    }}

    mat <- extract_exposure(fit$exposures)

    for (sid in idx) {{
      vals <- tryCatch(coerce_exposure_vector(mat, sid, visible), error = function(e) e)
      if (inherits(vals, "error")) {{
        out <- set_status(out, out$status$sample_id == sid, "failed", conditionMessage(vals), dt)
        next
      }}

      oo <- ood_from_info(fit, sid, sum(as.numeric(d$X[sid, ]), na.rm = TRUE))
      out$pred[sid, visible] <- scale_known(vals[visible], oo)
      out$ood[sid, "OOD"] <- oo
      out <- set_status(out, out$status$sample_id == sid, status_from_info(fit, sid), "", dt)
    }}
  }}

  write_outputs(out, output_dir, "sig_tool_lib", nrow(d$X), proc.time()[["elapsed"]] - t_all)
}}

run_stl(input_dir, output_dir)
'''

####################################################################################################
# Single-sample linear refinement
####################################################################################################

class CLASS_wrapper_linear_refinement:
    """Conservative single-sample linear relaxation initialized by model attribution.

    Counts and normalized SBS96 profiles are both accepted. Candidate signatures are
    restricted by ``df_refmask`` when supplied, and every proposed change must improve
    reconstruction relative to the raw attribution. The trust region prevents the
    optimizer from replacing the model result with an unrelated NNLS solution.
    """

    def __init__(self, df_ref_sig=None, hard_composition=0.005, hard_count=3,
                 prior_strength=12.0, max_rescue=2,
                 candidate_top_k=6, max_l1_change=0.20, max_ood_change=0.15,
                 max_ood_increase=0.005, rescue_min_composition=0.01, n_channel_folds=4,
                 normalized_effective_depth=1000, random_state=717,
                 version=None, **legacy_parameters):
        self.hard_composition = float(hard_composition)
        self.hard_count = int(hard_count)
        self.prior_strength = float(prior_strength)
        self.max_rescue = int(max_rescue)
        self.candidate_top_k = int(candidate_top_k)
        self.max_l1_change = float(max_l1_change)
        self.max_ood_change = float(max_ood_change)
        self.max_ood_increase = float(max_ood_increase)
        self.rescue_min_composition = float(rescue_min_composition)
        self.n_channel_folds = int(n_channel_folds)
        self.normalized_effective_depth = float(normalized_effective_depth)
        self.random_state = int(random_state)
        self.df_ref_sig = None
        self.last_diagnostics = pd.DataFrame()
        self.last_history = []
        if df_ref_sig is not None:
            self.set_reference(df_ref_sig)

    def set_reference(self, df_ref_sig):
        ref = df_ref_sig.copy().astype(float).clip(lower=0.0)
        self.df_ref_sig = ref.div(ref.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        self.signature_names = self.df_ref_sig.index
        self.context_names = self.df_ref_sig.columns
        self.R = self.df_ref_sig.to_numpy(float)
        self.R_norm = np.maximum(np.linalg.norm(self.R, axis=1), 1e-12)
        rng = np.random.default_rng(self.random_state)
        self.channel_folds = np.arange(self.R.shape[1]) % self.n_channel_folds
        for block in np.array_split(np.arange(self.R.shape[1]), 6):
            self.channel_folds[block] = rng.permutation(self.channel_folds[block])
        return self

    @staticmethod
    def _normalize(values):
        values = np.clip(np.asarray(values, dtype=float), 0.0, None)
        return values / values.sum() if values.sum() > 0 else values

    @staticmethod
    def _cosine(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

    @staticmethod
    def _loss(profile, reconstruction):
        return float(np.mean((profile - reconstruction) ** 2 / (profile + 1 / 96)))

    def _fit(self, profile, prior, active, known_mass, channels=None,
             effective_depth=1000.0, allow_mass_change=False):
        active = np.asarray(sorted(set(map(int, active))), dtype=int)
        if not len(active):
            return np.zeros_like(prior)
        use = np.ones(profile.size, dtype=bool) if channels is None else np.asarray(channels, dtype=bool)
        p = prior[active]
        order = np.argsort(p)[::-1]
        dominant = np.zeros(len(active), dtype=bool)
        target, running = 0.80 * max(known_mass, 1e-12), 0.0
        for position in order:
            if p[position] >= 0.05 and running < target:
                dominant[position] = True
                running += p[position]
        if len(order):
            dominant[order[0]] = True

        lower = np.zeros(len(active))
        upper = np.ones(len(active))
        for position, value in enumerate(p):
            if value <= 1e-12:
                upper[position] = 0.20
            elif dominant[position]:
                lower[position] = max(0.01, 0.35 * value, value - 0.25)
                upper[position] = min(1.0, value + 0.25)
            else:
                upper[position] = min(0.35, max(0.05, 2.5 * value + 0.03))
        if lower.sum() >= 1.0:
            lower *= 0.95 / lower.sum()

        scales = np.maximum(0.02, 0.35 * np.maximum(p, 0.01))
        depth = float(np.clip(effective_depth, 200, 5000))
        base_weight = 1.0 / np.sqrt(profile[use] + 1.0 / depth)
        weight = base_weight.copy()
        solution = np.clip(p, lower, upper)
        mass_strength = max(2.0, 0.25 * self.prior_strength) * (0.25 if allow_mass_change else 1.0)
        reference = self.R[active][:, use]
        for _ in range(3):
            matrix = np.vstack([
                np.sqrt(depth) * weight[:, None] * reference.T,
                np.diag(np.sqrt(self.prior_strength) / scales),
                np.sqrt(mass_strength) * np.ones((1, len(active))),
            ])
            response = np.r_[
                np.sqrt(depth) * weight * profile[use],
                np.sqrt(self.prior_strength) * p / scales,
                np.sqrt(mass_strength) * known_mass,
            ]
            try:
                solution = lsq_linear(matrix, response, bounds=(lower, upper), method="trf",
                                      lsmr_tol="auto", max_iter=200).x
            except Exception:
                solution = np.clip(p, lower, upper)
            if solution.sum() > 1.0:
                room = solution - lower
                if room.sum() > 1e-12:
                    solution -= (solution.sum() - 1.0) * room / room.sum()
                solution = np.maximum(solution, lower)
            fitted = solution @ reference
            weight = base_weight * np.where(fitted > profile[use], np.sqrt(2.0), 1.0)

        delta = self.max_ood_change if allow_mass_change else min(0.02, self.max_ood_change)
        low, high = max(0.0, known_mass - delta), min(1.0, known_mass + delta)
        if solution.sum() > high:
            solution *= high / solution.sum()
        elif 0 < solution.sum() < low:
            solution *= low / solution.sum()
        out = np.zeros_like(prior)
        out[active] = solution
        return out


    def _candidate_pool(self, profile, prior, allowed):
        residual = np.clip(profile - prior @ self.R, 0.0, None)
        norm = np.linalg.norm(residual)
        if norm <= 1e-12:
            return []
        scores = self.R @ residual / (self.R_norm * norm + 1e-12)
        candidates = np.flatnonzero((prior <= self.hard_composition) & allowed)
        return candidates[np.argsort(scores[candidates])[::-1][:self.candidate_top_k]].tolist()

    def _cv_gain(self, profile, prior, active, effective_depth, allow_mass_change):
        gains = []
        baseline = prior @ self.R
        for fold in range(self.n_channel_folds):
            test = self.channel_folds == fold
            fitted = self._fit(profile, prior, active, prior.sum(), ~test,
                               effective_depth, allow_mass_change)
            gains.append(self._loss(profile[test], baseline[test]) -
                         self._loss(profile[test], (fitted @ self.R)[test]))
        return np.asarray(gains)

    def refine_sample(self, values, initial_composition, allowed=None):
        values = np.asarray(values, dtype=float)
        total = float(values.sum())
        profile = self._normalize(values)
        prior = np.clip(np.asarray(initial_composition, dtype=float), 0.0, None)
        if prior.sum() > 1.0:
            prior /= prior.sum()
        allowed = np.ones(len(prior), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
        prior[~allowed] = 0.0
        known_mass = float(prior.sum())
        normalized_input = total <= 2.0
        depth = None if normalized_input else total
        effective_depth = self.normalized_effective_depth if normalized_input else total
        threshold = self.hard_composition if depth is None else max(self.hard_composition, self.hard_count / max(depth, 1.0))
        active = np.flatnonzero((prior >= threshold) & allowed).tolist()
        if prior.max(initial=0.0) > 0:
            active.append(int(np.argmax(prior)))
        baseline_loss = self._loss(profile, prior @ self.R)
        for j in np.flatnonzero((prior > 0) & allowed & (prior < threshold)):
            trial = prior.copy()
            trial[j] = 0.0
            if self._loss(profile, trial @ self.R) > baseline_loss * 1.005 + 1e-8:
                active.append(int(j))
        active = sorted(set(active))
        allow_mass_relaxation = (1.0 - known_mass) >= 0.08
        if allow_mass_relaxation:
            mass_cv = self._cv_gain(profile, prior, active, effective_depth, True)
            allow_mass_relaxation = bool(np.mean(mass_cv > 0) >= 0.75 and mass_cv.mean() > 0)
        fitted = self._fit(profile, prior, active, known_mass,
                           effective_depth=effective_depth,
                           allow_mass_change=allow_mass_relaxation)
        fitted_loss = self._loss(profile, fitted @ self.R)
        rescued = []
        candidates = self._candidate_pool(profile, prior, allowed)
        for _ in range(self.max_rescue):
            trials = []
            for candidate in candidates:
                if candidate in active:
                    continue
                trial = self._fit(profile, prior, active + [candidate], known_mass,
                                  effective_depth=effective_depth, allow_mass_change=True)
                gain = fitted_loss - self._loss(profile, trial @ self.R)
                if trial[candidate] >= max(self.rescue_min_composition, threshold):
                    trials.append((gain, candidate, trial))
            if not trials:
                break
            gain, candidate, trial = max(trials, key=lambda item: item[0])
            if gain <= max(1e-8, 0.002 * baseline_loss):
                break
            cv_gain = self._cv_gain(profile, prior, active + [candidate], effective_depth, True)
            if np.mean(cv_gain > 0) < 0.75 or cv_gain.mean() <= 0:
                break
            active.append(candidate)
            rescued.append(candidate)
            fitted, fitted_loss = trial, self._loss(profile, trial @ self.R)
        cosine_before = self._cosine(profile, prior @ self.R)
        cosine_fit = self._cosine(profile, fitted @ self.R)
        core_cv = self._cv_gain(profile, prior, active, effective_depth, allow_mass_relaxation)
        accepted = (fitted_loss < baseline_loss and cosine_fit + 1e-10 >= cosine_before
                    and np.mean(core_cv > 0) >= 0.75 and core_cv.mean() > 0)
        if accepted:
            relaxation = float(np.clip((0.995 - cosine_before) / 0.005, 0.0, 1.0))
            result = prior + relaxation * (fitted - prior)
            l1 = float(np.abs(result - prior).sum())
            if l1 > self.max_l1_change:
                result = prior + (result - prior) * self.max_l1_change / l1
            min_known = max(0.0, known_mass - self.max_ood_increase)
            if 0 < result.sum() < min_known:
                result *= min_known / result.sum()
            if result.sum() > 1.0:
                result /= result.sum()
            if (self._loss(profile, result @ self.R) > baseline_loss + 1e-12 or
                    self._cosine(profile, result @ self.R) + 1e-10 < cosine_before):
                result, accepted = prior, False
        else:
            result = prior
        result = np.clip(result, 0.0, None)
        before_sub1 = (prior > 0) & (prior < 0.01)
        after_sub1 = (result > 0) & (result < 0.01)
        before_1to5 = (prior >= 0.01) & (prior < 0.05)
        after_1to5 = (result >= 0.01) & (result < 0.05)
        history = {
            "input_kind": "normalized" if normalized_input else "counts",
            "input_total": total, "threshold": threshold,
            "rescued": [str(self.signature_names[i]) for i in rescued],
            "accepted": bool(accepted), "cosine_before": cosine_before,
            "cosine_after": self._cosine(profile, result @ self.R),
            "cv_gain": float(core_cv.mean()),
            "cv_positive_fraction": float(np.mean(core_cv > 0)),
            "l1_change": float(np.abs(result - prior).sum()),
            "known_mass_before": float(prior.sum()),
            "known_mass_after": float(result.sum()),
            "ood_before": float(max(0.0, 1.0 - prior.sum())),
            "ood_after": float(max(0.0, 1.0 - result.sum())),
            "ood_change": float(prior.sum() - result.sum()),
            "sub1_count_before": int(before_sub1.sum()),
            "sub1_count_after": int(after_sub1.sum()),
            "sub1_mass_before": float(prior[before_sub1].sum()),
            "sub1_mass_after": float(result[after_sub1].sum()),
            "one_to_five_count_before": int(before_1to5.sum()),
            "one_to_five_count_after": int(after_1to5.sum()),
            "one_to_five_mass_before": float(prior[before_1to5].sum()),
            "one_to_five_mass_after": float(result[after_1to5].sum()),
            "minor_removed_mass": float(np.clip(prior[prior < 0.05] - result[prior < 0.05], 0, None).sum()),
            "minor_added_mass": float(np.clip(result[prior < 0.05] - prior[prior < 0.05], 0, None).sum()),
        }
        return result, history

    def __call__(self, df_3nt_raw, df_initial_composition, df_ref_sig=None, df_refmask=None):
        if df_ref_sig is not None and (self.df_ref_sig is None or not self.df_ref_sig.equals(df_ref_sig)):
            self.set_reference(df_ref_sig)
        if self.df_ref_sig is None:
            raise ValueError("A reference matrix is required at initialization or call time.")
        missing = self.context_names.difference(df_3nt_raw.columns)
        if len(missing):
            raise KeyError(f"Input matrix is missing {len(missing)} reference contexts; first={list(missing[:5])}")
        raw = df_3nt_raw.loc[:, self.context_names].astype(float)
        initial = df_initial_composition.reindex(index=raw.index, columns=self.signature_names, fill_value=0.0).astype(float)
        mask = ensure_ref_mask(df_refmask, raw.index, self.signature_names)
        rows, history = [], []
        for sample in raw.index:
            result, item = self.refine_sample(raw.loc[sample], initial.loc[sample], mask.loc[sample])
            rows.append(result)
            history.append({"sample_id": sample, **item})
        composition = pd.DataFrame(rows, index=raw.index, columns=self.signature_names)
        reconstruction = reconstruct_from_compo(composition, self.df_ref_sig)
        ood = pd.DataFrame({"OOD": (1.0 - composition.sum(axis=1)).clip(lower=0.0)}, index=raw.index)
        self.last_history = history
        self.last_diagnostics = pd.DataFrame(history).set_index("sample_id")
        return composition, reconstruction, ood
