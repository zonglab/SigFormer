#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-only utilities for SigFormer.

This module contains curriculum configuration, reference-bank construction,
training/evaluation batch generation, metrics, logging, and timing helpers.
It does not configure global plotting or warning behavior at import time.
"""

from __future__ import annotations

import os, sys, copy, json, math, time, shutil, argparse, contextlib, pickle, random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import datetime as _dt
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import s02_Data as data_v9

SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = SCRIPT_DIR.parent / "resource"


# ============================================================================
# General training helpers
# ============================================================================
def str2bool(x: Any) -> bool:
    """Parse shell-friendly booleans."""
    if isinstance(x, bool):
        return x
    sx = str(x).strip().lower()
    if sx in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if sx in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot parse boolean from {x!r}")

def current_stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def seed_everything(seed: int) -> None:
    """Make the run reasonably repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def copy_run_sources(out_scripts: Path, path_log: Optional[str] = None) -> None:
    """Snapshot all files needed to reproduce this training run."""
    ensure_dir(out_scripts)
    names = [
        "s01_Core.py",
        "s02_Data.py",
        "s03_Util_train.py",
        "s04_Util_apply.py",
        "s05_Train.py",
    ]
    for name in names:
        src = SCRIPT_DIR / name
        if src.exists():
            shutil.copy2(src, out_scripts / name)
    for src in sorted(RESOURCE_DIR.glob("COSMIC_v*.txt")):
        shutil.copy2(src, out_scripts / src.name)
    print_log(f"source snapshot copied to {out_scripts}", path_log, print_time=True)

def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def model_without_parallel(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def should_use_data_parallel(flag: Any, device: torch.device) -> Tuple[bool, str]:
    """Return whether DataParallel should be enabled for the current device."""
    want_dp = str2bool(flag)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not want_dp:
        return False, f"disabled by --use_data_parallel={flag}; visible_cuda_devices={n_gpu}"
    if device.type != "cuda":
        return False, f"device={device}; DataParallel requires cuda"
    if n_gpu <= 1:
        return False, f"visible_cuda_devices={n_gpu}; single-GPU run skips DataParallel"
    return True, f"requested=True; visible_cuda_devices={n_gpu}"

def model_parameter_summary(model: nn.Module) -> pd.DataFrame:
    """Count trainable/non-trainable parameters and approximate size."""
    rows = []
    for name, par in model_without_parallel(model).named_parameters():
        n = int(par.numel())
        rows.append({
            "name": name,
            "shape": "x".join(map(str, par.shape)),
            "trainable": bool(par.requires_grad),
            "n_parameters": n,
            "size_MB_fp32": n * 4 / (1024 ** 2),
        })
    return pd.DataFrame(rows)

class StepTimer:
    """Lightweight timing recorder for training and evaluation stages."""

    def __init__(self, enabled: bool, out_dir: Path, path_log: str, device: Optional[torch.device] = None, batch_every: int = 1):
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.path_log = path_log
        self.device = device
        self.batch_every = max(1, int(batch_every))
        self.rows: List[Dict[str, Any]] = []
        self.path_tsv = self.out_dir / "timing_profile.tsv"

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def _sync(self) -> None:
        if self.enabled and self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def should_record(self, batch: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if batch is None:
            return True
        b = int(batch)
        return b == 1 or b % self.batch_every == 0

    @contextlib.contextmanager
    def block(
        self,
        script: str,
        function: str,
        stage: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        mode: str = "",
        **extra: Any,
    ):
        if not self.should_record(batch):
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            self._sync()
            sec = time.perf_counter() - t0
            row = {
                "wall_time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": -1 if epoch is None else int(epoch),
                "batch": -1 if batch is None else int(batch),
                "mode": str(mode),
                "script": script,
                "function": function,
                "stage": stage,
                "seconds": float(sec),
                "ok": bool(ok),
            }
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None:
                    row[k] = v
                else:
                    row[k] = str(v)
            self.rows.append(row)

    def flush(self) -> None:
        if not self.enabled or not self.rows:
            return
        df = pd.DataFrame(self.rows)
        self.path_tsv.parent.mkdir(parents=True, exist_ok=True)
        if self.path_tsv.exists():
            df.to_csv(self.path_tsv, sep="	", mode="a", header=False, index=False)
        else:
            df.to_csv(self.path_tsv, sep="	", index=False)
        self.rows.clear()

    def log_epoch_summary(self, epoch: int, top_n: int = 12) -> None:
        if not self.enabled:
            return
        frames = []
        if self.path_tsv.exists():
            try:
                frames.append(pd.read_csv(self.path_tsv, sep="	"))
            except Exception:
                pass
        if self.rows:
            frames.append(pd.DataFrame(self.rows))
        if not frames:
            return
        df = pd.concat(frames, axis=0, ignore_index=True)
        df = df[df["epoch"].eq(int(epoch))]
        if df.empty:
            return
        grp = (df.groupby(["script", "function", "stage"], dropna=False)["seconds"]
                 .agg(["count", "sum", "mean", "max"])
                 .reset_index()
                 .sort_values("sum", ascending=False))
        path_summary = self.out_dir / f"timing_profile_epoch_{int(epoch):04d}_summary.tsv"
        grp.to_csv(path_summary, sep="	", index=False)
        print_log(f"timing profile epoch={int(epoch):04d} top stages:", self.path_log, print_time=True)
        for _, r in grp.head(int(top_n)).iterrows():
            print_log(
                f"  {r['sum']:9.3f}s total | {r['mean']:8.4f}s mean × {int(r['count']):5d} | {r['script']}::{r['function']}::{r['stage']}",
                self.path_log,
                print_time=False,
            )

@dataclass
class CurriculumConfig:
    """Epoch-dependent generator and optimizer schedule."""
    ep_total: int = 500
    bch_per_ep: int = 1000
    bch_size: int = 64
    lr_base: float = 4e-4
    lr_warm_ep: int = 5
    lr_cool_ep: int = 150
    lmda_compo: float = 3.0
    lmda_recon: float = 1.0
    lmda_ood: float = 0.20
    ood_clean_ep: int = 40
    ood_ramp_end_ep: int = 180
    ood_stable_rate: float = 0.50
    pcOOD_cosmic_fraction: float = 0.20
    ood_min_compo: float = 0.05
    perturb_basis: bool = True
    basis_perturb_conc: float = 50000.0
    basis_perturb_mix: float = 0.020
    basis_perturb_mix_denovo: Optional[float] = None
    n_cosmic_per_batch: int = 65
    n_seen_denovo_per_batch: int = 65

class curriculum:
    """Callable training curriculum.

    The schedule keeps the first 60 epochs clean of sample-level OOD, then adds
    OOD pressure, then cools down with a stable nonzero OOD rate. The training
    objective remains COSMIC-centered, while modest reference perturbation keeps
    the model from memorizing a single matrix with the devotion of a spreadsheet.
    """

    def __init__(
        self,
        ep_total: int = 500,
        bch_per_ep: int = 1000,
        bch_size: int = 64,
        lr_base: float = 4e-4,
        lr_warm_ep: int = 5,
        lr_cool_ep: int = 150,
        lmda_compo: float = 3.0,
        lmda_recon: float = 1.0,
        lmda_ood: float = 0.20,
        ood_clean_ep: int = 40,
        ood_ramp_end_ep: int = 180,
        ood_stable_rate: float = 0.50,
        pcOOD_cosmic_fraction: float = 0.20,
        ood_min_compo: float = 0.05,
        perturb_basis: bool = True,
        basis_perturb_conc: float = 50000.0,
        basis_perturb_mix: float = 0.020,
        basis_perturb_mix_denovo: Optional[float] = None,
        n_cosmic_per_batch: int = 65,
        n_seen_denovo_per_batch: int = 65,
    ):
        self.cfg = CurriculumConfig(
            ep_total=ep_total,
            bch_per_ep=bch_per_ep,
            bch_size=bch_size,
            lr_base=lr_base,
            lr_warm_ep=lr_warm_ep,
            lr_cool_ep=lr_cool_ep,
            lmda_compo=lmda_compo,
            lmda_recon=lmda_recon,
            lmda_ood=lmda_ood,
            ood_clean_ep=ood_clean_ep,
            ood_ramp_end_ep=ood_ramp_end_ep,
            ood_stable_rate=ood_stable_rate,
            pcOOD_cosmic_fraction=pcOOD_cosmic_fraction,
            ood_min_compo=ood_min_compo,
            perturb_basis=perturb_basis,
            basis_perturb_conc=basis_perturb_conc,
            basis_perturb_mix=basis_perturb_mix,
            basis_perturb_mix_denovo=basis_perturb_mix_denovo,
            n_cosmic_per_batch=int(n_cosmic_per_batch),
            n_seen_denovo_per_batch=int(n_seen_denovo_per_batch),
        )

    def ood_ratio(self, epoch: int) -> float:
        """OOD curriculum: clean start, ramp, then stable nonzero OOD pressure."""
        ep = int(epoch)
        clean = int(self.cfg.ood_clean_ep)
        ramp_end = max(clean + 1, int(self.cfg.ood_ramp_end_ep))
        stable = float(self.cfg.ood_stable_rate)
        if ep <= clean:
            return 0.0
        if ep <= ramp_end:
            return stable * (ep - clean) / max(ramp_end - clean, 1)
        return stable

    def lr(self, epoch: int) -> float:
        ep = int(epoch)
        if ep <= max(self.cfg.lr_warm_ep, 1):
            return self.cfg.lr_base * ep / max(self.cfg.lr_warm_ep, 1)
        cool_start = max(1, self.cfg.ep_total - self.cfg.lr_cool_ep + 1)
        if ep < cool_start:
            return self.cfg.lr_base
        t = (ep - cool_start) / max(self.cfg.lr_cool_ep - 1, 1)
        return self.cfg.lr_base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0))))

    def __call__(self, epoch: int) -> Dict[str, Any]:
        cfg = copy.deepcopy(data_v9.DEFAULT_CONFIG_SINGLE)
        cfg["BSize"] = int(self.cfg.bch_size)
        cfg["pcOOD"] = float(self.ood_ratio(epoch))
        cfg["pcOOD_cosmic_fraction"] = float(self.cfg.pcOOD_cosmic_fraction)
        cfg["ood_min_compo"] = float(self.cfg.ood_min_compo)
        # Keep all bins available for stochastic training. Fixed eval configs
        # below will isolate depth/noise combinations.
        cfg["DEPTH"] = {"100-400": 1, "401-2000": 1, "2001-7000": 1, "7000-100000": 1}
        cfg["NOISE"] = {"0.85-0.90": 140, "0.90-0.95": 240, "0.95-1.00": 1800}
        cfg["ACTVE"] = {"1-3": 1, "4-6": 2, "7-10": 1, "11-20": 1}
        cfg["PRIOR"] = {"4-6": 1, "7-10": 1, "11-20": 1, "21-40": 1, "41-60": 1, "61-120": 1, "130": 1}
        cfg["COMPO"] = {0.1: 1, 1.0: 2, 10.0: 1}
        cfg["n_cosmic_per_batch"] = int(self.cfg.n_cosmic_per_batch)
        cfg["n_seen_denovo_per_batch"] = int(self.cfg.n_seen_denovo_per_batch)
        return {
            "epoch": int(epoch),
            "lr": float(self.lr(epoch)),
            "data_config": cfg,
            "lmda_compo": float(self.cfg.lmda_compo),
            "lmda_recon": float(self.cfg.lmda_recon),
            "lmda_ood": float(self.cfg.lmda_ood),
            "perturb_basis": bool(self.cfg.perturb_basis),
            "basis_perturb_conc": float(self.cfg.basis_perturb_conc),
            "basis_perturb_mix": float(self.cfg.basis_perturb_mix),
            "basis_perturb_mix_denovo": None if self.cfg.basis_perturb_mix_denovo is None else float(self.cfg.basis_perturb_mix_denovo),
        }

def locate_default_cosmic() -> Path:
    """Find the newest bundled hg38 COSMIC reference in the package resources."""
    roots = [RESOURCE_DIR, SCRIPT_DIR]
    candidates: List[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("COSMIC_v*_SBS_GRCh38.txt")))
            candidates.extend(sorted(root.glob("COSMIC_v*_SBS_hg38.txt")))
    if candidates:
        return sorted(candidates, key=lambda p: p.name, reverse=True)[0]
    raise FileNotFoundError("No bundled COSMIC_v*_SBS_GRCh38.txt reference was found.")

def parse_cosmic_paths(args: argparse.Namespace) -> List[Path]:
    """Parse --cosmic_paths/--cosmic_path with optional 'all' bundled mode."""
    raw = str(getattr(args, "cosmic_paths", "") or "").strip()
    if raw.lower() == "all":
        return sorted(RESOURCE_DIR.glob("COSMIC_v*.txt"), reverse=True)
    if raw:
        return [Path(x.strip()) for x in raw.split(",") if x.strip()]
    if str(getattr(args, "cosmic_path", "") or "").strip():
        return [Path(args.cosmic_path)]
    return [locate_default_cosmic()]

def build_cosmic_bank_from_paths(paths: Sequence[Path], path_log: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read one/many COSMIC matrices and keep unique signature names."""
    frames: List[pd.DataFrame] = []
    source_rows: List[Dict[str, str]] = []
    seen: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = RESOURCE_DIR / p
        print_log(f"reading COSMIC source: {p}", path_log, print_time=True)
        df = data_v9.standardize_refsig(data_v9.get_COSMIC(path=str(p)))
        keep = [idx for idx in df.index if str(idx) not in seen]
        skipped = df.shape[0] - len(keep)
        if skipped:
            print_log(f"  skipped duplicate COSMIC signatures from {p.name}: {skipped}", path_log, print_time=True)
        if keep:
            frames.append(df.loc[keep])
            for idx in keep:
                seen.add(str(idx))
                source_rows.append({"signature": str(idx), "cosmic_source_file": p.name})
    if not frames:
        raise RuntimeError("No COSMIC signatures were loaded after duplicate removal.")
    R_cosmic = data_v9.standardize_refsig(pd.concat(frames, axis=0))
    M_cosmic = data_v9.summarize_refsig(R_cosmic)
    M_cosmic["ref_type"] = "COSMIC"
    M_cosmic["ref_role"] = "cosmic"
    M_cosmic["is_cosmic"] = True
    M_cosmic["is_mock"] = False
    M_cosmic["is_seen_denovo"] = False
    M_cosmic["is_leaveout_denovo"] = False
    src = pd.DataFrame(source_rows).drop_duplicates("signature").set_index("signature")
    M_cosmic = M_cosmic.join(src, how="left")
    return R_cosmic, M_cosmic

def perturb_reference_basis(
    R: pd.DataFrame,
    rng: np.random.Generator,
    concentration: float = 50000.0,
    mix: float = 0.020,
    M_ref: Optional[pd.DataFrame] = None,
    mix_denovo: Optional[float] = None,
) -> pd.DataFrame:
    """Perturb only the sampled reference pool rather than the full reference bank."""
    mix_cosmic = float(mix)
    mix_denovo_val = mix_cosmic if mix_denovo is None else float(mix_denovo)
    if max(mix_cosmic, mix_denovo_val) <= 0:
        return R.copy()
    arr = data_v9.normalize_rows(R.to_numpy(dtype=float), axis=1)
    out = np.empty_like(arr)
    if M_ref is not None and M_ref.shape[0] == R.shape[0]:
        M_use = M_ref.reindex(R.index)
        denovo_mask = np.zeros(R.shape[0], dtype=bool)
        for col in ["is_mock", "is_seen_denovo", "is_leaveout_denovo"]:
            if col in M_use.columns:
                denovo_mask |= M_use[col].fillna(False).astype(bool).to_numpy()
        mix_vec = np.where(denovo_mask, mix_denovo_val, mix_cosmic)
    else:
        mix_vec = np.full(R.shape[0], mix_cosmic, dtype=float)
    for i in range(arr.shape[0]):
        row_mix = float(mix_vec[i])
        if row_mix <= 0:
            out[i] = arr[i]
            continue
        alpha = np.maximum(arr[i] * float(concentration), 1e-8)
        noisy = rng.dirichlet(alpha)
        out[i] = data_v9.normalize_vec((1.0 - row_mix) * arr[i] + row_mix * noisy)
    return pd.DataFrame(out, index=R.index.copy(), columns=R.columns.copy())

def build_ref_bank(args: argparse.Namespace, out_dir: Path, path_log: str, timer: Optional[StepTimer] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read COSMIC, build train-seen mock de novo, and reserve leaveout de novo."""
    cosmic_paths = parse_cosmic_paths(args)
    with (timer.block("s05_Train.py", "build_ref_bank", "read_cosmic") if timer else contextlib.nullcontext()):
        R_cosmic, M_cosmic = build_cosmic_bank_from_paths(cosmic_paths, path_log)

    n_seen = int(args.n_mock)
    n_leaveout = int(args.n_mock_leaveout)
    n_total_mock = max(0, n_seen + n_leaveout)
    frames = [R_cosmic]
    metas = [M_cosmic]
    if n_total_mock > 0:
        print_log(f"building mock de novo signatures: seen={n_seen}, leaveout={n_leaveout}", path_log, print_time=True)
        with (timer.block("s02_Data.py", "build_mock_denovo_signatures", "build_mock_bank", n_total_mock=n_total_mock) if timer else contextlib.nullcontext()):
            R_mock, M_mock = data_v9.build_mock_denovo_signatures(
                R_cosmic,
                n_mock=n_total_mock,
                random_state=int(args.seed) + 17,
                verbose=False,
                max_trials=int(args.mock_max_trials),
                combo_bank_size=int(args.mock_combo_bank_size),
                cosine_max=float(args.mock_cosine_max),
                combo_cosine_max=float(args.mock_combo_cosine_max),
            )
        roles = ["seen_denovo"] * n_seen + ["leaveout_denovo"] * n_leaveout
        M_mock = M_mock.copy()
        M_mock["ref_type"] = ["mock_denovo_seen" if r == "seen_denovo" else "mock_denovo_leaveout" for r in roles]
        M_mock["ref_role"] = roles
        M_mock["is_cosmic"] = False
        M_mock["is_mock"] = True
        M_mock["is_seen_denovo"] = [r == "seen_denovo" for r in roles]
        M_mock["is_leaveout_denovo"] = [r == "leaveout_denovo" for r in roles]
        M_mock["cosmic_source_file"] = "mock_generated"
        frames.append(R_mock)
        metas.append(M_mock)
    R = data_v9.standardize_refsig(pd.concat(frames, axis=0))
    M = pd.concat(metas, axis=0).reindex(R.index).copy()
    M["ref_index"] = np.arange(R.shape[0])
    save_pickle({"R_grand": R, "M_grand": M, "cosmic_paths": [str(p) for p in cosmic_paths]}, out_dir / "ref_bank.pkl")
    R.to_csv(out_dir / "ref_bank__R_grand.tsv", sep="\t")
    M.to_csv(out_dir / "ref_bank__M_grand.tsv", sep="\t")
    print_log(
        f"ref bank ready: n_ref={R.shape[0]}, cosmic={int(M['is_cosmic'].sum())}, "
        f"seen_denovo={int(M['is_seen_denovo'].sum())}, leaveout_denovo={int(M['is_leaveout_denovo'].sum())}",
        path_log,
        print_time=True,
    )
    return R, M

def training_ref_bank(R_grand: pd.DataFrame, M_grand: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove leaveout de novo from training; keep COSMIC + train-seen de novo."""
    keep = ~M_grand.get("is_leaveout_denovo", pd.Series(False, index=M_grand.index)).astype(bool)
    return R_grand.loc[keep].copy(), M_grand.loc[keep].copy()

def select_batch_refpool_v9(
    R_grand: pd.DataFrame,
    M_grand: pd.DataFrame,
    rng: np.random.Generator,
    n_cosmic: int = 65,
    n_seen_denovo: int = 65,
    include_leaveout: bool = False,
    forced_role: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int], str]:
    """Batch pool sampler with optional forced OOD source role."""
    M = M_grand.copy()
    for col in ["is_cosmic", "is_seen_denovo", "is_leaveout_denovo"]:
        if col not in M.columns:
            M[col] = False
    forced_name: Optional[str] = None
    forced_role = str(forced_role or "").strip()
    if forced_role in {"cosmic", "seen_denovo", "leaveout_denovo"}:
        role_col = {"cosmic": "is_cosmic", "seen_denovo": "is_seen_denovo", "leaveout_denovo": "is_leaveout_denovo"}[forced_role]
        candidates = M.index[M[role_col].astype(bool)].to_numpy()
        if len(candidates) > 0:
            forced_name = str(rng.choice(candidates))
    cosmic = M.index[M["is_cosmic"].astype(bool)].to_numpy()
    seen = M.index[M["is_seen_denovo"].astype(bool)].to_numpy()
    leave = M.index[M["is_leaveout_denovo"].astype(bool)].to_numpy()
    take_cos = list(rng.choice(cosmic, size=min(n_cosmic, len(cosmic)), replace=False)) if len(cosmic) else []
    take_seen = list(rng.choice(seen, size=min(n_seen_denovo, len(seen)), replace=False)) if len(seen) else []
    take_leave: List[str] = []
    if include_leaveout or forced_role == "leaveout_denovo":
        if forced_name and forced_role == "leaveout_denovo":
            take_leave = [forced_name]
        elif len(leave):
            take_leave = [str(rng.choice(leave))]
    names = list(dict.fromkeys([str(x) for x in take_cos + take_seen + take_leave]))
    if forced_name is not None and forced_name not in names:
        # Replace a same-role random member when possible; otherwise append.
        names.append(forced_name)
    rng.shuffle(names)
    R = data_v9.standardize_refsig(R_grand.loc[names])
    M_batch = M_grand.reindex(names).copy()
    M_batch["batch_ref_index"] = np.arange(len(names))
    forced_idx = names.index(forced_name) if forced_name in names else None
    actual_role = forced_role if forced_idx is not None else "none"
    return R, M_batch, forced_idx, actual_role

def choose_train_ood_source(rng: np.random.Generator, pc_ood: float, p_cosmic: float = 0.50) -> str:
    """Training OOD source: none, COSMIC leave-out, or train-seen mock de novo."""
    if rng.random() >= float(pc_ood):
        return "none"
    return "cosmic" if rng.random() < float(p_cosmic) else "seen_denovo"

def forced_idx_from_batch_role(M_batch: pd.DataFrame, role: str, rng: np.random.Generator) -> Optional[int]:
    """Pick one batch-local reference index from an OOD source role."""
    role = str(role or "")
    col = {"cosmic": "is_cosmic", "seen_denovo": "is_seen_denovo", "leaveout_denovo": "is_leaveout_denovo"}.get(role)
    if col is None or col not in M_batch.columns:
        return None
    names = M_batch.index[M_batch[col].astype(bool)].to_numpy()
    if len(names) == 0:
        return None
    name = str(rng.choice(names))
    return int(list(M_batch.index).index(name))

def generate_single_batch_v9(
    R_grand: pd.DataFrame,
    M_grand: pd.DataFrame,
    config: Dict[str, Any],
    random_state: int,
    batch_id: str,
    ood_source_mode: str = "train_mixed",
    basis_perturb: bool = False,
    basis_perturb_conc: float = 50000.0,
    basis_perturb_mix: float = 0.020,
    basis_perturb_mix_denovo: Optional[float] = None,
    timer: Optional[StepTimer] = None,
    epoch: Optional[int] = None,
    batch_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate one batch after sampling a compact reference pool.

    Basis perturbation is applied only to the sampled batch reference pool,
    which preserves the data-generation logic while avoiding unnecessary work
    on references that cannot appear in the batch.
    """
    rng = np.random.default_rng(random_state)
    cfg = {**data_v9.DEFAULT_CONFIG_SINGLE, **({} if config is None else config)}
    cfg["pcOOD"] = 0.0  # OOD is controlled explicitly through force_ood_idx.
    include_leaveout = str(ood_source_mode) == "leaveout_denovo"
    with (timer.block("s05_Train.py", "select_batch_refpool_v9", "select_refpool", epoch=epoch, batch=batch_num, mode=ood_source_mode) if timer else contextlib.nullcontext()):
        R, M, _, _ = select_batch_refpool_v9(
            R_grand,
            M_grand,
            rng=rng,
            n_cosmic=int(config.get("n_cosmic_per_batch", 65)),
            n_seen_denovo=int(config.get("n_seen_denovo_per_batch", 65)),
            include_leaveout=include_leaveout,
            forced_role="leaveout_denovo" if include_leaveout else "",
        )
    if basis_perturb:
        with (timer.block("s05_Train.py", "perturb_reference_basis", "perturb_sampled_refpool", epoch=epoch, batch=batch_num, mode=ood_source_mode, n_ref=R.shape[0]) if timer else contextlib.nullcontext()):
            R = perturb_reference_basis(
                R,
                rng=rng,
                concentration=float(basis_perturb_conc),
                mix=float(basis_perturb_mix),
                M_ref=M,
                mix_denovo=basis_perturb_mix_denovo,
            )
    rows = []
    with (timer.block("s02_Data.py", "finalize_sample_row", "generate_samples", epoch=epoch, batch=batch_num, mode=ood_source_mode, n_ref=R.shape[0], n_samples=int(cfg["BSize"])) if timer else contextlib.nullcontext()):
        for i in range(int(cfg["BSize"])):
            if ood_source_mode == "train_mixed":
                src = choose_train_ood_source(rng, float(config.get("pcOOD", 0.0)), p_cosmic=float(config.get("pcOOD_cosmic_fraction", 0.50)))
            else:
                src = str(ood_source_mode)
            forced_idx = forced_idx_from_batch_role(M, src, rng)
            actual_src = src if forced_idx is not None else "none"
            n_active = data_v9.sample_int_from_weighted_ranges(cfg["ACTVE"], rng, default_hi=R.shape[0])
            n_active = max(1, min(int(n_active), R.shape[0]))
            if forced_idx is not None:
                n_extra = max(0, n_active - 1)
                extra = data_v9.sample_active_indices(R.shape[0], n_extra, rng, exclude=[forced_idx]) if n_extra > 0 else np.array([], dtype=int)
                active_idx = np.unique(np.append(extra, forced_idx)).astype(int)
                force = [forced_idx]
            else:
                active_idx = data_v9.sample_active_indices(R.shape[0], n_active, rng)
                force = []
            cohort_meta = {"ood_source_label": actual_src, "ood_design_source": src}
            row = data_v9.finalize_sample_row(f"S{i:06d}", R, cfg, rng, active_idx, cohort_meta=cohort_meta, force_ood_idx=force)
            rows.append(row)
    with (timer.block("s02_Data.py", "assemble_batch", "assemble_dataframes", epoch=epoch, batch=batch_num, mode=ood_source_mode, n_ref=R.shape[0], n_samples=int(cfg["BSize"])) if timer else contextlib.nullcontext()):
        return data_v9.assemble_batch(rows, R, M, batch_kind=f"single_{ood_source_mode}", batch_id=batch_id)

def batch_from_curriculum(
    R_grand: pd.DataFrame,
    M_grand: pd.DataFrame,
    cur_cfg: Dict[str, Any],
    seed: int,
    batch_id: str,
    timer: Optional[StepTimer] = None,
    epoch: Optional[int] = None,
    batch_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate one curriculum batch without perturbing the full reference bank."""
    return generate_single_batch_v9(
        R_grand,
        M_grand,
        config=cur_cfg["data_config"],
        random_state=seed + 1009,
        batch_id=batch_id,
        ood_source_mode="train_mixed",
        basis_perturb=bool(cur_cfg.get("perturb_basis", False)),
        basis_perturb_conc=float(cur_cfg.get("basis_perturb_conc", 50000.0)),
        basis_perturb_mix=float(cur_cfg.get("basis_perturb_mix", 0.020)),
        basis_perturb_mix_denovo=cur_cfg.get("basis_perturb_mix_denovo", None),
        timer=timer,
        epoch=epoch,
        batch_num=batch_num,
    )

def eval_configs(eval_bch_size: int, quick: bool = False, ood_min_compo: float = 0.05) -> List[Dict[str, Any]]:
    """Return depth × noise × OOD-source fixed evaluation configs."""
    depth_bins = ["100-400", "401-2000", "2001-7000", "7000-100000"]
    noise_bins = ["0.85-0.90", "0.90-0.95", "0.95-1.00"]
    ood_sources = ["none", "cosmic", "seen_denovo", "leaveout_denovo"]
    if quick:
        # Smoke/quick eval keeps all OOD-source categories but only one depth/noise tile.
        depth_bins = depth_bins[:1]
        noise_bins = noise_bins[:1]
    cfgs = []
    for d in depth_bins:
        for n in noise_bins:
            for src in ood_sources:
                cfg = copy.deepcopy(data_v9.DEFAULT_CONFIG_SINGLE)
                cfg["BSize"] = int(eval_bch_size)
                cfg["DEPTH"] = {d: 1}
                cfg["NOISE"] = {n: {"0.85-0.90": 140, "0.90-0.95": 240, "0.95-1.00": 1800}[n]}
                cfg["PRIOR"] = {"61-120": 1, "130": 1}
                cfg["ACTVE"] = {"1": 1, "2-5": 2, "6-10": 1}
                cfg["COMPO"] = {0.1: 1, 1.0: 2, 10.0: 0}
                cfg["pcOOD"] = 0.0 if src == "none" else 1.0
                cfg["ood_min_compo"] = float(ood_min_compo)
                cfgs.append({"depth": d, "noise": n, "ood_source": src, "config": cfg})
    return cfgs

def build_eval_data(
    R_grand: pd.DataFrame,
    M_grand: pd.DataFrame,
    out_eval_data: Path,
    args: argparse.Namespace,
    path_log: str,
    timer: Optional[StepTimer] = None,
) -> List[Dict[str, Any]]:
    """Build fixed evaluation batches and export pkl/tsv for later reuse."""
    ensure_dir(out_eval_data)
    manifest_path = out_eval_data / "eval_manifest.tsv"
    if manifest_path.exists() and not args.rebuild_eval:
        manifest = pd.read_csv(manifest_path, sep="\t")
        paths = manifest.to_dict("records")
        print_log(f"eval data manifest loaded: {len(paths)} batches", path_log, print_time=True)
        return paths

    cfgs = eval_configs(eval_bch_size=int(args.eval_bch_size or args.bch_size), quick=bool(args.eval_quick), ood_min_compo=float(args.ood_min_compo))
    for _rec in cfgs:
        _rec["config"]["n_cosmic_per_batch"] = int(args.n_cosmic_per_batch)
        _rec["config"]["n_seen_denovo_per_batch"] = int(args.n_seen_denovo_per_batch)
    total = len(cfgs) * int(args.eval_n_bch)
    print_log(f"building eval data: {len(cfgs)} source/depth/noise combos × {args.eval_n_bch} batches", path_log, print_time=True)
    meter = ProgressMeter(total=total, label="build eval", path_log=path_log, every=max(1, total // 12))
    records = []
    k = 0
    for combo in cfgs:
        combo_dir = out_eval_data / (
            f"src_{combo['ood_source']}__depth_{combo['depth'].replace('-', '_')}__noise_{combo['noise'].replace('.', 'p').replace('-', '_')}"
        )
        ensure_dir(combo_dir)
        for j in range(int(args.eval_n_bch)):
            seed = int(args.seed) + 200000 + k
            with (timer.block("s05_Train.py", "build_eval_data", "generate_eval_batch", mode=str(combo["ood_source"]), batch=k) if timer else contextlib.nullcontext()):
                batch = generate_single_batch_v9(
                    R_grand,
                    M_grand,
                    config=combo["config"],
                    random_state=seed + 919,
                    batch_id=f"eval_{k:05d}",
                    ood_source_mode=combo["ood_source"],
                    basis_perturb=bool(args.eval_perturb_basis),
                    basis_perturb_conc=float(args.basis_perturb_conc),
                    basis_perturb_mix=float(args.basis_perturb_mix),
                    basis_perturb_mix_denovo=args.basis_perturb_mix_denovo,
                    timer=timer,
                    batch_num=k,
                )
            with (timer.block("s02_Data.py", "export_batch", "write_eval_batch", mode=str(combo["ood_source"]), batch=k) if timer else contextlib.nullcontext()):
                written = data_v9.export_batch(batch, str(combo_dir), prefix=f"eval_{k:05d}", formats=("pkl", "tsv"), sigprofiler=True)
            records.append({
                "depth": combo["depth"],
                "noise": combo["noise"],
                "ood_source": combo["ood_source"],
                "batch_id": f"eval_{k:05d}",
                "pkl": written["pkl"],
                "sigprofiler": written.get("sigprofiler", ""),
            })
            k += 1
            meter.update(k)
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    meter.update(total, force=True)
    print_log(f"eval manifest saved: {manifest_path}", path_log, print_time=True)
    return records


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def print_log(msg: str,
              path_log: Optional[os.PathLike | str] = None,
              print_time: bool = False,
              also_stdout: bool = True,) -> str:
    """Print a message and append it to ``path_log``.

    If ``print_time`` is true, prepend a compact timestamp formatted as
    ``[yyyy-mm/dd hh-mm:ss]`` because colons in filenames and logs love making
    shell scripting just slightly worse.
    """
    prefix = ""
    if print_time:
        prefix = _dt.datetime.now().strftime("[%Y-%m/%d %H-%M:%S] ")
    line = prefix + str(msg)
    if also_stdout:
        print(line, flush=True)
    if path_log is not None:
        Path(path_log).parent.mkdir(parents=True, exist_ok=True)
        with open(path_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return line

def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def loss__mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Numerically plain MSE wrapper for naming consistency across scripts."""
    return F.mse_loss(pred, target, reduction=reduction)


def format_seconds(seconds: float) -> str:
    """Human-readable duration string."""
    seconds = float(max(seconds, 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


class ProgressMeter:
    """Small progress logger with elapsed time and ETA."""

    def __init__(self, total: int, label: str = "task", path_log: Optional[str] = None, every: int = 1):
        self.total = max(int(total), 1)
        self.label = label
        self.path_log = path_log
        self.every = max(int(every), 1)
        self.t0 = time.time()
        self.last = 0

    def update(self, done: int, extra: str = "", force: bool = False) -> None:
        done = int(done)
        if not force and done < self.total and (done - self.last) < self.every:
            return
        self.last = done
        elapsed = time.time() - self.t0
        rate = done / max(elapsed, 1e-9)
        remain = (self.total - done) / max(rate, 1e-9)
        pct = 100.0 * done / self.total
        suffix = f" | {extra}" if extra else ""
        print_log(
            f"{self.label}: {done}/{self.total} ({pct:5.1f}%) elapsed={format_seconds(elapsed)} eta={format_seconds(remain)}{suffix}",
            path_log=self.path_log,
            print_time=True,
        )


def nearest_ge_choice(x: int, choices: Sequence[int] = (1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500)) -> int:
    """Return the smallest configured value >= x, or the largest if x is beyond the list."""
    x = max(int(x), 1)
    for c in choices:
        if c >= x:
            return int(c)
    return int(choices[-1])


def smooth_series(y: Sequence[float], window: int = 25) -> np.ndarray:
    """Centered rolling mean with graceful behavior for short curves."""
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return arr
    win = max(1, min(int(window), arr.size))
    if win <= 2:
        return arr.copy()
    return pd.Series(arr).rolling(win, min_periods=max(1, win // 4), center=True).mean().bfill().ffill().to_numpy()


def _get_col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(df.shape[0], default, dtype=float)


def plot_learn_curve(
    df_curve: pd.DataFrame,
    path_out: os.PathLike | str,
    df_eval: Optional[pd.DataFrame] = None,
    finished_ep: Optional[int] = None,
    total_samples: Optional[int] = None,
    loss_cols: Sequence[str] = ("loss_total", "loss_compo", "loss_recon", "loss_ood"),
    grad_col: str = "grad_norm",
) -> plt.Figure:
    """Plot training learning curve with readable evaluation panels.

    Upper panel: batch-level losses and gradient norm over sample count.
    Lower panels: evaluation R² split into high-prior and full-reference panels.
    Inside each panel, depth is averaged away and curves are grouped by
    OOD-source population × noise band. This keeps the figure useful instead of
    turning the legend into a bureaucratic census of every possible subgroup.
    """
    df_curve = df_curve.copy()
    if df_curve.empty:
        df_curve = pd.DataFrame({"sample_seen": [0], "epoch": [0], "loss_total": [np.nan], grad_col: [np.nan]})
    if "sample_seen" not in df_curve.columns:
        df_curve["sample_seen"] = np.arange(df_curve.shape[0])
    if "epoch" not in df_curve.columns:
        df_curve["epoch"] = 0

    x = _get_col(df_curve, "sample_seen", 0.0)
    x_max = max(float(np.nanmax(x)) if x.size else 1.0, float(total_samples or 1)) * 1.20

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.05])
    ax_loss = fig.add_subplot(gs[0, :])
    ax_eval_high = fig.add_subplot(gs[1, 0], sharex=ax_loss)
    ax_eval_full = fig.add_subplot(gs[1, 1], sharex=ax_loss)
    ax_grad = ax_loss.twinx()
    eval_axes = {"high_prior": ax_eval_high, "full": ax_eval_full}

    # Epoch shading. The label ticks mark the first epoch of each shaded block.
    ep_done = int(finished_ep if finished_ep is not None else max(1, int(np.nanmax(_get_col(df_curve, "epoch", 1)))))
    block_n = nearest_ge_choice(max(1, ep_done // 6))
    if "epoch" in df_curve.columns and "sample_seen" in df_curve.columns:
        ep_first = df_curve.groupby("epoch", as_index=True)["sample_seen"].min().sort_index()
        ep_time = df_curve.groupby("epoch", as_index=True).tail(1).set_index("epoch")
        ticks, tick_labels = [], []
        for i, ep in enumerate(range(1, ep_done + 1, block_n)):
            ep_end = min(ep + block_n - 1, ep_done)
            x0 = float(ep_first.loc[ep]) if ep in ep_first.index else None
            next_ep = ep_end + 1
            if next_ep in ep_first.index:
                x1 = float(ep_first.loc[next_ep])
            else:
                x1 = x_max
            if x0 is None:
                continue
            ax_loss.axvspan(x0, x1, alpha=0.06 if i % 2 == 0 else 0.11, zorder=0)
            ticks.append(x0)
            sample_done = int(x0)
            tval = ""
            if "wall_time" in ep_time.columns and ep in ep_time.index:
                tval = str(ep_time.loc[ep, "wall_time"])
            tick_labels.append(f"ep{ep}\n{sample_done}\n{tval}")
        if ticks:
            for ax in eval_axes.values():
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)

    # Batch-level losses plus smoothed curves.
    handles, labels = [], []
    for col in loss_cols:
        if col not in df_curve.columns:
            continue
        y = df_curve[col].to_numpy(dtype=float)
        h = ax_loss.plot(x, y, alpha=0.16, linewidth=0.75, label=f"{col} raw")[0]
        hs = ax_loss.plot(x, smooth_series(y, window=max(5, min(101, len(y) // 12 or 5))), linewidth=1.7, label=f"{col} smooth")[0]
        handles.extend([h, hs])
        labels.extend([h.get_label(), hs.get_label()])

    if grad_col in df_curve.columns:
        y_grad = df_curve[grad_col].to_numpy(dtype=float)
        hg = ax_grad.plot(x, smooth_series(y_grad, window=max(5, min(101, len(y_grad) // 12 or 5))), linestyle=":", linewidth=1.5, label="grad_norm smooth")[0]
        handles.append(hg)
        labels.append(hg.get_label())

    ax_loss.set_xlim(0, x_max)
    ax_loss.set_ylabel("loss")
    ax_grad.set_ylabel("gradient norm")
    ax_loss.grid(alpha=0.18)
    ax_loss.set_title("SigFormer learning curve")

    # Evaluation R2 panels: group by OOD-source population × noise, average over depth/batches.
    if df_eval is not None and not df_eval.empty:
        eval_df = df_eval.copy()
        if "sample_seen" not in eval_df.columns and "epoch" in eval_df.columns:
            ep_to_x = df_curve.groupby("epoch")["sample_seen"].max().to_dict()
            eval_df["sample_seen"] = eval_df["epoch"].map(ep_to_x).fillna(0)
        r2_col = "R2" if "R2" in eval_df.columns else "r2" if "r2" in eval_df.columns else None
        if r2_col is not None:
            if "mode" not in eval_df.columns:
                eval_df["mode"] = "eval"
            if "noise" not in eval_df.columns:
                eval_df["noise"] = "noise"
            if "ood_source" not in eval_df.columns:
                eval_df["ood_source"] = "mixed"
            grouped = eval_df.groupby(["mode", "sample_seen", "noise", "ood_source"], as_index=False)[r2_col].mean()
            for mode, ax in eval_axes.items():
                gm = grouped[grouped["mode"].astype(str).eq(mode)]
                for (noise, src), g in gm.groupby(["noise", "ood_source"]):
                    g = g.sort_values("sample_seen")
                    label = f"{src}|{noise}"
                    ax.plot(g["sample_seen"].to_numpy(dtype=float), g[r2_col].to_numpy(dtype=float), marker="o", ms=3, lw=1.15, label=label)

    for mode, ax in eval_axes.items():
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("data seen / samples")
        ax.set_ylabel("eval R²")
        ax.set_title("high-prior eval R²" if mode == "high_prior" else "full-reference eval R²")
        ax.grid(alpha=0.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.4, ncol=2, loc="lower right")
    if handles:
        ax_loss.legend(handles, labels, fontsize=8, ncol=2, loc="upper right")

    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=150)
    return fig
