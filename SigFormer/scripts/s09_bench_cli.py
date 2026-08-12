#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one benchmark condition using the same wrappers and helpers as j01_benchmark."""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
RESOURCE_DIR = PACKAGE_DIR / "resource"

from .s06_wrapper import (
    CLASS_wrapper_MuSiCal,
    CLASS_wrapper_SigFormer,
    CLASS_wrapper_SigProfilerAssignment,
    CLASS_wrapper_sigfit,
    CLASS_wrapper_SigLASSO,
    CLASS_wrapper_sig_tool_lib,
)
from .s07_bench_helper import (
    BENCHMARK_MODES,
    METHOD_ORDER,
    OOD_PRED_METHODS,
    R_env,
    MAKE_config_batch,
    RUN_method_timed,
    benchmark_root,
    combine_benchmark_units,
    condition_path,
    cosmic_version_label,
    ensure_dir,
    generate_benchmark_batch,
    load_or_build_bank,
    normalize_mode,
    parse_ref_size,
    parse_signatures,
    parse_steps,
    ref_mask_from_benchmark,
    save_benchmark_outputs,
    stable_seed,
)


# ============================================================================
# CLI configuration and runner construction
# ============================================================================


def resolve_methods(mode, text):
    default = OOD_PRED_METHODS if normalize_mode(mode) == "ood_titration" else METHOD_ORDER
    methods = default if text in {None, "", "auto"} else [x for x in str(text).replace(";", ",").replace(" ", ",").split(",") if x]
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; valid={METHOD_ORDER}")
    return methods


def resolve_cosmic_targets(text, metadata):
    signatures = parse_signatures(text)
    return [str(x) for x in metadata.index[metadata["is_cosmic"].astype(bool)]] if signatures in [["auto"], ["all_cosmic"]] else signatures


def write_run_config(path, cfg, args, methods):
    ensure_dir(path)
    payload = {"config": cfg, "args": vars(args), "methods": methods}
    (Path(path) / "run_config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_runners(methods, tmp_root, r_env, cosmic_version, fail_soft, model_path):
    version = float(str(cosmic_version).replace("v", ""))
    runners = {}
    if {"SgF_raw", "SgF"}.intersection(methods):
        refine = dict(candidate_top_k=6, max_rescue=2, hard_composition=0.005, hard_count=3, prior_strength=12.0, max_l1_change=0.20, max_ood_change=0.15, max_ood_increase=0.005, rescue_min_composition=0.01, n_channel_folds=4, normalized_effective_depth=1000, random_state=717)
        model = CLASS_wrapper_SigFormer(model_path, epoch=None, device=None, simplex="softmax", strict=False, refine=True, refinement_kwargs=refine)
        if "SgF_raw" in methods:
            runners["SgF_raw"] = model.predict_raw
        if "SgF" in methods:
            runners["SgF"] = model
    builders = {
        "Mus": lambda: CLASS_wrapper_MuSiCal(method="likelihood_bidirectional", thresh=0.001),
        "SPA": lambda: CLASS_wrapper_SigProfilerAssignment(PATH_tmp=str(tmp_root / "SigProfilerAssignment"), hg_ver="GRCh38", cos_ver=version),
        "sft": lambda: CLASS_wrapper_sigfit(conda_env=r_env, work_dir=str(tmp_root / "sigfit"), fail_soft=fail_soft, iter=1400, warmup=None, chains=3, cores=1, seed=19970717),
        "sLS": lambda: CLASS_wrapper_SigLASSO(conda_env=r_env, work_dir=str(tmp_root / "sigLASSO"), fail_soft=fail_soft, conf=0.1, adaptive=True, gamma=1, alpha_min=400, iter_max=None, sd_multiplier=1.0, elastic_net=False, normalize="none"),
        "stl": lambda: CLASS_wrapper_sig_tool_lib(conda_env=r_env, work_dir=str(tmp_root / "signature_tools_lib"), nmuts_threshold=300, pvalue_threshold=0.15, pvalue_method="normErrorSAD", nmuts_method="residualSSD", fail_soft=fail_soft),
    }
    for name in methods:
        if name in builders:
            runners[name] = builders[name]()
    return runners



# ============================================================================
# One-condition execution and result persistence
# ============================================================================


def run_condition(bank, bank_meta, cfg, methods, runners, out_dir, n_batches, seed, make_plots, verbose):
    records = []
    for batch_number in range(int(n_batches)):
        unit = f"bch{batch_number:03d}"
        batch_seed = stable_seed(seed, cfg["mode"], unit, cfg.get("titer_signature"))
        batch = generate_benchmark_batch(bank, bank_meta, cfg=cfg, mode=cfg["mode"], random_state=batch_seed, batch_id=f"{cfg['mode']}_{unit}")
        mask = ref_mask_from_benchmark(batch, full_reference=True)
        results, timing = {}, []
        for name in methods:
            RUN_method_timed(name, runners[name], batch, mask, results, timing, unit_label=unit, verbose=verbose)
        records.append({"unit_label": unit, "batch": batch, "method_results": results, "timing": pd.DataFrame(timing)})
    combined_batch, combined_results, combined_timing = combine_benchmark_units(records)
    save_benchmark_outputs(combined_batch, combined_results, out_dir, timing=combined_timing, make_plots=make_plots)
    return Path(out_dir) / "summary.tsv"



# ============================================================================
# Command-line interface
# ============================================================================


def build_parser():
    p = argparse.ArgumentParser(description="Run one SigFormer benchmark condition.")
    p.add_argument("--mode", required=True, choices=BENCHMARK_MODES + ["denovo_titration", "no_OOC", "random_OOC", "titration_COSMIC", "titration_OOC"])
    p.add_argument("--depth-bin", required=True)
    p.add_argument("--noise-bin", required=True)
    p.add_argument("--compo-bin", required=True, type=float)
    p.add_argument("--active-bin", default="1-3")
    p.add_argument("--n-samples", default=12, type=int)
    p.add_argument("--n-batches", default=1, type=int)
    p.add_argument("--titer-nsmps", default=3, type=int)
    p.add_argument("--titer-steps", default="0,0.1,0.3,0.6")
    p.add_argument("--titer-std", default=0.02, type=float)
    p.add_argument("--titer-signatures", default="auto")
    p.add_argument("--random-ood-frac", default=1.0, type=float)
    p.add_argument("--random-ood-max-compo", default=0.60, type=float)
    p.add_argument("--denovo-source", default="leaveout_mock", choices=["seen_mock", "leaveout_mock"])
    p.add_argument("--ref-size", default="COSMIC:80,MOCK:20")
    p.add_argument("--cosmic-version", default="v3.4")
    p.add_argument("--cosmic-path", default=str(RESOURCE_DIR / "COSMIC_v3.4_SBS_GRCh38.txt"))
    p.add_argument("--n-mock-bank", default=1000, type=int)
    p.add_argument("--ref-bank-cache", default=None)
    p.add_argument("--model-path", default=str(RESOURCE_DIR / "sigformer_v9_epoch_1600.pt"))
    p.add_argument("--out-root", default=None)
    p.add_argument("--date-tag", default=_dt.datetime.now().strftime("%Y%m%d"))
    p.add_argument("--r-env", default=R_env)
    p.add_argument("--random-state", default=19970717, type=int)
    p.add_argument("--methods", default="auto")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--fail-soft", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--tmp-root", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    mode = normalize_mode(args.mode)
    root = Path(args.out_root) if args.out_root else benchmark_root(args.date_tag, args.cosmic_version)
    tmp_root = ensure_dir(Path(args.tmp_root) if args.tmp_root else root / "_tmp" / mode) / f"pid{os.getpid()}_{uuid.uuid4().hex[:10]}"
    tmp_root = ensure_dir(tmp_root)
    methods = resolve_methods(mode, args.methods)
    cfg0 = MAKE_config_batch(mode=mode, BSize=args.n_samples, depth_bin=args.depth_bin, noise_bin=args.noise_bin, compo_bin=args.compo_bin, active_bin=args.active_bin, REF_size=parse_ref_size(args.ref_size), random_ood_frac=args.random_ood_frac, random_ood_max_compo=args.random_ood_max_compo, titer_steps=parse_steps(args.titer_steps), titer_std=args.titer_std, titer_nsmps=args.titer_nsmps, denovo_source=args.denovo_source, cosmic_version=args.cosmic_version)
    cache = args.ref_bank_cache or root / "_cache" / f"refbank_COSMIC_{cosmic_version_label(args.cosmic_version)}_seed{args.random_state}_nmock{args.n_mock_bank}.pkl"
    need_bank = mode == "cosmic_titration" and parse_signatures(args.titer_signatures) in [["auto"], ["all_cosmic"]]
    bank = bank_meta = None
    if not args.dry_run or need_bank:
        bank, bank_meta = load_or_build_bank(cosmic_path=args.cosmic_path, cosmic_version=args.cosmic_version, n_mock=args.n_mock_bank, random_state=args.random_state, cache_path=cache, verbose=verbose)
    targets = resolve_cosmic_targets(args.titer_signatures, bank_meta) if mode == "cosmic_titration" and bank_meta is not None else ([None] if mode != "cosmic_titration" else parse_signatures(args.titer_signatures))
    runners = {} if args.dry_run else build_runners(methods, tmp_root, args.r_env, args.cosmic_version, args.fail_soft, args.model_path)
    summaries = []
    try:
        for signature in targets:
            cfg = copy.deepcopy(cfg0)
            cfg["titer_signature"] = None if signature in {"auto", "all_cosmic"} else signature
            out_dir = condition_path(root, mode, args.noise_bin, args.depth_bin, args.active_bin, args.compo_bin, signature=cfg["titer_signature"])
            write_run_config(out_dir, cfg, args, methods)
            if args.dry_run:
                print(out_dir, flush=True)
                continue
            summaries.append(run_condition(bank, bank_meta, cfg, methods, runners, out_dir, args.n_batches, args.random_state, not args.no_plots, verbose))
    except Exception:
        (tmp_root / "python_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
    for path in summaries:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
