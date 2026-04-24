# SigFormer V6

SigFormer V6 is a cleaned and accuracy-first rewrite of the V5.x training stack for mutational signature decomposition.
It keeps the original model idea intact, but fixes several training inconsistencies and adds more realistic confidence/OOD diagnostics.

## What changed from V5.x

1. **Simplex schedule is explicit and stable**
   - `entmax_alpha` is no longer trainable.
   - Training starts with `softmax`, moves to `entmax15`, and finishes with a sparse tail stage.

2. **Reconstruction loss is aligned with training**
   - In V5.x pretraining, reconstruction loss was computed but not backpropagated.
   - In V6, reconstruction can contribute gradients through a staged curriculum.

3. **False-positive control is built into main training**
   - Weak inactive-signature penalties are used during the main training loop.
   - Late sparse simplex stages reduce diffuse tiny outputs.

4. **Confidence is more useful at inference time**
   - Confidence target considers both absolute composition error and support mismatch.
   - Confidence-aware masking is supported for post-processing.
   - Backbone-to-confidence gradient is detached by default to protect decomposition accuracy.

5. **OOD / de novo suspicion is surfaced explicitly**
   - The model now exports residual-based novelty diagnostics.
   - A holdout-style OOD evaluation is run during training.

6. **Evaluation is more complete**
   - Full evaluation is run every `--eval_every` epochs.
   - Per-batch loss curves are saved for each epoch.
   - Raw and confidence-masked performance are both reported.

## File layout

- `s01_core_v6.py`
  - Core model definition.
  - Composition head, confidence head, simplex control, masking helper.

- `s02_data_v6.py`
  - Simulation utilities.
  - COSMIC summary, de novo bank generation, profile/noise simulation.

- `s03_utils_v6.py`
  - Logging, metrics, plotting, confidence target builder, evaluation helpers.

- `s03_train_v6.py`
  - Main training entry point.
  - One-stage curriculum, checkpointing, periodic evaluation, OOD holdout diagnostics.

- `j04_quickstart_v6.ipynb`
  - Example notebook for loading pretrained weights and running inference on simulated data.

## Environment

Python 3.10+ is recommended.

Required packages:
- numpy
- pandas
- matplotlib
- torch
- entmax
- jupyter

You also need your local `YZ_vis_sig.py` helper that provides `get_COSMIC()`.

## Training

### Standard one-stage training

```bash
python s03_train_v6.py \
  --dir runs/SigFormer_V6_main \
  --device cuda \
  --n_epochs 300 \
  --n_batches 4000 \
  --batch_size 64 \
  --lr_base 3e-4 \
  --eval_every 5
```

### Resume from a checkpoint for an extra polishing stage

`--n_epochs` is the **final total epoch index target**, not “extra epochs”.
So if you resume from epoch 200 and want 100 more epochs, set `--n_epochs 300`.

```bash
python s03_train_v6.py \
  --dir runs/SigFormer_V6_stage2 \
  --device cuda \
  --resume_ckpt runs/SigFormer_V6_main/3_model_wts/SigFormer_v6_epoch200.pt \
  --n_epochs 300 \
  --lr_base 8e-5 \
  --eval_every 5
```

### Optional custom curriculum

You can pass a TSV with these columns:
- `epoch_stt`
- `epoch_end`
- `simplex`
- `lambda_recon_scale`
- `lambda_fp_scale`
- `depth_low`
- `depth_mid`
- `depth_hig`
- `norm_frac`
- `use_bucketed_refs`
- `balanced_tail`

Then:

```bash
python s03_train_v6.py --dir runs/custom --curriculum my_curriculum.tsv
```

## Outputs

Inside the run directory:

- `2_eval_figs/`
  - evaluation grids
  - OOD holdout scatter
  - summary plots

- `3_model_wts/`
  - checkpoint `.pt` files

- `4_batch_loss/`
  - per-epoch batch loss TSV and PNG

- `summary.tsv`
  - epoch-level summary table

## Inference / notebook usage

Open `j04_quickstart_v6.ipynb` and update:
- the path to your trained checkpoint
- any run-specific paths if needed

The notebook demonstrates:
- loading a pretrained model
- building a reference bank
- simulating samples
- running inference
- applying confidence-aware masking
- inspecting novelty/OOD diagnostics

## Notes on interpretation

- Use **raw composition** for debugging and ablation.
- Use **masked composition** for user-facing interpretation when the goal is cleaner support recovery.
- Treat novelty / OOD outputs as a **warning signal**, not as a guaranteed discovery of a new biological process.
- Residual-based novelty profiles are only a first approximation of hidden de novo processes.

## License

A conservative default for academic release is `MIT` if you want maximum reuse, or `BSD-3-Clause` if you want a slightly more formal academic-style permissive license.
If the project depends on institutional policy or unpublished components, decide that before public release instead of letting GitHub chaos choose for you.
