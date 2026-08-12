# SigFormer

SigFormer is a mutational-signature decomposition and analysis toolkit for SBS96 profiles. The repository contains the SigFormer model, synthetic benchmark generation, wrappers for comparison methods, and reproducible cohort-analysis notebooks for PCAWG and normal tissues.

## Repository layout

```text
SigFormer/
├── SigFormer/
│   ├── __init__.py
│   ├── scripts/
│   │   ├── s01_Core.py          # model architecture
│   │   ├── s02_Data.py          # synthetic data generation
│   │   ├── s03_Util_train.py    # training-only utilities and losses
│   │   ├── s04_Util_apply.py    # downstream analysis and plotting helpers
│   │   ├── s05_Train.py         # training entry point
│   │   ├── s06_wrapper.py       # SigFormer and comparison-method wrappers
│   │   ├── s07_bench_helper.py  # benchmark-only helpers
│   │   └── s09_bench_cli.py     # benchmark CLI
│   └── resource/                # local reference data and model checkpoint
├── j01_benchmark.ipynb
├── j02_demo_PCAWG.ipynb
├── j03_demo_normal.ipynb
├── pyproject.toml
├── LICENSE
└── THIRD_PARTY_DATA.md
```

---

## Installation

### 1. Create the Conda environment

The tested research environment is named `SgF`:

```bash
mamba create -n SgF -y -c pytorch -c nvidia \
    python=3.10 ipykernel pytorch-gpu=2.3 \
    r-base=4.5.1 r-devtools r-remotes r-biocmanager r-rstan

conda activate SgF
python -m ipykernel install --user --name py_SgF
```

### 2. Install Python and Bioconductor dependencies

```bash
mamba install -y pandas matplotlib leidenalg scikit-misc scikit-learn umap-learn seaborn
mamba install -y bioconda::sigprofilerassignment

mamba install -y -c conda-forge -c bioconda \
    bioconductor-variantannotation \
    bioconductor-summarizedexperiment \
    bioconductor-bsgenome \
    bioconductor-bsgenome.hsapiens.ucsc.hg38 \
    bioconductor-bsgenome.hsapiens.1000genomes.hs37d5 \
    r-factoextra
```

### 3. Install the R comparison methods

Start R inside the activated `SgF` environment:

```bash
R
```

Then install:

```r
remotes::install_github("gersteinlab/siglasso", upgrade = "never")
remotes::install_github("kgori/sigfit", upgrade = "never", build_vignettes = FALSE)
remotes::install_github("Nik-Zainal-Group/signature.tools.lib", dependencies = TRUE, upgrade = "never")
```

Exit R after installation.

### 4. Install MuSiCal

After the environment dependencies are already installed:

```bash
python -m pip install --no-deps git+https://github.com/parklab/MuSiCal.git
```

### 5. Clone and install SigFormer

```bash
git clone https://github.com/zonglab/SigFormer.git
cd SigFormer
python -m pip install --no-deps ./
```

The installed package name is:

```python
import SigFormer
```

---

## Notebook workflow

Start Jupyter from the repository root and select the `py_SgF` kernel.

```bash
jupyter lab
```

Every notebook begins with explicit runtime configuration, including:

```python
%matplotlib inline
import logging, warnings
logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
```

### `j01_benchmark.ipynb`

The first half is an on-the-fly demonstration of four benchmark modes:

1. `no_OOC`
2. `random_OOC`
3. `titration_COSMIC`
4. `titration_OOC`

Each mode has four cells: configuration, synthetic data generation + inference, visualization, and result saving. All selected methods are stored in one `DICT_runner` and executed through the same loop. `titration_OOC` is restricted to methods that explicitly estimate OOC/OOD mass.

The second half demonstrates batch submission with `s09_bench_cli.py`, checklist generation, result loading, and plotting.

### `j02_demo_PCAWG.ipynb`

Runs the PCAWG workflow using bundled raw profiles and cached comparison-method outputs. SigFormer is inferred from the checkpoint in 500-sample chunks; comparison methods load cache first and are only executed when a cache is absent.

### `j03_demo_normal.ipynb`

Runs the normal-tissue workflow using the same shared analysis helpers. In addition to the common reconstruction, UMAP, clustering, and per-cluster workflow, it includes the normal background continuum, selected tissue/process stacked bars, OOC-depth analysis, and external residual-profile comparisons.

---

## Wrapper names and method tags

Short tags are used only for compact internal identifiers such as UMAP columns. Logs and figure labels use full method names.

| Tag | Full name |
| --- | --- |
| `SgF` | SigFormer, refined by default |
| `Mus` | MuSiCal |
| `SPA` | SigProfilerAssignment |
| `sft` | sigfit |
| `sLS` | sigLASSO |
| `stl` | signature.tools.lib |

`R_env=None` resolves to the currently active Conda environment through `CONDA_DEFAULT_ENV`. Therefore, when Python is launched from `conda activate SgF`, `R_env=None` has the same effect as `R_env="SgF"`.

All tunable wrapper parameters are exposed in the notebook configuration cells rather than hidden inside helper modules.

---

## Python API example

```python
import pandas as pd
import SigFormer
from SigFormer.scripts.s06_wrapper import CLASS_wrapper_SigFormer

model = CLASS_wrapper_SigFormer(
    PATH_model=str(SigFormer.DEFAULT_MODEL_PATH),
    device=None,
    refine=True,
)

composition, reconstruction, ooc = model(
    df_3nt_raw,
    df_reference,
    df_refmask,
)
```

The wrapper returns refined SigFormer output by default. Use `model.predict_raw(...)` when the unrefined model output is required.

---

---

## Signature colors

Stacked-bar colors are defined centrally in `s04_Util_apply.py`, but notebooks copy the palette before use:

```python
SIG_COLORS = analysis.DEFAULT_SIGNATURE_COLORS.copy()
SIG_COLORS["SBS9"] = "#your_color"
```

Passing this dictionary to `plot_cluster_stackbar(..., sig_colors=SIG_COLORS)` changes that notebook's figures without mutating module-level global state.

---

## Resource licensing before a public GitHub release

The source code is prepared under the MIT License. **Do not assume the same MIT licence can be applied to bundled third-party data.** The local handoff contains the current resources for testing, but `SigFormer/resource/**` and `example_data/template_mock/**` are intentionally ignored by `.gitignore` until their redistribution rights are confirmed.

Read `THIRD_PARTY_DATA.md` before making the repository public. In particular, independently verify the redistribution terms for COSMIC reference data, PCAWG/normal-tissue resources, generated data derived from those references, and the pretrained model checkpoint.

---

## Reproducibility notes

- Training-time utilities and the model architecture are separated from downstream application helpers.
- Cohort inference is cache-first and processed in 500-sample chunks when computation is required.
- Notebook-specific global settings are explicit and do not execute on module import.
- Plotting is implemented with matplotlib.

---
