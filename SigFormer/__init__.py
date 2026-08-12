"""SigFormer: mutational-signature decomposition and analysis utilities."""

from pathlib import Path

__version__ = "0.1.0"
PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resource"
DEFAULT_MODEL_PATH = RESOURCE_DIR / "sigformer_v9_epoch_1600.pt"

from .scripts.s01_Core import SigFormerCore
from .scripts.s06_wrapper import (
    CLASS_wrapper_MuSiCal,
    CLASS_wrapper_SigFormer,
    CLASS_wrapper_SigLASSO,
    CLASS_wrapper_SigProfilerAssignment,
    CLASS_wrapper_sig_tool_lib,
    CLASS_wrapper_sigfit,
)

__all__ = [
    "SigFormerCore",
    "CLASS_wrapper_SigFormer",
    "CLASS_wrapper_MuSiCal",
    "CLASS_wrapper_SigProfilerAssignment",
    "CLASS_wrapper_sigfit",
    "CLASS_wrapper_SigLASSO",
    "CLASS_wrapper_sig_tool_lib",
    "PACKAGE_DIR",
    "RESOURCE_DIR",
    "DEFAULT_MODEL_PATH",
]
