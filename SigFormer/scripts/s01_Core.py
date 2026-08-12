import math
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# Prefer the external entmax package, but keep inference portable.
try:
    from entmax import entmax_bisect
except Exception:
    def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
        """PyTorch-only forward implementation matching entmax.entmax_bisect."""
        if not torch.is_tensor(alpha):
            alpha = X.new_tensor(float(alpha))
        alpha = alpha.to(device=X.device, dtype=X.dtype)
        d = X.shape[dim]
        X_scaled = X * (alpha - 1)
        max_val = X_scaled.max(dim=dim, keepdim=True).values
        tau_lo = max_val - 1
        tau_hi = max_val - (1.0 / d) ** (alpha - 1)
        inv = 1.0 / (alpha - 1)
        for _ in range(int(n_iter)):
            tau_m = (tau_lo + tau_hi) / 2
            p_m = torch.clamp(X_scaled - tau_m, min=0) ** inv
            f_m = p_m.sum(dim=dim, keepdim=True) - 1
            ge = f_m >= 0
            tau_lo = torch.where(ge, tau_m, tau_lo)
            tau_hi = torch.where(ge, tau_hi, tau_m)
        p = torch.clamp(X_scaled - tau_hi, min=0) ** inv
        if ensure_sum_one:
            p = p / p.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        return p


# ============================================================
# Basic blocks
# ============================================================

class MLP(nn.Module):
    """Small Transformer feed-forward block with pre-LN and residual."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.gelu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y


class S_Attn_blk(nn.Module):
    """Self-attention over either sample tokens or reference tokens."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                            dropout=dropout,
                                        batch_first=True,)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attn(query=h, key=h, value=h,
                         key_padding_mask=key_padding_mask, need_weights=False,)
        x = x + self.dropout(h)
        x = self.mlp(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class X_Attn_blk(nn.Module):
    """Cross-attention block: query tokens attend to key/value tokens."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_qq = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                            dropout=dropout,
                                        batch_first=True,)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, tok_qq: torch.Tensor, tok_kv: torch.Tensor,
                     qq_mask: Optional[torch.Tensor] = None,
                     kv_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        qq = self.norm_qq(tok_qq)
        kv = self.norm_kv(tok_kv)
        h, _ = self.attn(query=qq, key=kv, value=kv,
                         key_padding_mask=kv_mask, need_weights=False,)
        tok_qq = tok_qq + self.dropout(h)
        tok_qq = self.mlp(tok_qq)
        if qq_mask is not None:
            tok_qq = tok_qq.masked_fill(qq_mask.unsqueeze(-1), 0.0)
        return tok_qq


# ============================================================
# SigFormer core
# ============================================================

class SigFormerCore(nn.Module):
    """
    Core network for reference-conditioned mutational signature attribution.
    
    Input
    X_smp : [B, 96]
        Counts or normalized 96-channel mutational profile.
    X_ref : [B, K, 96]
        Reference signatures aligned to the same 96 channels.
    ref_mask : optional [B, K] or [K]
        True/1 means the corresponding reference signature is available under
        the sample-specific prior. False/0 hides it from reference self-attn,
        sample->reference cross-attn, and the simplex output.

    Output
    ------
    composition : [B, K + 1] if use_tok_ood else [B, K]
        Simplex weights. When use_tok_ood=True, composition[:, :K] are
        absolute known-signature masses and composition[:, K] is predicted OOD
        residual mass.
    """

    _SIMPLEX_ALPHA = {"softmax": 1.0,
                       "entmax": 1.5,
                    "sparsemax": 2.0,}

    def __init__(self, n_chann: int=96, d_model: int=96, n_heads: int=4,
                       n_L_smp: int=2,  n_L_ref: int=4,  n_L_smp_ref: int=1,
                       mlp_ratio: float = 4.0, dropout: float = 0.1,
                       simplex: Optional[Union[str, float, int]] = "entmax",
                       use_tok_ood: bool = True,
                       residual_init: str = "zero_sample_depth",
                       ood_lg_bias_init: float = -2.0,):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if residual_init not in {"zero", "zero_sample", "zero_depth", "zero_sample_depth"}:
            raise ValueError("residual_init must be one of: 'zero', 'zero_sample', "
                             "'zero_depth', 'zero_sample_depth'")
        
        self.n_chann = n_chann
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_L_smp = n_L_smp
        self.n_L_ref = n_L_ref
        self.n_L_smp_ref = n_L_smp_ref
        self.use_tok_ood = use_tok_ood
        self.residual_init = residual_init
        
        ### encoding
        self.smp_encode = nn.Linear(1, d_model)
        self.ref_encode = nn.Linear(n_chann, d_model)
        self.ctx__embed = nn.Parameter(torch.empty(14, d_model)) ### 96-channel context = 4*6*4 factorized embedding
        ctx = torch.arange(n_chann)
        self.register_buffer("ctx__idx", torch.stack((ctx // 24, (ctx // 4) % 6 + 4, ctx % 4 + 10), dim=1), persistent=False)
        
        ### attentions
        param = dict(dim=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.ss_s_attn = nn.ModuleList([S_Attn_blk(**param) for _ in range(n_L_smp)])
        self.rr_s_attn = nn.ModuleList([S_Attn_blk(**param) for _ in range(n_L_ref)])
        self.rs_x_attn = nn.ModuleList([X_Attn_blk(**param) for _ in range(n_L_ref)])
        self.sr_x_attn = nn.ModuleList([X_Attn_blk(**param) for _ in range(n_L_smp_ref)])
        
        ### OOD token
        self.tok_ood = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ood_seed = nn.Linear(d_model, d_model)
        self.ood_seed_LN = nn.LayerNorm(d_model)
        self.dep_encode = nn.Linear(1, d_model)
        self.ood_lg_bias = nn.Parameter(torch.tensor(float(ood_lg_bias_init)))
        
        ### composition head
        self.compo_head = nn.Linear(d_model, 1)
        self.compo_norm = nn.LayerNorm(d_model)
        
        ### simplex
        self.simplex_amin = 1.0
        self.simplex_amax = 2.0
        self.register_buffer("simplex", torch.tensor(self._simplex_to_alpha(simplex), dtype=torch.float32))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.trunc_normal_(self.smp_encode.weight, std=0.02)
        nn.init.zeros_(self.smp_encode.bias)
        nn.init.trunc_normal_(self.ref_encode.weight, std=0.02)
        nn.init.zeros_(self.ref_encode.bias)
        nn.init.trunc_normal_(self.compo_head.weight, std=0.02)
        nn.init.zeros_(self.compo_head.bias)
        nn.init.trunc_normal_(self.ctx__embed, std=0.02)
        
        nn.init.zeros_(self.tok_ood)
        nn.init.zeros_(self.ood_seed.weight)
        nn.init.zeros_(self.ood_seed.bias)
        nn.init.zeros_(self.dep_encode.weight)
        nn.init.zeros_(self.dep_encode.bias)

    @classmethod
    def _simplex_to_alpha(cls, simplex: Optional[Union[str, float, int]]) -> float:
        if simplex is None:
            return 1.5
        if isinstance(simplex, (float, int)):
            return float(simplex)
        mode = str(simplex).lower()
        if mode not in cls._SIMPLEX_ALPHA:
            raise ValueError(f"Unknown simplex mode: {simplex}")
        return cls._SIMPLEX_ALPHA[mode]

    def _apply_simplex(self, logits: torch.Tensor, simplex: Optional[Union[str, float, int]] = None) -> torch.Tensor:
        alpha = self.simplex if simplex is None else logits.new_tensor(self._simplex_to_alpha(simplex))
        alpha = alpha.to(device=logits.device, dtype=logits.dtype).clamp(self.simplex_amin, self.simplex_amax,)
        
        if float(alpha.detach().cpu()) <= self.simplex_amin + 1e-6:
            return F.softmax(logits, dim=-1)
        if entmax_bisect is None:
            raise ImportError("simplex != softmax requires the entmax package: pip install entmax")
        return entmax_bisect(logits, alpha=alpha, dim=-1)

    @staticmethod
    def _format_ref_pad_mask(ref_mask: Optional[torch.Tensor],
                             B: int, K: int, device: torch.device,) -> torch.Tensor:
        """Return bool padding mask: True means hidden/unusable. 
           Input ref_mask True/1 means available."""

        if ref_mask is None:
            avail = torch.ones(B, K, dtype=torch.bool, device=device)

        elif ref_mask.dim() == 1 and ref_mask.shape[0] == K:
            avail = ref_mask.to(device=device).bool().unsqueeze(0).expand(B, -1).clone()

        elif ref_mask.dim() == 2 and tuple(ref_mask.shape) == (B, K):
            avail = ref_mask.to(device=device).bool().clone()

        else:
            warnings.warn(f"Invalid ref_mask shape {tuple(ref_mask.shape)}; "
                          f"expected [K] = ({K},) or [B, K] = ({B}, {K}). "
                          "Falling back to no masking.", UserWarning, stacklevel=2,)
            avail = torch.ones(B, K, dtype=torch.bool, device=device)

        # Sanity check: each sample must have at least one usable ref.
        # A row with all False means the sample is fully masked.
        all_masked = ~avail.any(dim=1)
        if all_masked.any():
            bad_indices = all_masked.nonzero(as_tuple=False).flatten().tolist()
            warnings.warn(f"ref_mask has fully masked sample(s) at batch indices {bad_indices}; "
                           "falling back to no masking for those sample(s).",
                           UserWarning, stacklevel=2,)
            avail[all_masked] = True
        return ~avail

    def _build_tok_ood(self, tok_smp: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        B = tok_smp.shape[0]
        tok_ood = self.tok_ood.expand(B, -1, -1)

        if "sample" in self.residual_init:
            smp_summary = tok_smp.mean(dim=1, keepdim=True)
            tok_ood = tok_ood + self.ood_seed(self.ood_seed_LN(smp_summary))

        if "depth" in self.residual_init:
            # depth is useful mainly for uncertainty/calibration, not as a direct
            # oracle for OOD fraction. The /10 keeps log-counts in a tame range.
            log_depth = torch.log1p(depth).to(dtype=tok_smp.dtype) / 10.0
            tok_ood = tok_ood + self.dep_encode(log_depth).unsqueeze(1)

        return tok_ood

    def forward(self, X_smp: torch.Tensor,
                      X_ref: torch.Tensor,
                      simplex: Optional[Union[str, float, int]] = None,
                      ref_mask: Optional[torch.Tensor] = None,
                      return_extra: bool = False,):
        device = X_smp.device
        B, C   = X_smp.shape
        B2, K, C2 = X_ref.shape

        if C != self.n_chann:
            raise ValueError(f"X_smp expected {self.n_chann} channels, got {C}")
        if B2 != B:
            raise ValueError("X_ref batch size must match X_smp")
        if C2 != self.n_chann:
            raise ValueError(f"X_ref expected {self.n_chann} channels, got {C2}")

        depth = X_smp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        X_smp = (X_smp / depth * 50000.0).unsqueeze(-1)
        tok_smp = self.smp_encode(X_smp)

        tok_smp = tok_smp + self.ctx__embed[self.ctx__idx].sum(dim=1).unsqueeze(0)

        for blk in self.ss_s_attn:
            tok_smp = blk(tok_smp)

        ref_pad = self._format_ref_pad_mask(ref_mask, B, K, device)  ### True means hidden everywhere.
        tok_ref_real = self.ref_encode(X_ref).masked_fill(ref_pad.unsqueeze(-1), 0.0)

        if self.use_tok_ood:
            tok_ood = self._build_tok_ood(tok_smp, depth)
            tok_ref = torch.cat([tok_ref_real, tok_ood], dim=1)
            ref_pad_ext = torch.cat([ref_pad, ref_pad.new_zeros(B, 1)], dim=1)
        else:
            tok_ref = tok_ref_real
            ref_pad_ext = ref_pad

        # Reference path: real refs and residual token compete/cooperate through
        # self-attn, then each token queries the sample evidence. The optional
        # sample->reference path lets sample tokens become reference-aware before
        # later ref->sample layers, while excluding tok_ood from its keys.
        for i, (blk_s, blk_x) in enumerate(zip(self.rr_s_attn, self.rs_x_attn)):
            tok_ref = blk_s(tok_ref, key_padding_mask=ref_pad_ext)
            tok_ref = blk_x(tok_ref, tok_smp, qq_mask=ref_pad_ext)

            if i < len(self.sr_x_attn):              ### excludes tok_ood.
                tok_smp = self.sr_x_attn[i](tok_smp, tok_ref[:,:K,:], kv_mask=ref_pad,)

        logits = self.compo_head(self.compo_norm(tok_ref)).squeeze(-1)
        if self.use_tok_ood:
            logits[:, K] = logits[:, K] + self.ood_lg_bias
        logits = logits.masked_fill(ref_pad_ext, torch.finfo(logits.dtype).min)

        composition = self._apply_simplex(logits, simplex=simplex)

        if return_extra:
            extra = {"composition": composition,
                     "known_composition": composition[:, :K],
                     "ood_mass": composition[:, K] if self.use_tok_ood else composition.new_zeros(B),
                     "logits": logits,
                     "known_ref_mask": ~ref_pad,
                     "extended_ref_mask": ~ref_pad_ext,
                     "depth": depth.squeeze(-1),}
            return extra

        return composition
