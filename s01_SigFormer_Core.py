import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect

#################################################################################
# Basic building blocks: MLP, Self-Attn, Cross-Attn blocks
#################################################################################

class MLP(nn.Module):
    """
    Simple position-wise feedforward:
    x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout + Residual
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        """
        y = self.norm(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y  # residual


class Self__Attn_Blk(nn.Module):
    """
    Standard Transformer self-attention block:
    - LayerNorm
    - Multihead self-attention
    - Dropout + residual
    - Position-wise MLP
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        """
        # Self-attention
        h = self.LayerNorm(x)
        h, _ = self.attn(query=h, key=h, value=h, need_weights=False,)
        x = x + self.dropout(h)  # residual
        x = self.mlp(x) #        Position-wise feedforward
        return x


class Cross_Attn_Blk(nn.Module):
    """
    Cross-attention block:
      query: reference tokens
      key/value: sample tokens
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_qq = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, tok_ref: torch.Tensor, tok_smp: torch.Tensor,) -> torch.Tensor:
        """
        tok_ref:  [batch, n_ref, dim]     (queries)
        tok_smp:  [batch, n_smp, dim]    (keys/values)
        """
        qq = self.norm_qq(tok_ref)
        kv = self.norm_kv(tok_smp)

        attn_out, _ = self.attn(query=qq, key=kv, value=kv, need_weights=False,)
        tok_ref = tok_ref + self.dropout(attn_out)  # residual on ref tokens

        # Position-wise FFN on ref tokens
        tok_ref = self.mlp(tok_ref)
        return tok_ref


class Self_Cross_Block(nn.Module):
    """
    A block acting on reference tokens:
      1) self-attention within reference tokens
      2) cross-attention from reference tokens to sample tokens

    Objective:
      - ref tokens internal information mixing
      - Then cross-attention between sample tokens to taking sample info to ref tokens
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        param=dict(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.self__attn = Self__Attn_Blk(**param)
        self.cross_attn = Cross_Attn_Blk(**param)

    def forward(self, tok_ref: torch.Tensor, tok_smp: torch.Tensor,) -> torch.Tensor:
        """
        tok_ref: [batch, n_ref, dim]
        tok_smp: [batch, n_smp, dim]
        """
        tok_ref = self.self__attn(tok_ref)
        tok_ref = self.cross_attn(tok_ref, tok_smp,)
        return tok_ref


# ============================================================
# Core model: SigFormerCore
#   - sample tokens: 96 channels (+ 1 CLS)
#   - reference tokens: each signature is一个 token
#   - backbone: sample encoder + ref blocks (self+cross)
#   - heads: composition + confidence
# ============================================================

class SigFormerCore(nn.Module):
    """
    The architecture core of SigFormer.

    Input:
      X_smp_raw: [B, 96]
          - 96-trinucleotide profile(s)
      X_ref_sig: [B, N_ref, 96]
          - reference signatures

    Output:
      composition: [B, N_ref]: Signature composition for all input samples
      confidence:  [B, N_ref]: Confidence for each composition assignments (0~1)
    """
    def __init__(self,
                 n_chann: int = 96,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_L_smp: int = 2,
                 n_L_ref: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,):
        """
        Args:
            n_chann:    96-trinucleotides
            d_model:    token embedding
            n_heads:    multi-head attention
            n_L_smp:    number of layers for    sample token encoder
            n_L_ref:    number of layers for reference token encoder
            mlp_ratio:  FFN ratio
            dropout:    dropout ratio
        """
        super().__init__()

        self.n_chann = n_chann
        self.d_model = d_model
        self.n_heads = n_heads

        # ---- Embeddings for sample tokens ----
        self.ctx__embed = nn.Embedding(num_embeddings=n_chann,
                                       embedding_dim=d_model)
        self.smp_encode = nn.Linear(1, d_model)
        self.ref_encode = nn.Linear(n_chann, d_model)

        # self-attn blocks for sample "denoising"
        self.s_attn_tower = nn.ModuleList([
            Self__Attn_Blk(dim=d_model, num_heads=n_heads,
                           mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_L_smp)
        ])

        # self+cross-attn blocks for reference tokens
        self.x_attn_tower = nn.ModuleList([
            Self_Cross_Block(dim=d_model, num_heads=n_heads,
                             mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_L_ref)
        ])

        # ---- Heads ----
        # composition head
        self.compo_norm = nn.LayerNorm(d_model)
        self.compo_head = nn.Linear(d_model, 1)  # [B, N_ref, 1] -> [B, N_ref]

        # confidence head
        self.confi_extra_dim = 2
        self.confi_input_dim = d_model + self.confi_extra_dim
        self.confi_norm = nn.LayerNorm(self.confi_input_dim)
        self.confi_head = nn.Linear(self.confi_input_dim, 1)

        # ---- Entmax alpha (learnable) ----
        # starts with 1.0, ==softmax
        # - first few epochs: entmax_alpha.requires_grad = False
        # - after getting stable, start trainign
        self.entmax_alpha = nn.Parameter(torch.tensor(1.0))
        self.entmax_alpha_min = 1.0
        self.entmax_alpha_max = 2.0

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.smp_encode.weight, std=0.02)
        nn.init.zeros_(       self.smp_encode.bias)
        nn.init.trunc_normal_(self.ref_encode.weight, std=0.02)
        nn.init.zeros_(       self.ref_encode.bias)
        nn.init.trunc_normal_(self.compo_head.weight, std=0.02)
        nn.init.zeros_(       self.compo_head.bias)
        nn.init.trunc_normal_(self.confi_head.weight, std=0.02)
        nn.init.zeros_(       self.confi_head.bias)
        nn.init.trunc_normal_(self.ctx__embed.weight, std=0.02)
        nn.init.constant_(self.entmax_alpha, 1.0)

    # --------------------------------------------------------
    # Core forward
    # --------------------------------------------------------
    def forward(self,
                X_smp_raw: torch.Tensor,
                X_ref_sig: torch.Tensor,
                simplex: Optional[str] = None,):
        """
        Args:
            X_smp_raw: [B, 96]
            X_ref_sig: [B, N_ref, 96]
            simplex:
                - None / "learned": try to make entmax_alpha learnable (clamp[1,2])
                - "softmax":        softmax
                - "entmax15":       entmax(alpha=1.5)
                - "sparsemax":      entmax(alpha=2.0)
                - "none"/"logits":  directly output logits
                - float/int:        forced to use alpha (clampped to [1,2])

        Returns:
            composition: [B, N_ref]
            confidence:  [B, N_ref]
        """
        device = X_smp_raw.device
        B, C = X_smp_raw.shape
        B2, N_ref, C2 = X_ref_sig.shape

        assert C == self.n_chann,  f"X_smp_raw expected channels={self.n_chann}, got {C}"
        assert B2 == B,            "X_ref_sig batch size must match X_smp_raw"
        assert C2 == self.n_chann, f"X_ref_sig expected channels={self.n_chann}, got {C2}"

        ########################################################################
        # 1) Build sample tokens
        ########################################################################
        depth = X_smp_raw.sum(dim=1, keepdim=True)               # [B, 1]
        X_smp_nom = X_smp_raw / depth * 50000                    # [B, 96]
        X_smp_val = X_smp_nom.unsqueeze(-1)                      # [B, 96, 1]
        tok_smp = self.smp_encode(X_smp_val)                     # [B, 96, d_model]

        emb_ctx = torch.arange(self.n_chann, device=device)      # [96]
        emb_ctx = emb_ctx.unsqueeze(0).expand(B, -1)             # [B, 96]
        tok_ctx = self.ctx__embed(emb_ctx)                       # [B, 96, d_model]

        tok_smp = tok_ctx + tok_smp                              # [B, 96, d_model]

        ########################################################################
        # 2) Build reference tokens
        ########################################################################
        tok_ref = self.ref_encode(X_ref_sig)                     # [B, N_ref, d_model]

        ########################################################################
        # 3) Attn iteration
        ########################################################################
        for blk in self.s_attn_tower:
            tok_smp = blk(tok_smp)                               # sample denoising

        for blk in self.x_attn_tower:
            tok_ref = blk(tok_ref, tok_smp)                      # cross attn

        ########################################################################
        # 4.1 Composition logits + entmax simplex
        ########################################################################
        logit_compo = self.compo_norm(tok_ref)                   # [B, N_ref, d_model]
        logit_compo = self.compo_head(logit_compo)               # [B, N_ref, 1]
        logit_compo = logit_compo.squeeze(-1)                    # [B, N_ref]

        # Unified simplex / entmax controlling logic
        if simplex is None or simplex == "learned":
            alpha = self.entmax_alpha.clamp(self.entmax_alpha_min, self.entmax_alpha_max)
            composition = entmax_bisect(logit_compo, alpha=alpha, dim=-1)

        elif isinstance(simplex, (float, int)):
            alpha = float(simplex)
            alpha = max(self.entmax_alpha_min,
                        min(self.entmax_alpha_max, alpha))
            composition = entmax_bisect(logit_compo, alpha=alpha, dim=-1)

        elif simplex == "entmax15":
            composition = entmax_bisect(logit_compo, alpha=1.5, dim=-1)

        elif simplex == "sparsemax":
            composition = entmax_bisect(logit_compo, alpha=2.0, dim=-1)

        elif simplex == "softmax":
            composition = F.softmax(logit_compo, dim=-1)

        elif simplex in ("none", "logits"):
            composition = logit_compo

        else:
            alpha = self.entmax_alpha.clamp(self.entmax_alpha_min, self.entmax_alpha_max)
            print(f"[SigFormerCore] unrecognized simplex='{simplex}', "
                  f"fallback to entmax with learned alpha={alpha.item():.3f}")
            composition = entmax_bisect(logit_compo, alpha=alpha, dim=-1)

        ########################################################################
        # 4.2 Confidence head: tok_ref + depth + reconstruction residual
        ########################################################################
        ### make sure this is detached from mainstream gradient that this won't affect normal inference
        tok_ref_det = tok_ref.detach()                           # [B, N_ref, d_model]

        ### Depth features
        depth_safe = depth.clamp(min=1.0)                        # [B, 1]
        depth_log = depth_safe.log()                             # [B, 1]

        # reconstruction
        comp_for_recon = composition.detach().unsqueeze(1)       # [B, N_ref]

        # [B, 1, N_ref] @ [B, N_ref, 96] -> [B, 1, 96] -> [B, 96]
        recon_profile = torch.bmm(comp_for_recon, X_ref_sig).squeeze(1)  # [B, 96]

        # residual score: 1 - cosine(sample_normed, recon)
        cos_sim = F.cosine_similarity(
            X_smp_nom, recon_profile, dim=-1, eps=1e-8)          # [B]
        resid_score = 1.0 - cos_sim                              # [B]

        # Extra, like residues
        extra_feat = torch.cat(
            [depth_log, resid_score.unsqueeze(-1)], dim=1)       # [B, 2]
        extra_feat = extra_feat.unsqueeze(1).expand(B, N_ref, 2) # [B, N_ref, 2]

        # combine ref token + extra features
        confi_in = torch.cat([tok_ref_det, extra_feat], dim=-1)  # [B, N_ref, d_model+2]
        confi_in = self.confi_norm(confi_in)
        logit_confi = self.confi_head(confi_in).squeeze(-1)      # [B, N_ref]

        confidence = torch.sigmoid(logit_confi)                  # [B, N_ref], 0~1

        return composition, confidence


### if __name__ == "__main__":
###     # Simple sanity check
###     B, N_ref, C = 4, 10, 96
###     model = SigFormerCore(n_chann=C,
###                           d_model=128,
###                           n_heads=4,
###                           n_L_smp=2,
###                           n_L_ref=3,
###                           mlp_ratio=4.0,
###                           dropout=0.1,
###                           use_sparsemax_default=False)
### 
###     X_smp_raw = torch.rand(B, C)
###     X_ref_sig = torch.rand(B, N_ref, C)
### 
###     composition, logit_compo, confidence = model(X_smp_raw, X_ref_sig)
### 
###     print("composition:", composition.shape)  # [B, N_ref]
###     print("logit_compo:", logit_compo.shape)  # [B, N_ref]
###     print("confidence:", confidence.shape)    # [B, N_ref]
