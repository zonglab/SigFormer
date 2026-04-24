import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect


class MLP(nn.Module):
    """Simple transformer FFN block with pre-norm and residual."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y


class SelfAttentionBlock(nn.Module):
    """Standard self-attention block used for sample tokens or reference tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y, _ = self.attn(query=y, key=y, value=y, need_weights=False)
        x = x + self.dropout(y)
        x = self.mlp(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Reference tokens attend to sample tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.norm_keyval = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, ref_tokens: torch.Tensor, sample_tokens: torch.Tensor) -> torch.Tensor:
        query = self.norm_query(ref_tokens)
        keyval = self.norm_keyval(sample_tokens)
        attn_out, _ = self.attn(query=query, key=keyval, value=keyval, need_weights=False)
        ref_tokens = ref_tokens + self.dropout(attn_out)
        ref_tokens = self.mlp(ref_tokens)
        return ref_tokens


class RefInteractionBlock(nn.Module):
    """Reference self-attention followed by cross-attention to the sample."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.cross_attn = CrossAttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, ref_tokens: torch.Tensor, sample_tokens: torch.Tensor) -> torch.Tensor:
        ref_tokens = self.self_attn(ref_tokens)
        ref_tokens = self.cross_attn(ref_tokens, sample_tokens)
        return ref_tokens


class SigFormerCore(nn.Module):
    """
    SigFormer core model.

    Main ideas kept from your V5 design:
    1. sample profile -> 96 sample tokens
    2. each reference signature -> one reference token
    3. sample tower denoises sample tokens
    4. reference tower mixes signatures and reads sample context
    5. composition head predicts mixture weights on simplex
    6. confidence head is kept detached from the main trunk by default

    V6 changes:
    - entmax alpha is no longer trainable
    - confidence head uses a few more explicit diagnostic features
    - optional auxiliary outputs for OOD / residual diagnostics
    """

    def __init__(
        self,
        n_chann: int = 96,
        d_model: int = 256,
        n_heads: int = 8,
        n_L_smp: int = 2,
        n_L_ref: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        confidence_detach_backbone: bool = True,
    ):
        super().__init__()

        self.n_chann = n_chann
        self.d_model = d_model
        self.n_heads = n_heads
        self.confidence_detach_backbone = confidence_detach_backbone

        self.ctx_embed = nn.Embedding(num_embeddings=n_chann, embedding_dim=d_model)
        self.sample_value_proj = nn.Linear(1, d_model)
        self.ref_proj = nn.Linear(n_chann, d_model)

        self.sample_blocks = nn.ModuleList([
            SelfAttentionBlock(dim=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_L_smp)
        ])
        self.ref_blocks = nn.ModuleList([
            RefInteractionBlock(dim=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_L_ref)
        ])

        self.composition_norm = nn.LayerNorm(d_model)
        self.composition_head = nn.Linear(d_model, 1)

        # extra features for confidence: log_depth, residual_cos, sample_entropy,
        # predicted_weight, predicted_logit
        self.conf_extra_dim = 5
        self.conf_input_dim = d_model + self.conf_extra_dim
        self.confidence_norm = nn.LayerNorm(self.conf_input_dim)
        self.confidence_head = nn.Linear(self.conf_input_dim, 1)

        # keep the old key name for checkpoint friendliness, but it is a buffer now.
        self.register_buffer("entmax_alpha", torch.tensor(1.5, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.sample_value_proj.weight, std=0.02)
        nn.init.zeros_(self.sample_value_proj.bias)
        nn.init.trunc_normal_(self.ref_proj.weight, std=0.02)
        nn.init.zeros_(self.ref_proj.bias)
        nn.init.trunc_normal_(self.composition_head.weight, std=0.02)
        nn.init.zeros_(self.composition_head.bias)
        nn.init.trunc_normal_(self.confidence_head.weight, std=0.02)
        nn.init.zeros_(self.confidence_head.bias)
        nn.init.trunc_normal_(self.ctx_embed.weight, std=0.02)

    @staticmethod
    def normalize_profile(x_raw: torch.Tensor, target_depth: float = 50000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize raw counts or already-normalized input to a common scale."""
        depth = x_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_norm = x_raw / depth * target_depth
        return x_norm, depth

    @staticmethod
    def _vector_entropy(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        prob = x / x.sum(dim=-1, keepdim=True).clamp(min=eps)
        prob = prob.clamp(min=eps)
        return -(prob * prob.log()).sum(dim=-1)

    def _apply_simplex(self, logits: torch.Tensor, simplex: Optional[Union[str, float, int]]) -> torch.Tensor:
        if simplex is None:
            simplex = "entmax15"

        if isinstance(simplex, (float, int)):
            alpha = float(simplex)
            alpha = max(1.0, min(2.0, alpha))
            return entmax_bisect(logits, alpha=alpha, dim=-1)

        simplex = str(simplex).lower()
        if simplex in {"entmax", "entmax15"}:
            return entmax_bisect(logits, alpha=1.5, dim=-1)
        if simplex == "sparsemax":
            return entmax_bisect(logits, alpha=2.0, dim=-1)
        if simplex == "softmax":
            return F.softmax(logits, dim=-1)
        if simplex in {"none", "logits"}:
            return logits
        if simplex == "learned":
            # Backward compatibility. The alpha is no longer trainable.
            return entmax_bisect(logits, alpha=float(self.entmax_alpha.item()), dim=-1)

        raise ValueError(f"Unsupported simplex mode: {simplex}")

    @staticmethod
    def build_masked_composition(
        composition: torch.Tensor,
        confidence: torch.Tensor,
        min_composition: float = 1e-3,
        confidence_threshold: float = 0.35,
        mode: str = "soft",
        soft_power: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Confidence-aware post-processing for tiny tails.
        This is intentionally inference-side by default, so it does not destabilize training.
        """
        if mode == "hard":
            keep = (composition >= min_composition) & (confidence >= confidence_threshold)
            masked = composition * keep.to(composition.dtype)
        elif mode == "soft":
            gate = torch.clamp(confidence, min=0.0, max=1.0) ** soft_power
            masked = composition * gate
            masked = torch.where(masked >= min_composition, masked, torch.zeros_like(masked))
        else:
            raise ValueError(f"Unsupported mask mode: {mode}")

        mass = masked.sum(dim=-1, keepdim=True)
        fallback = composition / composition.sum(dim=-1, keepdim=True).clamp(min=eps)
        masked = torch.where(mass > eps, masked / mass.clamp(min=eps), fallback)
        return masked

    def forward(
        self,
        X_smp_raw: torch.Tensor,
        X_ref_sig: torch.Tensor,
        simplex: Optional[Union[str, float, int]] = "entmax15",
        return_aux: bool = False,
    ):
        device = X_smp_raw.device
        B, C = X_smp_raw.shape
        B_ref, N_ref, C_ref = X_ref_sig.shape

        assert C == self.n_chann, f"X_smp_raw expected {self.n_chann} channels, got {C}"
        assert B_ref == B, "X_ref_sig batch size must match X_smp_raw"
        assert C_ref == self.n_chann, f"X_ref_sig expected {self.n_chann} channels, got {C_ref}"

        sample_norm, depth = self.normalize_profile(X_smp_raw)

        sample_values = sample_norm.unsqueeze(-1)
        sample_tokens = self.sample_value_proj(sample_values)
        channel_index = torch.arange(self.n_chann, device=device).unsqueeze(0).expand(B, -1)
        sample_tokens = sample_tokens + self.ctx_embed(channel_index)

        ref_tokens = self.ref_proj(X_ref_sig)

        for block in self.sample_blocks:
            sample_tokens = block(sample_tokens)
        for block in self.ref_blocks:
            ref_tokens = block(ref_tokens, sample_tokens)

        composition_logits = self.composition_head(self.composition_norm(ref_tokens)).squeeze(-1)
        composition = self._apply_simplex(composition_logits, simplex=simplex)

        if composition.dim() != 2:
            raise RuntimeError("Composition output is expected to be [batch, n_ref].")

        recon_profile = torch.bmm(composition.unsqueeze(1), X_ref_sig).squeeze(1)
        residual = sample_norm - recon_profile
        residual_cosine = 1.0 - F.cosine_similarity(sample_norm, recon_profile, dim=-1, eps=1e-8)
        residual_l1 = residual.abs().sum(dim=-1)
        positive_residual = torch.relu(residual)
        novelty_profile = positive_residual / positive_residual.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        sample_entropy = self._vector_entropy(sample_norm)

        if self.confidence_detach_backbone:
            conf_ref_tokens = ref_tokens.detach()
            conf_logits = composition_logits.detach()
            conf_weight = composition.detach()
            conf_depth = depth.detach()
            conf_residual = residual_cosine.detach()
            conf_entropy = sample_entropy.detach()
        else:
            conf_ref_tokens = ref_tokens
            conf_logits = composition_logits
            conf_weight = composition
            conf_depth = depth
            conf_residual = residual_cosine
            conf_entropy = sample_entropy

        log_depth = conf_depth.clamp(min=1.0).log()
        extra_feat = torch.stack([
            log_depth.squeeze(-1),
            conf_residual,
            conf_entropy,
            torch.zeros_like(conf_residual),
            torch.zeros_like(conf_residual),
        ], dim=-1)
        extra_feat = extra_feat.unsqueeze(1).expand(B, N_ref, self.conf_extra_dim).clone()
        extra_feat[:, :, 3] = conf_weight
        extra_feat[:, :, 4] = conf_logits

        confidence_input = torch.cat([conf_ref_tokens, extra_feat], dim=-1)
        confidence_input = self.confidence_norm(confidence_input)
        confidence_logits = self.confidence_head(confidence_input).squeeze(-1)
        confidence = torch.sigmoid(confidence_logits)

        if not return_aux:
            return composition, confidence

        aux: Dict[str, torch.Tensor] = {
            "composition_logits": composition_logits,
            "recon_profile": recon_profile,
            "sample_profile_norm": sample_norm,
            "depth": depth.squeeze(-1),
            "residual": residual,
            "residual_cosine": residual_cosine,
            "residual_l1": residual_l1,
            "novelty_profile": novelty_profile,
            "positive_residual_mass": positive_residual.sum(dim=-1),
            "sample_entropy": sample_entropy,
        }
        return composition, confidence, aux
