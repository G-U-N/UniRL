# Copyright 2025 Fu-Yun Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Simplified RMSNorm implementation:
        - Computes: x * rsqrt(mean(x^2) + eps) along the last dimension.
        - Then applies a learnable weight per feature.
        """
        super().__init__()
        self.eps = eps
        # Learnable gain parameter (weight) of shape (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS = sqrt(mean(x^2) + eps) along the last dimension
        # keepdim=True so that broadcasting back to x’s shape works correctly
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to float32 for stable norm calculation if necessary
        normalized = self._norm(x.float()).type_as(x)
        # Multiply by learnable weight (per-feature gain)
        return normalized * self.weight



class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        """
        SwiGLU feed-forward:
        - w1: projects to hidden_features, then apply SiLU.
        - w2: projects to hidden_features (for gating).
        - Multiply SiLU(w1(x)) * w2(x), then project back to out_features via w3.
        """
        super().__init__()
        out_features = out_features or in_features
        # Default hidden_features = 2 * in_features (common choice)
        hidden_features = hidden_features or (2 * in_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(w1(x)) * w2(x) --> then final projection
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int = None,
        use_rope: bool = True,
    ):
        """
        Single self-attention block with optional RoPE and Grouped Q/K heads (GQA).
        - embed_dim: total embedding dimension.
        - num_query_heads: number of Q heads.
        - num_kv_heads: number of K/V heads (<= num_query_heads). If None, defaults to num_query_heads.
        """
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads or num_query_heads

        # Ensure divisibility for head dimensions
        assert embed_dim % self.num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert embed_dim % self.num_kv_heads == 0, "embed_dim must be divisible by num_kv_heads"

        self.head_dim = embed_dim // self.num_query_heads      # Q head dimension
        self.kv_head_dim = embed_dim // self.num_kv_heads      # K/V head dimension

        # Linear projections for Q, K, V (project to embed_dim and then reshape)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection (from concat of all heads back to embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Apply RMSNorm per-head dimension before RoPE
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.kv_head_dim)

        self.use_rope = use_rope
        if use_rope:
            assert 0
            # Separate RoPE modules for Q and K (they may have different head dims)
            # self.rope_q = RotaryPositionalEmbedding(self.head_dim)
            # self.rope_k = RotaryPositionalEmbedding(self.kv_head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: shape (batch, seq_len, embed_dim)
        attn_mask (optional): shape (batch, num_heads, seq_len, seq_len) or broadcastable to that.
        """
        batch, length, _ = x.shape

        # 1. Project input to Q, K, V and reshape to (batch, seq_len, num_heads, head_dim)
        q = self.q_proj(x).view(batch, length, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.kv_head_dim)

        # 2. Apply RMSNorm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. Apply rotary positional embeddings if enabled
        if self.use_rope:
            assert 0
            # q = self.rope_q(q, seq_len=length)
            # k = self.rope_k(k, seq_len=length)

        # 4. If using fewer K/V heads, repeat them to match query heads (GQA)
        if self.num_kv_heads != self.num_query_heads:
            # For simple repetition, require head_dim == kv_head_dim
            assert self.head_dim == self.kv_head_dim, "For GQA, head_dim and kv_head_dim must match"
            repeat_factor = self.num_query_heads // self.num_kv_heads
            # Expand dims and repeat
            k = k.unsqueeze(3).repeat(1, 1, 1, repeat_factor, 1)
            v = v.unsqueeze(3).repeat(1, 1, 1, repeat_factor, 1)
            # Reshape back to (batch, seq_len, num_query_heads, head_dim)
            k = k.view(batch, length, self.num_query_heads, self.kv_head_dim)
            v = v.view(batch, length, self.num_query_heads, self.kv_head_dim)

        # 5. Transpose to (batch, num_heads, seq_len, head_dim) for attention call
        q = q.transpose(1, 2)  # (batch, num_query_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch, num_query_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_query_heads, seq_len, head_dim)

        # 6. Perform scaled dot-product attention
        #    attn_mask should be (batch, num_heads, seq_len, seq_len) or broadcastable
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # 7. Concatenate heads: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch, length, -1)

        # 8. Final linear projection back to embed_dim
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int = None,
        use_rope: bool = True,
    ):
        """
        Single Transformer block with Pre-Norm:
        - RMSNorm → Self-Attention → Residual
        - RMSNorm → SwiGLU (FFN) → Residual
        """
        super().__init__()
        # Pre-attention normalization
        self.norm1 = RMSNorm(embed_dim)
        self.attn = SelfAttentionBlock(embed_dim, num_query_heads, num_kv_heads, use_rope=use_rope)
        # Pre-FFN normalization
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual
        x = x + self.attn(self.norm1(x))
        # Feed-forward sub-layer with residual
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformerProjector(nn.Module):
    def __init__(self, config):
        """
        Vision Transformer–style projector:
        - A stack of TransformerBlocks operating on config.hidden_size
        - Optionally uses fewer K/V heads (GQA) and RoPE
        - Final RMSNorm + linear output to config.gen_hidden_size
        """
        super().__init__()
        embed_dim_internal = config.hidden_size

        # Validate that hidden_size is divisible by num_query_heads
        self.num_query_heads = getattr(config, 'gen_projector_transformer_heads', 8)
        if embed_dim_internal % self.num_query_heads != 0:
            raise ValueError("hidden_size must be divisible by num_query_heads")

        # Optionally use fewer K/V heads
        self.num_kv_heads = getattr(config, 'gen_projector_transformer_kv_heads', None)
        if self.num_kv_heads is not None:
            if self.num_query_heads % self.num_kv_heads != 0:
                raise ValueError("num_query_heads must be a multiple of num_kv_heads")
            if embed_dim_internal % self.num_kv_heads != 0:
                raise ValueError("hidden_size must be divisible by num_kv_heads")
            # Ensure head_dim == kv_head_dim for simple repetition
            if (embed_dim_internal // self.num_query_heads) != (embed_dim_internal // self.num_kv_heads):
                raise ValueError("For GQA, head_dim must equal kv_head_dim")

        self.use_rope = getattr(config, 'gen_projector_transformer_use_rope', False)
        self.num_layers = getattr(config, 'gen_projector_transformer_layers', 12)

        # Build a stack of TransformerBlocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim_internal,
                self.num_query_heads,
                self.num_kv_heads,
                self.use_rope
            )
            for _ in range(self.num_layers)
        ])

        # Final normalization and output projection
        self.norm_final = RMSNorm(embed_dim_internal)
        if not isinstance(embed_dim_internal, int) or not isinstance(config.sana_embeds_hidden_size, int):
            raise ValueError(f"Invalid dimensions: embed_dim_internal={embed_dim_internal}, sana_embeds_hidden_size={config.sana_embeds_hidden_size}")
        self.output_linear = nn.Linear(embed_dim_internal, config.sana_embeds_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, seq_len, hidden_size)
        Returns: (batch, seq_len, gen_hidden_size)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        return self.output_linear(x)
