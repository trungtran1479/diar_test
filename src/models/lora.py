"""Minimal LoRA linear that exposes the adapted matrix as `.weight`, not
just via `forward()`.

Why not `peft`: `peft.get_peft_model(..., target_modules=["q_proj",
"v_proj"])` works by replacing each target `nn.Linear` with a wrapper whose
`forward()` adds the LoRA delta. That is invisible to
`WavLMAttention.torch_multi_head_self_attention`, which does NOT call
`self.q_proj(x)` at all -- it reads the raw tensors directly:

    F.multi_head_attention_forward(..., use_separate_proj_weight=True,
        q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight)

so a peft-wrapped q_proj's `forward()` is simply never invoked -- `.weight`
resolves straight to the FROZEN base weight, the LoRA computation never
runs, and the whole encoder output ends up with no gradient path to
lora_A/lora_B at all (reproduced directly: layer-0 output requires_grad
was False even with LoRA "applied"). `LoRAWeightLinear` fixes this by
making `.weight` a computed PROPERTY (`base_weight + lora_B @ lora_A *
scaling`), not a stored tensor -- any caller that reads `.weight` directly,
not just one that calls `forward()`, gets the adapted matrix.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWeightLinear(nn.Module):
    """Wraps an existing `nn.Linear`. `lora_B` zero-initialized (the other
    factor, `lora_A`, uses the usual Kaiming init) so `.weight` exactly
    equals the base weight until any training happens."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_linear
        for p in self.base_layer.parameters():
            p.requires_grad = False
        out_features, in_features = base_linear.weight.shape
        # Match the base weight's device/dtype -- apply_lora() runs AFTER
        # model.to(device), so base_linear.weight is already on GPU; a
        # bare torch.zeros(...) here defaults to CPU/fp32 and breaks the
        # very next forward with a device-mismatch error.
        factory_kwargs = {"device": base_linear.weight.device, "dtype": base_linear.weight.dtype}
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only reached by a caller that DOES use forward() normally --
        # WavLM's attention doesn't, but this keeps the module usable
        # anywhere else `.weight` isn't read directly.
        return F.linear(self.lora_dropout(x), self.weight, self.bias)
