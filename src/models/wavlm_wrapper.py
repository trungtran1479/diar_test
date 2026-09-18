import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class WavLMBackbone(nn.Module):
    """Wraps a pretrained HuggingFace WavLM encoder for speaker counting —
    the OFFLINE counterpart to FastConformerBackbone (see
    fastconformer_wrapper.py for the causal-streaming version this mirrors).

    WavLM is bidirectional by construction (full self-attention, no causal
    masking option in HF's implementation), so unlike FastConformerBackbone
    there is no `_check_causal` step: any `microsoft/wavlm-*` checkpoint is
    offline/full-context by architecture, not by a config flag that could
    silently point at a streaming variant.

    Like FastConformer, WavLM is a FLAT encoder (uniform width/resolution at
    every layer, no U-shaped multi-stack structure), so the hypercolumn is
    built over evenly-spaced transformer LAYERS via `output_hidden_states`,
    the same pattern as FastConformerBackbone's layer taps.

    Frame rate: WavLM-Large's CNN feature encoder downsamples 16 kHz audio by
    320x -> native 50 Hz, neither Zipformer's 25 Hz nor FastConformer's
    12.5 Hz-native. Downsampled (not upsampled) to the project's 25 Hz
    convention via average-pooling, so the existing TCN head config
    (dilations 1,2,4,8,16,32 @ 25 Hz -> 5.08 s receptive field) plugs in
    completely unchanged and stays numerically comparable to every other
    reported table. This is a deliberate choice, not the only defensible one
    (keeping native 50 Hz and doubling the dilation schedule would also
    preserve the 5.08 s real-time receptive field) — see
    zipcount-wavlm-offline-handoff memory for the reasoning.

    Takes RAW WAVEFORM input (WavLM's own CNN front-end does the framing),
    NOT log-mel fbank — see extract_wavlm_waveform_features in
    src/data/feature_extractor.py. Do not feed this any of the project's
    fbank extractors; WavLM has no concept of an external mel filterbank.
    """

    def __init__(
        self,
        pretrained_model: str = "microsoft/wavlm-large",
        freeze: bool = True,
        collect_layers: bool = True,
        num_hypercolumn_layers: int = 6,
        target_frame_rate_hz: float = 25.0,
        tap_indices: Optional[List[int]] = None,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        from transformers import WavLMModel

        self.encoder = WavLMModel.from_pretrained(pretrained_model)
        self.is_causal = False  # WavLM is bidirectional by construction

        cfg = self.encoder.config
        self.collect_layers = collect_layers
        self.d_model = int(cfg.hidden_size)
        n_layers = int(cfg.num_hidden_layers)

        # Verify (don't assume) the CNN front-end's downsampling factor —
        # a different WavLM variant/config would silently desync frame-rate
        # arithmetic otherwise.
        conv_stride_product = 1
        for s in cfg.conv_stride:
            conv_stride_product *= int(s)
        self.native_frame_rate_hz = 16000.0 / conv_stride_product
        downsample = self.native_frame_rate_hz / target_frame_rate_hz
        if abs(downsample - round(downsample)) > 1e-6:
            raise ValueError(
                f"target_frame_rate_hz={target_frame_rate_hz} is not an "
                f"integer divisor of the encoder's native rate "
                f"{self.native_frame_rate_hz}Hz (conv_stride product="
                f"{conv_stride_product})"
            )
        self.downsample_factor = int(round(downsample))
        if self.downsample_factor < 1:
            raise ValueError(
                f"target_frame_rate_hz={target_frame_rate_hz} is above the "
                f"encoder's native rate {self.native_frame_rate_hz}Hz — "
                "upsampling is not implemented here (only the FastConformer "
                "wrapper upsamples; WavLM's native rate is already at or "
                "above the project's 25Hz convention)"
            )

        if self.collect_layers:
            if tap_indices is not None:
                bad = [i for i in tap_indices if not (0 <= i < n_layers)]
                if bad:
                    raise ValueError(
                        f"tap_indices {bad} out of range for {n_layers}-layer "
                        f"encoder (valid range 0..{n_layers - 1})"
                    )
                self.tap_indices = sorted(set(int(i) for i in tap_indices))
            else:
                idxs = torch.linspace(0, n_layers - 1, num_hypercolumn_layers)
                self.tap_indices = sorted(set(idxs.round().long().tolist()))
            if len(self.tap_indices) != num_hypercolumn_layers:
                print(
                    f"Warning: requested {num_hypercolumn_layers} hypercolumn "
                    f"taps, deduped/overridden to {len(self.tap_indices)} unique "
                    f"layer indices {self.tap_indices} (only {n_layers} layers total)"
                )
            self.stack_dims = [self.d_model] * len(self.tap_indices)
        else:
            self.tap_indices = []
            self.stack_dims = None

        self.output_dim = (
            sum(self.stack_dims) if self.stack_dims else self.d_model
        )

        if freeze:
            self._freeze_params(unfreeze_last_n)

        n_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(
            f"Built WavLM backbone: {pretrained_model}, {n_layers} layers, "
            f"d_model={self.d_model}, causal=False (bidirectional by "
            f"construction), native_rate={self.native_frame_rate_hz}Hz -> "
            f"downsample /{self.downsample_factor} -> {target_frame_rate_hz}Hz, "
            f"hypercolumn_taps={self.tap_indices if self.collect_layers else 'none (final layer only)'} "
            f"output_dim={self.output_dim}, backbone_trainable_params={n_trainable}"
        )

    def _freeze_params(self, unfreeze_last_n: int):
        """Freeze the whole encoder, then unfreeze the last `unfreeze_last_n`
        transformer layers (self.encoder.encoder.layers) if requested. The
        CNN feature extractor and feature projection stay frozen
        unconditionally -- same convention as StreamingZipformerEncoder's
        `encoder_embed`, which `_freeze_params` there never unfreezes either.

        Previously `unfreeze_last_n` was accepted by the YAML schema
        (`model.encoder.unfreeze_last_n`) and printed by train.py's
        stage2_unfreeze_last_stacks branch, but never actually reached this
        class -- WavLMBackbone had no such parameter and this method didn't
        exist, so `freeze=True` always froze the ENTIRE encoder regardless
        of what the config claimed. A `stage2_unfreeze_last_stacks` WavLM
        run was silently training head-only.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.trainable_layer_indices: List[int] = []
        if unfreeze_last_n > 0:
            layers = self.encoder.encoder.layers
            num_layers = len(layers)
            start_idx = max(0, num_layers - unfreeze_last_n)
            for i in range(start_idx, num_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True
            self.trainable_layer_indices = list(range(start_idx, num_layers))
            print(f"Unfrozen last {num_layers - start_idx} WavLM transformer layers "
                  f"(indices {self.trainable_layer_indices}).")

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint each transformer layer's activations. Must be called
        AFTER _freeze_params/train-mode setup, not before: HF's
        GradientCheckpointingLayer gates checkpointing per-layer on that
        layer's OWN `self.training` flag (see modeling_layers.py), so this
        correctly checkpoints only layers that are actually unfrozen/in
        train mode -- a frozen layer has `training=False` and is never
        checkpointed regardless of this call, matching where the memory
        saving is actually needed.

        `use_reentrant=False` is required, not the HF default
        (use_reentrant=True): with partial unfreezing, the hidden_states
        flowing INTO the first unfrozen layer come from a frozen
        predecessor and so do not themselves require grad. Reentrant
        checkpointing needs at least one requires-grad INPUT to the
        checkpointed function and raises `RuntimeError: element 0 of
        tensors does not require grad and does not have a grad_fn`
        otherwise -- reproduced directly with unfreeze_last_n=2. Non-
        reentrant checkpointing has no such requirement (verified
        gradients match a non-checkpointed run exactly, same seeds).
        """
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def apply_lora(
        self,
        target_modules: List[str],
        layer_indices: Optional[List[int]] = None,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        """Wrap `target_modules` (e.g. ["q_proj", "v_proj"]) in every layer
        in `layer_indices` (default: all layers) with `LoRAWeightLinear`
        (see src/models/lora.py), leaving every base WavLM weight frozen.
        Unlike `unfreeze_last_n` (which only ever reaches the last few
        layers), this can adapt the WHOLE encoder depth, including the
        early layers the hypercolumn's front taps (indices 0, 5, 9) draw
        from -- the gap `unfreeze_last_n` structurally could not test.

        Does NOT use `peft`: `peft.get_peft_model(...,
        target_modules=[...])` wraps each target Linear's `forward()`, but
        `WavLMAttention.torch_multi_head_self_attention` never calls
        `self.q_proj(x)` -- it reads `self.q_proj.weight` directly and
        feeds it into `F.multi_head_attention_forward(...,
        q_proj_weight=...)`. A peft-wrapped q_proj's `forward()` is simply
        never invoked, so its `.weight` resolves to the frozen base weight
        and the LoRA path is completely inert -- reproduced directly
        (layer-0 output had requires_grad=False despite "successful"
        peft wrapping). `LoRAWeightLinear` exposes `.weight` as a computed
        property instead, so direct-tensor-read call sites see the adapted
        matrix too. See src/models/lora.py's module docstring.

        MUST be called on an already-loaded checkpoint's backbone (i.e.
        AFTER `--init-from` copies pretrained weights into `self.encoder`,
        not before): a state_dict saved before wrapping wouldn't match a
        post-wrap module's renamed keys (`....base_layer.weight` etc).
        Loading first, then wrapping, means the checkpoint load path never
        has to know LoRA exists.

        Invariants verified here: the base encoder must already be fully
        frozen, exactly `len(layer_indices) * len(target_modules)` Linears
        get wrapped, and exactly 2x that many parameter TENSORS (lora_A +
        lora_B per wrapped Linear) are trainable. `lora_B` is zero-init, so
        every wrapped Linear's `.weight` is EXACTLY (not approximately)
        equal to its base weight at this point -- checked directly here;
        the caller should still additionally verify this holds at the
        full-model ACTIVATION level (forward pass, not just this one
        property), since that also confirms the wrapped modules are
        actually reachable from the model's real forward path -- exactly
        what was silently broken in the peft version.
        """
        from src.models.lora import LoRAWeightLinear

        if any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError(
                "apply_lora: the base encoder must be fully frozen before "
                "wrapping (found a trainable param) -- construct with "
                "freeze=True, unfreeze_last_n=0"
            )
        n_layers = len(self.encoder.encoder.layers)
        if layer_indices is None:
            layer_indices = list(range(n_layers))
        bad = [i for i in layer_indices if not (0 <= i < n_layers)]
        if bad:
            raise ValueError(f"apply_lora: layer_indices {bad} out of range for {n_layers} layers")

        wrapped_names = []
        for i in layer_indices:
            attn = self.encoder.encoder.layers[i].attention
            for name in target_modules:
                if not hasattr(attn, name):
                    raise ValueError(f"apply_lora: layer {i}'s attention has no module {name!r}")
                base = getattr(attn, name)
                if not isinstance(base, torch.nn.Linear):
                    raise ValueError(
                        f"apply_lora: layer {i}.attention.{name} is "
                        f"{type(base).__name__}, not nn.Linear"
                    )
                setattr(attn, name, LoRAWeightLinear(base, rank=rank, alpha=alpha, dropout=dropout))
                wrapped_names.append(f"layers.{i}.attention.{name}")

        self.lora_applied = True
        self.lora_target_modules = list(target_modules)
        self.lora_layer_indices = list(layer_indices)

        expected_wrapped = len(layer_indices) * len(target_modules)
        if len(wrapped_names) != expected_wrapped:
            raise RuntimeError(
                f"apply_lora: expected exactly {expected_wrapped} wrapped Linears "
                f"({len(layer_indices)} layers x {len(target_modules)} target_modules "
                f"{target_modules}), found {len(wrapped_names)}"
            )
        trainable_names = [n for n, p in self.encoder.named_parameters() if p.requires_grad]
        if len(trainable_names) != expected_wrapped * 2:
            raise RuntimeError(
                f"apply_lora: expected exactly {expected_wrapped * 2} trainable tensors "
                f"(lora_A + lora_B per wrapped Linear), found {len(trainable_names)}: "
                f"{trainable_names}"
            )
        non_lora_trainable = [n for n in trainable_names if "lora_" not in n]
        if non_lora_trainable:
            raise RuntimeError(
                f"apply_lora: non-LoRA parameters ended up trainable -- topology drift: "
                f"{non_lora_trainable}"
            )
        # lora_B zero-init -> every wrapped Linear's .weight must be EXACTLY
        # its base weight right now (not just close), checked directly
        # rather than trusting the zero-init call to have actually run.
        for i in layer_indices:
            attn = self.encoder.encoder.layers[i].attention
            for name in target_modules:
                mod = getattr(attn, name)
                if not torch.equal(mod.weight, mod.base_layer.weight):
                    raise RuntimeError(
                        f"apply_lora: layers.{i}.attention.{name}.weight != base_layer.weight "
                        "at init despite lora_B being zero-initialized"
                    )
        n_lora_params = sum(p.numel() for n, p in self.encoder.named_parameters() if p.requires_grad)
        print(
            f"LoRA applied: target_modules={target_modules} "
            f"layers={layer_indices} rank={rank} alpha={alpha} dropout={dropout} -> "
            f"{len(wrapped_names)} wrapped Linears, {len(trainable_names)} trainable "
            f"tensors, {n_lora_params} trainable LoRA params"
        )

    def forward_features(
        self, features: torch.Tensor, feature_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: [B, T_samples] raw waveform, already zero-mean/unit-var
            normalized per-utterance (see extract_wavlm_waveform_features —
            do NOT feed unnormalized waveform or log-mel fbank here).
        feature_lens: [B] valid sample counts (NOT frame counts).

        Returns [B, T_enc, D] at target_frame_rate_hz, D = sum(stack_dims)
        hypercolumn (e.g. 6*1024=6144) or d_model (1024) if
        collect_layers=False.
        """
        device = features.device
        T = features.size(1)
        attention_mask = (
            torch.arange(T, device=device)[None, :] < feature_lens.to(device)[:, None]
        ).long()

        out = self.encoder(
            input_values=features,
            attention_mask=attention_mask,
            output_hidden_states=self.collect_layers,
        )
        encoder_out_lens = self.encoder._get_feat_extract_output_lengths(
            feature_lens.to(device)
        )

        if self.collect_layers:
            # hidden_states[0] is the pre-transformer (post feature-projection)
            # embedding, not a transformer layer output; layer i's output is
            # hidden_states[i + 1].
            pieces = [out.hidden_states[i + 1] for i in self.tap_indices]  # each [B, T_enc, d_model]
            h = torch.cat(pieces, dim=-1)  # [B, T_enc, output_dim]
        else:
            h = out.last_hidden_state  # [B, T_enc, d_model]

        if self.downsample_factor > 1:
            h_t = h.transpose(1, 2)  # [B, D, T_enc]
            # Trim to a multiple of the downsample factor first: avg_pool1d
            # would otherwise silently drop the same trailing remainder from
            # every sample regardless of its own valid length, which is fine
            # for lengths (floor-divide matches exactly) but would leave
            # untrimmed trailing samples out of any pooled window entirely
            # for the last group if T_enc isn't an exact multiple.
            t_trim = (h_t.size(-1) // self.downsample_factor) * self.downsample_factor
            h_t = h_t[..., :t_trim]
            h_t = F.avg_pool1d(h_t, kernel_size=self.downsample_factor,
                                stride=self.downsample_factor)
            h = h_t.transpose(1, 2)  # [B, T_enc // factor, D]
            encoder_out_lens = torch.div(
                encoder_out_lens, self.downsample_factor, rounding_mode="floor"
            )

        return h, encoder_out_lens
