import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class FastConformerBackbone(nn.Module):
    """Wraps a pretrained NeMo FastConformer encoder for speaker counting.

    Analogous to StreamingZipformerEncoder, but FastConformer is a FLAT
    encoder (uniform-width layers, no U-shaped multi-resolution stacks), so
    the Zipformer "hypercolumn over 6 stacks" doesn't map 1:1. Here the
    hypercolumn is built over evenly-spaced LAYERS instead: same resolution,
    same width (d_model) at every tap, so unlike the Zipformer wrapper no
    per-tap resampling to a common rate is needed.

    Default checkpoint (stt_en_fastconformer_hybrid_large_streaming_480ms,
    114.75M params) uses chunked_limited attention (att_context_size
    [70, 6]) + causal convolutions: a single non-cached forward() call
    already implements the same bounded-lookahead computation as true
    chunked streaming, exactly like the existing Zipformer wrapper's
    causal=True offline forward — so training/eval can call forward_features
    directly without a chunked loop. True chunk-by-chunk forward_streaming
    (for RTF measurement / production streaming) is not implemented here;
    NeMo exposes cache_aware_stream_step/get_initial_cache_state for that
    when needed later.

    The encoder's native output rate is 100Hz / subsampling_factor (12.5Hz
    for the 8x-subsampled fastconformer family) — not the project's 25Hz
    convention. Frames are simply repeated (nearest-neighbor upsample) to
    reach target_frame_rate_hz so the TCN head's dilation schedule keeps the
    same receptive field in seconds without any head changes.
    """

    def __init__(
        self,
        pretrained_model: str = "stt_en_fastconformer_hybrid_large_streaming_480ms",
        freeze: bool = True,
        collect_layers: bool = True,
        num_hypercolumn_layers: int = 6,
        target_frame_rate_hz: float = 25.0,
        tap_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        import nemo.collections.asr as nemo_asr

        full_model = nemo_asr.models.ASRModel.from_pretrained(
            pretrained_model, map_location="cpu"
        )
        self.encoder = full_model.encoder
        del full_model

        self.is_causal = self._check_causal(self.encoder, pretrained_model)
        self.collect_layers = collect_layers
        self.subsampling_factor = int(self.encoder.subsampling_factor)
        self.native_frame_rate_hz = 100.0 / self.subsampling_factor
        upsample = target_frame_rate_hz / self.native_frame_rate_hz
        if abs(upsample - round(upsample)) > 1e-6:
            raise ValueError(
                f"target_frame_rate_hz={target_frame_rate_hz} is not an "
                f"integer multiple of the encoder's native rate "
                f"{self.native_frame_rate_hz}Hz"
            )
        self.upsample_factor = int(round(upsample))
        self.d_model = int(self.encoder.d_model)

        n_layers = len(self.encoder.layers)
        self._tap_outputs = {}
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
            for i in self.tap_indices:
                self.encoder.layers[i].register_forward_hook(self._make_hook(i))
        else:
            self.tap_indices = []
            self.stack_dims = None

        self.output_dim = (
            sum(self.stack_dims) if self.stack_dims else self.d_model
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        print(
            f"Built FastConformer backbone: {pretrained_model}, "
            f"{n_layers} layers, d_model={self.d_model}, causal(chunked_limited)"
            f"={self.is_causal}, native_rate={self.native_frame_rate_hz}Hz -> "
            f"upsample x{self.upsample_factor} -> {target_frame_rate_hz}Hz, "
            f"hypercolumn_taps={self.tap_indices if self.collect_layers else 'none (final layer only)'} "
            f"output_dim={self.output_dim}"
        )

    @staticmethod
    def _check_causal(encoder, pretrained_model: str) -> bool:
        """Verify (don't assume) that this encoder's non-cached forward() is
        actually streaming-equivalent: chunked_limited attention with a
        bounded right-context, plus causal (right-context-0) convolutions.
        A different `pretrained_model` in config (e.g. an offline/full-
        context checkpoint) would otherwise silently inherit a hardcoded
        `is_causal=True` and every downstream "streaming" claim about it
        would be false."""
        att_context_size = list(getattr(encoder, "att_context_size", [-1, -1]))
        att_context_style = str(getattr(encoder, "att_context_style", "")).lower()
        conv_context_size = getattr(encoder, "conv_context_size", None)

        att_right = att_context_size[1] if len(att_context_size) > 1 else -1
        conv_right = 0
        if isinstance(conv_context_size, (list, tuple)) and len(conv_context_size) > 1:
            conv_right = conv_context_size[1]

        causal = (
            att_context_style == "chunked_limited"
            and att_right >= 0
            and conv_right == 0
        )
        if not causal:
            raise ValueError(
                f"pretrained_model={pretrained_model!r} does not look "
                f"causal/streaming-compatible: att_context_style="
                f"{att_context_style!r} att_context_size={att_context_size} "
                f"conv_context_size={conv_context_size}. FastConformerBackbone "
                "assumes a single offline forward() is streaming-equivalent "
                "(chunked_limited attention + causal convs); a full-context "
                "checkpoint would silently leak future context into every "
                "frame if treated the same way."
            )
        print(
            f"FastConformer causality check passed: att_context_style="
            f"{att_context_style} att_context_size={att_context_size} "
            f"conv_context_size={conv_context_size}"
        )
        return True

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self._tap_outputs[idx] = output
        return hook

    def forward_features(
        self, features: torch.Tensor, feature_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: [B, T, 80] NeMo-style log-mel (see extract_nemo_fbank_features
            in src/data/feature_extractor.py — do NOT feed Kaldi-style fbank
            here, it's a different mel/window implementation and would put
            this frozen encoder out of distribution).
        feature_lens: [B]

        Returns [B, T_enc, D] at target_frame_rate_hz, D = sum(stack_dims)
        hypercolumn (e.g. 6*512=3072) or d_model (512) if collect_layers=False.
        """
        if self.collect_layers:
            self._tap_outputs = {}

        x = features.transpose(1, 2)  # [B, 80, T] -- NeMo encoder wants channel-first
        encoder_out, encoder_out_lens = self.encoder(audio_signal=x, length=feature_lens)

        if self.collect_layers:
            pieces = [self._tap_outputs[i] for i in self.tap_indices]  # each [B, T_enc, d_model]
            # FastConformer is a FLAT encoder: every layer operates at the
            # SAME time resolution, unlike Zipformer's U-shaped stacks where
            # a resample-to-common-rate step is legitimately needed. Any
            # length mismatch here means something structural drifted
            # (topology change, NeMo version change) — silently cropping or
            # zero-padding would hide that bug and, worse, zero-padding
            # could leak an artificial "no speech" signal into otherwise
            # valid frames. Fail loudly instead.
            t_enc = pieces[0].size(1)
            for i, p in zip(self.tap_indices, pieces):
                if p.size(1) != t_enc:
                    raise RuntimeError(
                        f"FastConformer hypercolumn tap length mismatch: "
                        f"layer {self.tap_indices[0]} has T={t_enc} but "
                        f"layer {i} has T={p.size(1)}. Flat-encoder taps "
                        "must all share one time resolution."
                    )
            final_t = encoder_out.size(-1)  # encoder_out is [B, D, T] channel-first
            if final_t != t_enc:
                raise RuntimeError(
                    f"FastConformer hypercolumn tap length ({t_enc}) does not "
                    f"match the encoder's own final output length ({final_t})."
                )
            h = torch.cat(pieces, dim=-1)  # [B, T_enc, output_dim]
        else:
            h = encoder_out.transpose(1, 2)  # [B, D, T] -> [B, T, D]

        if self.upsample_factor > 1:
            h = h.repeat_interleave(self.upsample_factor, dim=1)
            encoder_out_lens = encoder_out_lens * self.upsample_factor

        return h, encoder_out_lens
