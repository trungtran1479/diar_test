import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Any, Dict

# To use icefall's zipformer without modifying it
def setup_zipformer_imports(zipformer_dir: str):
    if zipformer_dir not in sys.path:
        sys.path.insert(0, zipformer_dir)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Local copy of icefall.utils.make_pad_mask.

    lengths: [B]
    Returns bool mask [B, max_len]; True marks padded positions.
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max().item())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.expand(n, max_len) >= lengths.unsqueeze(-1)


class MockZipformerEncoder(nn.Module):
    """Mock encoder for shape testing without loading the real 280MB model."""
    def __init__(self, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(80, output_dim)

    def forward_features(self, features: torch.Tensor, feature_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mock downsampling by 4 (similar to typical Conv2dSubsampling)
        B, T, _ = features.shape
        T_enc = (T - 3) // 2
        T_enc = (T_enc - 1) // 2

        # Ensure we don't go negative
        T_enc = max(1, T_enc)

        # [B, T, 80] -> [B, T, D]
        h = self.proj(features)

        # Strided subsampling so mock frames stay time-aligned with labels
        # (taking the first T_enc frames would cover only the first quarter
        # of the utterance): [B, T, D] -> [B, T_enc, D]
        h = h[:, ::4, :][:, :T_enc, :]
        if h.shape[1] < T_enc:
            pad = h[:, -1:, :].expand(B, T_enc - h.shape[1], self.output_dim)
            h = torch.cat([h, pad], dim=1)

        h_lens = ((feature_lens - 3) // 2 - 1) // 2
        h_lens = torch.clamp(h_lens, min=1)

        return h, h_lens


class StreamingZipformerEncoder(nn.Module):
    """Wraps icefall Zipformer2 encoder for speaker counting.

    Includes:
    1. encoder_embed (Conv2dSubsampling) — converts [B,T,80] → [T_sub, B, encoder_dim[0]]
    2. encoder (Zipformer2) — 6-stack streaming encoder → [T_enc, B, 512]

    Never uses decoder/joiner/predictor/tokenizer.

    model_args: optional dict of finetune.py CLI options that override its
    defaults when building the encoder, e.g.
        {"causal": True, "chunk_size": "32", "left_context_frames": "128",
         "encoder_dim": "192,256,384,512,384,256", ...}
    Keys may use underscores; they are converted to --dashed-flags.
    """

    def __init__(
        self,
        zipformer_dir: str,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        strict_load: bool = False,
        min_load_ratio: float = 0.90,
        model_args: Optional[Dict[str, Any]] = None,
        collect_stacks: bool = False,
        multiscale_alignment: str = "average",
        multiscale_include_final: bool = False,
    ):
        super().__init__()
        setup_zipformer_imports(zipformer_dir)
        self.collect_stacks = collect_stacks
        self.multiscale_alignment = str(multiscale_alignment).lower()
        self.multiscale_include_final = bool(multiscale_include_final)
        if self.multiscale_alignment not in {"average", "encoder_learned"}:
            raise ValueError(
                "multiscale_alignment must be 'average' or "
                f"'encoder_learned', got {multiscale_alignment!r}"
            )
        if self.multiscale_include_final and not self.collect_stacks:
            raise ValueError(
                "multiscale_include_final=True requires collect_stacks=True"
            )

        finetune_path = os.path.join(zipformer_dir, "finetune.py")
        if not os.path.exists(finetune_path):
            raise FileNotFoundError(
                f"Could not find finetune.py in {zipformer_dir}. "
                "Please provide a full Icefall recipe directory (e.g. icefall/egs/librispeech/ASR/zipformer) "
                "that contains finetune.py, train.py, zipformer.py, etc."
            )

        import argparse
        import finetune as recipe_finetune

        # Build argv from model_args so yaml-configured architecture/causal
        # options actually reach the recipe (defaults alone build a
        # NON-causal encoder, which silently breaks the streaming story).
        argv: List[str] = []
        for k, v in (model_args or {}).items():
            flag = "--" + str(k).replace("_", "-")
            if isinstance(v, bool):
                v = "true" if v else "false"
            argv += [flag, str(v)]

        p = argparse.ArgumentParser()
        recipe_finetune.add_model_arguments(p)
        args, unused = p.parse_known_args(argv)
        if unused:
            print(f"Warning: model_args not recognized by recipe (ignored): {unused}")
        params = recipe_finetune.get_params()
        params.update(vars(args))

        # Icefall get_model often needs vocab_size and blank_id to build the decoder
        # even though we only want the encoder
        if not hasattr(params, "vocab_size"):
            params.vocab_size = 500
        if not hasattr(params, "blank_id"):
            params.blank_id = 0
        if not hasattr(params, "context_size"):
            params.context_size = 2

        # Build the full AsrModel to get the encoder part
        self.full_model = recipe_finetune.get_model(params)

        self.encoder_embed = getattr(self.full_model, "encoder_embed", None)
        self.encoder = self.full_model.encoder

        # Drop the RNNT/CTC branches entirely: forward_encoder only needs
        # encoder_embed + encoder, and keeping them would leave ~1.5M unused
        # trainable params in the optimizer and checkpoint.
        for attr in ("decoder", "joiner", "simple_am_proj", "simple_lm_proj", "ctc_output"):
            if hasattr(self.full_model, attr):
                setattr(self.full_model, attr, None)

        self.is_causal = bool(getattr(self.encoder, "causal", False))
        print(f"Built Zipformer encoder: causal={self.is_causal}, "
              f"chunk_size={getattr(self.encoder, 'chunk_size', None)}, "
              f"left_context_frames={getattr(self.encoder, 'left_context_frames', None)}")

        if self.encoder_embed is None:
            print("Warning: full_model.encoder_embed not found. Assuming self.encoder handles subsampling.")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path, strict=strict_load, min_ratio=min_load_ratio)

        if freeze:
            self._freeze_params(unfreeze_last_n)

        # Multi-scale hypercolumn: capture the per-stack outputs of the
        # U-Net-like encoder. Each DownsampledZipformer2Encoder upsamples back
        # to the shared 50 Hz base rate, so every hooked output is
        # [T_50, B, C_i] with C_i in encoder_dim (e.g. 192,256,384,512,384,256).
        self._stack_outputs: List[torch.Tensor] = []
        if self.collect_stacks:
            if not hasattr(self.encoder, "encoders"):
                raise ValueError("collect_stacks=True requires a Zipformer2 encoder with .encoders")
            self.stack_dims = [int(d) for d in self.encoder.encoder_dim]
            self.final_dim = max(self.stack_dims)
            for enc_stack in self.encoder.encoders:
                enc_stack.register_forward_hook(self._stack_hook)
            print(f"Multi-scale hypercolumn enabled: stack dims {self.stack_dims} "
                  f"+ final={self.multiscale_include_final} "
                  f"(alignment={self.multiscale_alignment}) -> concat dim "
                  f"{sum(self.stack_dims) + (self.final_dim if self.multiscale_include_final else 0)}")

        # Detect output_dim using a dummy forward pass
        dummy_feats = torch.zeros(1, 100, 80)
        dummy_lens = torch.tensor([100], dtype=torch.long)
        with torch.no_grad():
            try:
                dummy_out, _ = self.forward_features(dummy_feats, dummy_lens)
                self.output_dim = dummy_out.shape[-1]
                print(f"Detected backbone output_dim: {self.output_dim}")
            except Exception as e:
                print(f"Warning: Failed to auto-detect output_dim: {e}")
                self.output_dim = 512

    def _load_checkpoint(self, path: str, strict: bool, min_ratio: float):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        full_sd = ckpt.get("model", ckpt)

        my_sd_embed = {}
        my_sd_encoder = {}

        expected_embed_sd = self.encoder_embed.state_dict() if self.encoder_embed else {}
        expected_encoder_sd = self.encoder.state_dict()

        for k, v in full_sd.items():
            if k.startswith("encoder_embed.") and self.encoder_embed is not None:
                new_k = k[14:]
                if new_k in expected_embed_sd:
                    if expected_embed_sd[new_k].shape == v.shape:
                        my_sd_embed[new_k] = v
                    else:
                        print(f"Shape mismatch for encoder_embed {new_k}: expected {expected_embed_sd[new_k].shape}, got {v.shape}")

            elif k.startswith("encoder."):
                new_k = k[8:]
                if new_k in expected_encoder_sd:
                    if expected_encoder_sd[new_k].shape == v.shape:
                        my_sd_encoder[new_k] = v
                    else:
                        print(f"Shape mismatch for encoder {new_k}: expected {expected_encoder_sd[new_k].shape}, got {v.shape}")

        if self.encoder_embed is not None:
            missing_em, unexpected_em = self.encoder_embed.load_state_dict(my_sd_embed, strict=strict)
        else:
            missing_em, unexpected_em = [], []

        missing_en, unexpected_en = self.encoder.load_state_dict(my_sd_encoder, strict=strict)

        # Summary by numel
        loaded_numel = sum(v.numel() for v in my_sd_embed.values()) + sum(v.numel() for v in my_sd_encoder.values())
        expected_numel = sum(v.numel() for v in expected_embed_sd.values()) + sum(v.numel() for v in expected_encoder_sd.values())
        ratio = loaded_numel / max(expected_numel, 1)

        print(f"Loaded: {loaded_numel}/{expected_numel} params ({ratio:.1%})")
        print(f"Missing keys: {len(missing_em) + len(missing_en)}")
        print(f"Unexpected keys: {len(unexpected_em) + len(unexpected_en)}")

        if ratio < min_ratio:
            raise RuntimeError(
                f"Only {ratio:.1%} params loaded! Config mismatch suspected."
            )

    def _freeze_params(self, unfreeze_last_n: int):
        if self.encoder_embed is not None:
            for p in self.encoder_embed.parameters():
                p.requires_grad = False

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0 and hasattr(self.encoder, "encoders"):
            num_encoders = len(self.encoder.encoders)
            start_idx = max(0, num_encoders - unfreeze_last_n)
            for i in range(start_idx, num_encoders):
                for p in self.encoder.encoders[i].parameters():
                    p.requires_grad = True
            print(f"Unfrozen last {num_encoders - start_idx} stacks of Zipformer.")

    def _stack_hook(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self._stack_outputs.append(out)

    @staticmethod
    def _match_time(x: torch.Tensor, t_enc: int) -> torch.Tensor:
        """Crop/right-zero-pad a [B,T,C] tensor to the encoder time axis."""
        if x.size(1) > t_enc:
            return x[:, :t_enc]
        if x.size(1) < t_enc:
            return torch.nn.functional.pad(x, (0, 0, 0, t_enc - x.size(1)))
        return x

    def _hypercolumn(
        self,
        t_enc: int,
        final_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate captured per-stack outputs into a hypercolumn.

        Each captured tensor: [T_50, B, C_i] at the 50 Hz base rate.
        ``average`` is the exact legacy avg_pool1d path.  The new
        ``encoder_learned`` mode invokes the encoder's actual
        ``downsample_output`` module separately for every stack, sharing its
        trained temporal weights.  The actual final Zipformer output can be
        appended after the aligned six-stack hypercolumn.
        """
        assert len(self._stack_outputs) == len(self.stack_dims), (
            f"captured {len(self._stack_outputs)} stack outputs, expected {len(self.stack_dims)}")
        pieces = []
        for out in self._stack_outputs:
            if self.multiscale_alignment == "average":
                # Preserve the historical operation order exactly for legacy
                # configs and checkpoints.
                x = out.permute(1, 2, 0)                               # [B, C_i, T_50]
                x = torch.nn.functional.avg_pool1d(x, 2, stride=2, ceil_mode=True)
                x = x.transpose(1, 2)                                  # [B, ceil(T_50/2), C_i]
            else:
                # SimpleDownsample is channel-agnostic. Calling the module
                # itself (not copying its bias) also preserves the correct
                # gradient path when the backbone is unfrozen.
                x = self.encoder.downsample_output(out).transpose(0, 1)
            pieces.append(self._match_time(x, t_enc))

        if self.multiscale_include_final:
            if final_output is None:
                raise RuntimeError(
                    "actual final Zipformer output is required when "
                    "multiscale_include_final=True"
                )
            pieces.append(self._match_time(final_output, t_enc))

        h = torch.cat(pieces, dim=-1)                                  # [B, ~T_enc, sum(C_i)]
        return h

    def forward_features(self, features: torch.Tensor, feature_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: [B, T, 80]
        feature_lens: [B]

        Returns [B, T_enc, D] with D = 512 (final stack mix) or, when
        collect_stacks=True, D = sum(stack_dims) hypercolumn (e.g. 1984).
        """
        if self.collect_stacks:
            self._stack_outputs = []
        if hasattr(self.full_model, "forward_encoder"):
            # Best path: use the recipe's native forward_encoder which handles
            # all mask generation, transposes, and subsampling correctly.
            encoder_out, encoder_out_lens = self.full_model.forward_encoder(features, feature_lens)
        else:
            # Fallback path if forward_encoder is missing
            if self.encoder_embed is not None:
                x, x_lens = self.encoder_embed(features, feature_lens)

                # In many recipes, x is transposed to [T, B, C] for Zipformer
                if hasattr(self.encoder, "encoders") and x.dim() == 3:
                    # check if we need to transpose (if the model expects [T, B, C])
                    # usually Zipformer2 expects [T, B, C]
                    x = x.transpose(0, 1)  # [B, T, C] -> [T, B, C]

                encoder_out, encoder_out_lens = self.encoder(x, x_lens)
            else:
                encoder_out, encoder_out_lens = self.encoder(features, feature_lens)

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Ensure output is [B, T_enc, D]
        if encoder_out.dim() == 3 and encoder_out.shape[1] == features.shape[0] and encoder_out.shape[0] != features.shape[0]:
            # It's [T_enc, B, D], transpose it back to [B, T_enc, D]
            encoder_out = encoder_out.transpose(0, 1)

        if self.collect_stacks:
            encoder_out = self._hypercolumn(
                encoder_out.size(1),
                final_output=(
                    encoder_out if self.multiscale_include_final else None
                ),
            )

        return encoder_out, encoder_out_lens

    # ------------------------------------------------------------------
    # Streaming API (mirrors icefall zipformer/streaming_decode.py)
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        """Chunk size in encoder-embed frames (50 Hz)."""
        return int(self.encoder.chunk_size[0])

    @property
    def left_context_len(self) -> int:
        """Left context in encoder-embed frames (50 Hz)."""
        return int(self.encoder.left_context_frames[0])

    @property
    def pad_length(self) -> int:
        """Extra fbank lookahead frames each chunk needs: 7 for the
        Conv2dSubsampling convs + 2*3 for ConvNeXt right padding."""
        return 7 + 2 * 3

    def get_init_states(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> List[torch.Tensor]:
        """states[:-2] per-layer caches, states[-2] embed ConvNeXt left pad,
        states[-1] processed_lens [B] (50 Hz frames seen so far)."""
        assert self.is_causal, "Streaming requires an encoder built with causal=True"
        states = self.encoder.get_init_states(batch_size, device)
        states.append(self.encoder_embed.get_init_states(batch_size, device))
        states.append(torch.zeros(batch_size, dtype=torch.int32, device=device))
        return states

    def forward_streaming(
        self,
        features_chunk: torch.Tensor,
        states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Process one chunk with cached states.

        features_chunk: [B, T_chunk, 80] where T_chunk = 2*chunk_size + pad_length
                        (fbank at 100 Hz; the pad_length tail is lookahead that
                        will be re-consumed by the next chunk).
        states: from get_init_states() or the previous call.

        Returns:
            h: [B, chunk_size//2, D]   (25 Hz encoder frames)
            h_lens: [B]
            new_states
        """
        assert self.is_causal, "Streaming requires an encoder built with causal=True"
        B = features_chunk.size(0)
        chunk_size = self.chunk_size
        left_context_len = self.left_context_len
        device = features_chunk.device

        feature_lens = torch.full((B,), features_chunk.size(1), dtype=torch.long, device=device)

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=features_chunk,
            x_lens=feature_lens,
            cached_left_pad=cached_embed_left_pad,
        )  # x: [B, chunk_size, D0]
        assert x.size(1) == chunk_size, (x.size(1), chunk_size)

        src_key_padding_mask = make_pad_mask(x_lens)

        # Mask out the not-yet-filled part of the initial left-context cache
        processed_mask = torch.arange(left_context_len, device=device).expand(B, left_context_len)
        processed_lens = states[-1]  # [B]
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        new_processed_lens = processed_lens + x_lens

        # [B, left_context_len + chunk_size]
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
        encoder_states = states[:-2]
        if self.collect_stacks:
            encoder_out, encoder_out_lens, new_encoder_states = self._streaming_encoder_collect(
                x, x_lens, encoder_states, src_key_padding_mask
            )
        else:
            encoder_out, encoder_out_lens, new_encoder_states = self.encoder.streaming_forward(
                x=x,
                x_lens=x_lens,
                states=encoder_states,
                src_key_padding_mask=src_key_padding_mask,
            )
            encoder_out = encoder_out.permute(1, 0, 2)  # [T, B, D] -> [B, T, D]

        new_states = new_encoder_states + [new_cached_embed_left_pad, new_processed_lens]
        return encoder_out, encoder_out_lens, new_states

    def _streaming_encoder_collect(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: List[torch.Tensor],
        src_key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Replica of Zipformer2.streaming_forward that also captures per-stack
        outputs (forward hooks do not fire on the plain streaming_forward
        method calls). Returns the aligned multiscale tensor at 25 Hz."""
        from scaling import convert_num_channels  # recipe dir already on sys.path

        enc = self.encoder
        self._stack_outputs = []
        new_states: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        layer_offset = 0

        for i, module in enumerate(enc.encoders):
            num_layers = module.num_layers
            ds = int(enc.downsampling_factor[i])
            x = convert_num_channels(x, enc.encoder_dim[i])
            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6:(layer_offset + num_layers) * 6],
                left_context_len=enc.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            self._stack_outputs.append(x)   # [T_50, B, C_i]
            outputs.append(x)
            new_states += new_layer_states

        lengths = (x_lens + 1) // 2
        final_output = None
        if self.multiscale_include_final:
            final_output = enc.downsample_output(
                enc._get_full_dim_output(outputs)
            ).transpose(0, 1)
        h = self._hypercolumn(
            int(lengths.max().item()),
            final_output=final_output,
        )
        return h, lengths, new_states
