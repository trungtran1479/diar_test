import random
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .zipformer_wrapper import StreamingZipformerEncoder, MockZipformerEncoder
from .fastconformer_wrapper import FastConformerBackbone
from .wavlm_wrapper import WavLMBackbone
from .crnn_baseline import CRNNBackbone
from .heads import (
    CausalAttentionCountHead,
    CausalGRUCountHead,
    CausalSSMCountHead,
    DeformableCountHead,
    LinearCountHead,
    TCNCountHead,
    TCNEventStateCountHead,
    TemporalAdaptiveCountHead,
)
from .pyramid_head import GatedPyramidOrdinalHead

class ZipCountModel(nn.Module):
    """ZipCount v1 Main Model.
    
    Composes:
    1. Backbone (Zipformer, Mock, or CRNN)
    2. Head (Linear or TemporalAdaptive)
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss_fn: Optional[nn.Module] = None
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss_fn
        
    def forward(
        self, 
        features: torch.Tensor, 
        feature_lens: torch.Tensor
    ):
        """
        Args:
            features: [B, T, 80] log-mel
            feature_lens: [B] lengths
        Returns:
            logits tuple and h_lens
        """
        h, h_lens = self.backbone.forward_features(features, feature_lens)
        logits = self.head(h)
        return logits, h_lens

def build_model(config: dict) -> ZipCountModel:
    """Build the model from a YAML config dict."""
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    head_cfg = model_cfg.get("head", {})
    
    encoder_type = encoder_cfg.get("type", "mock")
    
    # 1. Build Backbone
    if encoder_type == "mock":
        output_dim = encoder_cfg.get("output_dim", 512)
        backbone = MockZipformerEncoder(output_dim=output_dim)
    elif encoder_type == "crnn":
        bidirectional = encoder_cfg.get("bidirectional", False)
        output_dim = encoder_cfg.get("output_dim", 512)
        backbone = CRNNBackbone(output_dim=output_dim, bidirectional=bidirectional)
    elif encoder_type == "zipformer":
        # Architecture / streaming options forwarded to the recipe's
        # finetune.py argparse (yaml is the single source of truth; without
        # this the recipe defaults win, e.g. causal=False).
        recipe_arg_keys = [
            "num_encoder_layers", "downsampling_factor", "feedforward_dim",
            "num_heads", "encoder_dim", "query_head_dim", "value_head_dim",
            "pos_head_dim", "pos_dim", "encoder_unmasked_dim",
            "cnn_module_kernel", "causal", "chunk_size", "left_context_frames",
        ]
        model_args = {k: encoder_cfg[k] for k in recipe_arg_keys if k in encoder_cfg}
        backbone = StreamingZipformerEncoder(
            zipformer_dir=encoder_cfg.get("zipformer_dir", ""),
            checkpoint_path=encoder_cfg.get("checkpoint"),
            freeze=encoder_cfg.get("freeze_backbone", True),
            unfreeze_last_n=encoder_cfg.get("unfreeze_last_n", 0),
            strict_load=encoder_cfg.get("strict_load", False),
            min_load_ratio=encoder_cfg.get("min_load_ratio", 0.90),
            model_args=model_args,
            collect_stacks=encoder_cfg.get("multiscale", False),
            multiscale_alignment=encoder_cfg.get(
                "multiscale_alignment", "average"
            ),
            multiscale_include_final=encoder_cfg.get(
                "multiscale_include_final", False
            ),
        )
    elif encoder_type == "fastconformer":
        backbone = FastConformerBackbone(
            pretrained_model=encoder_cfg.get(
                "pretrained_model", "stt_en_fastconformer_hybrid_large_streaming_480ms"
            ),
            freeze=encoder_cfg.get("freeze_backbone", True),
            collect_layers=encoder_cfg.get("multiscale", True),
            num_hypercolumn_layers=encoder_cfg.get("num_hypercolumn_layers", 6),
            target_frame_rate_hz=encoder_cfg.get("target_frame_rate_hz", 25.0),
            tap_indices=encoder_cfg.get("tap_indices", None),
        )
    elif encoder_type == "wavlm":
        backbone = WavLMBackbone(
            pretrained_model=encoder_cfg.get(
                "pretrained_model", "microsoft/wavlm-large"
            ),
            freeze=encoder_cfg.get("freeze_backbone", True),
            collect_layers=encoder_cfg.get("multiscale", True),
            num_hypercolumn_layers=encoder_cfg.get("num_hypercolumn_layers", 6),
            target_frame_rate_hz=encoder_cfg.get("target_frame_rate_hz", 25.0),
            tap_indices=encoder_cfg.get("tap_indices", None),
            unfreeze_last_n=encoder_cfg.get("unfreeze_last_n", 0),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    output_dim = getattr(backbone, "output_dim", encoder_cfg.get("output_dim", 512))

    # Backbone construction (esp. loading a pretrained checkpoint) consumes a
    # backbone-family-specific, non-portable amount of RNG (e.g. every
    # nn.Linear's reset_parameters draws before being overwritten by loaded
    # weights) — a Zipformer build and a FastConformer build leave the RNG at
    # different states even at the "same seed", so the head's random init
    # would silently differ between backbone families even when nothing
    # about the head changed. Reseeding here anchors head init to the seed
    # alone, independent of which backbone came before it — the same fix as
    # `reset_rng_after_init` in train.py (see its comment; that one protects
    # the TRAINING-time stochastic path, e.g. dropout masks, this one
    # protects the head's own INITIAL weights). Residual caveat: this makes
    # head init reproducible per-config, but does NOT make it bit-identical
    # across configs whose head has a different-shaped first layer (e.g.
    # StackGatedProjection's input dim depends on stack_dims) — a
    # differently-sized weight tensor consumes a different-sized slice of
    # the RNG stream, so every subsequent module still drifts. True
    # tensor-level pairing across differently-shaped heads would need
    # per-module seeding, not implemented here.
    seed = config.get("training", {}).get("seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 2. Build Head
    head_type = head_cfg.get("type", "linear")
    num_classes = model_cfg.get("num_count_classes", 4)
    
    if head_type == "linear":
        head = LinearCountHead(d_model=output_dim, num_classes=num_classes)
    elif head_type == "temporal_adaptive":
        head = TemporalAdaptiveCountHead(
            d_model=output_dim,
            num_classes=num_classes,
            adapter_dim=head_cfg.get("adapter_dim", 256),
            kernel_size=head_cfg.get("adapter_kernel_size", 5),
            dropout=head_cfg.get("adapter_dropout", 0.1)
        )
    elif head_type == "deformable_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = DeformableCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            num_layers=head_cfg.get("num_layers", 2),
            num_points=head_cfg.get("num_points", 4),
            num_groups=head_cfg.get("num_groups", 4),
            max_offset=head_cfg.get("max_offset", 16),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            stack_input_mask=head_cfg.get("stack_input_mask"),
            renormalize_active_gate=head_cfg.get("renormalize_active_gate", False),
        )
    elif head_type == "tcn_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = TCNCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            dilations=tuple(head_cfg.get("dilations", [1, 2, 4, 8, 16, 32])),
            kernel=head_cfg.get("kernel", 3),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            causal=head_cfg.get("causal", True),
            stack_indices=head_cfg.get("stack_indices"),
            stack_input_mask=head_cfg.get("stack_input_mask"),
            renormalize_active_gate=head_cfg.get("renormalize_active_gate", False),
        )
    elif head_type == "gru_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = CausalGRUCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            num_layers=head_cfg.get("num_layers", 2),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            stack_indices=head_cfg.get("stack_indices"),
            stack_input_mask=head_cfg.get("stack_input_mask"),
            renormalize_active_gate=head_cfg.get("renormalize_active_gate", False),
        )
    elif head_type == "ssm_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = CausalSSMCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            state_dim=head_cfg.get("state_dim"),
            num_layers=head_cfg.get("num_layers", 3),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            stack_indices=head_cfg.get("stack_indices"),
            stack_input_mask=head_cfg.get("stack_input_mask"),
            renormalize_active_gate=head_cfg.get("renormalize_active_gate", False),
        )
    elif head_type == "attention_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = CausalAttentionCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            num_heads=head_cfg.get("num_heads", 4),
            num_layers=head_cfg.get("num_layers", 3),
            max_context=head_cfg.get("max_context", 128),
            ffn_multiplier=head_cfg.get("ffn_multiplier", 2),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            stack_indices=head_cfg.get("stack_indices"),
            stack_input_mask=head_cfg.get("stack_input_mask"),
            renormalize_active_gate=head_cfg.get("renormalize_active_gate", False),
        )
    elif head_type == "tcn_state_ordinal":
        stack_dims = getattr(backbone, "stack_dims", None) if encoder_cfg.get("multiscale", False) else None
        head = TCNEventStateCountHead(
            d_in=output_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 256),
            dilations=tuple(head_cfg.get("dilations", [1, 2, 4, 8, 16, 32])),
            kernel=head_cfg.get("kernel", 3),
            dropout=head_cfg.get("dropout", 0.1),
            stack_dims=stack_dims,
            head_hidden_dim=head_cfg.get("head_hidden_dim", 0),
            event_stay_bias=head_cfg.get("event_stay_bias", 2.9),
            state_emission_scale=head_cfg.get("state_emission_scale", 1.0),
            use_state_filter=head_cfg.get("use_state_filter", True),
            event_filter_class_weights=head_cfg.get(
                "event_filter_class_weights", [1.0, 1.0, 1.0]),
        )
    elif head_type == "gated_pyramid_ordinal":
        if not encoder_cfg.get("multiscale", False):
            raise ValueError(
                "gated_pyramid_ordinal requires model.encoder.multiscale=true"
            )
        stack_dims = getattr(backbone, "stack_dims", None)
        if stack_dims is None:
            raise ValueError(
                "gated_pyramid_ordinal requires a backbone exposing stack_dims"
            )
        include_final = encoder_cfg.get("multiscale_include_final", False)
        final_dim = getattr(backbone, "final_dim", None) if include_final else None
        head = GatedPyramidOrdinalHead(
            d_in=output_dim,
            stack_dims=stack_dims,
            final_dim=final_dim,
            num_classes=num_classes,
            d_model=head_cfg.get("d_model", 192),
            dilations=tuple(
                head_cfg.get("dilations", [1, 2, 4, 8, 16, 32])
            ),
            kernel=head_cfg.get("kernel", 3),
            dropout=head_cfg.get("dropout", 0.1),
            use_framewise_gate=head_cfg.get("use_framewise_gate", True),
            gate_mode=head_cfg.get("gate_mode"),
            ordinal_mode=head_cfg.get("ordinal_mode", "corn"),
            edge_hidden_dim=head_cfg.get("edge_hidden_dim", 128),
            use_stage2=head_cfg.get("use_stage2", True),
            stack_input_mask=head_cfg.get("stack_input_mask"),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    head_params = sum(p.numel() for p in head.parameters())
    print(
        f"Head '{head_type}': {head_params:,} params, input dim (backbone "
        f"output_dim / hypercolumn width) = {output_dim}. NOTE: the head's "
        "own param count scales with this input dim (e.g. StackGatedProjection's "
        "first Linear is stack_dims -> d_model), so two systems with the same "
        "head TYPE but a different backbone output_dim are NOT parameter-"
        "matched — report cross-backbone comparisons as a system comparison, "
        "not an architecture-isolated one, unless a param-matched control is run."
    )

    return ZipCountModel(backbone=backbone, head=head)
