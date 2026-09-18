import torch.nn as nn


def _train_uniformly_trainable_subtrees(module: nn.Module) -> None:
    params = list(module.parameters(recurse=True))
    if params and all(p.requires_grad for p in params):
        # Every parameter in this ENTIRE subtree requires grad: flip the
        # whole subtree to train() in one call. nn.Module.train() recurses
        # into children, so this correctly re-enables dropout submodules
        # too -- a plain nn.Dropout owns no parameters of its own
        # (`parameters(recurse=False)` is always empty for it), so it can
        # only be reached correctly via this subtree-level cascade, never
        # by checking a module's own direct parameters in isolation.
        module.train()
        return
    # Mixed or fully-frozen subtree: don't flip this level (a fully-frozen
    # subtree stays in eval() from the earlier backbone.eval() call; a
    # mixed one needs its trainable boundary found further down), recurse.
    for child in module.children():
        _train_uniformly_trainable_subtrees(child)


def set_backbone_partial_train_mode(backbone: nn.Module) -> None:
    """Put `backbone` in eval() everywhere, then train() every maximal
    subtree whose parameters ALL require grad.

    A plain `model.train()` puts the ENTIRE model -- including any frozen
    early layers of a partially-unfrozen backbone -- into train mode, so
    dropout/whitening/balancer noise randomizes "frozen" features every
    step even though their weights never update. That silently confounds a
    head_only vs unfreeze_last_n comparison: the two arms would then differ
    in both adaptation AND stochastic-backbone noise, not adaptation alone.

    Operates at the SUBTREE level (not per-module direct-parameter checks)
    specifically so parameterless children like nn.Dropout -- which own no
    parameters themselves and so can never be individually detected as
    "trainable" -- correctly inherit train() from their nearest fully-
    trainable ancestor (e.g. one whole unfrozen transformer layer) via
    nn.Module.train()'s built-in recursion into children. An earlier version
    of this function checked each module's own `parameters(recurse=False)`
    directly and never flipped Dropout submodules at all, silently leaving
    dropout OFF inside "unfrozen" layers too -- caught by a test asserting
    actual stochastic behavior, not just the `.training` flag on the layer
    container. A fully-frozen backbone is the special case where the
    recursion finds nothing to flip and the backbone simply stays in
    eval().
    """
    backbone.eval()
    _train_uniformly_trainable_subtrees(backbone)
