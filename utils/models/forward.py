import contextlib
import contextvars

import torch
import torch.nn as nn


_SELECTED_LOGIT_POSITIONS = contextvars.ContextVar("selected_logit_positions", default=None)


class PositionSelectiveLinear(nn.Linear):
    """A state-dict-compatible Linear that can project selected sequence positions only."""

    @classmethod
    def from_linear(cls, linear):
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear).__name__}")

        # Avoid allocating another (potentially very large) vocabulary projection.
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.weight = linear.weight
        module.bias = linear.bias
        module.training = linear.training
        return module

    def forward(self, inputs):
        positions = _SELECTED_LOGIT_POSITIONS.get()
        if positions is not None:
            if inputs.ndim != 3:
                raise ValueError(
                    "Selective logits require output-head inputs shaped [batch, sequence, hidden]"
                )
            if isinstance(positions, tuple):
                batch_indices, positions = positions
            else:
                positions = torch.as_tensor(positions, device=inputs.device, dtype=torch.long)
                if positions.ndim == 0:
                    positions = positions.expand(inputs.shape[0])
                if positions.shape != (inputs.shape[0],):
                    raise ValueError(
                        f"Expected one logit position per row ({inputs.shape[0]}), "
                        f"got {tuple(positions.shape)}"
                    )
                batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
            inputs = inputs[batch_indices, positions]
        return nn.functional.linear(inputs, self.weight, self.bias)


@contextlib.contextmanager
def select_logit_positions(positions):
    token = _SELECTED_LOGIT_POSITIONS.set(positions)
    try:
        yield
    finally:
        _SELECTED_LOGIT_POSITIONS.reset(token)


def _find_position_selective_output_head(model, model_family="llada"):
    visited = set()
    pending = [model]
    while pending:
        module = pending.pop(0)
        if module is None or id(module) in visited:
            continue
        visited.add(id(module))

        if model_family == "llada":
            transformer = getattr(module, "transformer", None)
            output_head = getattr(transformer, "ff_out", None)
            if isinstance(output_head, nn.Linear):
                return transformer, "ff_out", output_head
        elif model_family == "sdar":
            output_head = getattr(module, "lm_head", None)
            if isinstance(output_head, nn.Linear):
                return module, "lm_head", output_head
        else:
            raise ValueError(f"Selective logits do not support model_family={model_family!r}")

        for name in ("module", "_fsdp_wrapped_module", "base_model", "model"):
            child = getattr(module, name, None)
            if isinstance(child, nn.Module) and id(child) not in visited:
                pending.append(child)
    return None


def install_position_selective_output_head(model, model_family="llada"):
    """Install selective projection without changing state-dict keys or Parameter objects."""
    config = _get_model_config(model)
    if model_family == "llada" and getattr(config, "weight_tying", False):
        raise ValueError("Selective logits do not support LLaDA checkpoints with a tied output head")

    found = _find_position_selective_output_head(model, model_family=model_family)
    if found is None:
        if model_family == "llada":
            raise ValueError("Could not find LLaDA transformer.ff_out output projection")
        if model_family == "sdar":
            raise ValueError("Could not find SDAR lm_head output projection")
        raise ValueError(f"Could not find output projection for model_family={model_family!r}")
    parent, name, output_head = found
    if isinstance(output_head, PositionSelectiveLinear):
        return output_head

    replacement = PositionSelectiveLinear.from_linear(output_head)
    setattr(parent, name, replacement)
    return replacement


def _get_model_config(model):
    for module in (model, getattr(model, "module", None), getattr(model, "_fsdp_wrapped_module", None)):
        config = getattr(module, "config", None)
        if config is not None:
            return config
    return None


def _sdar_attention_mask(input_ids, block_size):
    """Build SDAR's eval/generation block-causal mask for each batch row."""
    batch_size, seq_len = input_ids.shape
    token_blocks = torch.arange(seq_len, device=input_ids.device) // block_size
    mask = token_blocks[:, None] >= token_blocks[None, :]
    return mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)


@contextlib.contextmanager
def _sdar_logits_forward_mode(model):
    """Use SDAR's logits-producing forward path without dropping autograd.

    SDARForCausalLM switches to its native diffusion-loss branch based on the
    top-level module's ``training`` flag. Flip only that module flag so decoder
    layers remain in train mode for dropout/gradient checkpointing.
    """
    if not isinstance(model, nn.Module):
        yield
        return

    targets = [
        module
        for module in model.modules()
        if hasattr(module, "prepare_for_bd_training")
    ] or [model]
    training_states = [
        (module, module.training)
        for module in targets
        if getattr(module, "training", False)
    ]
    if not training_states:
        yield
        return

    for module, _ in training_states:
        module.training = False
    try:
        yield
    finally:
        for module, training in training_states:
            module.training = training


def forward_logits(model, input_ids, model_family="llada"):
    if model_family == "sdar":
        config = _get_model_config(model)
        block_size = getattr(config, "block_size", None) or 4
        # SDARForCausalLM.forward() switches to its native denoising-loss branch
        # whenever self.training is true, and that branch requires labels and
        # returns no logits. RePara/GRPO replay needs differentiable logits, so
        # keep the normal wrapped call path but temporarily use the eval/logits
        # branch. Autograd remains enabled because this is not a no_grad context.
        with _sdar_logits_forward_mode(model):
            outputs = model(
                input_ids,
                attention_mask=_sdar_attention_mask(input_ids, block_size),
                position_ids=torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1),
                use_cache=False,
            )
    else:
        outputs = model(input_ids)

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError(
            f"Model forward did not return logits for model_family={model_family!r}. "
            "Use --model_family sdar for SDARForCausalLM checkpoints."
        )
    return logits


def forward_selected_logits(model, input_ids, positions, model_family="llada"):
    """Return vocabulary logits for selected sequence positions in each batch row."""
    positions = torch.as_tensor(positions, device=input_ids.device, dtype=torch.long)
    if positions.ndim == 0:
        positions = positions.expand(input_ids.shape[0])
    if positions.ndim < 1 or positions.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"Expected selected positions with leading batch dimension {input_ids.shape[0]}, "
            f"got {tuple(positions.shape)}"
        )

    batch_shape = (input_ids.shape[0],) + (1,) * (positions.ndim - 1)
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device) \
        .reshape(batch_shape).expand_as(positions)

    selected_indices = (batch_indices.reshape(-1), positions.reshape(-1))
    with select_logit_positions(selected_indices):
        logits = forward_logits(model, input_ids, model_family=model_family)

    # Safe fallback for callers that have not installed the selective output head.
    if logits.ndim == 3:
        logits = logits[batch_indices, positions]
    elif logits.ndim == 2 and positions.ndim > 1:
        logits = logits.reshape(*positions.shape, logits.shape[-1])

    expected_ndim = positions.ndim + 1
    if logits.ndim != expected_ndim or logits.shape[:-1] != positions.shape:
        raise RuntimeError(
            f"Expected selected logits shaped {tuple(positions.shape)} + [vocab], "
            f"got {tuple(logits.shape)}"
        )
    return logits


def forward_masked_logits(model, input_ids, selection_mask, model_family="llada"):
    """Project logits at a dynamic set of selected positions and return their row mapping."""
    if selection_mask.shape != input_ids.shape or selection_mask.dtype != torch.bool:
        raise ValueError("selection_mask must be boolean and have the same shape as input_ids")
    batch_indices, positions = selection_mask.nonzero(as_tuple=True)
    if positions.numel() == 0:
        raise ValueError("selection_mask must select at least one position")

    with select_logit_positions((batch_indices, positions)):
        logits = forward_logits(model, input_ids, model_family=model_family)

    if logits.ndim == 3:
        logits = logits[batch_indices, positions]
    if logits.ndim != 2:
        raise RuntimeError(f"Expected masked logits shaped [positions, vocab], got {tuple(logits.shape)}")
    return logits, batch_indices, positions
