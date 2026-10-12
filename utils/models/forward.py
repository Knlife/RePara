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








def forward_logits(model, input_ids, model_family="llada"):
    outputs = model(input_ids)

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError(
            'LLaDA model and evaluation support.'
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
