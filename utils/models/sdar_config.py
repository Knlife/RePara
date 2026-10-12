"""Configure SDAR's native blocks separately from RePara commitment blocks."""


def configure_sdar_native_block_size(model_config, requested=None):
    declared = getattr(model_config, "block_size", None)
    for name, value in (("model block_size", declared), ("sdar_native_block_size", requested)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if requested is not None and declared is not None and requested != declared:
        raise ValueError(
            f"sdar_native_block_size ({requested}) disagrees with model block_size ({declared})"
        )
    # Preserve legacy b4 behavior for existing configurations. Explicitly save
    # the resolved size with the checkpoint so future replay/evaluation uses it.
    resolved = requested if requested is not None else (declared if declared is not None else 4)
    model_config.block_size = resolved
    return resolved
