"""Tensor shape assertions and common operations."""

import torch


def assert_shape(
    tensor: torch.Tensor,
    expected: tuple[int | None, ...],
    name: str,
) -> None:
    """Assert that a tensor has the expected shape.

    ``None`` in the expected shape means "any value" for that dimension.

    :param tensor: The tensor to check.
    :param expected: Expected shape tuple. Use ``None`` for unconstrained dims.
    :param name: Name of the tensor (used in error messages).
    :raises AssertionError: If the tensor shape does not match the expected shape.

    Example::

        assert_shape(q_values, (None, n_actions), "q_values")
    """
    actual = tensor.shape
    if len(actual) != len(expected):
        raise AssertionError(
            f"Tensor '{name}' has {len(actual)} dims, expected {len(expected)}. "
            f"Shape: {tuple(actual)}, expected: {expected}"
        )
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if e is not None and a != e:
            raise AssertionError(
                f"Tensor '{name}' dim {i} is {a}, expected {e}. "
                f"Shape: {tuple(actual)}, expected: {expected}"
            )
