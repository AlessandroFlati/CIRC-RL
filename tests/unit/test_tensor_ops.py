"""Tests for circ_rl.utils.tensor_ops."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from circ_rl.utils.tensor_ops import assert_shape


class TestAssertShape:
    def test_exact_match_passes(self) -> None:
        t = torch.zeros(3, 4, 5)
        assert_shape(t, (3, 4, 5), "test_tensor")

    def test_none_wildcard_passes(self) -> None:
        t = torch.zeros(3, 4, 5)
        assert_shape(t, (None, 4, None), "test_tensor")

    def test_all_none_passes(self) -> None:
        t = torch.zeros(3, 4, 5)
        assert_shape(t, (None, None, None), "test_tensor")

    def test_wrong_ndim_raises(self) -> None:
        t = torch.zeros(3, 4)
        with pytest.raises(AssertionError, match="2 dims, expected 3"):
            assert_shape(t, (3, 4, 5), "test_tensor")

    def test_wrong_dim_value_raises(self) -> None:
        t = torch.zeros(3, 4, 5)
        with pytest.raises(AssertionError, match="dim 1 is 4, expected 7"):
            assert_shape(t, (3, 7, 5), "test_tensor")

    def test_scalar_tensor(self) -> None:
        t = torch.tensor(1.0)
        assert_shape(t, (), "scalar")

    def test_1d_tensor(self) -> None:
        t = torch.zeros(10)
        assert_shape(t, (10,), "1d")

    def test_error_message_includes_tensor_name(self) -> None:
        t = torch.zeros(3, 4)
        with pytest.raises(AssertionError, match="my_tensor"):
            assert_shape(t, (3, 5), "my_tensor")

    @given(
        dims=st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=1,
            max_size=5,
        )
    )
    def test_hypothesis_exact_shape_always_passes(
        self, dims: list[int]
    ) -> None:
        t = torch.zeros(*dims)
        assert_shape(t, tuple(dims), "hyp_tensor")

    @given(
        dims=st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=1,
            max_size=5,
        )
    )
    def test_hypothesis_all_none_always_passes(
        self, dims: list[int]
    ) -> None:
        t = torch.zeros(*dims)
        assert_shape(t, tuple(None for _ in dims), "hyp_tensor")
