"""Composite policy combining analytic + optional residual.

The primary policy is the analytic derivation from validated hypotheses.
The residual (if present) adds a bounded correction for unexplained
variance.

See ``CIRC-RL_Framework.md`` Section 3.7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
    from circ_rl.policy.residual_policy import ResidualPolicy


class CompositePolicy:
    r"""Combines an analytic policy with an optional residual correction.

    .. math::

        \pi(a \mid s; e) = \pi_{\text{analytic}}(a \mid s; e) + \delta\pi(a \mid s)

    :param analytic_policy: The analytic (LQR/MPC) policy.
    :param residual_policy: Optional bounded residual correction.
        If None, the composite policy is purely analytic.
    :param explained_variance: Fraction of variance explained by the
        analytic policy (:math:`\eta^2`). Used for diagnostics.
    """

    def __init__(
        self,
        analytic_policy: AnalyticPolicy,
        residual_policy: ResidualPolicy | None = None,
        explained_variance: float = 1.0,
    ) -> None:
        self._analytic = analytic_policy
        self._residual = residual_policy
        self._explained_variance = explained_variance

    @property
    def analytic_policy(self) -> AnalyticPolicy:
        """The analytic component."""
        return self._analytic

    @property
    def residual_policy(self) -> ResidualPolicy | None:
        """The residual correction component (None if purely analytic)."""
        return self._residual

    @property
    def explained_variance(self) -> float:
        r"""Fraction of variance explained by the analytic policy (:math:`\eta^2`)."""
        return self._explained_variance

    @property
    def has_residual(self) -> bool:
        """Whether a residual correction is active."""
        return self._residual is not None

    def get_action(
        self,
        state: np.ndarray,
        env_idx: int,
    ) -> np.ndarray:
        """Compute the composite action.

        :param state: Current state, shape ``(state_dim,)``.
        :param env_idx: Environment index.
        :returns: Composite action, shape ``(action_dim,)``.
        """
        analytic_action = self._analytic.get_action(state, env_idx)

        if self._residual is not None:
            import torch

            state_t = torch.from_numpy(state).float().unsqueeze(0)
            analytic_t = torch.from_numpy(analytic_action).float().unsqueeze(0)

            with torch.no_grad():
                output = self._residual(state_t, analytic_t)

            delta = output.delta_action.squeeze(0).numpy()
            return analytic_action + delta

        return analytic_action

    @property
    def complexity(self) -> int:
        """Total complexity (analytic + residual parameter count)."""
        c = self._analytic.complexity
        if self._residual is not None:
            c += sum(
                p.numel() for p in self._residual.parameters()
            )
        return c

    def __repr__(self) -> str:
        return (
            f"CompositePolicy(analytic={self._analytic!r}, "
            f"has_residual={self.has_residual}, "
            f"explained_variance={self._explained_variance:.3f})"
        )
