"""Action normalization for varying dynamics scales.

Normalizes actions by the ratio D_e / D_ref so that the analytic
policy operates in a normalized action space.

See ``CIRC-RL_Framework.md`` Section 3.6.3 (Action Normalization
for Varying Dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ActionNormalizer:
    r"""Normalize actions by dynamics scale ratio.

    For environment :math:`e` with dynamics scale :math:`D_e`:

    .. math::

        a^*_t(e) = r_e \cdot \tilde{a}^*_t

    where :math:`r_e = D_e / D_{\text{ref}}` and
    :math:`\tilde{a}^*_t = \pi^*_{\text{abs}}(s_t)` is the action
    in normalized space.

    See ``CIRC-RL_Framework.md`` Section 3.6.3.

    :param dynamics_scales: Per-environment dynamics scales, shape
        ``(n_envs,)``.
    :param reference_scale: Reference dynamics scale
        :math:`D_{\text{ref}}`.
    """

    dynamics_scales: np.ndarray
    reference_scale: float

    def normalize_action(
        self,
        action: np.ndarray,
        env_idx: int,
    ) -> np.ndarray:
        r"""Convert an abstract action to a physical action for env_idx.

        Applies: :math:`a_{\text{physical}} = r_e \cdot a_{\text{abstract}}`

        :param action: Abstract action, shape ``(action_dim,)``.
        :param env_idx: Environment index.
        :returns: Physical action for environment env_idx.
        """
        r_e = self._scale_ratio(env_idx)
        return action * r_e

    def denormalize_action(
        self,
        action: np.ndarray,
        env_idx: int,
    ) -> np.ndarray:
        r"""Convert a physical action back to abstract action space.

        Applies: :math:`a_{\text{abstract}} = a_{\text{physical}} / r_e`

        :param action: Physical action, shape ``(action_dim,)``.
        :param env_idx: Environment index.
        :returns: Abstract action.
        """
        r_e = self._scale_ratio(env_idx)
        if abs(r_e) < 1e-10:
            logger.warning(
                "Dynamics scale ratio near zero for env {}: r_e={:.6f}",
                env_idx, r_e,
            )
            return action
        return action / r_e

    def _scale_ratio(self, env_idx: int) -> float:
        r"""Compute :math:`r_e = D_e / D_{\text{ref}}`."""
        if self.reference_scale < 1e-10:
            return 1.0
        d_e = float(self.dynamics_scales[env_idx])
        return d_e / self.reference_scale

    def scale_ratio(self, env_idx: int) -> float:
        r"""Public access to :math:`r_e = D_e / D_{\text{ref}}`.

        :param env_idx: Environment index.
        :returns: Scale ratio for this environment.
        """
        return self._scale_ratio(env_idx)
