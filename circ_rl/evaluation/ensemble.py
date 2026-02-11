"""MDL-weighted ensemble policy construction.

Implements Phase 4 of ``CIRC-RL_Framework.md``: combine multiple policies
into an ensemble where weights are inversely proportional to MDL scores.

    w_i ~ exp(-MDL(pi_i))
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from circ_rl.evaluation.mdl_scorer import MDLScore, MDLScorer
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.trajectory_buffer import Trajectory


class EnsemblePolicy:
    """MDL-weighted mixture of policies.

    Each policy's weight is proportional to exp(-MDL(pi_i)), giving
    higher weight to policies with better fit-complexity trade-off.

    :param policies: List of trained policies.
    :param weights: Normalized weights for each policy.
    :param scores: MDL scores for each policy.
    """

    def __init__(
        self,
        policies: list[CausalPolicy],
        weights: np.ndarray,
        scores: list[MDLScore],
    ) -> None:
        if len(policies) != len(weights):
            raise ValueError(
                f"Number of policies ({len(policies)}) must match "
                f"number of weights ({len(weights)})"
            )
        if len(policies) == 0:
            raise ValueError("Cannot create ensemble with zero policies")

        self._policies = list(policies)
        self._weights = weights / weights.sum()  # Normalize
        self._scores = list(scores)

    @classmethod
    def from_mdl_scores(
        cls,
        policies: list[CausalPolicy],
        evaluation_trajectory: Trajectory,
        complexity_weight: float = 0.01,
    ) -> EnsemblePolicy:
        """Build an ensemble from policies, weighting by MDL scores.

        :param policies: List of trained policies.
        :param evaluation_trajectory: Trajectory for MDL scoring.
        :param complexity_weight: MDL complexity weight.
        :returns: A new EnsemblePolicy.
        """
        scorer = MDLScorer(complexity_weight=complexity_weight)
        scores = [scorer.score(p, evaluation_trajectory) for p in policies]

        mdl_values = np.array([s.total for s in scores])
        # Softmax weighting: w_i ~ exp(-MDL_i)
        # Subtract max for numerical stability
        shifted = -mdl_values - np.max(-mdl_values)
        weights = np.exp(shifted)
        weights = weights / weights.sum()

        logger.info(
            "Ensemble built from {} policies. MDL range: [{:.4f}, {:.4f}]. "
            "Max weight: {:.4f}",
            len(policies),
            float(mdl_values.min()),
            float(mdl_values.max()),
            float(weights.max()),
        )

        return cls(policies, weights, scores)

    @property
    def n_policies(self) -> int:
        """Number of policies in the ensemble."""
        return len(self._policies)

    @property
    def weights(self) -> np.ndarray:
        """Normalized ensemble weights."""
        return self._weights.copy()

    @property
    def scores(self) -> list[MDLScore]:
        """MDL scores for each policy."""
        return list(self._scores)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> int | np.ndarray:
        """Get an action by weighted sampling from the ensemble.

        With probability w_i, use policy i to select the action.

        :param state: State tensor.
        :param deterministic: If True, use the highest-weighted policy.
        :returns: Integer action for discrete, numpy array for continuous.
        """
        if deterministic:
            best_idx = int(np.argmax(self._weights))
            return self._policies[best_idx].get_action(state, deterministic=True)

        # Sample a policy according to weights
        idx = int(np.random.choice(len(self._policies), p=self._weights))
        return self._policies[idx].get_action(state, deterministic=False)

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted log-probability across the ensemble.

        :param states: State tensor of shape ``(batch, state_dim)``.
        :param actions: Action tensor of shape ``(batch,)``.
        :returns: Weighted log-probabilities of shape ``(batch,)``.
        """
        log_probs = []
        for policy in self._policies:
            with torch.no_grad():
                output = policy.evaluate_actions(states, actions)
            log_probs.append(output.log_prob)

        # Weighted average in probability space
        stacked = torch.stack(log_probs, dim=0)  # (n_policies, batch)
        weights_tensor = torch.from_numpy(self._weights).float().to(states.device)
        # log(sum(w_i * exp(log_p_i)))
        weighted = stacked + weights_tensor.unsqueeze(1).log()
        return torch.logsumexp(weighted, dim=0)  # (batch,)
