"""Concrete pipeline stages for the CIRC-RL v2 framework.

Implements the scientific discovery pipeline as pipeline stages:
1. Causal Discovery
2. Feature Selection
2.5. Transition Analysis
3. Hypothesis Generation (symbolic regression)
4. Hypothesis Falsification
5. Analytic Policy Derivation (LQR/MPC)
6. Residual Learning (bounded correction)
7. Diagnostic Validation (premise/derivation/conclusion)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.causal_discovery.mechanism_validator import MechanismValidator
from circ_rl.environments.data_collector import DataCollector
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.orchestration.pipeline import PipelineStage, hash_config

if TYPE_CHECKING:
    from circ_rl.analytic_policy.ilqr_solver import ILQRSolution, ILQRSolver
    from circ_rl.analytic_policy.robust_mpc import RobustILQRPlanner
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.hypothesis.derived_features import DerivedFeatureSpec
    from circ_rl.hypothesis.expression import SymbolicExpression
    from circ_rl.hypothesis.hypothesis_register import HypothesisEntry
    from circ_rl.hypothesis.symbolic_regressor import SymbolicRegressionConfig


class CausalDiscoveryStage(PipelineStage):
    """Phase 1: Discover causal structure from multi-environment data.

    :param env_family: The environment family.
    :param n_transitions_per_env: Transitions to collect per environment.
    :param discovery_method: Algorithm for causal discovery (pc, ges, fci).
    :param alpha: Significance level for CI tests.
    :param seed: Random seed for data collection.
    :param include_env_params: If True, augment the causal graph with
        environment-parameter nodes for env-param causal discovery.
    :param ep_correlation_threshold: p-value threshold for Pearson
        correlation pre-screen that adds ``ep -> reward`` edges.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_transitions_per_env: int = 5000,
        discovery_method: str = "pc",
        alpha: float = 0.05,
        seed: int = 42,
        include_env_params: bool = False,
        ep_correlation_threshold: float = 0.05,
    ) -> None:
        super().__init__(name="causal_discovery")
        self._env_family = env_family
        self._n_transitions = n_transitions_per_env
        self._method = discovery_method
        self._alpha = alpha
        self._seed = seed
        self._include_env_params = include_env_params
        self._ep_correlation_threshold = ep_correlation_threshold

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run causal discovery.

        :returns: Dict with keys: graph, dataset, node_names, state_names,
            validation_result, env_param_node_names.
        """
        collector = DataCollector(
            self._env_family,
            include_env_params=self._include_env_params,
        )
        dataset = collector.collect(
            n_transitions_per_env=self._n_transitions, seed=self._seed
        )

        state_dim = dataset.state_dim
        action_dim = 1 if dataset.actions.ndim == 1 else dataset.actions.shape[1]

        state_names = [f"s{i}" for i in range(state_dim)]
        action_names = (
            ["action"]
            if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )
        next_state_names = [f"s{i}_next" for i in range(state_dim)]
        node_names = state_names + action_names + ["reward"] + next_state_names

        # Augment with env-param nodes if enabled
        env_param_node_names: list[str] = []
        if self._include_env_params and self._env_family.param_names:
            env_param_node_names = [
                f"ep_{name}" for name in self._env_family.param_names
            ]
            node_names = node_names + env_param_node_names

        builder = CausalGraphBuilder()
        graph = builder.discover(
            dataset,
            node_names,
            method=self._method,
            alpha=self._alpha,
            env_param_names=env_param_node_names if env_param_node_names else None,
            ep_correlation_threshold=self._ep_correlation_threshold,
        )

        validator = MechanismValidator(alpha=self._alpha)
        validation = validator.validate_invariance(
            dataset, graph, node_names, target_node="reward"
        )

        logger.info(
            "Causal discovery complete: {} nodes, {} edges, invariant={}, "
            "env_param_nodes={}",
            len(graph.nodes),
            len(graph.edges),
            validation.is_invariant,
            env_param_node_names,
        )

        return {
            "graph": graph,
            "dataset": dataset,
            "node_names": node_names,
            "state_names": state_names,
            "validation_result": validation,
            "env_param_node_names": env_param_node_names,
        }

    def config_hash(self) -> str:
        return hash_config({
            "n_transitions": self._n_transitions,
            "method": self._method,
            "alpha": self._alpha,
            "seed": self._seed,
            "n_envs": self._env_family.n_envs,
            "include_env_params": self._include_env_params,
            "ep_correlation_threshold": self._ep_correlation_threshold,
        })


class FeatureSelectionStage(PipelineStage):
    """Phase 2: Select invariant causal features.

    :param epsilon: Maximum cross-environment ATE variance (ATE mode).
    :param min_ate: Minimum absolute ATE.
    :param use_mechanism_invariance: Use Chow-test mechanism invariance
        with soft weighting instead of ATE-variance hard filter.
    :param enable_conditional_invariance: Legacy flag for ATE variance mode.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        min_ate: float = 0.01,
        use_mechanism_invariance: bool = False,
        enable_conditional_invariance: bool = False,
    ) -> None:
        super().__init__(name="feature_selection", dependencies=["causal_discovery"])
        self._epsilon = epsilon
        self._min_ate = min_ate
        self._use_mechanism_invariance = use_mechanism_invariance
        self._enable_conditional_invariance = enable_conditional_invariance

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run feature selection.

        :returns: Dict with keys: feature_mask, feature_weights,
            selected_features, result, context_param_names.
        """
        cd_output = inputs["causal_discovery"]
        graph = cd_output["graph"]
        dataset = cd_output["dataset"]
        state_names = cd_output["state_names"]
        env_param_node_names: list[str] = cd_output.get("env_param_node_names", [])

        selector = InvFeatureSelector(
            epsilon=self._epsilon,
            min_ate=self._min_ate,
            use_mechanism_invariance=self._use_mechanism_invariance,
            enable_conditional_invariance=self._enable_conditional_invariance,
        )
        result = selector.select(
            dataset,
            graph,
            state_names,
            env_param_names=env_param_node_names if env_param_node_names else None,
        )

        # If no features selected, fall back to all state features
        if len(result.selected_features) == 0:
            logger.warning(
                "No features selected by invariance filter; "
                "falling back to all state features"
            )
            feature_weights = np.ones(len(state_names), dtype=np.float32)
            feature_mask = np.ones(len(state_names), dtype=bool)
        else:
            feature_weights = result.feature_weights
            feature_mask = result.feature_mask

        logger.info(
            "Feature selection: {}/{} features selected, "
            "context_dependent={}, context_params={}",
            int(feature_mask.sum()),
            len(state_names),
            list(result.context_dependent_features.keys()),
            result.context_param_names,
        )

        return {
            "feature_mask": feature_mask,
            "feature_weights": feature_weights,
            "selected_features": result.selected_features,
            "result": result,
            "context_param_names": result.context_param_names,
        }

    def config_hash(self) -> str:
        return hash_config({
            "epsilon": self._epsilon,
            "min_ate": self._min_ate,
            "use_mechanism_invariance": self._use_mechanism_invariance,
            "enable_conditional_invariance": self._enable_conditional_invariance,
        })


class TransitionAnalysisStage(PipelineStage):
    """Phase 2.5: Analyze transition dynamics for dynamics normalization.

    Estimates per-environment dynamics scales and tests whether transition
    mechanisms are invariant using LOEO R^2.

    :param loeo_r2_threshold: Minimum R^2 for invariant transitions.
    """

    def __init__(self, loeo_r2_threshold: float = 0.9) -> None:
        super().__init__(
            name="transition_analysis",
            dependencies=["causal_discovery"],
        )
        self._loeo_r2_threshold = loeo_r2_threshold

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run transition dynamics analysis.

        :returns: Dict with keys: dynamics_scales, reference_scale,
            transition_result.
        """
        from circ_rl.feature_selection.transition_analyzer import TransitionAnalyzer

        cd_output = inputs["causal_discovery"]
        dataset = cd_output["dataset"]
        state_names = cd_output["state_names"]

        # Infer action_dim from dataset
        action_dim = 1 if dataset.actions.ndim == 1 else dataset.actions.shape[1]

        analyzer = TransitionAnalyzer(loeo_r2_threshold=self._loeo_r2_threshold)
        result = analyzer.analyze(dataset, state_names, action_dim)

        return {
            "dynamics_scales": result.dynamics_scales,
            "reference_scale": result.reference_scale,
            "transition_result": result,
        }

    def config_hash(self) -> str:
        return hash_config({
            "loeo_r2_threshold": self._loeo_r2_threshold,
        })


class ObservationAnalysisStage(PipelineStage):
    """Phase 2.6: Analyze observation space for algebraic constraints.

    Detects algebraic constraints among observation dimensions
    (e.g., ``s0^2 + s1^2 = 1`` for cos/sin encodings) and builds
    canonical coordinate mappings (e.g., ``theta = atan2(s1, s0)``).

    When constraints are found and mapped to canonical coordinates,
    downstream dynamics SR operates in the lower-dimensional canonical
    space where dynamics are typically simpler and more exact.

    :param singular_value_threshold: SVD threshold for constraint detection.
    :param circle_tolerance: Tolerance for circle manifold classification.
    """

    def __init__(
        self,
        singular_value_threshold: float = 1e-3,
        circle_tolerance: float = 0.1,
    ) -> None:
        super().__init__(
            name="observation_analysis",
            dependencies=["causal_discovery"],
        )
        self._sv_threshold = singular_value_threshold
        self._circle_tolerance = circle_tolerance

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run observation space analysis.

        :returns: Dict with keys: analysis_result, canonical_dataset,
            canonical_state_names. If no mappable constraints are found,
            analysis_result is None and no canonical dataset is produced.
        """
        from circ_rl.environments.data_collector import ExploratoryDataset
        from circ_rl.observation_analysis.observation_analyzer import (
            ObservationAnalysisConfig,
            ObservationAnalyzer,
        )

        cd_output = inputs["causal_discovery"]
        dataset = cd_output["dataset"]
        state_names: list[str] = cd_output["state_names"]

        config = ObservationAnalysisConfig(
            singular_value_threshold=self._sv_threshold,
            circle_tolerance=self._circle_tolerance,
        )
        analyzer = ObservationAnalyzer(config)
        result = analyzer.analyze(dataset, state_names)

        if not result.mappings:
            logger.info("No canonical mappings found; dynamics will use obs space")
            return {"analysis_result": None}

        logger.info(
            "Observation analysis: {} mapping(s), canonical dims: {}",
            len(result.mappings),
            result.canonical_state_names,
        )

        # Build canonical dataset for downstream dynamics SR
        canonical_dataset = ExploratoryDataset(
            states=result.canonical_states,
            actions=dataset.actions,
            next_states=result.canonical_next_states,
            rewards=dataset.rewards,
            env_ids=dataset.env_ids,
            env_params=dataset.env_params,
        )

        return {
            "analysis_result": result,
            "canonical_dataset": canonical_dataset,
            "canonical_state_names": result.canonical_state_names,
        }

    def config_hash(self) -> str:
        return hash_config({
            "sv_threshold": self._sv_threshold,
            "circle_tolerance": self._circle_tolerance,
        })


class HypothesisGenerationStage(PipelineStage):
    """Phase 3: Generate symbolic hypotheses via symbolic regression.

    Runs PySR on variant state dimensions (dynamics) and reward to
    discover analytic functional forms.

    See ``CIRC-RL_Framework.md`` Section 3.4.

    :param include_env_params: Include env params as SR features.
    :param sr_config: Symbolic regression config for dynamics hypotheses.
        If None, uses SymbolicRegressionConfig defaults.
    :param reward_sr_config: Symbolic regression config for reward
        hypotheses. If None, uses ``sr_config`` (or defaults if both
        are None).
    :param reward_derived_features: Derived feature specs for reward SR.
        These features are computed from state variables and included
        as additional input columns for reward symbolic regression
        (e.g., ``theta = atan2(s1, s0)``).
    """

    def __init__(
        self,
        include_env_params: bool = True,
        sr_config: SymbolicRegressionConfig | None = None,
        reward_sr_config: SymbolicRegressionConfig | None = None,
        reward_derived_features: list[DerivedFeatureSpec] | None = None,
    ) -> None:
        super().__init__(
            name="hypothesis_generation",
            dependencies=[
                "causal_discovery",
                "feature_selection",
                "transition_analysis",
            ],
        )
        self._include_env_params = include_env_params
        self._sr_config = sr_config
        self._reward_sr_config = reward_sr_config or sr_config
        self._reward_derived_features = reward_derived_features or []

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate dynamics and reward hypotheses.

        When ``observation_analysis`` output is present with canonical
        mappings, dynamics SR runs in the canonical coordinate space
        (simpler dynamics). Reward SR always runs in observation space.

        :returns: Dict with keys: register, variable_names,
            dynamics_variable_names, dynamics_state_names, state_names,
            action_names, derived_columns, reward_derived_features.
        """
        from circ_rl.hypothesis.dynamics_hypotheses import DynamicsHypothesisGenerator
        from circ_rl.hypothesis.hypothesis_register import HypothesisRegister
        from circ_rl.hypothesis.reward_hypotheses import RewardHypothesisGenerator

        cd_output = inputs["feature_selection"]  # FeatureSelection produces result
        # We need the dataset and state_names from causal_discovery,
        # which is a dependency of feature_selection
        fs_output = inputs["feature_selection"]
        ta_output = inputs["transition_analysis"]

        # Retrieve artifacts via transitive deps: dataset/state_names
        # flow through the pipeline's all_artifacts dict.
        # Access causal_discovery output via the transition_analysis stage's input.
        # Actually, we need the dataset from the pipeline. The simplest
        # approach: require causal_discovery as a dependency too.
        cd_output = inputs.get("causal_discovery", {})
        dataset = cd_output.get("dataset")
        state_names: list[str] = cd_output.get("state_names", [])
        env_param_node_names: list[str] = cd_output.get("env_param_node_names", [])

        if dataset is None:
            raise ValueError(
                "HypothesisGenerationStage requires causal_discovery output "
                "(dataset, state_names)"
            )

        transition_result = ta_output["transition_result"]
        fs_result = fs_output["result"]
        validation_result = cd_output.get("validation_result")

        # Env param names (raw, not ep_-prefixed)
        from circ_rl.causal_discovery.causal_graph import CausalGraph

        raw_env_param_names: list[str] | None = None
        if env_param_node_names:
            raw_env_param_names = [
                name.removeprefix(CausalGraph.ENV_PARAM_PREFIX)
                for name in env_param_node_names
            ]

        # Check for observation analysis: canonical space for dynamics
        oa_output = inputs.get("observation_analysis", {})
        analysis_result = oa_output.get("analysis_result")

        dynamics_angular_dims: tuple[int, ...] = ()
        if analysis_result is not None:
            dynamics_dataset = oa_output["canonical_dataset"]
            dynamics_state_names = oa_output["canonical_state_names"]
            dynamics_angular_dims = analysis_result.angular_dims
            logger.info(
                "Using canonical coordinates for dynamics SR: {} "
                "(angular_dims={})",
                dynamics_state_names,
                dynamics_angular_dims,
            )
        else:
            dynamics_dataset = dataset
            dynamics_state_names = state_names

        register = HypothesisRegister()

        # Dynamics hypotheses (in canonical space if available)
        dyn_gen = DynamicsHypothesisGenerator(
            sr_config=self._sr_config,
            include_env_params=self._include_env_params,
        )
        dyn_ids = dyn_gen.generate(
            dataset=dynamics_dataset,
            transition_result=transition_result,
            state_feature_names=dynamics_state_names,
            register=register,
            env_param_names=raw_env_param_names,
            angular_dims=dynamics_angular_dims,
            parallel=True,
        )

        # Reward hypotheses (always in observation space)
        reward_is_invariant = (
            validation_result.is_invariant if validation_result else True
        )

        # Pre-compute derived feature columns for reward SR
        reward_derived_cols: dict[str, np.ndarray] | None = None
        if self._reward_derived_features:
            from circ_rl.hypothesis.derived_features import (
                compute_derived_columns,
            )

            reward_derived_cols = compute_derived_columns(
                self._reward_derived_features,
                dataset.states,
                state_names,
            )

        reward_gen = RewardHypothesisGenerator(sr_config=self._reward_sr_config)
        reward_ids = reward_gen.generate(
            dataset=dataset,
            feature_selection_result=fs_result,
            state_feature_names=state_names,
            register=register,
            reward_is_invariant=reward_is_invariant,
            env_param_names=raw_env_param_names,
            derived_columns=reward_derived_cols,
        )

        # Build variable names for reward (observation space)
        actions_2d = (
            dataset.actions if dataset.actions.ndim == 2
            else dataset.actions[:, np.newaxis]
        )
        action_dim = actions_2d.shape[1]
        action_names = (
            ["action"] if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )
        variable_names = list(state_names) + action_names
        # Env params BEFORE derived features (to match _build_features
        # positional env param matching)
        if raw_env_param_names and self._include_env_params:
            variable_names.extend(raw_env_param_names)
        # Derived features LAST
        if reward_derived_cols:
            variable_names.extend(reward_derived_cols.keys())

        # Build dynamics variable names (canonical space if available)
        dynamics_variable_names = list(dynamics_state_names) + action_names
        if raw_env_param_names and self._include_env_params:
            dynamics_variable_names.extend(raw_env_param_names)

        logger.info(
            "Hypothesis generation complete: {} dynamics + {} reward hypotheses",
            len(dyn_ids),
            len(reward_ids),
        )

        return {
            "register": register,
            "variable_names": variable_names,
            "dynamics_variable_names": dynamics_variable_names,
            "dynamics_state_names": dynamics_state_names,
            "state_names": state_names,
            "action_names": action_names,
            "derived_columns": reward_derived_cols,
            "reward_derived_features": self._reward_derived_features,
        }

    def config_hash(self) -> str:
        from dataclasses import asdict

        sr_dict = (
            asdict(self._sr_config) if self._sr_config else None
        )
        reward_sr_dict = (
            asdict(self._reward_sr_config)
            if self._reward_sr_config
            else None
        )
        derived_names = [
            spec.name for spec in self._reward_derived_features
        ]
        return hash_config({
            "include_env_params": self._include_env_params,
            "sr_config": sr_dict,
            "reward_sr_config": reward_sr_dict,
            "reward_derived_features": derived_names,
        })


class HypothesisFalsificationStage(PipelineStage):
    """Phase 4: Falsify and score hypotheses.

    Runs structural consistency, OOD prediction, and trajectory
    prediction tests, then selects the best via symbolic MDL.

    See ``CIRC-RL_Framework.md`` Section 3.5.

    :param structural_p_threshold: p-value for structural consistency.
    :param structural_min_relative_improvement: Practical significance
        threshold for structural consistency. Default 0.01 (1%).
    :param ood_confidence: OOD confidence interval level.
    :param held_out_fraction: Fraction of envs held out for OOD test.
    """

    def __init__(
        self,
        structural_p_threshold: float = 0.01,
        structural_min_relative_improvement: float = 0.01,
        ood_confidence: float = 0.99,
        held_out_fraction: float = 0.2,
    ) -> None:
        super().__init__(
            name="hypothesis_falsification",
            dependencies=["hypothesis_generation", "causal_discovery"],
        )
        self._structural_p = structural_p_threshold
        self._structural_min_ri = structural_min_relative_improvement
        self._ood_confidence = ood_confidence
        self._held_out_fraction = held_out_fraction

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run falsification on all hypotheses.

        When observation analysis produced canonical coordinates,
        dynamics hypotheses are tested against canonical data while
        reward hypotheses are tested against observation-space data.

        :returns: Dict with keys: register, falsification_result,
            best_dynamics, best_reward, dynamics_state_names.
        """
        from circ_rl.hypothesis.falsification_engine import (
            FalsificationConfig,
            FalsificationEngine,
        )
        from circ_rl.hypothesis.hypothesis_register import HypothesisStatus

        hg_output = inputs["hypothesis_generation"]
        cd_output = inputs["causal_discovery"]

        register = hg_output["register"]
        variable_names = hg_output["variable_names"]
        state_names = hg_output["state_names"]
        dynamics_state_names = hg_output.get("dynamics_state_names", state_names)
        dynamics_variable_names = hg_output.get(
            "dynamics_variable_names", variable_names,
        )
        derived_columns = hg_output.get("derived_columns")
        reward_derived_features = hg_output.get(
            "reward_derived_features", [],
        )
        dataset = cd_output["dataset"]

        # Check if canonical dataset is available for dynamics
        oa_output = inputs.get("observation_analysis", {})
        canonical_dataset = oa_output.get("canonical_dataset")
        analysis_result = oa_output.get("analysis_result")
        use_canonical = (
            canonical_dataset is not None
            and dynamics_state_names != state_names
        )
        canonical_angular_dims: tuple[int, ...] = ()
        if analysis_result is not None:
            canonical_angular_dims = analysis_result.angular_dims

        config = FalsificationConfig(
            structural_p_threshold=self._structural_p,
            structural_min_relative_improvement=self._structural_min_ri,
            ood_confidence=self._ood_confidence,
            held_out_fraction=self._held_out_fraction,
        )
        engine = FalsificationEngine(config)

        if use_canonical:
            # Dynamics hypotheses use canonical data.
            # Build derived_columns so _build_features can resolve canonical
            # state names (e.g. "phi_0", "s2") to the correct columns in the
            # canonical dataset -- the sN naming convention doesn't match
            # canonical column indices.
            canonical_derived: dict[str, np.ndarray] = {}
            for i, cname in enumerate(dynamics_state_names):
                canonical_derived[cname] = canonical_dataset.states[:, i]

            # Separate dynamics vs reward targets so the first pass
            # (canonical variable names) doesn't falsify reward hypotheses
            # that use observation-space variable names.
            dynamics_targets = {
                e.target_variable
                for e in register.get_by_status(HypothesisStatus.UNTESTED)
                if e.target_variable != "reward"
            }

            logger.info(
                "Running falsification with canonical data for dynamics "
                "(state_names={}, targets={})",
                dynamics_state_names,
                dynamics_targets,
            )
            result = engine.run(
                register=register,
                dataset=canonical_dataset,
                state_feature_names=dynamics_state_names,
                variable_names=dynamics_variable_names,
                derived_columns=canonical_derived,
                angular_dims=canonical_angular_dims,
                target_filter=dynamics_targets,
            )
            # Reward hypotheses use observation-space data
            # (run a second pass for any remaining untested reward hypotheses)
            result_reward = engine.run(
                register=register,
                dataset=dataset,
                state_feature_names=state_names,
                variable_names=variable_names,
                derived_columns=derived_columns,
                target_filter={"reward"},
            )
            # Merge results: combine counts from both passes
            total_tested = result.n_tested + result_reward.n_tested
            total_validated = result.n_validated + result_reward.n_validated
            total_falsified = result.n_falsified + result_reward.n_falsified
        else:
            result = engine.run(
                register=register,
                dataset=dataset,
                state_feature_names=state_names,
                variable_names=variable_names,
                derived_columns=derived_columns,
            )
            result_reward = result
            total_tested = result.n_tested
            total_validated = result.n_validated
            total_falsified = result.n_falsified

        # Extract best validated hypotheses (from both passes)
        best_dynamics: dict[str, Any] = {}
        best_reward = None

        # Dynamics from first result
        for target, hyp_id in result.best_per_target.items():
            if target == "reward":
                continue
            if hyp_id is None:
                fallback_id = result.best_effort_per_target.get(target)
                if fallback_id is not None:
                    entry = register.get(fallback_id)
                    best_dynamics[target] = entry
                    logger.warning(
                        "Using best-effort (unvalidated) hypothesis for "
                        "'{}': {} (R2={:.4f})",
                        target,
                        entry.hypothesis_id,
                        entry.training_r2,
                    )
                continue
            best_dynamics[target] = register.get(hyp_id)

        # Reward from second result (or same if no canonical)
        reward_target = result_reward.best_per_target.get("reward")
        if reward_target is not None:
            best_reward = register.get(reward_target)

        logger.info(
            "Falsification complete: {}/{} validated, best_dynamics={}, "
            "best_reward={}",
            total_validated,
            total_tested,
            list(best_dynamics.keys()),
            best_reward.hypothesis_id if best_reward else None,
        )

        return {
            "register": register,
            "falsification_result": result,
            "best_dynamics": best_dynamics,
            "best_reward": best_reward,
            "variable_names": variable_names,
            "dynamics_variable_names": dynamics_variable_names,
            "dynamics_state_names": dynamics_state_names,
            "state_names": state_names,
            "reward_derived_features": reward_derived_features,
        }

    def config_hash(self) -> str:
        return hash_config({
            "structural_p": self._structural_p,
            "structural_min_ri": self._structural_min_ri,
            "ood_confidence": self._ood_confidence,
            "held_out_fraction": self._held_out_fraction,
        })


class AnalyticPolicyDerivationStage(PipelineStage):
    """Phase 5: Derive analytic policy from validated hypotheses.

    Classifies dynamics as linear/nonlinear, then applies LQR or MPC
    solver per environment.

    See ``CIRC-RL_Framework.md`` Section 3.6.

    :param env_family: Environment family for per-env solving.
    :param gamma: Discount factor for LQR DARE.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        gamma: float = 0.99,
        adaptive_replan_multiplier: float = 3.0,
        min_replan_interval: int = 3,
        robust_n_scenarios: int = 0,
        robust_confidence_level: float = 0.95,
        robust_min_uncertainty: float = 1e-6,
    ) -> None:
        super().__init__(
            name="analytic_policy_derivation",
            dependencies=[
                "hypothesis_falsification",
                "transition_analysis",
                "causal_discovery",
            ],
        )
        self._env_family = env_family
        self._gamma = gamma
        self._adaptive_replan_multiplier = adaptive_replan_multiplier
        self._min_replan_interval = min_replan_interval
        self._robust_n_scenarios = robust_n_scenarios
        self._robust_confidence_level = robust_confidence_level
        self._robust_min_uncertainty = robust_min_uncertainty

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Derive the analytic policy.

        :returns: Dict with keys: analytic_policy, explained_variance,
            solver_type, dynamics_expressions.
        """
        from circ_rl.analytic_policy.analytic_policy import (
            AnalyticPolicy,
            extract_linear_dynamics,
            extract_quadratic_cost,
        )
        from circ_rl.analytic_policy.hypothesis_classifier import HypothesisClassifier
        from circ_rl.analytic_policy.lqr_solver import (
            LinearDynamics,
            LQRSolver,
            QuadraticCost,
        )

        hf_output = inputs["hypothesis_falsification"]
        ta_output = inputs["transition_analysis"]
        cd_output = inputs.get("causal_discovery", {})

        best_dynamics = hf_output["best_dynamics"]
        best_reward = hf_output["best_reward"]
        state_names = hf_output["state_names"]
        dynamics_state_names = hf_output.get(
            "dynamics_state_names", state_names,
        )
        reward_derived_features = hf_output.get(
            "reward_derived_features", [],
        )

        # Check for observation analysis (canonical space)
        oa_output = inputs.get("observation_analysis", {})
        analysis_result = oa_output.get("analysis_result")
        obs_to_canonical_fn = None
        canonical_to_obs_fn = None
        angular_dims: tuple[int, ...] = ()
        if analysis_result is not None:
            obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
            canonical_to_obs_fn = analysis_result.canonical_to_obs_fn
            angular_dims = analysis_result.angular_dims
            logger.info(
                "iLQR will plan in canonical space: {} (angular={})",
                dynamics_state_names,
                angular_dims,
            )

        if not best_dynamics:
            raise ValueError(
                "No validated dynamics hypotheses found. "
                "Cannot derive analytic policy."
            )

        ta_output["transition_result"]

        # Get action info
        import gymnasium as gym

        action_space = self._env_family.action_space
        if isinstance(action_space, gym.spaces.Box):
            action_dim = int(action_space.shape[0])
            action_names = (
                ["action"] if action_dim == 1
                else [f"action_{i}" for i in range(action_dim)]
            )
            action_low = action_space.low
            action_high = action_space.high
        else:
            raise TypeError(
                "Analytic policy derivation requires continuous (Box) "
                f"action space, got {type(action_space)}"
            )

        # Use canonical state names for dynamics if available
        effective_state_names = dynamics_state_names
        state_dim = len(effective_state_names)

        # Classify dynamics and derive policy
        classifier = HypothesisClassifier()

        # Collect validated dynamics expressions by dim index
        dynamics_expressions: dict[int, SymbolicExpression] = {}
        for target, entry in best_dynamics.items():
            # target = "delta_<dim_name>" -> dim_idx
            dim_name = target.removeprefix("delta_")
            dim_idx = effective_state_names.index(dim_name)
            dynamics_expressions[dim_idx] = entry.expression

        # Spurious term detection: prune discretization artifacts
        spurious_dataset = cd_output.get("dataset")
        if spurious_dataset is not None:
            from circ_rl.hypothesis.spurious_detection import (
                SpuriousTermDetector,
            )

            detector = SpuriousTermDetector(
                r2_contribution_threshold=0.005,
            )
            # Build variable names matching what SR used
            spvar_names = list(effective_state_names) + list(action_names)
            if self._env_family.param_names:
                spvar_names.extend(self._env_family.param_names)

            # Use canonical dataset if available
            sp_dataset = spurious_dataset
            oa_canonical = inputs.get("observation_analysis", {}).get(
                "canonical_dataset",
            )
            if oa_canonical is not None:
                sp_dataset = oa_canonical

            for dim_idx, expr in list(dynamics_expressions.items()):
                try:
                    sp_result = detector.detect(
                        expr, sp_dataset, dim_idx, spvar_names,
                        wrap_angular=(dim_idx in angular_dims),
                    )
                    if sp_result.n_spurious > 0:
                        from circ_rl.hypothesis.expression import (
                            SymbolicExpression as _SPE,
                        )

                        dynamics_expressions[dim_idx] = (
                            _SPE.from_sympy(sp_result.pruned_expr)
                        )
                        logger.info(
                            "Pruned {} spurious terms from dim {} "
                            "(R2: {:.6f} -> {:.6f})",
                            sp_result.n_spurious, dim_idx,
                            sp_result.original_r2, sp_result.pruned_r2,
                        )
                except Exception:
                    logger.debug(
                        "Spurious detection skipped for dim {} "
                        "(evaluation error)",
                        dim_idx,
                    )

        # Determine solver type from all expressions
        all_linear = all(
            classifier.classify(
                entry.expression.sympy_expr,
                effective_state_names,
                action_names,
            ) == "lqr"
            for entry in best_dynamics.values()
        )
        solver_type = "lqr" if all_linear else "mpc"

        # Warn about missing dynamics dimensions
        missing_dims = set(range(state_dim)) - set(dynamics_expressions.keys())
        if missing_dims:
            missing_names = [
                effective_state_names[d] for d in sorted(missing_dims)
            ]
            logger.warning(
                "Analytic policy: no validated dynamics for dimensions {} "
                "({}). These will use identity dynamics (delta=0).",
                sorted(missing_dims),
                missing_names,
            )

        logger.info(
            "Analytic policy: solver={}, {}/{} dynamics dimensions covered",
            solver_type,
            len(dynamics_expressions),
            state_dim,
        )

        explained_variance = 0.0
        analytic_policy: AnalyticPolicy

        if solver_type == "lqr":
            # Build LQR solutions per environment
            lqr_solver = LQRSolver()
            lqr_solutions: dict[int, Any] = {}

            for env_idx in range(self._env_family.n_envs):
                env_params = self._env_family.get_env_params(env_idx)

                # Stack dynamics rows into full A, B matrices
                a_rows = []
                b_rows = []
                c_vals = []

                for dim_idx in range(state_dim):
                    if dim_idx in dynamics_expressions:
                        ld = extract_linear_dynamics(
                            dynamics_expressions[dim_idx],
                            effective_state_names,
                            action_names,
                            env_params=env_params if env_params else None,
                        )
                        a_rows.append(ld.a_matrix)
                        b_rows.append(ld.b_matrix)
                        c_vals.append(ld.c_vector[0])
                    else:
                        # Invariant dimension: identity dynamics
                        a_row = np.zeros((1, state_dim))
                        b_row = np.zeros((1, action_dim))
                        a_rows.append(a_row)
                        b_rows.append(b_row)
                        c_vals.append(0.0)

                a_matrix = np.eye(state_dim) + np.vstack(a_rows)
                b_matrix = np.vstack(b_rows)

                # Cost matrices: try extracting from reward hypothesis
                q_cost = None
                if best_reward is not None:
                    q_cost = extract_quadratic_cost(
                        best_reward.expression.sympy_expr,
                        state_names,
                        action_names,
                    )

                if q_cost is None:
                    # Default: penalize all state deviations equally
                    q_cost = QuadraticCost(
                        q_matrix=np.eye(state_dim),
                        r_matrix=np.eye(action_dim) * 0.01,
                    )

                full_dynamics = LinearDynamics(
                    a_matrix=a_matrix,
                    b_matrix=b_matrix,
                    c_vector=np.array(c_vals),
                )

                sol = lqr_solver.solve(
                    full_dynamics, q_cost, gamma=self._gamma,
                )
                lqr_solutions[env_idx] = sol

            # Use best dynamics entry for policy metadata
            first_entry = next(iter(best_dynamics.values()))
            analytic_policy = AnalyticPolicy(
                dynamics_hypothesis=first_entry,
                reward_hypothesis=best_reward,
                solver_type="lqr",
                state_dim=state_dim,
                action_dim=action_dim,
                n_envs=self._env_family.n_envs,
                lqr_solutions=lqr_solutions,
                action_low=action_low,
                action_high=action_high,
            )

            # Estimate explained variance from training R^2
            r2_values = [e.training_r2 for e in best_dynamics.values()]
            explained_variance = float(np.mean(r2_values))

        else:
            # iLQR solver for nonlinear dynamics
            from circ_rl.analytic_policy.ilqr_solver import (
                ILQRConfig,
                ILQRSolver,
                make_quadratic_terminal_cost,
            )

            first_entry = next(iter(best_dynamics.values()))

            # Build per-env iLQR solvers
            ilqr_solvers: dict[int, ILQRSolver] = {}

            # Obs bounds only used when NOT in canonical space
            obs_low_clamp: np.ndarray | None = None
            obs_high_clamp: np.ndarray | None = None
            if obs_to_canonical_fn is None:
                _ref_env = self._env_family.make_env(0)
                obs_low_clamp = np.asarray(
                    _ref_env.observation_space.low,  # type: ignore[attr-defined]
                    dtype=np.float64,
                )
                obs_high_clamp = np.asarray(
                    _ref_env.observation_space.high,  # type: ignore[attr-defined]
                    dtype=np.float64,
                )
                _ref_env.close()

            for env_idx in range(self._env_family.n_envs):
                env_params = self._env_family.get_env_params(env_idx)

                # Extract per-dim calibration coefficients for this env
                per_dim_cal: dict[int, tuple[float, float]] = {}
                for target, entry in best_dynamics.items():
                    dim_name = target.removeprefix("delta_")
                    dim_idx = effective_state_names.index(dim_name)
                    if (
                        entry.calibration_coefficients
                        and env_idx in entry.calibration_coefficients
                    ):
                        per_dim_cal[dim_idx] = (
                            entry.calibration_coefficients[env_idx]
                        )
                cal_arg = per_dim_cal or None

                # Build dynamics callable in effective (canonical) space
                dynamics_fn = _build_dynamics_fn(
                    dynamics_expressions,
                    effective_state_names,
                    action_names,
                    state_dim,
                    env_params,
                    obs_low=obs_low_clamp,
                    obs_high=obs_high_clamp,
                    angular_dims=angular_dims,
                    calibration_coefficients=cal_arg,
                )

                # Build reward callable (always in obs space)
                reward_fn = None
                if best_reward is not None:
                    reward_fn = _build_reward_fn(
                        best_reward.expression,
                        state_names,
                        action_names,
                        env_params,
                        reward_derived_features,
                        canonical_to_obs_fn=canonical_to_obs_fn,
                    )

                # Build analytic Jacobians in effective (canonical) space
                jac_state_fn, jac_action_fn = _build_dynamics_jacobian_fns(
                    dynamics_expressions,
                    effective_state_names,
                    action_names,
                    state_dim,
                    env_params,
                    calibration_coefficients=cal_arg,
                )

                # Build analytic reward derivatives if possible
                reward_deriv_fn = None
                if best_reward is not None:
                    reward_deriv_fn = _build_reward_derivative_fns(
                        best_reward.expression,
                        effective_state_names,
                        action_names,
                        env_params,
                        reward_derived_features or None,
                        obs_state_names=state_names,
                    )

                effective_reward = reward_fn or _default_reward
                terminal_cost_fn = make_quadratic_terminal_cost(
                    reward_fn=effective_reward,
                    action_dim=action_dim,
                    gamma=self._gamma,
                    state_dim=state_dim,
                    scale_override=10.0,
                )

                # Compute adaptive replan threshold from dynamics MSEs
                dynamics_mse_sum = sum(
                    e.training_mse for e in best_dynamics.values()
                )
                adaptive_tau: float | None = None
                if dynamics_mse_sum > 0:
                    adaptive_tau = (
                        self._adaptive_replan_multiplier
                        * np.sqrt(dynamics_mse_sum)
                    )

                ilqr_config = ILQRConfig(
                    horizon=100,
                    gamma=self._gamma,
                    max_action=float(action_high[0]),
                    n_random_restarts=8,
                    restart_scale=0.3,
                    replan_interval=10,
                    adaptive_replan_threshold=adaptive_tau,
                    min_replan_interval=self._min_replan_interval,
                )

                ilqr_solvers[env_idx] = ILQRSolver(
                    config=ilqr_config,
                    dynamics_fn=dynamics_fn,
                    reward_fn=effective_reward,
                    dynamics_jac_state_fn=jac_state_fn,
                    dynamics_jac_action_fn=jac_action_fn,
                    terminal_cost_fn=terminal_cost_fn,
                    reward_derivatives_fn=reward_deriv_fn,
                )

            # Build robust planners if enabled and uncertainty
            # is significant
            robust_planners: dict[int, RobustILQRPlanner] | None = None
            if self._robust_n_scenarios >= 2:
                robust_planners = self._build_robust_planners(
                    best_dynamics=best_dynamics,
                    cd_output=cd_output,
                    ilqr_solvers=ilqr_solvers,
                    dynamics_expressions=dynamics_expressions,
                    effective_state_names=effective_state_names,
                    action_names=action_names,
                    state_dim=state_dim,
                    env_params_per_env={
                        i: self._env_family.get_env_params(i)
                        for i in range(self._env_family.n_envs)
                    },
                    obs_low_clamp=obs_low_clamp,
                    obs_high_clamp=obs_high_clamp,
                    angular_dims=angular_dims,
                )

            # Wrap the per-env iLQR solvers
            analytic_policy = _ILQRAnalyticPolicy(
                dynamics_hypothesis=first_entry,
                reward_hypothesis=best_reward,
                state_dim=state_dim,
                action_dim=action_dim,
                n_envs=self._env_family.n_envs,
                ilqr_solvers=ilqr_solvers,
                action_low=action_low,
                action_high=action_high,
                obs_to_canonical_fn=obs_to_canonical_fn,
                robust_planners=robust_planners,
            )

            solver_type = "ilqr"
            r2_values = [e.training_r2 for e in best_dynamics.values()]
            explained_variance = float(np.mean(r2_values))

        logger.info(
            "Analytic policy derived: solver={}, explained_variance={:.4f}",
            solver_type,
            explained_variance,
        )

        return {
            "analytic_policy": analytic_policy,
            "explained_variance": explained_variance,
            "solver_type": solver_type,
            "dynamics_expressions": dynamics_expressions,
        }

    def _build_robust_planners(
        self,
        best_dynamics: dict[str, HypothesisEntry],
        cd_output: dict[str, Any],
        ilqr_solvers: dict[int, ILQRSolver],
        dynamics_expressions: dict[int, SymbolicExpression],
        effective_state_names: list[str],
        action_names: list[str],
        state_dim: int,
        env_params_per_env: dict[int, dict[str, float]],
        obs_low_clamp: np.ndarray | None,
        obs_high_clamp: np.ndarray | None,
        angular_dims: tuple[int, ...],
    ) -> dict[int, RobustILQRPlanner] | None:
        """Build per-env robust planners from coefficient uncertainty.

        Returns None if uncertainty is below threshold or dataset is
        unavailable.
        """
        from circ_rl.analytic_policy.coefficient_calibrator import (
            CoefficientCalibrator,
        )
        from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolver
        from circ_rl.analytic_policy.robust_mpc import (
            RobustILQRPlanner,
            RobustMPCConfig,
            ScenarioSampler,
            build_scenario_dynamics_fn,
            build_scenario_jacobian_fns,
            check_uncertainty_significant,
        )

        dataset = cd_output.get("dataset")
        if dataset is None:
            logger.warning(
                "Robust MPC: dataset not available, skipping"
            )
            return None

        robust_config = RobustMPCConfig(
            n_scenarios=self._robust_n_scenarios,
            confidence_level=self._robust_confidence_level,
            min_uncertainty_threshold=self._robust_min_uncertainty,
        )

        # Run calibration per dynamics dimension
        calibrator = CoefficientCalibrator(
            confidence_level=self._robust_confidence_level,
        )

        from circ_rl.analytic_policy.coefficient_calibrator import (
            CoefficientUncertainty,
        )

        per_dim_uncertainty: dict[int, CoefficientUncertainty] = {}
        for target, entry in best_dynamics.items():
            from circ_rl.hypothesis.expression import (
                SymbolicExpression as _SE,
            )

            expr = entry.expression
            if not isinstance(expr, _SE):
                continue
            dim_name = target.removeprefix("delta_")
            dim_idx = effective_state_names.index(dim_name)
            wrap_angular = dim_idx in angular_dims

            variable_names = effective_state_names + action_names
            try:
                unc = calibrator.calibrate(
                    expr, dataset, dim_idx, variable_names,
                    wrap_angular=wrap_angular,
                )
                per_dim_uncertainty[dim_idx] = unc
            except Exception:
                logger.debug(
                    "Robust MPC: calibration failed for dim {}, "
                    "skipping",
                    dim_idx,
                )

        if not per_dim_uncertainty:
            return None

        sampler = ScenarioSampler(robust_config)
        rng = np.random.default_rng(42)

        robust_planners: dict[int, RobustILQRPlanner] = {}

        for env_idx in range(self._env_family.n_envs):
            if not check_uncertainty_significant(
                per_dim_uncertainty, env_idx,
                robust_config.min_uncertainty_threshold,
            ):
                continue

            base_solver = ilqr_solvers.get(env_idx)
            if base_solver is None:
                continue

            scenarios = sampler.sample_scenarios(
                per_dim_uncertainty, env_idx, rng,
            )

            # Build base dynamics fn (without calibration -- robust
            # MPC applies its own scenario coefficients)
            env_params = env_params_per_env[env_idx]
            base_dyn_fn = _build_dynamics_fn(
                dynamics_expressions,
                effective_state_names,
                action_names,
                state_dim,
                env_params,
                obs_low=obs_low_clamp,
                obs_high=obs_high_clamp,
                angular_dims=angular_dims,
                calibration_coefficients=None,
            )
            base_jac_state = base_solver._jac_state_fn
            base_jac_action = base_solver._jac_action_fn

            # Build per-scenario solvers
            scenario_solvers: list[ILQRSolver] = []
            scenario_dynamics: list[
                Callable[[np.ndarray, np.ndarray], np.ndarray]
            ] = []

            for scenario_coeffs in scenarios:
                sc_dyn = build_scenario_dynamics_fn(
                    base_dyn_fn, scenario_coeffs, state_dim,
                )
                sc_jac_state, sc_jac_action = build_scenario_jacobian_fns(
                    base_jac_state, base_jac_action,
                    scenario_coeffs, state_dim,
                )

                sc_config = ILQRConfig(
                    horizon=base_solver.config.horizon,
                    gamma=base_solver.config.gamma,
                    max_action=base_solver.config.max_action,
                    n_random_restarts=robust_config.reduced_restarts,
                    restart_scale=base_solver.config.restart_scale,
                    replan_interval=base_solver.config.replan_interval,
                )

                sc_solver = ILQRSolver(
                    config=sc_config,
                    dynamics_fn=sc_dyn,
                    reward_fn=base_solver._reward_fn,
                    dynamics_jac_state_fn=sc_jac_state,
                    dynamics_jac_action_fn=sc_jac_action,
                    terminal_cost_fn=base_solver._terminal_cost_fn,
                    reward_derivatives_fn=base_solver._reward_derivatives_fn,
                )
                scenario_solvers.append(sc_solver)
                scenario_dynamics.append(sc_dyn)

            robust_planners[env_idx] = RobustILQRPlanner(
                config=robust_config,
                base_solver=base_solver,
                scenario_solvers=scenario_solvers,
                scenario_dynamics=scenario_dynamics,
            )

        if robust_planners:
            logger.info(
                "Robust MPC enabled for {} envs with {} scenarios",
                len(robust_planners), self._robust_n_scenarios,
            )
        else:
            logger.info(
                "Robust MPC: uncertainty below threshold for all "
                "envs, using standard MPC"
            )
            return None

        return robust_planners

    def config_hash(self) -> str:
        return hash_config({
            "gamma": self._gamma,
            "n_envs": self._env_family.n_envs,
        })


class ResidualLearningStage(PipelineStage):
    """Phase 6: Train bounded residual correction.

    Trains a small neural network correction on top of the analytic
    policy, bounded by eta_max * |a_analytic|.

    See ``CIRC-RL_Framework.md`` Section 3.7.

    :param env_family: Environment family for rollouts.
    :param n_iterations: Number of residual PPO iterations.
    :param eta_max: Maximum correction fraction.
    :param skip_if_eta2_above: Skip if explained variance exceeds this.
    :param abort_if_eta2_below: Abort if explained variance below this.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_iterations: int = 50,
        eta_max: float = 0.1,
        skip_if_eta2_above: float = 0.98,
        abort_if_eta2_below: float = 0.70,
    ) -> None:
        super().__init__(
            name="residual_learning",
            dependencies=["analytic_policy_derivation"],
        )
        self._env_family = env_family
        self._n_iterations = n_iterations
        self._eta_max = eta_max
        self._skip_threshold = skip_if_eta2_above
        self._abort_threshold = abort_if_eta2_below

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run residual training.

        :returns: Dict with keys: composite_policy, residual_policy,
            residual_metrics, skipped.
        """
        from circ_rl.policy.composite_policy import CompositePolicy
        from circ_rl.policy.residual_policy import ResidualPolicy
        from circ_rl.training.residual_trainer import (
            ResidualTrainer,
            ResidualTrainingConfig,
        )

        apd_output = inputs["analytic_policy_derivation"]
        analytic_policy = apd_output["analytic_policy"]
        explained_variance = apd_output["explained_variance"]

        import gymnasium as gym

        action_space = self._env_family.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"Residual learning requires Box action space, "
                f"got {type(action_space)}"
            )
        state_dim = int(self._env_family.observation_space.shape[0])  # type: ignore[union-attr]
        action_dim = int(action_space.shape[0])

        config = ResidualTrainingConfig(
            n_iterations=self._n_iterations,
            eta_max=self._eta_max,
            explained_variance=explained_variance,
            skip_if_eta2_above=self._skip_threshold,
            abort_if_eta2_below=self._abort_threshold,
        )

        residual = ResidualPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            eta_max=self._eta_max,
        )

        trainer = ResidualTrainer(
            analytic_policy=analytic_policy,
            residual_policy=residual,
            env_family=self._env_family,
            config=config,
        )

        skipped = False
        metrics: list[Any] = []

        if trainer.should_skip():
            logger.info(
                "Residual learning skipped: explained_variance={:.4f} > {:.4f}",
                explained_variance,
                self._skip_threshold,
            )
            skipped = True
            residual_out = None
        elif trainer.should_abort():
            logger.warning(
                "Residual learning aborted: explained_variance={:.4f} < {:.4f}",
                explained_variance,
                self._abort_threshold,
            )
            residual_out = None
        else:
            metrics = trainer.run()
            residual_out = residual

        composite = CompositePolicy(
            analytic_policy=analytic_policy,
            residual_policy=residual_out,
            explained_variance=explained_variance,
        )

        return {
            "composite_policy": composite,
            "residual_policy": residual_out,
            "residual_metrics": metrics,
            "skipped": skipped,
        }

    def config_hash(self) -> str:
        return hash_config({
            "n_iterations": self._n_iterations,
            "eta_max": self._eta_max,
            "skip_threshold": self._skip_threshold,
            "abort_threshold": self._abort_threshold,
        })


class DiagnosticValidationStage(PipelineStage):
    """Phase 7: Run diagnostic validation suite.

    Tests premise (dynamics hypothesis still valid), derivation
    (policy trajectories match predictions), and conclusion
    (observed returns match predicted).

    See ``CIRC-RL_Framework.md`` Section 3.9.

    :param env_family: Environment family for evaluation.
    :param premise_r2_threshold: R^2 threshold for premise test.
    :param derivation_divergence_threshold: Divergence threshold.
    :param conclusion_error_threshold: Relative error threshold.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        premise_r2_threshold: float = 0.5,
        derivation_divergence_threshold: float = 1.0,
        conclusion_error_threshold: float = 0.3,
    ) -> None:
        super().__init__(
            name="diagnostic_validation",
            dependencies=[
                "analytic_policy_derivation",
                "residual_learning",
                "causal_discovery",
                "hypothesis_falsification",
            ],
        )
        self._env_family = env_family
        self._premise_r2 = premise_r2_threshold
        self._derivation_div = derivation_divergence_threshold
        self._conclusion_err = conclusion_error_threshold

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the diagnostic suite.

        :returns: Dict with keys: diagnostic_result, recommended_action.
        """
        import gymnasium as gym

        from circ_rl.diagnostics.diagnostic_suite import DiagnosticSuite

        apd_output = inputs["analytic_policy_derivation"]
        inputs["residual_learning"]
        cd_output = inputs["causal_discovery"]
        hf_output = inputs["hypothesis_falsification"]

        analytic_policy = apd_output["analytic_policy"]
        dynamics_expressions = apd_output["dynamics_expressions"]
        dataset = cd_output["dataset"]
        state_names = cd_output["state_names"]
        variable_names = hf_output["variable_names"]
        best_reward = hf_output["best_reward"]
        reward_derived_features = hf_output.get(
            "reward_derived_features", [],
        )

        # Derive action_names from environment
        action_space = self._env_family.action_space
        if isinstance(action_space, gym.spaces.Box):
            action_dim = int(action_space.shape[0])
            action_names = (
                ["action"] if action_dim == 1
                else [f"action_{i}" for i in range(action_dim)]
            )
        else:
            action_names = ["action"]

        # Check for missing dynamics dimensions -- an incomplete dynamics
        # model means the premise is fundamentally violated; short-circuit
        # to RICHER_DYNAMICS without running the full diagnostic suite
        state_dim = len(state_names)
        covered_dims = set(dynamics_expressions.keys())
        all_dims = set(range(state_dim))
        missing_dims = all_dims - covered_dims

        if missing_dims:
            from circ_rl.diagnostics.diagnostic_suite import (
                DiagnosticResult,
                RecommendedAction,
            )
            from circ_rl.diagnostics.premise_test import PremiseTestResult

            missing_names = [state_names[d] for d in sorted(missing_dims)]
            logger.warning(
                "Dynamics model is INCOMPLETE: no validated hypotheses for "
                "dimensions {} ({}). Skipping full diagnostics -- "
                "recommendation: richer dynamics.",
                sorted(missing_dims),
                missing_names,
            )

            # Build a synthetic premise failure result
            premise_fail = PremiseTestResult(
                passed=False,
                per_env_r2={},
                overall_r2=0.0,
                per_env_rmse={},
            )
            result = DiagnosticResult(
                premise_result=premise_fail,
                derivation_result=None,
                conclusion_result=None,
                recommended_action=RecommendedAction.RICHER_DYNAMICS,
            )

            logger.info(
                "Diagnostic validation: recommended_action={}",
                result.recommended_action.value,
            )

            return {
                "diagnostic_result": result,
                "recommended_action": result.recommended_action,
            }

        # Use a representative subset of envs for diagnostics.
        # Testing all envs is O(n_envs) in iLQR solves and dominates
        # runtime.  Five evenly-spaced envs give good coverage of the
        # parametric range while keeping the stage under 15 min.
        n_envs = self._env_family.n_envs
        max_diag_envs = min(5, n_envs)
        step = max(1, n_envs // max_diag_envs)
        test_env_ids = list(range(0, n_envs, step))[:max_diag_envs]

        # Lighten the iLQR solvers for diagnostic evaluation:
        # - Disable robust MPC (diagnostics measure quality, not robustness)
        # - Reduce random restarts (diagnostics need representative, not
        #   peak performance)
        # - Widen replan interval (coarser but still valid for averages)
        # This yields ~10x speedup with negligible diagnostic quality loss.
        saved_robust = getattr(analytic_policy, "_robust_planners", None)
        saved_configs: dict[int, Any] = {}
        solvers = getattr(analytic_policy, "_ilqr_solvers", None)

        if saved_robust is not None:
            analytic_policy._robust_planners = None  # type: ignore[union-attr]

        # Force-disable adaptive replanning at the policy level
        if hasattr(analytic_policy, "_force_disable_adaptive_replan"):
            analytic_policy._force_disable_adaptive_replan = True  # type: ignore[union-attr]

        if solvers is not None:
            from dataclasses import replace as _dc_replace

            for env_id, solver in solvers.items():
                saved_configs[env_id] = solver._config
                solver._config = _dc_replace(
                    solver._config,
                    n_random_restarts=1,
                    replan_interval=25,
                    parallel_restarts=False,
                    adaptive_replan_threshold=None,
                    min_replan_interval=25,
                )
            logger.info(
                "Diagnostic validation: lightweight iLQR "
                "(restarts=1, replan_interval=25, no adaptive, no robust MPC)",
            )

        # Clear cached plans so the lighter config takes effect
        if hasattr(analytic_policy, "_solutions"):
            analytic_policy._solutions.clear()  # type: ignore[union-attr]
        if hasattr(analytic_policy, "_steps"):
            analytic_policy._steps.clear()  # type: ignore[union-attr]

        # Compute predicted returns by simulating the policy through
        # the learned dynamics + reward models from real initial states
        diag_episodes = 3
        predicted_returns = _compute_predicted_returns(
            analytic_policy=analytic_policy,
            dynamics_expressions=dynamics_expressions,
            reward_expression=(
                best_reward.expression if best_reward is not None else None
            ),
            state_names=state_names,
            action_names=action_names,
            env_family=self._env_family,
            test_env_ids=test_env_ids,
            n_episodes=diag_episodes,
            reward_derived_features=reward_derived_features,
        )

        suite = DiagnosticSuite(
            premise_r2_threshold=self._premise_r2,
            derivation_divergence_threshold=self._derivation_div,
            conclusion_error_threshold=self._conclusion_err,
            eval_episodes=diag_episodes,
        )

        try:
            result = suite.run(
                policy=analytic_policy,
                dynamics_expressions=dynamics_expressions,
                dataset=dataset,
                state_feature_names=state_names,
                variable_names=variable_names,
                env_family=self._env_family,
                predicted_returns=predicted_returns,
                test_env_ids=test_env_ids,
            )
        finally:
            # Restore deployment-grade solver configs
            if solvers is not None:
                for env_id, orig_config in saved_configs.items():
                    if env_id in solvers:
                        solvers[env_id]._config = orig_config
            if saved_robust is not None:
                analytic_policy._robust_planners = saved_robust  # type: ignore[union-attr]
            # Restore adaptive replanning
            if hasattr(analytic_policy, "_force_disable_adaptive_replan"):
                analytic_policy._force_disable_adaptive_replan = False  # type: ignore[union-attr]
            # Clear diagnostic-mode cached plans
            if hasattr(analytic_policy, "_solutions"):
                analytic_policy._solutions.clear()  # type: ignore[union-attr]
            if hasattr(analytic_policy, "_steps"):
                analytic_policy._steps.clear()  # type: ignore[union-attr]

        logger.info(
            "Diagnostic validation: recommended_action={}",
            result.recommended_action.value,
        )

        return {
            "diagnostic_result": result,
            "recommended_action": result.recommended_action,
        }

    def config_hash(self) -> str:
        return hash_config({
            "premise_r2": self._premise_r2,
            "derivation_div": self._derivation_div,
            "conclusion_err": self._conclusion_err,
        })


class ValidationFeedbackStage(PipelineStage):
    """Post-training diagnostic: check if per-env returns correlate with ep params.

    Detects environment parameters that may need to be included as policy
    context but were not identified during feature selection. This is a
    diagnostic-only stage -- it logs warnings and returns suggestions but
    does not re-run training.

    :param env_family: The environment family.
    :param correlation_alpha: p-value threshold for Pearson correlation test.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        correlation_alpha: float = 0.05,
    ) -> None:
        super().__init__(
            name="validation_feedback",
            dependencies=["policy_optimization", "feature_selection"],
        )
        self._env_family = env_family
        self._correlation_alpha = correlation_alpha

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run validation feedback.

        Correlates per-env returns from the last training iteration with
        per-env parameter values. Reports any significant correlations for
        parameters not already used as policy context.

        :returns: Dict with keys: suggested_context_params, correlations.
        """
        from scipy import stats

        po_output = inputs["policy_optimization"]
        fs_output = inputs["feature_selection"]
        context_param_names: list[str] = fs_output.get("context_param_names", [])

        # Get per-env returns from the last iteration of the first policy
        all_metrics = po_output.get("all_metrics", [])
        if not all_metrics or not all_metrics[0]:
            logger.warning("No training metrics available for validation feedback")
            return {"suggested_context_params": [], "correlations": {}}

        last_metrics = all_metrics[0][-1]
        per_env_returns = last_metrics.per_env_returns
        if per_env_returns is None or len(per_env_returns) < 3:
            logger.info(
                "Validation feedback: insufficient per-env returns "
                "(need >= 3, got {})",
                len(per_env_returns) if per_env_returns else 0,
            )
            return {"suggested_context_params": [], "correlations": {}}

        # Build per-env param values
        param_names = self._env_family.param_names
        if not param_names:
            return {"suggested_context_params": [], "correlations": {}}

        n_envs = min(len(per_env_returns), self._env_family.n_envs)
        returns_arr = np.array(per_env_returns[:n_envs])

        # Skip if returns are constant (pearsonr undefined)
        if np.std(returns_arr) < 1e-12:
            logger.info("Validation feedback: per-env returns are constant, skipping")
            return {"suggested_context_params": [], "correlations": {}}

        suggested: list[str] = []
        correlations: dict[str, dict[str, float]] = {}

        for param_name in param_names:
            ep_name = f"ep_{param_name}"

            # Skip params already used as context
            if ep_name in context_param_names:
                continue

            # Get per-env param values
            param_values = []
            for env_idx in range(n_envs):
                env_params = self._env_family.get_env_params(env_idx)
                param_values.append(env_params[param_name])

            param_arr = np.array(param_values)

            # Guard: skip constant params
            if np.std(param_arr) < 1e-12:
                continue

            corr, p_value = stats.pearsonr(param_arr, returns_arr)
            correlations[ep_name] = {
                "correlation": float(corr),
                "p_value": float(p_value),
            }

            if p_value < self._correlation_alpha:
                suggested.append(ep_name)
                logger.warning(
                    "Validation feedback: per-env returns significantly "
                    "correlate with {} (r={:.3f}, p={:.4f}) -- consider "
                    "adding as policy context",
                    ep_name,
                    corr,
                    p_value,
                )

        if not suggested:
            logger.info(
                "Validation feedback: no additional context params suggested"
            )

        return {
            "suggested_context_params": suggested,
            "correlations": correlations,
        }

    def config_hash(self) -> str:
        return hash_config({
            "correlation_alpha": self._correlation_alpha,
        })


# ---------------------------------------------------------------------------
# MPC helpers (used by AnalyticPolicyDerivationStage)
# ---------------------------------------------------------------------------


def _build_dynamics_fn(
    dynamics_expressions: dict[int, SymbolicExpression],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    angular_dims: tuple[int, ...] = (),
    calibration_coefficients: dict[int, tuple[float, float]] | None = None,
) -> Any:
    """Build a dynamics callable from symbolic expressions.

    Tries Numba JIT compilation first for faster per-call execution
    in the iLQR inner loop. Falls back to numpy lambdify if Numba
    is unavailable or compilation fails.

    :param obs_low: Lower observation bounds (clamp output). Optional.
    :param obs_high: Upper observation bounds (clamp output). Optional.
    :param angular_dims: State dimensions that are angular coordinates.
        After adding deltas, these are wrapped to ``[-pi, pi]``.
    :param calibration_coefficients: Per-dimension ``(alpha, beta)``
        calibration. When provided, each delta is transformed as
        ``delta' = alpha * delta + beta``. Used for per-environment
        coefficient calibration.
    :returns: Callable ``(state, action) -> next_state``.
    """
    # Try Numba-accelerated compilation first (no obs clamping in
    # canonical space, so fast path is suitable)
    if obs_low is None and obs_high is None:
        try:
            from circ_rl.analytic_policy.fast_dynamics import (
                build_fast_dynamics_fn,
            )

            fast_fn = build_fast_dynamics_fn(
                dynamics_expressions,
                state_names,
                action_names,
                state_dim,
                env_params,
                angular_dims=angular_dims,
            )
            if fast_fn is not None:
                logger.debug("Using Numba-accelerated dynamics function")
                return fast_fn
        except Exception:
            pass

    import sympy

    from circ_rl.hypothesis.expression import SymbolicExpression as _SE

    # Fallback: numpy lambdify
    dim_fns: dict[int, Any] = {}
    var_names = list(state_names) + list(action_names)

    for dim_idx, expr in dynamics_expressions.items():
        sympy_expr = expr.sympy_expr

        if env_params:
            subs = {sympy.Symbol(k): v for k, v in env_params.items()}
            sympy_expr = sympy_expr.subs(subs)

        compiled = _SE.from_sympy(sympy_expr)
        dim_fns[dim_idx] = compiled.to_callable(var_names)

    # Capture calibration for closure
    _cal = calibration_coefficients

    def dynamics_fn(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_state = state.copy()
        x = np.concatenate([state, action]).reshape(1, -1)
        for dim_idx, fn in dim_fns.items():
            delta = float(fn(x)[0])
            if _cal is not None and dim_idx in _cal:
                alpha, beta = _cal[dim_idx]
                delta = alpha * delta + beta
            next_state[dim_idx] += delta
        for d in angular_dims:
            next_state[d] = float(
                np.arctan2(np.sin(next_state[d]), np.cos(next_state[d]))
            )
        if obs_low is not None and obs_high is not None:
            np.clip(next_state, obs_low, obs_high, out=next_state)
        return next_state

    return dynamics_fn


def _build_reward_fn(
    reward_expression: SymbolicExpression,
    state_names: list[str],
    action_names: list[str],
    env_params: dict[str, float] | None,
    derived_feature_specs: list[Any] | None = None,
    canonical_to_obs_fn: Any | None = None,
    obs_state_names: list[str] | None = None,
) -> Any:
    """Build a reward callable from a symbolic expression.

    Tries Numba JIT compilation first for simple reward expressions
    (no derived features, no canonical-to-obs mapping). Falls back
    to numpy lambdify otherwise.

    :param derived_feature_specs: DerivedFeatureSpec list for computing
        features from the state at runtime (e.g., theta from cos/sin).
    :param canonical_to_obs_fn: If set, the callable receives canonical
        state but reward is expressed in observation space. This function
        converts canonical -> obs before evaluation.
    :param obs_state_names: Observation-space state names, required when
        ``canonical_to_obs_fn`` is set. Reward expressions and derived
        features use these names, not the canonical state names.
    :returns: Callable ``(state, action) -> float``.
    """
    # Try Numba-accelerated compilation for simple reward expressions
    if not derived_feature_specs and canonical_to_obs_fn is None:
        try:
            from circ_rl.analytic_policy.fast_dynamics import (
                build_fast_reward_fn,
            )

            fast_fn = build_fast_reward_fn(
                reward_expression,
                state_names,
                action_names,
                env_params,
            )
            if fast_fn is not None:
                logger.debug("Using Numba-accelerated reward function")
                return fast_fn
        except Exception:
            pass

    import sympy

    from circ_rl.hypothesis.expression import SymbolicExpression as _SE

    # When canonical_to_obs_fn is set, the reward expression is in
    # observation space, so variable names must match obs-space names.
    eval_state_names = (
        obs_state_names if obs_state_names is not None else state_names
    )

    sympy_expr = reward_expression.sympy_expr
    if env_params:
        subs = {sympy.Symbol(k): v for k, v in env_params.items()}
        sympy_expr = sympy_expr.subs(subs)

    var_names = list(eval_state_names) + list(action_names)
    # Add derived feature names to match the expression's variables
    if derived_feature_specs:
        var_names.extend(spec.name for spec in derived_feature_specs)
    compiled = _SE.from_sympy(sympy_expr)
    fn = compiled.to_callable(var_names)

    if derived_feature_specs:
        from circ_rl.hypothesis.derived_features import (
            compute_derived_single,
        )

        def reward_fn(state: np.ndarray, action: np.ndarray) -> float:
            obs_state = (
                canonical_to_obs_fn(state)
                if canonical_to_obs_fn is not None
                else state
            )
            derived = compute_derived_single(
                derived_feature_specs, obs_state, eval_state_names,
            )
            derived_vals = [
                derived[spec.name] for spec in derived_feature_specs
            ]
            x = np.concatenate(
                [obs_state, action, derived_vals],
            ).reshape(1, -1)
            return float(fn(x)[0])
    else:
        def reward_fn(state: np.ndarray, action: np.ndarray) -> float:
            obs_state = (
                canonical_to_obs_fn(state)
                if canonical_to_obs_fn is not None
                else state
            )
            x = np.concatenate([obs_state, action]).reshape(1, -1)
            return float(fn(x)[0])

    return reward_fn


def _default_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Default cost: negative squared state + action norm."""
    return -float(np.sum(state ** 2) + 0.01 * np.sum(action ** 2))


def _build_dynamics_jacobian_fns(
    dynamics_expressions: dict[int, SymbolicExpression],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    calibration_coefficients: dict[int, tuple[float, float]] | None = None,
) -> tuple[Any, Any]:
    r"""Build analytic Jacobian callables from symbolic expressions.

    Tries Numba JIT compilation first, falls back to numpy lambdify.

    Uses ``sympy.diff()`` on each dynamics expression to compute
    exact partial derivatives, then compiles them via ``lambdify``.

    The full dynamics are :math:`x_{t+1} = x_t + \Delta x_t`, so:

    - :math:`A = I + [\partial \Delta x_i / \partial x_j]_{i,j}`
    - :math:`B = [\partial \Delta x_i / \partial u_j]_{i,j}`

    :param dynamics_expressions: Per-dim symbolic delta expressions.
    :param state_names: State variable names.
    :param action_names: Action variable names.
    :param state_dim: State dimensionality.
    :param env_params: Environment parameters to substitute.
    :param obs_low: Observation lower bounds (unused, for API compat).
    :param obs_high: Observation upper bounds (unused, for API compat).
    :returns: Tuple ``(jac_state_fn, jac_action_fn)`` where each is
        a callable ``(state, action) -> matrix``.
    """
    # Try Numba-accelerated Jacobians first
    try:
        from circ_rl.analytic_policy.fast_dynamics import (
            build_fast_jacobian_fns,
        )

        fast_jac_s, fast_jac_a = build_fast_jacobian_fns(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
        )
        if fast_jac_s is not None and fast_jac_a is not None:
            logger.debug("Using Numba-accelerated Jacobian functions")
            return fast_jac_s, fast_jac_a
    except Exception:
        pass

    import sympy as sp

    all_var_names = list(state_names) + list(action_names)
    all_symbols = [sp.Symbol(n) for n in all_var_names]

    # Pre-compute substitution dict for env params
    param_subs: dict[sp.Symbol, float] = {}
    if env_params:
        param_subs = {sp.Symbol(k): v for k, v in env_params.items()}

    # Build symbolic Jacobian entries and compile to callables
    # A_entries[i][j] = compiled callable for d(delta_x_i)/d(x_j)
    # B_entries[i][j] = compiled callable for d(delta_x_i)/d(u_j)
    action_dim = len(action_names)
    state_syms = [sp.Symbol(n) for n in state_names]
    action_syms = [sp.Symbol(n) for n in action_names]

    # For each dim with an expression, differentiate symbolically
    a_fns: dict[tuple[int, int], Any] = {}
    b_fns: dict[tuple[int, int], Any] = {}

    for dim_idx, expr in dynamics_expressions.items():
        sympy_expr = expr.sympy_expr
        if param_subs:
            sympy_expr = sympy_expr.subs(param_subs)

        # Replace Abs with a smooth approximation before differentiating.
        # Abs(x) -> sqrt(x^2 + eps) avoids non-differentiable Derivative
        # nodes that sp.lambdify cannot compile.
        # NOTE: do NOT rewrite(Piecewise) first -- that converts Abs to
        # Piecewise which is no longer matched by .replace(sp.Abs, ...).
        _eps = sp.Rational(1, 10**10)
        sympy_expr = sympy_expr.replace(
            sp.Abs, lambda x: sp.sqrt(x**2 + _eps),  # type: ignore[arg-type]
        )

        # Derivatives w.r.t. state variables
        for j, s_sym in enumerate(state_syms):
            deriv = sp.diff(sympy_expr, s_sym)
            deriv_simplified = sp.simplify(deriv)
            if deriv_simplified != 0:
                try:
                    a_fns[(dim_idx, j)] = sp.lambdify(
                        all_symbols, deriv_simplified, modules=["numpy"],
                    )
                except Exception:
                    logger.warning(
                        "Cannot compile d(delta_x{})/d(x{}) -- "
                        "falling back to finite differences",
                        dim_idx, j,
                    )
                    return None, None

        # Derivatives w.r.t. action variables
        for j, a_sym in enumerate(action_syms):
            deriv = sp.diff(sympy_expr, a_sym)
            deriv_simplified = sp.simplify(deriv)
            if deriv_simplified != 0:
                try:
                    b_fns[(dim_idx, j)] = sp.lambdify(
                        all_symbols, deriv_simplified, modules=["numpy"],
                    )
                except Exception:
                    logger.warning(
                        "Cannot compile d(delta_x{})/d(u{}) -- "
                        "falling back to finite differences",
                        dim_idx, j,
                    )
                    return None, None

    # Capture calibration for Jacobian scaling
    _cal = calibration_coefficients

    def jac_state_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        """Evaluate df/dx at (state, action). Shape (n, n)."""
        a_mat = np.eye(state_dim)  # Identity for x_{t+1} = x_t + delta
        args = [*state, *action]
        for (di, dj), fn in a_fns.items():
            val = float(fn(*args))
            # d(alpha * delta + beta)/dx = alpha * d(delta)/dx
            if _cal is not None and di in _cal:
                val *= _cal[di][0]
            a_mat[di, dj] += val
        return a_mat

    def jac_action_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        """Evaluate df/du at (state, action). Shape (n, m)."""
        b_mat = np.zeros((state_dim, action_dim))
        args = [*state, *action]
        for (di, dj), fn in b_fns.items():
            val = float(fn(*args))
            # d(alpha * delta + beta)/du = alpha * d(delta)/du
            if _cal is not None and di in _cal:
                val *= _cal[di][0]
            b_mat[di, dj] = val
        return b_mat

    return jac_state_fn, jac_action_fn


def _build_reward_derivative_fns(
    reward_expression: SymbolicExpression,
    canonical_state_names: list[str],
    action_names: list[str],
    env_params: dict[str, float] | None,
    derived_feature_specs: list[Any] | None = None,
    obs_state_names: list[str] | None = None,
) -> (
    Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
    | None
):
    r"""Build analytic reward derivative callables from a symbolic expression.

    Uses ``sympy.diff()`` on the reward expression to compute exact
    first and second derivatives, then compiles them via ``lambdify``.

    When the reward is expressed in observation space with derived
    features (e.g., ``theta = atan2(sin, cos)``), composes the
    canonical-to-obs mapping symbolically before differentiating.

    :param reward_expression: Symbolic reward expression.
    :param canonical_state_names: Canonical state variable names
        (what iLQR uses internally).
    :param action_names: Action variable names.
    :param env_params: Environment parameters to substitute.
    :param derived_feature_specs: DerivedFeatureSpec list. Each spec
        with a ``sympy_fn`` attribute enables symbolic substitution.
    :param obs_state_names: Observation-space state names. When
        provided (and different from canonical), the reward is
        assumed to be in obs space and requires symbolic composition.
    :returns: Callable ``(state, action) -> (r_x, r_u, r_xx, r_uu, r_ux)``
        or ``None`` if symbolic composition is not possible.
    """
    import sympy as sp

    from circ_rl.hypothesis.expression import SymbolicExpression as _SE

    sympy_expr = reward_expression.sympy_expr

    # Substitute environment parameters
    if env_params:
        subs = {sp.Symbol(k): v for k, v in env_params.items()}
        sympy_expr = sympy_expr.subs(subs)

    # Step 1: Substitute derived features with their symbolic definitions
    if derived_feature_specs:
        for spec in derived_feature_specs:
            if spec.sympy_fn is None:
                logger.debug(
                    "Derived feature '{}' has no sympy_fn, "
                    "cannot build analytic reward derivatives",
                    spec.name,
                )
                return None
            source_syms = [sp.Symbol(n) for n in spec.source_names]
            sympy_expr = sympy_expr.subs(
                sp.Symbol(spec.name), spec.sympy_fn(*source_syms),
            )

    # Step 2: If reward uses obs-space names, substitute with canonical
    # equivalents. For circle constraints: s0 -> cos(phi_0), s1 -> sin(phi_0)
    # For unconstrained dims: obs_name -> canonical_name (rename)
    if obs_state_names is not None and obs_state_names != canonical_state_names:
        n_obs = len(obs_state_names)
        n_can = len(canonical_state_names)

        if n_can < n_obs:
            # Likely angular dims: canonical has fewer dims than obs.
            # For each pair of obs dims that map to one canonical dim
            # (circle constraint), substitute obs = cos/sin of canonical.
            # Heuristic: first (n_obs - n_can) canonical dims are angular,
            # each consuming 2 obs dims.
            n_angular = n_obs - n_can
            obs_subs: dict[sp.Symbol, sp.Expr] = {}

            for a_idx in range(n_angular):
                can_sym = sp.Symbol(canonical_state_names[a_idx])
                obs_cos_name = obs_state_names[2 * a_idx]
                obs_sin_name = obs_state_names[2 * a_idx + 1]
                obs_subs[sp.Symbol(obs_cos_name)] = sp.cos(can_sym)
                obs_subs[sp.Symbol(obs_sin_name)] = sp.sin(can_sym)

            # Unconstrained dims: simple rename
            obs_offset = 2 * n_angular
            can_offset = n_angular
            for i in range(n_obs - obs_offset):
                obs_name = obs_state_names[obs_offset + i]
                can_name = canonical_state_names[can_offset + i]
                if obs_name != can_name:
                    obs_subs[sp.Symbol(obs_name)] = sp.Symbol(can_name)

            sympy_expr = sympy_expr.subs(obs_subs)
        else:
            # Same dimension count, just rename
            for obs_n, can_n in zip(
                obs_state_names, canonical_state_names, strict=True,
            ):
                if obs_n != can_n:
                    sympy_expr = sympy_expr.subs(
                        sp.Symbol(obs_n), sp.Symbol(can_n),
                    )

    # Simplify the composed expression
    sympy_expr = sp.simplify(sympy_expr)

    # Build the variable list in canonical space
    all_var_names = list(canonical_state_names) + list(action_names)
    all_symbols = [sp.Symbol(n) for n in all_var_names]
    state_dim = len(canonical_state_names)
    action_dim = len(action_names)

    # Compute symbolic derivatives
    # First derivatives (gradient)
    grad_state_exprs = [sp.diff(sympy_expr, sp.Symbol(n)) for n in canonical_state_names]
    grad_action_exprs = [sp.diff(sympy_expr, sp.Symbol(n)) for n in action_names]

    # Second derivatives (Hessians)
    hess_xx_exprs = [
        [sp.diff(sympy_expr, sp.Symbol(si), sp.Symbol(sj))
         for sj in canonical_state_names]
        for si in canonical_state_names
    ]
    hess_uu_exprs = [
        [sp.diff(sympy_expr, sp.Symbol(ai), sp.Symbol(aj))
         for aj in action_names]
        for ai in action_names
    ]
    hess_ux_exprs = [
        [sp.diff(sympy_expr, sp.Symbol(ai), sp.Symbol(sj))
         for sj in canonical_state_names]
        for ai in action_names
    ]

    # Compile all derivatives via lambdify
    grad_s_fns = [
        sp.lambdify(all_symbols, sp.simplify(e), modules=["numpy"])
        for e in grad_state_exprs
    ]
    grad_a_fns = [
        sp.lambdify(all_symbols, sp.simplify(e), modules=["numpy"])
        for e in grad_action_exprs
    ]
    hess_xx_fns = [
        [sp.lambdify(all_symbols, sp.simplify(e), modules=["numpy"])
         for e in row]
        for row in hess_xx_exprs
    ]
    hess_uu_fns = [
        [sp.lambdify(all_symbols, sp.simplify(e), modules=["numpy"])
         for e in row]
        for row in hess_uu_exprs
    ]
    hess_ux_fns = [
        [sp.lambdify(all_symbols, sp.simplify(e), modules=["numpy"])
         for e in row]
        for row in hess_ux_exprs
    ]

    logger.debug(
        "Built analytic reward derivatives for expression: {}",
        sp.simplify(sympy_expr),
    )

    def reward_derivatives_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        args = [*state, *action]

        r_x = np.array([float(fn(*args)) for fn in grad_s_fns])
        r_u = np.array([float(fn(*args)) for fn in grad_a_fns])

        r_xx = np.array([
            [float(fn(*args)) for fn in row]
            for row in hess_xx_fns
        ])  # (state_dim, state_dim)
        r_uu = np.array([
            [float(fn(*args)) for fn in row]
            for row in hess_uu_fns
        ])  # (action_dim, action_dim)
        r_ux = np.array([
            [float(fn(*args)) for fn in row]
            for row in hess_ux_fns
        ])  # (action_dim, state_dim)

        return r_x, r_u, r_xx, r_uu, r_ux

    return reward_derivatives_fn


def _compute_predicted_returns(
    analytic_policy: Any,
    dynamics_expressions: dict[int, SymbolicExpression],
    reward_expression: SymbolicExpression | None,
    state_names: list[str],
    action_names: list[str],
    env_family: EnvironmentFamily,
    test_env_ids: list[int],
    n_episodes: int = 5,
    max_steps: int = 200,
    reward_derived_features: list[Any] | None = None,
) -> dict[int, float]:
    """Compute predicted returns by simulating the policy in the learned model.

    For each test environment, runs episodes using the analytic policy with
    the learned dynamics and reward models, starting from real initial states
    (obtained by resetting the actual environment). The total undiscounted
    return per episode is averaged across episodes.

    :param analytic_policy: The analytic policy (LQR or MPC).
    :param dynamics_expressions: Per-dim symbolic dynamics expressions.
    :param reward_expression: Reward symbolic expression (or None).
    :param state_names: State feature names.
    :param action_names: Action feature names.
    :param env_family: Environment family.
    :param test_env_ids: Environment IDs to simulate.
    :param n_episodes: Number of episodes per environment.
    :param max_steps: Maximum steps per episode.
    :param reward_derived_features: Derived feature specs for reward.
    :returns: Dict mapping env_id -> mean predicted return.
    """
    state_dim = len(state_names)
    predicted: dict[int, float] = {}

    for env_id in test_env_ids:
        env_params = env_family.get_env_params(env_id)

        env = env_family.make_env(env_id)
        obs_low = np.asarray(
            env.observation_space.low,  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        obs_high = np.asarray(
            env.observation_space.high,  # type: ignore[attr-defined]
            dtype=np.float64,
        )

        dynamics_fn = _build_dynamics_fn(
            dynamics_expressions, state_names, action_names,
            state_dim, env_params,
            obs_low=obs_low, obs_high=obs_high,
        )

        reward_fn = None
        if reward_expression is not None:
            reward_fn = _build_reward_fn(
                reward_expression, state_names, action_names,
                env_params, reward_derived_features,
            )

        episode_returns: list[float] = []

        for ep in range(n_episodes):
            # Use same seed pattern as ConclusionTest for matched comparisons
            obs, _ = env.reset(seed=env_id * 100 + ep)
            state = np.asarray(obs, dtype=np.float64)
            total_reward = 0.0

            for _ in range(max_steps):
                action = analytic_policy.get_action(
                    np.asarray(state, dtype=np.float32), env_id,
                )

                # Predicted reward from the reward model
                if reward_fn is not None:
                    r = reward_fn(state, action)
                else:
                    r = 0.0
                total_reward += r

                # Predicted next state from the dynamics model
                # (clamped to obs bounds inside dynamics_fn)
                next_state = dynamics_fn(state, action)

                # Safety: check for non-finite values
                if not np.all(np.isfinite(next_state)):
                    logger.warning(
                        "Predicted state diverged at env={}, step; "
                        "terminating episode early",
                        env_id,
                    )
                    break
                state = next_state

            episode_returns.append(total_reward)

        env.close()
        predicted[env_id] = float(np.mean(episode_returns))

        logger.debug(
            "Predicted return for env {}: {:.2f} (over {} episodes)",
            env_id, predicted[env_id], n_episodes,
        )

    return predicted


class _ILQRAnalyticPolicy:
    """Stateful policy wrapper for per-env iLQR solvers.

    Plans a full trajectory on first call per env (or when the
    step counter exceeds the horizon), then executes using the
    time-varying feedback gains for closed-loop correction:

        u_t = u*_t + K_t @ (x_t - x*_t)

    Call ``reset(env_idx)`` before each new episode to clear
    cached plans.
    """

    def __init__(
        self,
        dynamics_hypothesis: HypothesisEntry,
        reward_hypothesis: HypothesisEntry | None,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        ilqr_solvers: dict[int, ILQRSolver],
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        obs_to_canonical_fn: Any | None = None,
        robust_planners: dict[int, RobustILQRPlanner] | None = None,
    ) -> None:
        self._dynamics_hypothesis = dynamics_hypothesis
        self._reward_hypothesis = reward_hypothesis
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._n_envs = n_envs
        self._ilqr_solvers = ilqr_solvers
        self._action_low = action_low
        self._action_high = action_high
        self._obs_to_canonical_fn = obs_to_canonical_fn
        self._robust_planners = robust_planners
        self.solver_type = "ilqr"

        # Stateful: cached plans and step counters per env
        self._solutions: dict[int, ILQRSolution] = {}
        self._steps: dict[int, int] = {}

        # Adaptive replanning diagnostics
        self._prediction_errors: dict[int, list[float]] = {}
        self._replan_counts: dict[int, int] = {}

        # Override flag: when True, adaptive replanning is disabled
        # regardless of solver config. Used by DiagnosticValidationStage.
        self._force_disable_adaptive_replan: bool = False

    @property
    def n_free_parameters(self) -> int:
        """iLQR has no learned parameters."""
        return 0

    @property
    def complexity(self) -> int:
        """Symbolic complexity of the underlying hypothesis."""
        from circ_rl.hypothesis.expression import SymbolicExpression

        expr = self._dynamics_hypothesis.expression
        if isinstance(expr, SymbolicExpression):
            return expr.complexity
        return self._dynamics_hypothesis.complexity

    def get_action(
        self,
        state: np.ndarray,
        env_idx: int,
    ) -> np.ndarray:
        """Compute the optimal action via iLQR trajectory + feedback.

        On first call for a given env_idx (or after reset), plans
        a full trajectory. Subsequent calls use the cached plan with
        feedback correction.

        When ``obs_to_canonical_fn`` is set, the input ``state`` is
        an observation-space vector which is converted to canonical
        coordinates before planning and feedback computation.

        :param state: Current state in observation space,
            shape ``(obs_dim,)``.
        :param env_idx: Environment index.
        :returns: Optimal action, shape ``(action_dim,)``.
        """
        # Convert obs -> canonical if needed
        if self._obs_to_canonical_fn is not None:
            canonical_state = self._obs_to_canonical_fn(
                np.asarray(state, dtype=np.float64),
            )
        else:
            canonical_state = np.asarray(state, dtype=np.float64)

        solver = self._ilqr_solvers.get(env_idx)
        if solver is None:
            solver = next(iter(self._ilqr_solvers.values()))

        replan_interval = solver.config.replan_interval
        if replan_interval is None:
            replan_interval = solver.config.horizon

        tau = (
            None if self._force_disable_adaptive_replan
            else solver.config.adaptive_replan_threshold
        )
        min_interval = solver.config.min_replan_interval

        # Plan if needed: first call
        needs_plan = env_idx not in self._solutions

        if not needs_plan:
            t = self._steps[env_idx]

            # Check max interval (existing fixed-interval logic)
            if t >= replan_interval:
                needs_plan = True

            # Check adaptive threshold (only if above min interval)
            elif tau is not None and t >= min_interval:
                sol = self._solutions[env_idx]
                if t < len(sol.nominal_states):
                    predicted_state = sol.nominal_states[t]  # (state_dim,)
                    deviation = float(
                        np.linalg.norm(canonical_state - predicted_state)
                    )
                    # Track for diagnostics
                    if env_idx not in self._prediction_errors:
                        self._prediction_errors[env_idx] = []
                    self._prediction_errors[env_idx].append(deviation)
                    if deviation > tau:
                        needs_plan = True
                        logger.debug(
                            "Adaptive replan triggered for env {}: "
                            "deviation={:.4f} > tau={:.4f} at step {}",
                            env_idx, deviation, tau, t,
                        )

        if needs_plan:
            self._replan_counts[env_idx] = (
                self._replan_counts.get(env_idx, 0) + 1
            )
            # Warm-start from tail of previous plan when replanning
            warm_start = None
            if env_idx in self._solutions:
                prev_sol = self._solutions[env_idx]
                t_prev = self._steps[env_idx]
                remaining = prev_sol.nominal_actions[t_prev:]
                pad = np.zeros((t_prev, self._action_dim))
                warm_start = np.vstack([remaining, pad])

            # Use robust planner if available for this env
            robust_planner = (
                self._robust_planners.get(env_idx)
                if self._robust_planners is not None
                else None
            )
            if robust_planner is not None:
                sol = robust_planner.plan(
                    canonical_state, self._action_dim, warm_start,
                )
            else:
                sol = solver.plan(
                    canonical_state, self._action_dim, warm_start,
                )
            self._solutions[env_idx] = sol
            self._steps[env_idx] = 0

        sol = self._solutions[env_idx]
        t = self._steps[env_idx]

        # Closed-loop: nominal + feedback correction
        dx = canonical_state - sol.nominal_states[t]  # (state_dim,)
        action = sol.nominal_actions[t] + sol.feedback_gains[t] @ dx

        # Clip to action bounds
        if self._action_low is not None and self._action_high is not None:
            action = np.clip(action, self._action_low, self._action_high)

        self._steps[env_idx] = t + 1
        return action

    @property
    def replan_stats(self) -> dict[int, dict[str, float]]:
        """Per-env replanning statistics for diagnostics.

        :returns: Dict mapping env_idx to stats dict with keys:
            total_replans, mean_prediction_error, max_prediction_error.
        """
        stats: dict[int, dict[str, float]] = {}
        for env_idx in set(
            list(self._replan_counts.keys())
            + list(self._prediction_errors.keys())
        ):
            errors = self._prediction_errors.get(env_idx, [])
            stats[env_idx] = {
                "total_replans": float(
                    self._replan_counts.get(env_idx, 0),
                ),
                "mean_prediction_error": (
                    float(np.mean(errors)) if errors else 0.0
                ),
                "max_prediction_error": (
                    float(np.max(errors)) if errors else 0.0
                ),
            }
        return stats

    def reset(self, env_idx: int | None = None) -> None:
        """Clear cached plan for re-planning.

        :param env_idx: Specific env to reset, or None to reset all.
        """
        if env_idx is None:
            self._solutions.clear()
            self._steps.clear()
            self._prediction_errors.clear()
            self._replan_counts.clear()
        else:
            self._solutions.pop(env_idx, None)
            self._steps.pop(env_idx, None)
            self._prediction_errors.pop(env_idx, None)
            self._replan_counts.pop(env_idx, None)
