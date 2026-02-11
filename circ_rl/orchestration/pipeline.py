"""DAG-based pipeline orchestration for the CIRC-RL framework.

Chains Phases 1-4 with content-addressed caching and partial re-runs.
Each stage declares its dependencies and produces artifacts that downstream
stages consume.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class PipelineStage(ABC):
    """Abstract base class for a pipeline stage.

    :param name: Unique name for this stage.
    :param dependencies: Names of stages that must run before this one.
    """

    def __init__(self, name: str, dependencies: list[str] | None = None) -> None:
        self._name = name
        self._dependencies = list(dependencies) if dependencies else []

    @property
    def name(self) -> str:
        """Stage name."""
        return self._name

    @property
    def dependencies(self) -> list[str]:
        """Names of stages this stage depends on."""
        return list(self._dependencies)

    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage.

        :param inputs: Dictionary of artifacts from dependency stages.
        :returns: Dictionary of output artifacts.
        """

    @abstractmethod
    def config_hash(self) -> str:
        """Return a hash of the stage configuration for cache invalidation.

        :returns: Hex digest string.
        """


class CIRCPipeline:
    """DAG-based pipeline executor for CIRC-RL.

    Executes stages in topological order, caching results to disk.
    On re-run, stages are skipped if their configuration hash matches
    the cached version.

    :param stages: List of pipeline stages.
    :param cache_dir: Directory for caching stage artifacts.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        cache_dir: Path | str = ".circ_cache",
    ) -> None:
        self._stages = {s.name: s for s in stages}
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._execution_order = self._topological_sort()

    @property
    def stage_names(self) -> list[str]:
        """Names of all stages in execution order."""
        return list(self._execution_order)

    def run(
        self,
        force_from: str | None = None,
    ) -> dict[str, Any]:
        """Execute the pipeline.

        :param force_from: If specified, invalidate cache for this stage
            and all downstream stages, forcing re-execution.
        :returns: Dictionary of all stage artifacts.
        """
        all_artifacts: dict[str, Any] = {}

        invalidated = set()
        if force_from is not None:
            invalidated = self._downstream_stages(force_from)
            invalidated.add(force_from)
            logger.info(
                "Forcing re-run from '{}': invalidating {}",
                force_from,
                sorted(invalidated),
            )

        for stage_name in self._execution_order:
            stage = self._stages[stage_name]

            # Gather inputs from dependencies
            inputs = {}
            for dep in stage.dependencies:
                if dep in all_artifacts:
                    inputs[dep] = all_artifacts[dep]

            # Check cache
            if stage_name not in invalidated:
                cached = self._load_cache(stage_name, stage.config_hash())
                if cached is not None:
                    logger.info("Stage '{}': cache hit, skipping", stage_name)
                    all_artifacts[stage_name] = cached
                    continue

            logger.info("Stage '{}': running", stage_name)
            artifacts = stage.run(inputs)
            all_artifacts[stage_name] = artifacts

            self._save_cache(stage_name, stage.config_hash(), artifacts)
            logger.info("Stage '{}': completed and cached", stage_name)

        return all_artifacts

    def _topological_sort(self) -> list[str]:
        """Compute execution order via topological sort.

        :returns: List of stage names in dependency order.
        :raises ValueError: If dependencies form a cycle or reference unknown stages.
        """
        # Kahn's algorithm
        in_degree: dict[str, int] = {name: 0 for name in self._stages}
        adjacency: dict[str, list[str]] = {name: [] for name in self._stages}

        for name, stage in self._stages.items():
            for dep in stage.dependencies:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{name}' depends on unknown stage '{dep}'"
                    )
                adjacency[dep].append(name)
                in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            queue.sort()  # Deterministic ordering
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._stages):
            raise ValueError("Pipeline has cyclic dependencies")

        return order

    def _downstream_stages(self, stage_name: str) -> set[str]:
        """Find all stages that transitively depend on a given stage."""
        downstream: set[str] = set()
        queue = [stage_name]
        while queue:
            current = queue.pop(0)
            for name, stage in self._stages.items():
                if current in stage.dependencies and name not in downstream:
                    downstream.add(name)
                    queue.append(name)
        return downstream

    def _cache_path(self, stage_name: str, config_hash: str) -> Path:
        """Return the cache file path for a stage."""
        return self._cache_dir / f"{stage_name}_{config_hash}.pkl"

    def _load_cache(self, stage_name: str, config_hash: str) -> Any | None:
        """Load cached artifacts if they exist."""
        path = self._cache_path(stage_name, config_hash)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, stage_name: str, config_hash: str, artifacts: Any) -> None:
        """Save stage artifacts to cache."""
        # Remove old caches for this stage
        for old_cache in self._cache_dir.glob(f"{stage_name}_*.pkl"):
            old_cache.unlink()

        path = self._cache_path(stage_name, config_hash)
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)


def hash_config(config: dict[str, Any]) -> str:
    """Compute a deterministic hash of a configuration dictionary.

    :param config: Configuration to hash.
    :returns: Hex digest string.
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
