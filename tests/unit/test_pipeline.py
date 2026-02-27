"""Tests for circ_rl.orchestration.pipeline."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from circ_rl.orchestration.pipeline import CIRCPipeline, PipelineStage, hash_config


class DummyStage(PipelineStage):
    """A simple stage that records execution and returns its name."""

    def __init__(
        self, name: str, dependencies: list[str] | None = None, value: str = ""
    ) -> None:
        super().__init__(name, dependencies)
        self._value = value or name
        self.executed = False

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.executed = True
        return {"result": self._value, "inputs": list(inputs.keys())}

    def config_hash(self) -> str:
        return hash_config({"name": self._name, "value": self._value})


class TestPipelineTopologicalSort:
    def test_linear_chain(self) -> None:
        stages = [
            DummyStage("C", dependencies=["B"]),
            DummyStage("A"),
            DummyStage("B", dependencies=["A"]),
        ]
        pipeline = CIRCPipeline(stages, cache_dir=tempfile.mkdtemp())
        assert pipeline.stage_names == ["A", "B", "C"]

    def test_diamond(self) -> None:
        stages = [
            DummyStage("A"),
            DummyStage("B", dependencies=["A"]),
            DummyStage("C", dependencies=["A"]),
            DummyStage("D", dependencies=["B", "C"]),
        ]
        pipeline = CIRCPipeline(stages, cache_dir=tempfile.mkdtemp())
        names = pipeline.stage_names
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("D")
        assert names.index("C") < names.index("D")

    def test_unknown_dependency_raises(self) -> None:
        stages = [DummyStage("A", dependencies=["UNKNOWN"])]
        with pytest.raises(ValueError, match="unknown stage"):
            CIRCPipeline(stages, cache_dir=tempfile.mkdtemp())

    def test_cyclic_dependency_raises(self) -> None:
        stages = [
            DummyStage("A", dependencies=["B"]),
            DummyStage("B", dependencies=["A"]),
        ]
        with pytest.raises(ValueError, match="cyclic"):
            CIRCPipeline(stages, cache_dir=tempfile.mkdtemp())


class TestPipelineExecution:
    def test_all_stages_execute(self) -> None:
        stages = [
            DummyStage("A"),
            DummyStage("B", dependencies=["A"]),
        ]
        with tempfile.TemporaryDirectory() as td:
            pipeline = CIRCPipeline(stages, cache_dir=td)
            results = pipeline.run()
            assert "A" in results
            assert "B" in results
            assert results["A"]["result"] == "A"
            assert results["B"]["inputs"] == ["A"]

    def test_cache_hit_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # First run
            stages1 = [DummyStage("A"), DummyStage("B", dependencies=["A"])]
            pipeline1 = CIRCPipeline(stages1, cache_dir=td)
            pipeline1.run()
            assert stages1[0].executed
            assert stages1[1].executed

            # Second run with same config
            stages2 = [DummyStage("A"), DummyStage("B", dependencies=["A"])]
            pipeline2 = CIRCPipeline(stages2, cache_dir=td)
            pipeline2.run()
            # Should be cached
            assert not stages2[0].executed
            assert not stages2[1].executed

    def test_config_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # First run
            stages1 = [DummyStage("A", value="v1")]
            pipeline1 = CIRCPipeline(stages1, cache_dir=td)
            pipeline1.run()
            assert stages1[0].executed

            # Second run with different value
            stages2 = [DummyStage("A", value="v2")]
            pipeline2 = CIRCPipeline(stages2, cache_dir=td)
            pipeline2.run()
            # Should re-execute due to different config hash
            assert stages2[0].executed

    def test_force_from_invalidates_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # First run
            stages1 = [
                DummyStage("A"),
                DummyStage("B", dependencies=["A"]),
                DummyStage("C", dependencies=["B"]),
            ]
            pipeline1 = CIRCPipeline(stages1, cache_dir=td)
            pipeline1.run()

            # Force from B
            stages2 = [
                DummyStage("A"),
                DummyStage("B", dependencies=["A"]),
                DummyStage("C", dependencies=["B"]),
            ]
            pipeline2 = CIRCPipeline(stages2, cache_dir=td)
            pipeline2.run(force_from="B")

            # A should be cached, B and C should re-execute
            assert not stages2[0].executed
            assert stages2[1].executed
            assert stages2[2].executed


class TestHashConfig:
    def test_deterministic(self) -> None:
        config = {"a": 1, "b": "hello"}
        assert hash_config(config) == hash_config(config)

    def test_order_independent(self) -> None:
        assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})

    def test_different_configs_different_hashes(self) -> None:
        assert hash_config({"a": 1}) != hash_config({"a": 2})
