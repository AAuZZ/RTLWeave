from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_REPOS = frozenset({"ibex", "cva6", "caliptra-rtl"})
VISIBLE_CASE_KEYS = frozenset(
    {
        "instance_id",
        "repo",
        "pr_number",
        "base_commit",
        "title",
        "problem_statement",
        "repo_path",
        "docker_image",
    }
)
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "tb_script",
        "test_patch",
        "fix_patch",
        "patch",
        "modified_files",
        "f2p_tests",
        "p2p_tests",
        "n2p_tests",
        "s2p_tests",
        "fixed_tests",
        "reproducer_path",
        "reproducer_signal",
        "official_test_output",
        "candidate_patch",
        "previous_patch",
    }
)


class MiniTBError(RuntimeError):
    """Base class for deterministic experiment failures."""


class InputBoundaryError(MiniTBError):
    pass


class SpecValidationError(MiniTBError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class HarnessValidationError(MiniTBError):
    pass


class BudgetExhausted(MiniTBError):
    pass


@dataclass
class RetryBudget:
    total_limit: int = 7
    limits: dict[str, int] = field(
        default_factory=lambda: {"behavior_spec": 4, "harness_fill": 2, "reproduction": 1}
    )
    used: dict[str, int] = field(default_factory=dict)

    def consume(self, stage: str) -> None:
        total_used = sum(self.used.values())
        if total_used >= self.total_limit:
            raise BudgetExhausted(f"total LLM call budget exhausted ({self.total_limit})")
        stage_limit = self.limits.get(stage, 0)
        stage_used = self.used.get(stage, 0)
        if stage_used >= stage_limit:
            raise BudgetExhausted(f"{stage} LLM call budget exhausted ({stage_limit})")
        self.used[stage] = stage_used + 1

    @property
    def total_used(self) -> int:
        return sum(self.used.values())


@dataclass(frozen=True)
class VisibleCase:
    instance_id: str
    repo: str
    pr_number: int
    base_commit: str
    title: str
    problem_statement: str
    repo_path: str
    docker_image: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VisibleCase":
        keys = set(value)
        if keys != VISIBLE_CASE_KEYS:
            extra = sorted(keys - VISIBLE_CASE_KEYS)
            missing = sorted(VISIBLE_CASE_KEYS - keys)
            raise InputBoundaryError(
                f"visible case keys must exactly match the whitelist; extra={extra}, missing={missing}"
            )
        _reject_forbidden_recursive(value)
        if value["repo"] not in SUPPORTED_REPOS:
            raise InputBoundaryError(
                f"unsupported repository subset {value['repo']!r}; expected one of {sorted(SUPPORTED_REPOS)}"
            )
        if not str(value["instance_id"]).strip() or not str(value["problem_statement"]).strip():
            raise InputBoundaryError("instance_id and problem_statement are required")
        try:
            pr_number = int(value["pr_number"])
        except (TypeError, ValueError) as exc:
            raise InputBoundaryError("pr_number must be an integer") from exc
        if pr_number <= 0:
            raise InputBoundaryError("pr_number must be positive")
        return cls(
            instance_id=str(value["instance_id"]),
            repo=str(value["repo"]),
            pr_number=pr_number,
            base_commit=str(value["base_commit"]),
            title=str(value["title"]),
            problem_statement=str(value["problem_statement"]),
            repo_path=str(value["repo_path"]),
            docker_image=str(value["docker_image"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PortInfo:
    name: str
    direction: str
    width: str = ""
    declaration: str = ""

    @property
    def is_input(self) -> bool:
        return self.direction == "input"

    @property
    def is_output(self) -> bool:
        return self.direction == "output"


@dataclass
class TargetContext:
    file: str
    module: str
    source: str
    ports: list[PortInfo]
    parameters: dict[str, str] = field(default_factory=dict)
    imports: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    rank: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return data


@dataclass(frozen=True)
class RunResult:
    phase: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Marker:
    kind: str
    fields: dict[str, str]
    raw: str


@dataclass
class QualificationResult:
    status: str
    reason: str
    build: RunResult | None = None
    runs: list[RunResult] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    repeatable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "build": self.build.to_dict() if self.build else None,
            "runs": [run.to_dict() for run in self.runs],
            "markers": [dataclasses.asdict(marker) for marker in self.markers],
            "repeatable": self.repeatable,
        }


def _reject_forbidden_recursive(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_INPUT_KEYS:
                raise InputBoundaryError(f"forbidden input field at {path}.{key}")
            _reject_forbidden_recursive(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_recursive(child, f"{path}[{index}]")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
