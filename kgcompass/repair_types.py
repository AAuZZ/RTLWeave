from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepairPlan:
    hypothesis: str = ""
    edit_cone: List[Dict[str, Any]] = field(default_factory=list)
    allowed_files: List[str] = field(default_factory=list)
    forbidden_files: List[str] = field(default_factory=list)
    validator_profile: Dict[str, Any] = field(default_factory=dict)
    context_budget: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatchDelta:
    edited_files: List[str] = field(default_factory=list)
    changed_spans: List[Dict[str, Any]] = field(default_factory=list)
    touched_symbols: List[str] = field(default_factory=list)
    touched_modules: List[str] = field(default_factory=list)
    raw_diff: str = ""
    policy_violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(v.get("severity") == "blocker" for v in self.policy_violations)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationEvidence:
    type: str
    severity: str
    message: str
    file: str = ""
    line: Optional[int] = None
    module: str = ""
    signal: str = ""
    check_id: str = ""
    expected: str = ""
    observed: str = ""
    cycle: Optional[int] = None
    stimulus: Dict[str, Any] = field(default_factory=dict)
    raw_log: str = ""
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    validator: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    passed: bool
    evidence: List[ValidationEvidence] = field(default_factory=list)
    feedback_text: str = ""
    blocker_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


def evidence_from_dict(data: Dict[str, Any]) -> ValidationEvidence:
    return ValidationEvidence(
        type=str(data.get("type") or "UNKNOWN"),
        severity=str(data.get("severity") or "warning"),
        message=str(data.get("message") or ""),
        file=str(data.get("file") or data.get("file_path") or ""),
        line=data.get("line"),
        module=str(data.get("module") or data.get("module_name") or ""),
        signal=str(data.get("signal") or data.get("signal_name") or ""),
        check_id=str(data.get("check_id") or ""),
        expected=str(data.get("expected") or ""),
        observed=str(data.get("observed") or ""),
        cycle=data.get("cycle"),
        stimulus=dict(data.get("stimulus") or {}),
        raw_log=str(data.get("raw_log") or ""),
        related_nodes=list(data.get("related_nodes") or []),
        validator=str(data.get("validator") or ""),
    )


@dataclass
class FunctionalCheckSpec:
    check_id: str
    kind: str
    target_file: str
    target_module: str
    target_signal: str = ""
    clock: str = ""
    reset: str = ""
    stimulus: Dict[str, Any] = field(default_factory=dict)
    expectations: List[Dict[str, Any]] = field(default_factory=list)
    requires_base_fail: bool = True
    source: str = ""
    confidence: float = 0.0
    testability: str = ""
    test_cone: Dict[str, Any] = field(default_factory=dict)
    kg_anchors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproTestResult:
    check_id: str
    status: str
    base_status: str = ""
    patched_status: str = ""
    artifact_dir: str = ""
    message: str = ""
    spec: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
