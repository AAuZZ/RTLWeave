"""Reusable validation of a frozen Mini-TB on an already patched checkout."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

try:
    from diagnostic_evidence import build_validation_report, write_evidence_trace
    from repair_types import ValidationEvidence, ValidationReport
except Exception:  # pragma: no cover - package import path
    from ..diagnostic_evidence import build_validation_report, write_evidence_trace
    from ..repair_types import ValidationEvidence, ValidationReport
from .docker_runner import MiniTBDockerRunner, QualificationEngine, save_run_logs
from .models import QualificationResult, VisibleCase, write_json


RunnerFactory = Callable[[VisibleCase, Path, Path, int], Any]
_SAFE_EVALUATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def verify_frozen_assets(
    frozen_dir: Path,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Verify a frozen asset set and return the canonical hash mapping.

    Passing ``expected_hashes`` is important for the post-run check: changing
    both an asset and ``sha256.json`` cannot redefine the expected state.
    """

    frozen = Path(frozen_dir).resolve()
    hashes_path = frozen / "sha256.json"
    if expected_hashes is None:
        if not hashes_path.is_file():
            raise ValueError("frozen Mini-TB is missing sha256.json")
        raw = json.loads(hashes_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("frozen Mini-TB sha256.json must contain an object")
        expected_hashes = {str(name): str(value) for name, value in raw.items()}
    if not expected_hashes:
        raise ValueError("frozen Mini-TB has no recorded asset hashes")

    verified: dict[str, str] = {}
    for name, expected in sorted(expected_hashes.items()):
        if Path(name).name != name or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"invalid frozen Mini-TB hash entry: {name!r}")
        path = frozen / name
        actual = _sha256(path) if path.is_file() else ""
        if actual.lower() != expected.lower():
            raise ValueError(f"frozen Mini-TB hash mismatch: {name}")
        verified[name] = expected.lower()
    return verified


def load_frozen_behavior_spec(frozen_dir: Path) -> dict[str, Any]:
    """Load the frozen BehaviorSpec after integrity verification."""

    verify_frozen_assets(frozen_dir)
    value = json.loads((Path(frozen_dir) / "behavior_spec.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen behavior_spec.json must contain an object")
    return value


def summarize_frozen_minitb(frozen_dir: Path) -> dict[str, Any]:
    """Return the compact contract consumed by repair and diagnosis agents."""

    frozen = Path(frozen_dir).resolve()
    hashes = verify_frozen_assets(frozen)
    spec = json.loads((frozen / "behavior_spec.json").read_text(encoding="utf-8"))
    qualification = json.loads(
        (frozen / "qualification_at_freeze.json").read_text(encoding="utf-8")
    )
    failures: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for marker in qualification.get("markers") or []:
        if not isinstance(marker, dict) or marker.get("kind") != "FAIL":
            continue
        fields = marker.get("fields") or {}
        signature = tuple(
            fields.get(key)
            for key in ("check", "module", "signal", "expected", "observed", "cycle")
        )
        if signature in seen:
            continue
        seen.add(signature)
        failures.append(
            {
                "check_id": fields.get("check", ""),
                "module": fields.get("module", ""),
                "signal": fields.get("signal", ""),
                "expected": fields.get("expected", ""),
                "observed": fields.get("observed", ""),
                "cycle": _optional_int(fields.get("cycle")),
            }
        )
    target = spec.get("target") or {}
    return {
        "schema_version": "kgveri-frozen-minitb-summary-v1",
        "instance_id": spec.get("instance_id", ""),
        "template_id": spec.get("template_id", ""),
        "target": {
            "file": target.get("file", ""),
            "module": target.get("module", ""),
        },
        "checks": [
            {
                "check_id": check.get("check_id", ""),
                "signal": check.get("signal", ""),
                "op": check.get("op", ""),
                "expected": check.get("expected", ""),
            }
            for check in spec.get("checks") or []
            if isinstance(check, dict)
        ],
        "base_qualification": {
            "status": qualification.get("status", ""),
            "reason": qualification.get("reason", ""),
            "repeatable": bool(qualification.get("repeatable")),
        },
        "base_failure_signature": failures,
        "asset_sha256": hashes,
    }


def validate_frozen_minitb(
    *,
    case: VisibleCase,
    candidate_repo: Path,
    frozen_dir: Path,
    output_root: Path,
    evaluation_id: str | None = None,
    timeout: int = 300,
    repeat_count: int = 2,
    runner_factory: RunnerFactory = MiniTBDockerRunner,
) -> ValidationReport:
    """Run a frozen Mini-TB on an already patched candidate repository.

    The function performs no checkout, clone, or patch application.  It stages
    immutable test assets in a fresh per-evaluation directory, verifies the
    source and staged hashes before and after execution, and emits the common
    ``ValidationReport``/evidence contract used by the repair loop.
    """

    candidate = Path(candidate_repo).resolve()
    if not candidate.is_dir():
        raise ValueError(f"candidate repository does not exist: {candidate}")
    if timeout <= 0 or repeat_count <= 0:
        raise ValueError("timeout and repeat_count must be positive")

    frozen = Path(frozen_dir).resolve()
    source_hashes = verify_frozen_assets(frozen)
    summary = summarize_frozen_minitb(frozen)
    evaluation_dir = _new_evaluation_dir(Path(output_root), evaluation_id)
    staged = evaluation_dir / "frozen"
    shutil.copytree(frozen, staged, copy_function=shutil.copy2)
    verify_frozen_assets(staged, source_hashes)
    spec = json.loads((staged / "behavior_spec.json").read_text(encoding="utf-8"))

    result: QualificationResult | None = None
    execution_error = ""
    try:
        runner = runner_factory(case, candidate, evaluation_dir, timeout)
        result = QualificationEngine(runner).qualify(
            staged,
            evaluation_dir / "build",
            spec,
            expected_failure=False,
            repeat_count=repeat_count,
        )
        save_run_logs(evaluation_dir / "logs", result)
        write_json(evaluation_dir / "qualification.json", result.to_dict())
    except Exception as exc:  # validator failures become diagnostic evidence
        execution_error = f"{type(exc).__name__}: {exc}"

    integrity_errors: list[str] = []
    for label, path in (("source", frozen), ("staged", staged)):
        try:
            verify_frozen_assets(path, source_hashes)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            integrity_errors.append(f"{label}: {exc}")

    evidence = _validation_evidence(
        result=result,
        spec=spec,
        execution_error=execution_error,
        integrity_errors=integrity_errors,
    )
    metadata = {
        "validator": "frozen_minitb",
        "instance_id": case.instance_id,
        "repo": case.repo,
        "status": result.status if result else "VALIDATOR_ERROR",
        "reason": result.reason if result else execution_error,
        "repeatable": bool(result and result.repeatable),
        "candidate_repo": str(candidate),
        "frozen_dir": str(frozen),
        "evaluation_id": evaluation_dir.name,
        "artifact_dir": str(evaluation_dir),
        "frozen_hash_verified_before": True,
        "frozen_hash_verified_after": not integrity_errors,
        "frozen_summary": summary,
        "target_behavior_validated": (
            summary.get("base_qualification", {}).get("status") == "QUALIFIED_MINITB"
        ),
    }
    report = build_validation_report(evidence, metadata=metadata)
    write_json(evaluation_dir / "validation_report.json", report.to_dict())
    write_evidence_trace(
        str(evaluation_dir / "evidence_trace.json"),
        report,
        extra={"frozen_summary": summary},
    )
    return report


def _validation_evidence(
    *,
    result: QualificationResult | None,
    spec: dict[str, Any],
    execution_error: str,
    integrity_errors: list[str],
) -> list[ValidationEvidence]:
    evidence: list[ValidationEvidence] = []
    target = spec.get("target") or {}
    stimulus = _compact_stimulus(spec)
    for message in integrity_errors:
        evidence.append(
            ValidationEvidence(
                type="MINITB_ASSET_INTEGRITY",
                severity="blocker",
                message=message,
                file=str(target.get("file") or ""),
                module=str(target.get("module") or ""),
                validator="frozen_minitb",
            )
        )
    if execution_error:
        evidence.append(
            ValidationEvidence(
                type="MINITB_VALIDATOR_ERROR",
                severity="blocker",
                message=execution_error,
                file=str(target.get("file") or ""),
                module=str(target.get("module") or ""),
                validator="frozen_minitb",
            )
        )
        return evidence
    if result is None:
        return evidence

    if result.status in {"PATCH_ACCEPTED", "OFFICIAL_FIX_ACCEPTED"}:
        evidence.append(
            ValidationEvidence(
                type="MINITB_PASS",
                severity="pass",
                message="all frozen Mini-TB checks passed on the candidate repository",
                file=str(target.get("file") or ""),
                module=str(target.get("module") or ""),
                validator="frozen_minitb",
            )
        )
        return evidence

    if (
        spec.get("template_id") == "MINIMAL_MODULE_SMOKE"
        and result.status == "MODULE_SMOKE_ONLY"
    ):
        evidence.append(
            ValidationEvidence(
                type="MINITB_SMOKE_PASS",
                severity="pass",
                message=(
                    "target module compiled, elaborated, and completed the frozen smoke run; "
                    "no issue-specific behavior oracle was available"
                ),
                file=str(target.get("file") or ""),
                module=str(target.get("module") or ""),
                validator="frozen_minitb",
            )
        )
        return evidence

    failures = [marker for marker in result.markers if marker.kind == "FAIL"]
    seen: set[tuple[Any, ...]] = set()
    for marker in failures:
        fields = marker.fields
        key = tuple(fields.get(name) for name in ("check", "module", "signal", "expected", "observed", "cycle"))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            ValidationEvidence(
                type="MINITB_FUNCTIONAL_FAILURE",
                severity="blocker",
                message="frozen issue-directed check still fails on the candidate",
                file=str(target.get("file") or ""),
                module=str(fields.get("module") or target.get("module") or ""),
                signal=str(fields.get("signal") or ""),
                check_id=str(fields.get("check") or ""),
                expected=str(fields.get("expected") or ""),
                observed=str(fields.get("observed") or ""),
                cycle=_optional_int(fields.get("cycle")),
                stimulus=stimulus,
                raw_log=marker.raw,
                validator="frozen_minitb",
            )
        )
    if evidence:
        return evidence

    raw_log = ""
    failure_type = "MINITB_VALIDATION_FAILURE"
    if result.status == "INFRA_ERROR":
        failure_type = "MINITB_INFRA_ERROR"
        if result.build:
            raw_log = _run_log(result.build.stdout, result.build.stderr)
    elif result.status == "TB_BUILD_FAIL" and result.build:
        failure_type = "MINITB_BUILD_FAIL"
        raw_log = _run_log(result.build.stdout, result.build.stderr)
    elif result.runs:
        raw_log = _run_log(result.runs[-1].stdout, result.runs[-1].stderr)
    evidence.append(
        ValidationEvidence(
            type=failure_type,
            severity="blocker",
            message=f"{result.status}: {result.reason}",
            file=str(target.get("file") or ""),
            module=str(target.get("module") or ""),
            stimulus=stimulus,
            raw_log=raw_log,
            validator="frozen_minitb",
        )
    )
    return evidence


def _new_evaluation_dir(output_root: Path, evaluation_id: str | None) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if evaluation_id is None:
        evaluation_id = f"eval-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    if not _SAFE_EVALUATION_ID.fullmatch(evaluation_id):
        raise ValueError("evaluation_id must be a simple path-safe identifier")
    destination = root / evaluation_id
    destination.mkdir(parents=False, exist_ok=False)
    return destination


def _compact_stimulus(spec: dict[str, Any]) -> dict[str, Any]:
    stimulus: dict[str, Any] = dict(spec.get("defaults") or {})
    for step in spec.get("steps") or []:
        if isinstance(step, dict):
            stimulus.update(step.get("drive") or {})
    return dict(list(stimulus.items())[:16])


def _run_log(stdout: str, stderr: str, limit: int = 12_000) -> str:
    combined = (stdout + "\n" + stderr).strip()
    return combined[-limit:]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
