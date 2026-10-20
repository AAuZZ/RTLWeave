from __future__ import annotations

"""Small composition helpers for KGVeri's internal validation stack.

The project deliberately keeps validators independent.  This module only
combines their existing :class:`ValidationReport` values and determines where
to send a failure; it does not contain repository- or benchmark-specific test
logic.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

try:
    from diagnostic_evidence import build_validation_report
    from repair_types import ValidationEvidence, ValidationReport
except Exception:  # pragma: no cover - package import path
    from .diagnostic_evidence import build_validation_report
    from .repair_types import ValidationEvidence, ValidationReport

try:
    from minitb.frozen_validation import (
        summarize_frozen_minitb,
        validate_frozen_minitb,
        verify_frozen_assets,
    )
    from repository_validation import RepositoryRegressionValidator, RepositoryRunResult
    from verilog_internal_validation import VerilogValidationManager
except Exception:  # pragma: no cover - package import path
    from .minitb.frozen_validation import (
        summarize_frozen_minitb,
        validate_frozen_minitb,
        verify_frozen_assets,
    )
    from .repository_validation import RepositoryRegressionValidator, RepositoryRunResult
    from .verilog_internal_validation import VerilogValidationManager


VALIDATION_STACK_VERSION = "kgveri-lightweight-v1"

_INFRA_TYPES = {
    "CLEAN_REPLAY_CHECKOUT_FAILED",
    "DOCKER_UNAVAILABLE",
    "INFRA_ERROR",
    "MINITB_ASSET_INTEGRITY",
    "MINITB_VALIDATOR_ERROR",
    "REPOSITORY_INFRA_ERROR",
    "TOOL_UNAVAILABLE",
    "VALIDATOR_EXCEPTION",
    "VALIDATOR_ERROR",
    "VALIDATOR_TIMEOUT",
}
_PATCH_TYPES = {
    "EMPTY_PATCH",
    "FORBIDDEN_FILE_EDIT",
    "PATCH_APPLY_FAILED",
    "PATCH_DELTA_POLICY",
    "REPLACE_FORMAT_ERROR",
}


def merge_validation_reports(
    named_reports: Mapping[str, ValidationReport | None]
    | Sequence[tuple[str, ValidationReport | None]],
    *,
    required: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Merge validator reports without allowing a missing required gate to pass.

    A validator that returns ``passed=False`` but no blocker is also converted
    into an explicit blocker.  This guards against accidentally treating a
    skip-only or malformed report as a successful repair.
    """

    items = list(named_reports.items()) if isinstance(named_reports, Mapping) else list(named_reports)
    by_name = {str(name): report for name, report in items}
    evidence: list[ValidationEvidence] = []
    validator_meta: dict[str, Any] = {}

    for name, report in items:
        name = str(name)
        if report is None:
            validator_meta[name] = {"ran": False, "passed": False}
            continue
        report_evidence = list(report.evidence or [])
        evidence.extend(report_evidence)
        validator_meta[name] = {
            "ran": True,
            "passed": bool(report.passed),
            "blocker_count": int(report.blocker_count),
            "warning_count": int(report.warning_count),
            "metadata": dict(report.metadata or {}),
        }
        if not report.passed and not any(item.severity == "blocker" for item in report_evidence):
            evidence.append(
                ValidationEvidence(
                    type="VALIDATOR_FAILED_WITHOUT_EVIDENCE",
                    severity="blocker",
                    message=f"Required validator {name!r} failed without blocker evidence.",
                    validator=name,
                )
            )

    for name in required:
        report = by_name.get(name)
        if report is not None:
            continue
        evidence.append(
            ValidationEvidence(
                type="REQUIRED_VALIDATOR_NOT_RUN",
                severity="blocker",
                message=f"Required validator {name!r} did not run.",
                validator=str(name),
            )
        )

    merged_metadata = {
        "validation_stack_version": VALIDATION_STACK_VERSION,
        "validators": validator_meta,
        **dict(metadata or {}),
    }
    return build_validation_report(evidence, metadata=merged_metadata)


def combined_raw_feedback(
    named_reports: Mapping[str, ValidationReport | None]
    | Sequence[tuple[str, ValidationReport | None]],
    *,
    limit_per_validator: int = 12000,
    total_limit: int = 30000,
) -> str:
    """Return bounded validator output for the Diagnosis Agent prompt."""

    items = named_reports.items() if isinstance(named_reports, Mapping) else named_reports
    sections: list[str] = []
    for name, report in items:
        if report is None:
            continue
        raw_parts = [item.raw_log.strip() for item in report.evidence if item.raw_log.strip()]
        body = "\n\n".join(raw_parts).strip() or (report.feedback_text or "").strip()
        if not body:
            continue
        if len(body) > limit_per_validator:
            body = body[:limit_per_validator].rstrip() + "\n...[truncated]..."
        sections.append(f"=== {name} ===\n{body}")
    text = "\n\n".join(sections)
    if len(text) > total_limit:
        return text[:total_limit].rstrip() + "\n...[truncated]..."
    return text


def route_validation_failure(report: ValidationReport) -> dict[str, Any]:
    """Classify a failed report for lightweight deterministic orchestration."""

    blockers = [item for item in report.evidence if item.severity == "blocker"]
    types = {str(item.type or "").upper() for item in blockers}
    validators = {str(item.validator or "").lower() for item in blockers}

    infra = any(
        item_type in _INFRA_TYPES
        or "INFRA" in item_type
        or "TIMEOUT" in item_type
        or "UNAVAILABLE" in item_type
        or "ASSET_INTEGRITY" in item_type
        for item_type in types
    )
    if infra:
        return {
            "failure_class": "infrastructure",
            "retry_action": "retry_validator",
            "consume_patch_attempt": False,
            "validators": sorted(validators),
        }

    patch_format = bool(types & _PATCH_TYPES) or any(
        "PATCH_APPLY" in item_type or "POLICY" in item_type or "REPLACE" in item_type
        for item_type in types
    )
    if patch_format:
        return {
            "failure_class": "patch_format",
            "retry_action": "regenerate_patch",
            "consume_patch_attempt": True,
            "validators": sorted(validators),
        }

    build_failure = any(
        token in item_type
        for item_type in types
        for token in ("BUILD", "COMPILE", "SYNTAX", "INTERFACE", "IVERILOG", "VERILATOR")
    )
    if build_failure:
        return {
            "failure_class": "rtl_build",
            "retry_action": "diagnose_and_repair",
            "consume_patch_attempt": True,
            "validators": sorted(validators),
        }

    return {
        "failure_class": "rtl_behavior",
        "retry_action": "kg_trace_and_repair",
        "consume_patch_attempt": True,
        "validators": sorted(validators),
    }


def validator_exception_report(name: str, exc: BaseException) -> ValidationReport:
    """Turn an unexpected validator exception into routable infra evidence."""

    return build_validation_report(
        [
            ValidationEvidence(
                type="VALIDATOR_EXCEPTION",
                severity="blocker",
                message=f"{name} could not run: {exc}",
                raw_log=repr(exc),
                validator=name,
            )
        ],
        metadata={"validator": name, "exception_type": type(exc).__name__},
    )


@dataclass
class ValidationStackResult:
    report: ValidationReport
    reports: dict[str, ValidationReport | None]
    route: dict[str, Any]
    raw_feedback: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "reports": {
                name: value.to_dict() if value is not None else None
                for name, value in self.reports.items()
            },
            "route": dict(self.route),
            "raw_feedback": self.raw_feedback,
        }


class LightweightValidationStack:
    """Fail-fast validation for the three small RTL research subsets.

    The class profiles one fixed repository command on the exact buggy base,
    then reuses that immutable baseline for every candidate and clean replay.
    Mini-TB assets are supplied by the outer workflow and can never be changed
    by this class.
    """

    REQUIRED_STAGES = ("static", "frozen_minitb", "repository_regression")

    def __init__(
        self,
        *,
        repo_kind: str,
        base_repo: str | os.PathLike[str],
        base_commit: str,
        minitb_case: Any,
        frozen_dir: str | os.PathLike[str],
        artifact_root: str | os.PathLike[str],
        related_entities: Mapping[str, Any] | None = None,
        issue_text: str = "",
        minitb_timeout: int = 300,
        regression_timeout: int | None = None,
        regression_validator: RepositoryRegressionValidator | None = None,
        minitb_validator: Any = validate_frozen_minitb,
        static_validator_factory: Any = VerilogValidationManager,
    ) -> None:
        self.repo_kind = str(repo_kind)
        self.base_repo = Path(base_repo).resolve()
        self.base_commit = str(base_commit).strip()
        self.minitb_case = minitb_case
        self.frozen_dir = Path(frozen_dir).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.related_entities = dict(related_entities or {})
        self.issue_text = str(issue_text or "")
        self.minitb_timeout = int(minitb_timeout)
        self.regression = regression_validator or RepositoryRegressionValidator(
            self.repo_kind,
            timeout_seconds=regression_timeout,
            docker_image=str(getattr(minitb_case, "docker_image", "") or ""),
        )
        self._minitb_validator = minitb_validator
        self._static_validator_factory = static_validator_factory
        self.baseline: RepositoryRunResult | None = None
        self.frozen_summary = summarize_frozen_minitb(self.frozen_dir)

    @property
    def target_behavior_validated(self) -> bool:
        qualification = self.frozen_summary.get("base_qualification") or {}
        return bool(
            qualification.get("status") == "QUALIFIED_MINITB"
            and qualification.get("repeatable")
        )

    @property
    def baseline_dir(self) -> Path:
        return self.artifact_root / "repository_baseline"

    def profile_baseline(self, *, force: bool = False) -> RepositoryRunResult:
        """Load a matching cached baseline or profile the exact base once."""

        self._assert_exact_base_checkout()
        baseline_path = self.baseline_dir / "baseline.json"
        if not force and baseline_path.is_file():
            try:
                cached = RepositoryRunResult.from_dict(
                    json.loads(baseline_path.read_text(encoding="utf-8"))
                )
                # validate_candidate performs the authoritative command-id check.
                if cached.command_id == self.regression.command_id and cached.repo == self.regression.repo:
                    self.baseline = cached
                    return cached
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        self.baseline = self.regression.profile_base(self.base_repo, self.baseline_dir)
        return self.baseline

    def _assert_exact_base_checkout(self) -> None:
        head = subprocess.run(
            ["git", "-C", str(self.base_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != self.base_commit:
            raise ValueError(
                f"repository baseline is not checked out at exact base_commit {self.base_commit}"
            )
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            result = subprocess.run(
                ["git", "-C", str(self.base_repo), *args],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ValueError("repository baseline contains tracked modifications")

    def validate_candidate(
        self,
        *,
        candidate_repo: str | os.PathLike[str],
        patch_text: str,
        patch_delta: Any,
        repair_plan: Any,
        artifact_dir: str | os.PathLike[str],
        validation_id: str,
    ) -> ValidationStackResult:
        candidate = Path(candidate_repo).resolve()
        artifacts = Path(artifact_dir).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        reports: dict[str, ValidationReport | None] = {
            "static": None,
            "frozen_minitb": None,
            "repository_regression": None,
        }

        try:
            manager = self._static_validator_factory(
                str(candidate),
                related_entities=self.related_entities,
                issue_text=self.issue_text,
                artifact_dir=str(artifacts / "static"),
            )
            reports["static"] = manager.validate(
                patch_text=patch_text,
                patch_delta=patch_delta,
                repair_plan=repair_plan,
            )
        except Exception as exc:
            reports["static"] = validator_exception_report("static", exc)

        if reports["static"] and reports["static"].passed:
            try:
                reports["frozen_minitb"] = self._minitb_validator(
                    case=self.minitb_case,
                    candidate_repo=candidate,
                    frozen_dir=self.frozen_dir,
                    output_root=artifacts / "minitb_evaluations",
                    evaluation_id=validation_id,
                    timeout=self.minitb_timeout,
                )
            except Exception as exc:
                reports["frozen_minitb"] = validator_exception_report("frozen_minitb", exc)

        if reports["frozen_minitb"] and reports["frozen_minitb"].passed:
            try:
                baseline = self.baseline or self.profile_baseline()
                reports["repository_regression"] = self.regression.validate_candidate(
                    baseline,
                    candidate,
                    artifacts / "repository_regression",
                )
            except Exception as exc:
                reports["repository_regression"] = validator_exception_report(
                    "repository_regression", exc
                )

        report = merge_validation_reports(
            reports,
            required=self.REQUIRED_STAGES,
            metadata={
                "validation_id": validation_id,
                "repo": self.repo_kind,
                "base_commit": self.base_commit,
                "frozen_minitb_summary": self.frozen_summary,
                "target_behavior_validated": self.target_behavior_validated,
                "validation_level": (
                    "issue_minitb_and_lightweight_regression"
                    if self.target_behavior_validated
                    else "module_smoke_and_lightweight_regression"
                ),
                "fail_fast": True,
            },
        )
        route = (
            {"failure_class": "none", "retry_action": "accept", "consume_patch_attempt": False}
            if report.passed
            else route_validation_failure(report)
        )
        raw_feedback = combined_raw_feedback(reports)
        payload = {
            "report": report.to_dict(),
            "route": route,
            "stage_reports": {
                name: value.to_dict() if value is not None else None
                for name, value in reports.items()
            },
        }
        (artifacts / "validation_stack.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ValidationStackResult(report, reports, route, raw_feedback)

    def clean_replay(
        self,
        *,
        diff_text: str,
        patch_text: str,
        patch_delta: Any,
        repair_plan: Any,
        artifact_dir: str | os.PathLike[str],
    ) -> ValidationStackResult:
        """Apply the unified diff to a fresh exact-base clone and replay all gates."""

        artifacts = Path(artifact_dir).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        replay_repo = artifacts / "repo"
        if replay_repo.exists():
            shutil.rmtree(replay_repo)
        setup_report: ValidationReport | None = None
        frozen_hashes: dict[str, str] = {}
        try:
            frozen_hashes = verify_frozen_assets(self.frozen_dir)
        except Exception as exc:
            setup_report = build_validation_report(
                [
                    ValidationEvidence(
                        type="MINITB_ASSET_INTEGRITY",
                        severity="blocker",
                        message=f"clean replay rejected mutated frozen assets: {exc}",
                        raw_log=repr(exc),
                        validator="clean_replay",
                    )
                ]
            )

        if setup_report is None:
            clone = subprocess.run(
                ["git", "clone", "--quiet", "--shared", "--no-checkout", str(self.base_repo), str(replay_repo)],
                capture_output=True,
                text=True,
            )
            if clone.returncode == 0:
                checkout = subprocess.run(
                    ["git", "-C", str(replay_repo), "checkout", "--quiet", "--detach", self.base_commit],
                    capture_output=True,
                    text=True,
                )
            else:
                checkout = clone
            if clone.returncode != 0 or checkout.returncode != 0:
                message = (checkout.stderr or clone.stderr or "fresh clone failed").strip()
                setup_report = build_validation_report(
                    [
                        ValidationEvidence(
                            type="CLEAN_REPLAY_CHECKOUT_FAILED",
                            severity="blocker",
                            message=message,
                            raw_log=(clone.stdout + clone.stderr + checkout.stdout + checkout.stderr)[-12000:],
                            validator="clean_replay",
                        )
                    ]
                )
        if setup_report is None:
            apply_check = subprocess.run(
                ["git", "-C", str(replay_repo), "apply", "--check", "--whitespace=nowarn", "-"],
                input=diff_text,
                capture_output=True,
                text=True,
            )
            apply = apply_check
            if apply_check.returncode == 0:
                apply = subprocess.run(
                    ["git", "-C", str(replay_repo), "apply", "--whitespace=nowarn", "-"],
                    input=diff_text,
                    capture_output=True,
                    text=True,
                )
            if apply.returncode != 0:
                setup_report = build_validation_report(
                    [
                        ValidationEvidence(
                            type="CLEAN_REPLAY_PATCH_APPLY_FAILED",
                            severity="blocker",
                            message="Candidate diff does not apply to the exact base revision.",
                            raw_log=(apply.stdout + "\n" + apply.stderr)[-12000:],
                            validator="clean_replay",
                        )
                    ]
                )

        if setup_report is None:
            inner = self.validate_candidate(
                candidate_repo=replay_repo,
                patch_text=patch_text,
                patch_delta=patch_delta,
                repair_plan=repair_plan,
                artifact_dir=artifacts / "validators",
                validation_id="clean-replay",
            )
            setup_report = inner.report
            reports = inner.reports
            raw_feedback = inner.raw_feedback
        else:
            reports = {"clean_replay_setup": setup_report}
            raw_feedback = setup_report.feedback_text

        report = merge_validation_reports(
            {"clean_replay": setup_report},
            required=("clean_replay",),
            metadata={
                "clean_replay": True,
                "base_commit": self.base_commit,
                "patch_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
                "frozen_asset_sha256": frozen_hashes,
                "target_behavior_validated": self.target_behavior_validated,
            },
        )
        route = (
            {"failure_class": "none", "retry_action": "accept", "consume_patch_attempt": False}
            if report.passed
            else route_validation_failure(report)
        )
        manifest = {
            "schema_version": VALIDATION_STACK_VERSION,
            "passed": report.passed,
            "base_commit": self.base_commit,
            "base_repo": str(self.base_repo),
            "replay_workspace": str(replay_repo),
            "replay_workspace_retained": False,
            "patch_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
            "frozen_asset_sha256": frozen_hashes,
            "repository_command_id": self.regression.command_id,
            "validation_report": report.to_dict(),
            "route": route,
        }
        (artifacts / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (artifacts / "validation_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (artifacts / "applied.diff").write_text(diff_text, encoding="utf-8")
        shutil.rmtree(replay_repo, ignore_errors=True)
        return ValidationStackResult(report, reports, route, raw_feedback)
