from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .behavior_spec import (
    BehaviorSpecGenerator,
    BehaviorSpecValidator,
    OpenAICompatibleClient,
    SpecValidationError,
    load_and_validate_spec,
)
from .context_builder import ContextBuilder
from .docker_runner import MiniTBDockerRunner, QualificationEngine, save_run_logs
from .evidence_slice import evidence_review
from .experiment_manifest import generation_fingerprint
from .ibex_resolver import RTLSourceResolver, SourceResolution
from .models import (
    BudgetExhausted,
    HarnessValidationError,
    MiniTBError,
    QualificationResult,
    RetryBudget,
    write_json,
)
from .template_renderer import HarnessFillGenerator, TemplateRenderer, build_smoke_spec
from .visible_input import ensure_base_checkout, select_case


COMPONENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_ROOT.parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "minitb"
DEFAULT_VISIBLE_CASES = ARTIFACT_ROOT / "visible_cases.json"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated KGVeri Mini-Testbench experiment")
    parser.add_argument("--visible-cases", type=Path, default=DEFAULT_VISIBLE_CASES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="build visible context without model calls")
    _case_args(inspect)

    smoke = subparsers.add_parser(
        "smoke", help="compile and execute the selected module without model calls"
    )
    _case_args(smoke)
    smoke.add_argument("--timeout", type=int, default=300)

    generate = subparsers.add_parser("generate", help="generate and qualify a Mini-TB on the base")
    _case_args(generate)
    generate.add_argument("--behavior-spec", type=Path)
    generate.add_argument("--deterministic-fill", action="store_true", default=True)
    generate.add_argument("--llm-harness-fill", dest="deterministic_fill", action="store_false",
                          help="optional legacy harness call; deterministic rendering is the default")
    generate.add_argument("--repair-rounds", type=int, choices=(0, 1, 2, 3), default=None,
                          help="base-feedback corrections; default 2 for fresh generation, 0 for supplied specs")
    generate.add_argument("--timeout", type=int, default=300)

    qualify = subparsers.add_parser("qualify", help="replay a frozen Mini-TB on the base")
    _case_args(qualify, kg=False)
    qualify.add_argument("--timeout", type=int, default=300)

    offline = subparsers.add_parser(
        "offline-evaluate", help="evaluate a frozen Mini-TB against an explicit patch"
    )
    _case_args(offline, kg=False)
    offline.add_argument(
        "--patch",
        "--official-patch",
        dest="patch",
        type=Path,
        required=True,
        help="explicit candidate diff; --official-patch is retained as a compatibility alias",
    )
    offline.add_argument("--timeout", type=int, default=300)
    return parser.parse_args(argv)


def _case_args(parser: argparse.ArgumentParser, kg: bool = True) -> None:
    parser.add_argument("instance_id")
    if kg:
        parser.add_argument("--kg-location", type=Path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generator = None
    try:
        case = select_case(args.visible_cases.resolve(), args.instance_id)
        case_output = args.output_root.resolve() / case.instance_id
        case_output.mkdir(parents=True, exist_ok=True)
        write_json(case_output / "visible_input.json", case.to_dict())
        base_repo = ensure_base_checkout(case, case_output / "base_repo")

        if args.command == "inspect":
            context = _build_context(case, base_repo, case_output, args.kg_location)
            print(json.dumps(_inspection_view(context), indent=2, ensure_ascii=True))
            return 0
        if args.command == "smoke":
            return _smoke_only(case, base_repo, case_output, args)
        if args.command == "generate":
            return _generate(case, base_repo, case_output, args)
        if args.command == "qualify":
            result = _replay(case, base_repo, case_output, args.timeout)
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
            return 0 if result.status in {"QUALIFIED_MINITB", "MODULE_SMOKE_ONLY"} else 2
        if args.command == "offline-evaluate":
            result = _offline_evaluate(case, base_repo, case_output, args)
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
            return 0 if result.status in {
                "PATCH_ACCEPTED",
                "OFFICIAL_FIX_ACCEPTED",
                "MODULE_SMOKE_ONLY",
            } else 2
    except (MiniTBError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


def _generate(case, base_repo: Path, case_output: Path, args: argparse.Namespace) -> int:
    start = time.monotonic()
    _invalidate_previous_generation(case_output)
    context = _build_context(case, base_repo, case_output, args.kg_location)
    validator = BehaviorSpecValidator(case, context, base_repo)
    attempts_path = case_output / "attempts.jsonl"
    writer = _attempt_writer(attempts_path)
    writer({
        "stage": "generation_start",
        "status": "STARTED",
        "generation_policy": "kg-self-correction-v1",
        "generator_sha256": generation_fingerprint(),
    })
    preflight = MiniTBDockerRunner(case, base_repo, case_output, timeout=30).preflight()
    write_json(case_output / "preflight.json", preflight.to_dict())
    if preflight.returncode != 0:
        result = QualificationResult(
            status="INFRA_ERROR", reason="MiniTB toolchain preflight failed", build=preflight
        )
        writer({"stage": "preflight", "status": result.status, "reason": result.reason})
        _finalize_summary(case_output, result, start, "preflight_failed")
        print(json.dumps(result.to_dict(), indent=2))
        return 2
    client = None
    budget = RetryBudget(total_limit=10, limits={"behavior_spec": 5, "harness_fill": 2, "reproduction": 3})

    try:
        if args.behavior_spec:
            spec = load_and_validate_spec(args.behavior_spec.resolve(), validator, case_output / "behavior_spec.json")
            writer({"stage": "behavior_spec", "round": 0, "status": "VALID", "source": "explicit_file"})
        else:
            client = OpenAICompatibleClient(PROJECT_ROOT)
            generator = BehaviorSpecGenerator(
                validator,
                COMPONENT_ROOT / "prompts" / "behavior_spec.md",
                client,
                writer,
                budget,
            )
            spec = generator.generate(context, case_output / "behavior_spec.json")
    except (SpecValidationError, BudgetExhausted) as exc:
        errors = exc.errors if isinstance(exc, SpecValidationError) else [str(exc)]
        writer({"stage": "behavior_spec", "status": "SPEC_INVALID", "validator_errors": errors})
        result = _run_smoke(case, context, base_repo, case_output, args.timeout, writer)
        if result.status == "MODULE_SMOKE_ONLY":
            _freeze(
                case_output,
                case_output / "smoke",
                build_smoke_spec(case.instance_id, context["selected_target"]),
                result,
            )
        _finalize_summary(case_output, result, start, "spec_invalid_smoke")
        return 0 if result.status == "MODULE_SMOKE_ONLY" else 2

    if spec["bug_class"] == "unsupported" or spec["oracle"]["confidence"] == "low":
        smoke_target = _target_for_spec(context, spec)
        result = _run_smoke(case, context, base_repo, case_output, args.timeout, writer, smoke_target)
        if result.status == "MODULE_SMOKE_ONLY":
            _freeze(
                case_output,
                case_output / "smoke",
                build_smoke_spec(case.instance_id, smoke_target),
                result,
            )
        _finalize_summary(case_output, result, start, "low_confidence_smoke")
        return 0 if result.status == "MODULE_SMOKE_ONLY" else 2

    renderer = TemplateRenderer(COMPONENT_ROOT / "templates")
    target = _target_for_spec(context, spec)
    expected_fill = renderer.expected_fill(spec, target)
    try:
        if args.deterministic_fill:
            fill = expected_fill
            writer({"stage": "harness_fill", "round": 0, "status": "VALID", "source": "deterministic"})
        else:
            client = client or OpenAICompatibleClient(PROJECT_ROOT)
            fill = HarnessFillGenerator(
                COMPONENT_ROOT / "prompts" / "harness_fill.md",
                client,
                writer,
                budget,
            ).generate(spec, target, expected_fill)
    except (HarnessValidationError, BudgetExhausted) as exc:
        writer({"stage": "harness_fill", "status": "INVALID", "validator_error": str(exc)})
        result = _run_smoke(case, context, base_repo, case_output, args.timeout, writer, target)
        if result.status == "MODULE_SMOKE_ONLY":
            _freeze(
                case_output,
                case_output / "smoke",
                build_smoke_spec(case.instance_id, target),
                result,
            )
        _finalize_summary(case_output, result, start, "harness_invalid_smoke")
        return 0 if result.status == "MODULE_SMOKE_ONLY" else 2

    generated = case_output / "generated"
    write_json(case_output / "behavior_spec_initial.json", spec)
    candidates = [spec]
    if generator is not None and args.deterministic_fill:
        candidates = _rank_generation_candidates(generator.candidate_specs, spec, context)
        write_json(case_output / "behavior_spec_candidates.json", candidates)
    result = None
    for index, candidate in enumerate(candidates, start=1):
        candidate_fill = (
            fill if candidate == spec else renderer.expected_fill(candidate, _target_for_spec(context, candidate))
        )
        candidate_result = _qualify_spec(
            case, context, base_repo, case_output, candidate, args.timeout, candidate_fill
        )
        writer({
            "stage": "candidate_qualification",
            "candidate": index,
            "status": candidate_result.status,
            "reason": candidate_result.reason,
        })
        if result is None or candidate_result.status == "QUALIFIED_MINITB":
            spec, fill, result = candidate, candidate_fill, candidate_result
        if candidate_result.status == "QUALIFIED_MINITB":
            write_json(case_output / "behavior_spec.json", spec)
            break
    assert result is not None
    repair_rounds = getattr(args, "repair_rounds", None)
    if repair_rounds is None:
        repair_rounds = 0 if args.behavior_spec else 2
    for correction in range(repair_rounds):
        if result.status not in {"TB_BUILD_FAIL", "BASE_PASS", "IRRELEVANT_FAIL"}:
            break
        save_run_logs(case_output / "logs", result, prefix=f"correction_{correction}_")
        write_json(case_output / f"qualification_before_correction_{correction}.json", result.to_dict())
        writer({"stage": "qualification_initial", "status": result.status, "reason": result.reason})
        client = client or OpenAICompatibleClient(PROJECT_ROOT)
        try:
            revised = BehaviorSpecGenerator(
                validator, COMPONENT_ROOT / "prompts" / "behavior_spec.md", client, writer, budget
            ).repair(context, spec, _base_feedback(result, spec), case_output / "behavior_spec.json")
        except (SpecValidationError, BudgetExhausted) as exc:
            writer({"stage": "correction_exhausted", "status": "SPEC_INVALID", "reason": str(exc)})
            break
        spec = revised
        result = _qualify_spec(case, context, base_repo, case_output, spec, args.timeout)

    # Retain the complete generated contract. A buggy-base failure is not a
    # reason to delete other checks or move them to an observed failing cycle.
    save_run_logs(case_output / "logs", result)
    write_json(case_output / "qualification.json", result.to_dict())
    writer({"stage": "qualification", "status": result.status, "reason": result.reason})

    if result.status == "QUALIFIED_MINITB":
        _freeze(case_output, generated, spec, result)
        _finalize_summary(case_output, result, start, "qualified")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
        return 0

    functional_status = result.status
    write_json(case_output / "functional_qualification.json", result.to_dict())
    result = _run_smoke(case, context, base_repo, case_output, args.timeout, writer, target)
    if result.status == "MODULE_SMOKE_ONLY":
        _freeze(
            case_output,
            case_output / "smoke",
            build_smoke_spec(case.instance_id, target),
            result,
        )
    _finalize_summary(case_output, result, start, f"functional_{functional_status.lower()}_smoke")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0 if result.status == "MODULE_SMOKE_ONLY" else 2


def _smoke_only(case, base_repo: Path, case_output: Path, args: argparse.Namespace) -> int:
    start = time.monotonic()
    _invalidate_previous_generation(case_output)
    context = _build_context(case, base_repo, case_output, args.kg_location)
    writer = _attempt_writer(case_output / "attempts.jsonl")
    writer({"stage": "smoke_start", "status": "STARTED", "model_calls": 0})
    preflight = MiniTBDockerRunner(case, base_repo, case_output, timeout=30).preflight()
    write_json(case_output / "preflight.json", preflight.to_dict())
    if preflight.returncode != 0:
        result = QualificationResult(
            status="INFRA_ERROR", reason="MiniTB toolchain preflight failed", build=preflight
        )
        writer({"stage": "preflight", "status": result.status, "reason": result.reason})
        write_json(case_output / "qualification.json", result.to_dict())
        _finalize_summary(case_output, result, start, "smoke_preflight_failed")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
        return 2
    result = _run_smoke(
        case, context, base_repo, case_output, args.timeout, writer, context["selected_target"]
    )
    if result.status == "MODULE_SMOKE_ONLY":
        _freeze(
            case_output,
            case_output / "smoke",
            build_smoke_spec(case.instance_id, context["selected_target"]),
            result,
        )
    _finalize_summary(case_output, result, start, "smoke_only")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0 if result.status == "MODULE_SMOKE_ONLY" else 2


def _qualify_spec(case, context, base_repo, case_output, spec, timeout, fill=None):
    target = _target_for_spec(context, spec)
    renderer = TemplateRenderer(COMPONENT_ROOT / "templates")
    generated = case_output / "generated"
    if generated.exists():
        shutil.rmtree(generated)
    renderer.render(spec, target, fill or renderer.expected_fill(spec, target), generated)
    resolution = RTLSourceResolver(base_repo, repo=case.repo, target_file=target["file"]).resolve(target["file"], target["module"])
    write_json(generated / "source_resolution.json", resolution.to_dict())
    runner = MiniTBDockerRunner(case, base_repo, case_output, timeout=timeout)
    runner.prepare_run_script(generated, resolution)
    return QualificationEngine(runner).qualify(generated, case_output / "build", spec, expected_failure=True, repeat_count=2)


def _rank_generation_candidates(
    proposals: list[dict[str, Any]], preferred: dict[str, Any], context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Keep a tiny, issue-grounded beam without adding model calls or templates."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in [preferred, *proposals]:
        key = json.dumps(spec, sort_keys=True)
        if key in seen or not _nontrivial_candidate(spec, context):
            continue
        seen.add(key)
        unique.append(spec)

    def rank(spec: dict[str, Any]) -> tuple[int, int, int]:
        target = _target_for_spec(context, spec)
        hints = evidence_review(spec, target.get("minitb_evidence") or {})
        critical = sum(
            hint.startswith("no issue-named trigger input")
            or hint.startswith("no issue-named output")
            or "after the issue trigger is asserted" in hint
            for hint in hints
        )
        return critical, len(spec.get("checks") or []), len(spec.get("steps") or [])

    return sorted(unique, key=rank)[:3] or [preferred]


def _nontrivial_candidate(spec: dict[str, Any], context: dict[str, Any]) -> bool:
    if spec.get("bug_class") == "unsupported" or (spec.get("oracle") or {}).get("confidence") == "low":
        return False
    checks = spec.get("checks") or []
    steps = spec.get("steps") or []
    if not checks or not steps or not (spec.get("oracle") or {}).get("evidence_refs"):
        return False
    checked = {str(check.get("signal") or "") for check in checks}
    if not checked:
        return False
    issue_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", json.dumps(context.get("issue") or {}))
        if len(token) > 2
    }
    if issue_tokens and not any(
        len(issue_tokens & {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+", str(check.get("message") or ""))
            if len(token) > 2
        }) >= 2
        for check in checks
    ):
        return False
    timing = spec.get("timing") or {}
    structural = {timing.get("clock_port"), timing.get("reset_port"), ""}
    driven = {
        re.sub(r"\[.*$", "", str(name))
        for step in steps if isinstance(step, dict)
        for name in (step.get("drive") or {})
    }
    issue_text = json.dumps(context.get("issue") or {}).lower()
    if not (driven - structural) and "reset" not in issue_text:
        return False
    # Reject unknown four-state literals, but do not mistake letters in a
    # package-qualified enum (for example ibex_pkg::OP_B_IMM) for X/Z bits.
    return not any(_contains_unknown_literal(check.get("expected")) for check in checks)


def _contains_unknown_literal(value: Any) -> bool:
    text = str(value or "").replace("_", "")
    match = re.fullmatch(r"\d+'[sS]?[bBoOdDhH]([0-9a-fA-FxXzZ?]+)", text)
    if match:
        return bool(re.search(r"[xXzZ?]", match.group(1)))
    return text.lower() in {"'x", "'z"}


def _base_feedback(result: QualificationResult, spec: dict[str, Any]) -> dict[str, Any]:
    logs = list(result.runs[:1]) if result.runs else ([result.build] if result.build else [])
    compact_logs = []
    for entry in logs:
        marker_lines = [
            line for line in entry.stdout.splitlines() if line.startswith("KGVERI_MINITB_")
        ]
        compact_logs.append({
            "phase": entry.phase,
            "returncode": entry.returncode,
            "stdout": "\n".join(marker_lines)[-6000:],
            "stderr": entry.stderr[-3000:] if entry.returncode != 0 else "",
        })
    schedule = []
    cycle = 0
    for index, step in enumerate(spec.get("steps") or []):
        cycle += int(step.get("after_cycles") or 0)
        schedule.append({
            "step": index,
            "start_cycle": cycle,
            "drive": step.get("drive") or {},
        })
        cycle += int(step.get("hold_cycles") or 0)
    return {
        "status": result.status,
        "reason": result.reason,
        "step_schedule_after_reset": schedule,
        "logs": compact_logs,
    }


def _retime_unique_stable_transitions(
    spec: dict[str, Any], result: QualificationResult
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align a missed stable check with an unambiguous stimulus-boundary transition.

    This is deliberately conservative: it runs only when the buggy base passed,
    uses the first local execution trace, and changes a check only when its signal
    has exactly one adjacent-cycle transition at a cycle where a stimulus step
    begins. No expected value or functional oracle is synthesized here.
    """
    if result.status != "BASE_PASS" or not result.runs:
        return spec, []

    samples: dict[str, list[tuple[int, str]]] = {}
    pattern = re.compile(
        r"^KGVERI_MINITB_OBSERVE signal=(\S+) cycle=(\d+) value=(\S+)$"
    )
    for line in result.runs[0].stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        signal, cycle_text, value = match.groups()
        if re.search(r"[xXzZ?]", value):
            continue
        samples.setdefault(signal, []).append((int(cycle_text), value.lower()))

    step_start_cycles: set[int] = set()
    cycle = 0
    for step in spec.get("steps") or []:
        cycle += int(step.get("after_cycles") or 0)
        if step.get("drive"):
            step_start_cycles.add(cycle)
        cycle += int(step.get("hold_cycles") or 0)

    corrected = json.loads(json.dumps(spec))
    corrections: list[dict[str, Any]] = []
    for check in corrected.get("checks") or []:
        if check.get("op") != "stable":
            continue
        trace = sorted(samples.get(str(check.get("signal"))) or [])
        transitions = [
            current_cycle
            for (previous_cycle, previous_value), (current_cycle, current_value)
            in zip(trace, trace[1:])
            if current_cycle == previous_cycle + 1
            and current_cycle in step_start_cycles
            and current_value != previous_value
        ]
        if len(transitions) != 1:
            continue
        old_cycle = int(check.get("at_or_after_cycle") or 0)
        new_cycle = transitions[0]
        if old_cycle == new_cycle:
            continue
        check["at_or_after_cycle"] = new_cycle
        corrections.append({
            "check_id": check.get("check_id"),
            "signal": check.get("signal"),
            "old_cycle": old_cycle,
            "new_cycle": new_cycle,
            "basis": "unique_observed_transition_at_stimulus_boundary",
        })
    return corrected, corrections


def _minimize_to_reproducing_checks(
    spec: dict[str, Any], result: QualificationResult
) -> tuple[dict[str, Any], list[str]]:
    """Keep only checks that actually reproduce the qualified base failure.

    MiniTB is a minimal reproducer, not an exhaustive issue-level regression.
    Later checks cannot be qualified when an earlier check terminates the run,
    so retaining them would allow unexecuted assertions to reject a patch.
    """
    checks = spec.get("checks") or []
    if result.status != "QUALIFIED_MINITB" or len(checks) <= 1 or not result.runs:
        return spec, []
    failed_ids = {
        match.group(1)
        for line in result.runs[0].stdout.splitlines()
        if (match := re.match(r"^KGVERI_MINITB_FAIL check=(\S+)", line.strip()))
    }
    if not failed_ids:
        return spec, []
    kept = [check for check in checks if check.get("check_id") in failed_ids]
    if not kept or len(kept) == len(checks):
        return spec, []
    minimized = json.loads(json.dumps(spec))
    minimized["checks"] = json.loads(json.dumps(kept))
    removed = [
        str(check.get("check_id"))
        for check in checks
        if check.get("check_id") not in failed_ids
    ]
    return minimized, removed


def _run_smoke(
    case, context, base_repo: Path, case_output: Path, timeout: int, writer, target=None
) -> QualificationResult:
    target = target or context["selected_target"]
    spec = build_smoke_spec(case.instance_id, target)
    renderer = TemplateRenderer(COMPONENT_ROOT / "templates")
    fill = renderer.expected_fill(spec, target)
    generated = case_output / "smoke"
    if generated.exists():
        shutil.rmtree(generated)
    renderer.render(spec, target, fill, generated)
    resolution = RTLSourceResolver(
        base_repo, repo=case.repo, target_file=target["file"]
    ).resolve(target["file"], target["module"])
    write_json(generated / "source_resolution.json", resolution.to_dict())
    runner = MiniTBDockerRunner(case, base_repo, case_output, timeout=timeout)
    runner.prepare_run_script(generated, resolution)
    result = QualificationEngine(runner).qualify(
        generated, case_output / "smoke_build", spec, expected_failure=False, repeat_count=1
    )
    save_run_logs(case_output / "logs", result, prefix="smoke_")
    write_json(case_output / "qualification.json", result.to_dict())
    writer({"stage": "smoke_fallback", "status": result.status, "reason": result.reason})
    return result


def _replay(case, base_repo: Path, case_output: Path, timeout: int) -> QualificationResult:
    frozen = case_output / "frozen"
    _verify_frozen(frozen)
    spec = json.loads((frozen / "behavior_spec.json").read_text(encoding="utf-8"))
    resolution_data = json.loads((frozen / "source_resolution.json").read_text(encoding="utf-8"))
    resolution = _resolution_from_dict(resolution_data)
    runner = MiniTBDockerRunner(case, base_repo, case_output, timeout=timeout)
    result = QualificationEngine(runner).qualify(
        frozen, case_output / "replay_build", spec, expected_failure=True, repeat_count=2
    )
    save_run_logs(case_output / "logs", result, prefix="replay_")
    write_json(case_output / "qualification_replay.json", result.to_dict())
    return result


def _offline_evaluate(case, base_repo: Path, case_output: Path, args: argparse.Namespace) -> QualificationResult:
    frozen = case_output / "frozen"
    _verify_frozen(frozen)
    patch = args.patch.resolve()
    if not patch.is_file() or patch.stat().st_size == 0:
        raise ValueError(f"offline patch is missing or empty: {patch}")
    offline_repo = case_output / "offline_repo"
    if offline_repo.exists():
        shutil.rmtree(offline_repo)
    clone = subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(base_repo), str(offline_repo)],
        capture_output=True,
        text=True,
    )
    if clone.returncode != 0:
        raise ValueError(f"failed to create offline repository: {clone.stderr.strip()}")
    subprocess.run(
        ["git", "-C", str(offline_repo), "checkout", "--quiet", "--detach", case.base_commit],
        check=True,
    )
    apply = subprocess.run(
        ["git", "-C", str(offline_repo), "apply", "--whitespace=nowarn", str(patch)],
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        # HWE-Bench patches occasionally retain the original blob ids while the
        # packaged base checkout contains a nearby commit. Let Git reconstruct
        # the original blobs and perform a three-way application in that case.
        apply_3way = subprocess.run(
            ["git", "-C", str(offline_repo), "apply", "--3way", "--whitespace=nowarn", str(patch)],
            capture_output=True,
            text=True,
        )
        if apply_3way.returncode != 0:
            raise ValueError(
                "offline patch does not apply: "
                f"direct={apply.stderr.strip()}; three_way={apply_3way.stderr.strip()}"
            )

    spec = json.loads((frozen / "behavior_spec.json").read_text(encoding="utf-8"))
    resolution = _resolution_from_dict(
        json.loads((frozen / "source_resolution.json").read_text(encoding="utf-8"))
    )
    runner = MiniTBDockerRunner(case, offline_repo, case_output, timeout=args.timeout)
    result = QualificationEngine(runner).qualify(
        frozen, case_output / "offline_build", spec, expected_failure=False, repeat_count=2
    )
    interface_adapter_used = False
    if _is_interface_only_build_failure(result):
        # A correct patch may intentionally remove or rename unrelated DUT
        # ports. Keep the frozen BehaviorSpec, stimulus, checks, and oracle,
        # but deterministically regenerate only structural bindings against
        # the patched module interface.
        save_run_logs(case_output / "logs", result, prefix="offline_initial_")
        context = ContextBuilder(offline_repo, case).build()
        target = _target_for_spec(context, spec)
        base_context = ContextBuilder(base_repo, case).build()
        base_target = _target_for_spec(base_context, spec)
        target, interface_port_renames = _adapt_target_for_patch_port_renames(
            base_target,
            target,
            patch.read_text(encoding="utf-8", errors="replace"),
            spec["target"]["file"],
            spec,
        )
        adapter = case_output / "offline_adapter"
        if adapter.exists():
            shutil.rmtree(adapter)
        renderer = TemplateRenderer(COMPONENT_ROOT / "templates")
        renderer.render(spec, target, renderer.expected_fill(spec, target), adapter)
        resolution = RTLSourceResolver(
            offline_repo, repo=case.repo, target_file=target["file"]
        ).resolve(target["file"], target["module"])
        write_json(adapter / "source_resolution.json", resolution.to_dict())
        runner.prepare_run_script(adapter, resolution)
        result = QualificationEngine(runner).qualify(
            adapter,
            case_output / "offline_adapter_build",
            spec,
            expected_failure=False,
            repeat_count=2,
        )
        interface_adapter_used = True
        save_run_logs(case_output / "logs", result, prefix="offline_adapter_")
    else:
        save_run_logs(case_output / "logs", result, prefix="offline_")
        interface_port_renames = {}
    report = result.to_dict()
    report["patch_path"] = str(patch)
    report["patch_sha256"] = _sha256(patch)
    report["frozen_assets_unchanged"] = True
    report["interface_adapter_used"] = interface_adapter_used
    report["interface_port_renames"] = interface_port_renames
    report["behavior_spec_unchanged"] = True
    write_json(case_output / "offline_evaluation.json", report)
    return result


def _is_interface_only_build_failure(result: QualificationResult) -> bool:
    if result.status != "TB_BUILD_FAIL" or result.build is None:
        return False
    errors = [
        line.strip()
        for line in result.build.stderr.splitlines()
        if line.lstrip().startswith("%Error") and "Exiting due" not in line
    ]
    return bool(errors) and all("Pin not found:" in line for line in errors)


def _adapt_target_for_patch_port_renames(
    base_target: dict[str, Any],
    patched_target: dict[str, Any],
    patch_text: str,
    target_file: str,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Alias simple patch-declared DUT port renames without changing test semantics.

    Only one-to-one ANSI declarations removed and added in the same diff hunk
    are eligible. The old port must be referenced by the frozen specification,
    both ports must have the same direction, and the old/new names must be
    absent from the patched/base interfaces respectively.
    """
    base_ports = {port["name"]: port for port in base_target.get("ports") or []}
    patched_ports = {port["name"]: port for port in patched_target.get("ports") or []}
    referenced = _spec_port_names(spec)
    renames: dict[str, str] = {}
    removed: list[tuple[str, str]] = []
    added: list[tuple[str, str]] = []
    in_target = False

    def flush_hunk() -> None:
        nonlocal removed, added
        old = [item for item in removed if item[1] in base_ports and item[1] not in patched_ports]
        new = [item for item in added if item[1] in patched_ports and item[1] not in base_ports]
        if len(old) == len(new):
            for (old_direction, old_name), (new_direction, new_name) in zip(old, new):
                if old_name in referenced and old_direction == new_direction:
                    renames[old_name] = new_name
        removed = []
        added = []

    normalized_target = target_file.removeprefix("./")
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_hunk()
            in_target = False
        elif line.startswith("+++ "):
            candidate = line[4:].strip().removeprefix("b/").removeprefix("./")
            in_target = candidate == normalized_target
        elif line.startswith("@@"):
            flush_hunk()
        elif in_target and line.startswith("-") and not line.startswith("---"):
            declaration = _diff_port_declaration(line[1:])
            if declaration:
                removed.append(declaration)
        elif in_target and line.startswith("+") and not line.startswith("+++"):
            declaration = _diff_port_declaration(line[1:])
            if declaration:
                added.append(declaration)
    flush_hunk()

    if not renames:
        return patched_target, {}
    old_by_new = {new: old for old, new in renames.items()}
    adapted_ports = []
    for port in patched_target.get("ports") or []:
        physical_name = port["name"]
        old_name = old_by_new.get(physical_name)
        if old_name is None:
            adapted_ports.append(port)
            continue
        adapted = dict(port)
        adapted["name"] = old_name
        adapted["physical_name"] = physical_name
        adapted["declaration"] = re.sub(
            rf"\b{re.escape(physical_name)}\b",
            old_name,
            str(port.get("declaration") or ""),
        )
        adapted_ports.append(adapted)
    adapted_target = dict(patched_target)
    adapted_target["ports"] = adapted_ports
    return adapted_target, renames


def _diff_port_declaration(line: str) -> tuple[str, str] | None:
    content = re.sub(r"//.*$", "", line).strip()
    direction_match = re.match(r"(input|output|inout)\b", content)
    if not direction_match:
        return None
    name_match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*(?:=[^,]+)?\s*,?\s*$", content
    )
    return (direction_match.group(1), name_match.group(1)) if name_match else None


def _spec_port_names(spec: dict[str, Any]) -> set[str]:
    names = set(spec.get("defaults") or {})
    for step in spec.get("steps") or []:
        names.update(re.split(r"[\[.]", name, maxsplit=1)[0] for name in (step.get("drive") or {}))
    names.update(re.split(r"[\[.]", check["signal"], maxsplit=1)[0] for check in spec.get("checks") or [])
    names.update(re.split(r"[\[.]", name, maxsplit=1)[0] for name in (spec.get("observations") or []))
    timing = spec.get("timing") or {}
    names.update(filter(None, (timing.get("clock_port"), timing.get("reset_port"))))
    return names


def _build_context(case, base_repo: Path, case_output: Path, kg_location: Path | None) -> dict[str, Any]:
    builder = ContextBuilder(base_repo, case, kg_location)
    return builder.write(case_output / "context_summary.json")


def _target_for_spec(context: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    for target in context["candidate_targets"]:
        if target["file"] == spec["target"]["file"] and target["module"] == spec["target"]["module"]:
            return target
    raise ValueError("BehaviorSpec target is not present in context candidates")


def _freeze(case_output: Path, generated: Path, spec: dict[str, Any], result: QualificationResult) -> None:
    frozen = case_output / "frozen"
    if frozen.exists():
        shutil.rmtree(frozen)
    frozen.mkdir(parents=True)
    for name in (
        "minitb_top.sv",
        "sim_main.cpp",
        "sim_main_iverilog.sv",
        "run.sh",
        "source_resolution.json",
        "harness_fill.json",
    ):
        shutil.copy2(generated / name, frozen / name)
    write_json(frozen / "behavior_spec.json", spec)
    write_json(frozen / "qualification_at_freeze.json", result.to_dict())
    hashes = {
        path.name: _sha256(path)
        for path in sorted(frozen.iterdir())
        if path.is_file() and path.name != "sha256.json"
    }
    write_json(frozen / "sha256.json", hashes)
    for path in frozen.iterdir():
        if path.is_file():
            path.chmod(0o444 if path.name != "run.sh" else 0o555)


def _verify_frozen(frozen: Path) -> None:
    hashes_path = frozen / "sha256.json"
    if not hashes_path.is_file():
        raise ValueError("frozen Mini-TB is missing sha256.json")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        path = frozen / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen Mini-TB hash mismatch: {name}")


def _resolution_from_dict(value: dict[str, Any]):
    return SourceResolution(
        sources=list(value.get("sources") or []),
        include_dirs=list(value.get("include_dirs") or []),
        defines=list(value.get("defines") or []),
        packages=list(value.get("packages") or []),
        modules=list(value.get("modules") or []),
        unresolved=list(value.get("unresolved") or []),
        manifests_consulted=list(value.get("manifests_consulted") or []),
    )


def _attempt_writer(path: Path):
    def write(record: dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **record}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return write


def _invalidate_previous_generation(case_output: Path) -> None:
    """Prevent a failed regeneration from leaving a stale frozen success visible."""
    # Docker build trees can contain root-owned Verilator outputs. They are
    # cleaned by run.sh inside the container before each build, so host-side
    # invalidation only removes user-owned generated/frozen assets.
    for name in ("generated", "smoke", "frozen"):
        path = case_output / name
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "attempts.jsonl",
        "behavior_spec.json",
        "qualification.json",
        "qualification_replay.json",
        "offline_evaluation.json",
        "result_summary.json",
        "functional_qualification.json",
        "preflight.json",
        "initial_qualification.json",
        "behavior_spec_initial.json",
    ):
        (case_output / name).unlink(missing_ok=True)


def _finalize_summary(
    case_output: Path,
    result: QualificationResult,
    started_at: float,
    path: str,
) -> None:
    write_json(
        case_output / "result_summary.json",
        {
            "status": result.status,
            "reason": result.reason,
            "execution_path": path,
            "repeatable": result.repeatable,
            "wall_time_seconds": round(time.monotonic() - started_at, 3),
            "frozen": (case_output / "frozen" / "sha256.json").is_file(),
        },
    )


def _inspection_view(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": context["instance_id"],
        "repo": context["repo"],
        "kg_available": context["kg_available"],
        "candidate_targets": [
            {
                "rank": item["rank"],
                "file": item["file"],
                "module": item["module"],
                "ports": len(item["ports"]),
                "reason": item["reason"],
            }
            for item in context["candidate_targets"]
        ],
        "adjacent_assets": [item["path"] for item in context["adjacent_assets"]],
        "budgets": context["budgets"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
