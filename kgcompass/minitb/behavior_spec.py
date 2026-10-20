from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .evidence_slice import evidence_review
from .models import (
    BudgetExhausted,
    FORBIDDEN_INPUT_KEYS,
    RetryBudget,
    SpecValidationError,
    VisibleCase,
    write_json,
)
from .template_catalog import (
    TEMPLATE_BY_CLASS,
    classify_template,
    instruction_field_requirements,
    issue_facts,
    rank_observation_ports,
)


SPEC_KEYS = frozenset(
    {
        "schema_version",
        "instance_id",
        "bug_class",
        "target",
        "timing",
        "defaults",
        "steps",
        "checks",
        "observations",
        "oracle",
        "template_id",
        "assumptions",
    }
)
TARGET_KEYS = frozenset({"file", "module", "parameters"})
TIMING_KEYS = frozenset(
    {"clock_port", "reset_port", "reset_active", "reset_cycles", "timeout_cycles"}
)
TIMING_OPTIONAL_KEYS = frozenset({"sample_phase"})
INTENT_KEYS = frozenset(
    {
        "bug_class",
        "parameters",
        "sample_phase",
        "steps",
        "checks",
        "observations",
        "oracle_confidence",
        "assumptions",
    }
)
SV_VALUE = re.compile(
    r"^(?:\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|'[01xXzZ]|-?\d+|"
    r"[A-Za-z_][A-Za-z0-9_$]*(?:::[A-Za-z_][A-Za-z0-9_$]*)?)$"
)
PORT_REFERENCE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_$]*)(?:(?:\[\d+(?::\d+)?\])|(?:\.[A-Za-z_][A-Za-z0-9_$]*))*$"
)


def compact_generation_context(context: dict[str, Any]) -> dict[str, Any]:
    """Expose one full interface; alternative modules are selection metadata.

    A bounded target-selection response obtains a different full interface on
    the next call, instead of mixing several unrelated interfaces in one prompt.
    """
    issue = context.get("issue") or {}
    # Short RTL terms such as CSR, PMP and ALU are informative source anchors.
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(issue).lower()))
    words -= {
        "and", "are", "bug", "can", "for", "from", "has", "into", "issue",
        "logic", "must", "not", "output", "should", "that", "the", "their",
        "there", "these", "this", "when", "where", "which", "with",
    }
    candidates = []
    selected = context.get("selected_target")
    candidate_targets = [selected] if isinstance(selected, dict) else list(
        context.get("candidate_targets") or []
    )[:1]
    selected_evidence: dict[str, Any] = {}
    for target in candidate_targets:
        item = {key: target[key] for key in ("file", "module", "ports", "parameters", "imports", "includes", "rank", "reason") if key in target}
        selected_evidence = target.get("minitb_evidence") or context.get("minitb_evidence") or {}
        source = target.get("source", "")
        if len(source) <= 8000:
            item["source"] = source
        else:
            lines = source.splitlines()
            selected = set(range(min(15, len(lines))))
            scores = [(len(words & set(re.findall(r"[a-z_][a-z0-9_]*", line.lower()))), i) for i, line in enumerate(lines)]
            used = sum(len(lines[i]) + 1 for i in selected)
            for score, index in sorted(scores, key=lambda pair: (-pair[0], pair[1])):
                if not score:
                    break
                window = set(range(max(0, index - 4), min(len(lines), index + 6))) - selected
                cost = sum(len(lines[i]) + 1 for i in window)
                if used + cost <= 7000:
                    selected.update(window)
                    used += cost
            excerpt = []
            last = -1
            for index in sorted(selected):
                if index != last + 1:
                    excerpt.append("// ... omitted ...")
                excerpt.append(lines[index])
                last = index
            item["source"] = "\n".join(excerpt)
            item["source_is_excerpt"] = True
        candidates.append(item)
    if selected_evidence:
        selected_evidence = json.loads(json.dumps(selected_evidence))
        spans = selected_evidence.get("support_spans") or []
        selected_evidence["support_spans"] = spans[:6]
        for span in selected_evidence["support_spans"]:
            if isinstance(span, dict):
                span["source"] = str(span.get("source") or "")[:600]
    contract = _generation_contract(candidates[0], issue, selected_evidence) if candidates else {}

    def option(target: dict[str, Any]) -> dict[str, Any]:
        evidence = target.get("minitb_evidence") or {}
        inputs = [p["name"] for p in target.get("ports", []) if p.get("direction") == "input"]
        outputs = [p["name"] for p in target.get("ports", []) if p.get("direction") == "output"]
        grounded_inputs = evidence.get("controllable_inputs", [])
        grounded_outputs = evidence.get("observable_outputs", [])
        return {
            **{key: target[key] for key in ("file", "module", "rank", "reason") if key in target},
            "inputs": list(dict.fromkeys([*grounded_inputs, *inputs]))[:16],
            "outputs": list(dict.fromkeys([*grounded_outputs, *outputs]))[:16],
            "issue_grounded_inputs": grounded_inputs[:8],
            "issue_grounded_outputs": grounded_outputs[:8],
        }
    return {
        "instance_id": context.get("instance_id"), "repo": context.get("repo"), "issue": issue,
        "candidate_targets": candidates,
        "generation_contract": contract,
        "minitb_evidence": selected_evidence,
        "package_context": _compact_package_context(
            str(context.get("package_context") or ""), issue, candidates[0] if candidates else {}
        ),
        "issue_facts": issue_facts(issue),
        "target_options": [option(target) for target in (context.get("candidate_targets") or [])[:12]],
        "adjacent_assets": [{"path": asset.get("path"), "excerpt": asset.get("excerpt", "")[:1200]} for asset in (context.get("adjacent_assets") or [])[:2]],
    }


def _compact_package_context(
    text: str, issue: dict[str, Any], target: dict[str, Any], limit: int = 12_000
) -> str:
    if len(text) <= limit:
        return text
    terms = {
        token.lower()
        for token in re.findall(
            r"[A-Za-z_][A-Za-z0-9_]{3,}", json.dumps(issue, ensure_ascii=True)
        )
    }
    for port in target.get("ports") or []:
        declaration = str(port.get("declaration") or "")
        terms.update(
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:_t|_e)\b", declaration)
        )
    lines = text.splitlines()
    selected: set[int] = set(range(min(20, len(lines))))
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
    result: list[str] = []
    used = 0
    previous = -2
    for index in sorted(selected):
        line = lines[index]
        addition = ("// ... omitted ...\n" if index != previous + 1 else "") + line + "\n"
        if used + len(addition) > limit:
            break
        result.append(addition)
        used += len(addition)
        previous = index
    return "".join(result)


def _generation_contract(
    target: dict[str, Any], issue: dict[str, Any], evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    ports = [port for port in target.get("ports") or [] if isinstance(port, dict)]
    inputs = [str(port.get("name")) for port in ports if port.get("direction") == "input"]
    outputs = [str(port.get("name")) for port in ports if port.get("direction") == "output"]
    clock = next((name for name in inputs if _looks_clock(name)), "")
    reset = next((name for name in inputs if _looks_reset(name)), "")
    suggested = classify_template(issue, target)
    evidence = evidence or {}
    return {
        "allowed_input_ports": inputs,
        "allowed_output_ports": outputs,
        "clock_port": clock,
        "reset_port": reset,
        "suggested_bug_class": suggested,
        "ranked_observation_ports": rank_observation_ports(issue, target)[:8],
        "evidence_driven_inputs": evidence.get("controllable_inputs") or [],
        "evidence_observable_outputs": evidence.get("observable_outputs") or [],
        "required_guard_signals": evidence.get("required_guard_signals") or [],
        "event_port_candidates": [
            name for name in inputs + outputs
            if re.search(
                r"(?:valid|ready|req|gnt|grant|ret|retire|commit|done|save|restore|cause|err)",
                name,
                re.I,
            )
        ][:12],
    }


def materialize_behavior_intent(
    value: dict[str, Any],
    context: dict[str, Any],
    previous_spec: dict[str, Any] | None = None,
    *,
    enforce_suggested_class: bool = False,
) -> dict[str, Any]:
    """Turn the model's small semantic intent into a complete BehaviorSpec.

    The LLM chooses only bug-specific stimulus and observations. Repository
    identity, target identity, interface defaults, reset convention, timeout,
    template selection, and issue provenance are filled deterministically.
    Full BehaviorSpec responses remain accepted for backward compatibility.
    """
    if set(value) == set(SPEC_KEYS):
        return value
    requested_target = value.get("target")
    if requested_target is not None:
        if not isinstance(requested_target, dict) or set(requested_target) != {"file", "module"}:
            raise SpecValidationError(["intent target requires file and module only"])
        match = next((item for item in (context.get("candidate_targets") or [])[:12]
                      if item.get("file") == requested_target["file"]
                      and item.get("module") == requested_target["module"]), None)
        if match is None:
            raise SpecValidationError(["intent target is not a supplied target option"])
        if previous_spec and requested_target != {key: previous_spec["target"][key] for key in ("file", "module")}:
            raise SpecValidationError(["execution correction cannot change target"])
        context = {**context, "selected_target": match}
        value = {key: item for key, item in value.items() if key != "target"}
    if set(value) != set(INTENT_KEYS):
        extra = sorted(set(value) - set(INTENT_KEYS))
        missing = sorted(set(INTENT_KEYS) - set(value))
        raise SpecValidationError(
            [f"model intent keys mismatch: extra={extra}, missing={missing}"]
        )

    if previous_spec is not None:
        previous_target = previous_spec.get("target") or {}
        matching_target = next(
            (
                target
                for target in context.get("candidate_targets") or []
                if target.get("file") == previous_target.get("file")
                and target.get("module") == previous_target.get("module")
            ),
            None,
        )
        if matching_target is not None:
            context = {**context, "selected_target": matching_target}
    compact = compact_generation_context(context)
    targets = compact.get("candidate_targets") or []
    if not targets:
        raise SpecValidationError(["generation context has no localized target"])
    target = targets[0]
    ports = [port for port in target.get("ports") or [] if isinstance(port, dict)]
    inputs = [port for port in ports if port.get("direction") == "input"]
    outputs = {str(port.get("name")) for port in ports if port.get("direction") == "output"}

    bug_class = str(value.get("bug_class") or "")
    suggested_class = str((compact.get("generation_contract") or {}).get("suggested_bug_class") or "")
    if enforce_suggested_class and bug_class != "unsupported" and suggested_class in TEMPLATE_BY_CLASS:
        bug_class = suggested_class
    if previous_spec is not None:
        bug_class = str(previous_spec.get("bug_class") or bug_class)
    clock = next((str(port.get("name")) for port in inputs if _looks_clock(str(port.get("name")))), "")
    reset = next((str(port.get("name")) for port in inputs if _looks_reset(str(port.get("name")))), "")
    if bug_class == "combinational":
        timing_clock = ""
        timing_reset = ""
        reset_active = "none"
        reset_cycles = 0
    elif bug_class in {"clocked_state", "protocol", "event_trace"}:
        timing_clock = clock
        timing_reset = reset
        reset_active = "low" if reset.lower().endswith(("_ni", "_n")) else ("high" if reset else "none")
        reset_cycles = 2 if reset else 0
    else:
        timing_clock = ""
        timing_reset = ""
        reset_active = "none"
        reset_cycles = 0

    defaults: dict[str, str] = {}
    for port in inputs:
        name = str(port.get("name"))
        if name == timing_reset:
            defaults[name] = "1'b1" if reset_active == "low" else "1'b0"
        else:
            defaults[name] = "'0"

    steps = _normalize_intent_steps(value.get("steps"), inputs)
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("drive"), dict):
            # Clock and reset execution belongs to the fixed template. Silently
            # dropping these structural fields is safe and leaves the model's
            # bug-specific trigger untouched.
            step["drive"].pop(clock, None)
            step["drive"].pop(reset, None)
    steps = _ground_explicit_instruction_fields(steps, context.get("issue") or {}, inputs)
    checks = json.loads(json.dumps(value.get("checks"))) if isinstance(value.get("checks"), list) else []
    observations = value.get("observations") if isinstance(value.get("observations"), list) else []
    observations = list(dict.fromkeys(
        [str(item) for item in observations
         if re.split(r"[\[.]", str(item), maxsplit=1)[0] in outputs]
        + [str(check.get("signal")) for check in checks if isinstance(check, dict)
           and re.split(r"[\[.]", str(check.get("signal")), maxsplit=1)[0] in outputs]
    ))

    schedule_end = reset_cycles
    for step in steps:
        try:
            schedule_end += int(step.get("after_cycles", 0)) + int(step.get("hold_cycles", 0))
        except (AttributeError, TypeError, ValueError):
            pass
    check_end = 0
    for check in checks:
        if isinstance(check, dict):
            try:
                check_end = max(check_end, int(check.get("at_or_after_cycle", 0)))
            except (TypeError, ValueError):
                pass
    timeout = min(200, max(10, schedule_end + 4, check_end + 4))
    if any(isinstance(check, dict) and "when" in check for check in checks):
        timeout = 200

    parameters = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
    if previous_spec is not None:
        oracle = previous_spec.get("oracle")
    else:
        issue = context.get("issue") or {}
        evidence = str(issue.get("problem_statement") or issue.get("title") or "")
        oracle = {
            "confidence": value.get("oracle_confidence"),
            "evidence_refs": [{"source": "problem_statement", "text": evidence}],
        }

    return {
        "schema_version": "kgveri-minitb-v1",
        "instance_id": context.get("instance_id"),
        "bug_class": bug_class,
        "target": {
            "file": target.get("file"),
            "module": target.get("module"),
            "parameters": parameters,
        },
        "timing": {
            "clock_port": timing_clock,
            "reset_port": timing_reset,
            "reset_active": reset_active,
            "reset_cycles": reset_cycles,
            "timeout_cycles": timeout,
            "sample_phase": value.get("sample_phase"),
        },
        "defaults": defaults,
        "steps": steps,
        "checks": checks,
        "observations": observations,
        "oracle": oracle,
        "template_id": TEMPLATE_BY_CLASS.get(bug_class, "MINIMAL_MODULE_SMOKE"),
        "assumptions": value.get("assumptions") if isinstance(value.get("assumptions"), list) else [],
    }


def infer_intent_target(
    intent: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any] | None:
    """Bind a compact intent to the unique candidate exposing all used ports."""
    if not isinstance(intent, dict) or "target" in intent:
        return None
    driven = {
        re.split(r"[\[.]", str(name), maxsplit=1)[0]
        for step in (intent.get("steps") or [])
        if isinstance(step, dict)
        for name in (step.get("drive") or {})
    }
    checked = {
        re.split(r"[\[.]", str(check.get("signal") or ""), maxsplit=1)[0]
        for check in (intent.get("checks") or [])
        if isinstance(check, dict)
    }
    observed = {
        re.split(r"[\[.]", str(name), maxsplit=1)[0]
        for name in (intent.get("observations") or [])
    }
    if not driven or not checked:
        return None
    matches: list[dict[str, Any]] = []
    for target in (context.get("candidate_targets") or [])[:12]:
        inputs = {
            str(port.get("name") or "")
            for port in (target.get("ports") or [])
            if port.get("direction") == "input"
        }
        outputs = {
            str(port.get("name") or "")
            for port in (target.get("ports") or [])
            if port.get("direction") == "output"
        }
        if driven <= inputs and checked <= outputs and observed <= outputs:
            matches.append(target)
    if len(matches) != 1:
        return None
    target = matches[0]
    revised = json.loads(json.dumps(intent))
    revised["target"] = {"file": target["file"], "module": target["module"]}
    return revised


def _normalize_intent_steps(value: Any, inputs: list[dict[str, Any]]) -> list[Any]:
    if not isinstance(value, list):
        return []
    widths = {
        str(port.get("name")): _port_width(
            str(port.get("width") or ""), str(port.get("declaration") or "")
        )
        for port in inputs
    }
    normalized = json.loads(json.dumps(value))
    for step in normalized:
        if not isinstance(step, dict) or not isinstance(step.get("drive"), dict):
            continue
        for selection, raw in list(step["drive"].items()):
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_$]*)(?:\[(\d+)(?::(\d+))?\])?", str(selection))
            port_name = match.group(1) if match else str(selection)
            width = widths.get(port_name)
            if match and match.group(2) is not None:
                high = int(match.group(2))
                low = int(match.group(3) or match.group(2))
                width = high - low + 1
            text = str(raw).strip()
            hex_match = re.fullmatch(r"0[xX]([0-9a-fA-F]+)", text)
            bin_match = re.fullmatch(r"0[bB]([01]+)", text)
            if hex_match and width:
                step["drive"][selection] = f"{width}'h{hex_match.group(1)}"
            elif bin_match and width:
                step["drive"][selection] = f"{width}'b{bin_match.group(1)}"
            elif text.lower() in {"true", "false"} and width == 1:
                step["drive"][selection] = "1'b1" if text.lower() == "true" else "1'b0"
    return normalized


def _ground_explicit_instruction_fields(
    steps: list[Any], issue: dict[str, Any], inputs: list[dict[str, Any]]
) -> list[Any]:
    """Overlay only instruction fields explicitly stated by the issue.

    This is schema grounding, not a case recipe: it neither supplies an opcode
    omitted by the issue nor changes an expected output.  The model must still
    choose a complete trigger and oracle.
    """
    requirements = instruction_field_requirements(issue)
    if not steps or not requirements:
        return steps
    instruction_inputs: list[str] = []
    for port in inputs:
        name = str(port.get("name") or "")
        width = _port_width(str(port.get("width") or ""), str(port.get("declaration") or ""))
        if width and width >= 16 and re.search(r"(?:^|_)instr(?:_|$)", name, re.IGNORECASE):
            instruction_inputs.append(name)
    if not instruction_inputs:
        return steps
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("drive"), dict):
            continue
        drive = step["drive"]
        input_name = next(
            (name for name in instruction_inputs if name in drive or any(
                str(selection).startswith(f"{name}[") for selection in drive
            )),
            None,
        )
        if input_name is None:
            continue
        for requirement in requirements:
            high, low = int(requirement["high"]), int(requirement["low"])
            selection = f"{input_name}[{high}:{low}]" if high != low else f"{input_name}[{high}]"
            raw = str(requirement["value"])
            if "'" not in raw:
                width = high - low + 1
                raw = f"1'b{raw}" if width == 1 and raw in {"0", "1"} else f"{width}'d{raw}"
            drive[selection] = raw
    return steps


class BehaviorSpecValidator:
    def __init__(self, case: VisibleCase, context: dict[str, Any], repo_root: Path):
        self.case = case
        self.context = context
        self.repo_root = repo_root
        visible_rtl = "\n".join(
            str(item.get("source") or "")
            for item in context.get("candidate_targets") or []
            if isinstance(item, dict)
        )
        visible_rtl += "\n" + "\n".join(
            str(item.get("excerpt") or "")
            for item in context.get("adjacent_assets") or []
            if isinstance(item, dict)
        )
        visible_rtl += "\n" + str(context.get("package_context") or "")
        self.visible_symbols = set(
            re.findall(r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_$]*)(?![A-Za-z0-9_$])", visible_rtl)
        )

    def validate(self, value: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(value, dict):
            raise SpecValidationError(["BehaviorSpec must be a JSON object"])
        self._exact_keys(value, SPEC_KEYS, "$", errors)
        self._scan_forbidden(value, "$", errors)
        if value.get("schema_version") not in {"kgveri-minitb-v1", "ibex-minitb-v1"}:
            errors.append("schema_version must be kgveri-minitb-v1")
        if value.get("instance_id") != self.case.instance_id:
            errors.append("instance_id does not match the selected visible case")

        bug_class = value.get("bug_class")
        if bug_class not in TEMPLATE_BY_CLASS:
            errors.append(f"unsupported bug_class: {bug_class!r}")
        expected_template = TEMPLATE_BY_CLASS.get(str(bug_class))
        if value.get("template_id") != expected_template:
            errors.append(
                f"template_id must be {expected_template!r} for bug_class {bug_class!r}"
            )

        target = value.get("target")
        if not isinstance(target, dict):
            errors.append("target must be an object")
            target = {}
        else:
            self._exact_keys(target, TARGET_KEYS, "$.target", errors)
        target_context = self._target_context(target, errors)
        ports = {
            item.get("name"): item
            for item in (target_context.get("ports") or [])
            if isinstance(item, dict) and item.get("name")
        }
        inputs = {name: port for name, port in ports.items() if port.get("direction") == "input"}
        outputs = {name: port for name, port in ports.items() if port.get("direction") == "output"}
        self._validate_parameters(target, target_context, errors)

        timing = value.get("timing")
        if not isinstance(timing, dict):
            errors.append("timing must be an object")
            timing = {}
        else:
            timing_keys = TIMING_KEYS | (TIMING_OPTIONAL_KEYS if "sample_phase" in timing else set())
            self._exact_keys(timing, timing_keys, "$.timing", errors)
        self._validate_timing(timing, inputs, bug_class, errors)

        defaults = value.get("defaults")
        if not isinstance(defaults, dict):
            errors.append("defaults must be an object")
            defaults = {}
        self._validate_drives(defaults, inputs, "$.defaults", errors)

        steps = value.get("steps")
        if not isinstance(steps, list):
            errors.append("steps must be a list")
            steps = []
        if len(steps) > 16:
            errors.append("steps exceeds the first-version limit of 16")
        for index, step in enumerate(steps):
            self._validate_step(step, inputs, index, errors)
            if isinstance(step, dict) and isinstance(step.get("drive"), dict):
                reserved = {
                    str(timing.get("clock_port") or ""),
                    str(timing.get("reset_port") or ""),
                } - {""}
                for selection in step["drive"]:
                    base = str(selection).split("[", 1)[0]
                    if base in reserved:
                        errors.append(
                            f"$.steps[{index}].drive.{selection} is managed by the template renderer"
                        )
        if bug_class in {"combinational", "protocol", "event_trace"} and not any(
            isinstance(step, dict) and isinstance(step.get("drive"), dict) and step["drive"]
            for step in steps
        ):
            errors.append(f"{bug_class} tests must drive at least one trigger input in steps")
        self._validate_explicit_instruction_fields(steps, inputs, errors)

        checks = value.get("checks")
        if not isinstance(checks, list) or (not checks and bug_class != "unsupported"):
            errors.append("at least one check is required")
            checks = []
        if len(checks) > 4:
            errors.append(
                "MiniTB permits at most 4 checks; keep the primary issue property and only "
                "directly necessary supporting properties"
            )
        if bug_class == "combinational" and len(steps) > 8:
            errors.append("COMB_DIRECTED permits at most 8 stimulus vectors")
        check_ids: set[str] = set()
        for index, check in enumerate(checks):
            self._validate_check(check, outputs, index, check_ids, errors)
            if isinstance(check, dict):
                if "when" in check:
                    condition = check["when"]
                    if not isinstance(condition, dict) or not 1 <= len(condition) <= 4:
                        errors.append(f"$.checks[{index}].when requires 1..4 port/value predicates")
                    else:
                        self._validate_drives(condition, ports, f"$.checks[{index}].when", errors)
                        if check.get("signal") in condition:
                            errors.append("event predicate cannot gate a check on its own checked signal")
                        if any(
                            not (match := PORT_REFERENCE.fullmatch(str(name)))
                            or match.group(1) not in ports
                            for name in condition
                        ):
                            errors.append("event predicates require valid target port references")
                        if any(re.search(r"'[sS]?[bBoOdDhH]?[0-9a-fA-F_]*[xXzZ?]", str(v)) for v in condition.values()):
                            errors.append("event predicates cannot use X/Z values")
                if "occurrence" in check:
                    occurrence = check["occurrence"]
                    if "when" not in check or type(occurrence) is not int or not 1 <= occurrence <= 200:
                        errors.append("occurrence requires when and an integer in [1, 200]")
                try:
                    check_cycle = int(check.get("at_or_after_cycle"))
                    timeout_cycle = int(timing.get("timeout_cycles"))
                    if check_cycle >= timeout_cycle:
                        errors.append(
                            f"$.checks[{index}].at_or_after_cycle must be less than timeout_cycles"
                        )
                except (TypeError, ValueError):
                    pass

        self._validate_schedule(bug_class, steps, checks, errors)
        if bug_class == "event_trace":
            event_checks = [
                check for check in checks
                if isinstance(check, dict) and "when" in check
            ]
            if not event_checks:
                errors.append("EVENT_TRACE requires at least one event-qualified check")

        observations = value.get("observations")
        if not isinstance(observations, list):
            errors.append("observations must be a list")
        else:
            for signal in observations:
                reference = PORT_REFERENCE.fullmatch(str(signal))
                if not reference or reference.group(1) not in outputs:
                    errors.append(f"observation {signal!r} is not a target output port")

        self._validate_oracle(value.get("oracle"), errors)
        assumptions = value.get("assumptions")
        if not isinstance(assumptions, list) or any(not isinstance(item, str) for item in assumptions):
            errors.append("assumptions must be a list of strings")

        if errors:
            raise SpecValidationError(errors)
        normalized = json.loads(json.dumps(value))
        normalized["schema_version"] = "kgveri-minitb-v1"
        return normalized

    def _target_context(self, target: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        candidates = self.context.get("candidate_targets") or []
        for candidate in candidates:
            if candidate.get("file") == target.get("file") and candidate.get("module") == target.get("module"):
                path = self.repo_root / str(target.get("file"))
                if not path.is_file():
                    errors.append("target.file does not exist in the base repository")
                return candidate
        errors.append("target file/module is not one of the validated context candidates")
        return {}

    def _validate_parameters(
        self, target: dict[str, Any], context: dict[str, Any], errors: list[str]
    ) -> None:
        values = target.get("parameters")
        if not isinstance(values, dict):
            errors.append("target.parameters must be an object")
            return
        available = context.get("parameters") or {}
        for name, value in values.items():
            if name not in available:
                errors.append(f"target parameter {name!r} is not declared by the module")
            if not _is_safe_sv_value(value):
                errors.append(f"target parameter {name!r} has an unsafe or invalid value")
            else:
                self._validate_symbolic_value(value, f"$.target.parameters.{name}", errors)

    def _validate_timing(
        self,
        timing: dict[str, Any],
        inputs: dict[str, dict[str, Any]],
        bug_class: Any,
        errors: list[str],
    ) -> None:
        clock = str(timing.get("clock_port") or "")
        reset = str(timing.get("reset_port") or "")
        reset_active = timing.get("reset_active")
        if clock and clock not in inputs:
            errors.append(f"clock_port {clock!r} is not an input port")
        elif clock and not _looks_clock(clock):
            errors.append(f"clock_port {clock!r} has no clock naming/source evidence")
        if reset and reset not in inputs:
            errors.append(f"reset_port {reset!r} is not an input port")
        elif reset and not _looks_reset(reset):
            errors.append(f"reset_port {reset!r} has no reset naming/source evidence")
        if reset_active not in {"low", "high", "none"}:
            errors.append("reset_active must be low, high, or none")
        if timing.get("sample_phase", "post_edge") not in {"pre_edge", "post_edge"}:
            errors.append("sample_phase must be pre_edge or post_edge")
        if reset_active == "none" and (reset or int(timing.get("reset_cycles") or 0) != 0):
            errors.append("reset_active=none requires empty reset_port and zero reset_cycles")
        if reset_active != "none" and not reset:
            errors.append("an active reset requires reset_port")
        if reset:
            inferred = "low" if reset.lower().endswith(("_ni", "_n")) else "high"
            if reset_active != inferred:
                errors.append(
                    f"reset_active={reset_active!r} conflicts with reset port naming evidence ({inferred})"
                )
        if bug_class in {"clocked_state", "protocol", "event_trace"} and not clock:
            errors.append(f"{bug_class} requires a validated clock_port")
        if bug_class == "combinational" and (clock or reset):
            errors.append("combinational tests must not declare clock/reset timing; use a clocked template")
        for key, lower, upper in (
            ("reset_cycles", 0, 50),
            ("timeout_cycles", 1, 200),
        ):
            try:
                number = int(timing.get(key))
            except (TypeError, ValueError):
                errors.append(f"{key} must be an integer")
                continue
            if not lower <= number <= upper:
                errors.append(f"{key} must be in [{lower}, {upper}]")
        if reset_active in {"low", "high"}:
            try:
                if int(timing.get("reset_cycles")) < 1:
                    errors.append("an active reset requires at least one reset cycle")
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _validate_schedule(
        bug_class: Any,
        steps: list[Any],
        checks: list[Any],
        errors: list[str],
    ) -> None:
        if bug_class == "combinational":
            for index, step in enumerate(steps):
                if isinstance(step, dict):
                    try:
                        delayed = int(step.get("after_cycles") or 0) != 0
                    except (TypeError, ValueError):
                        delayed = False
                    if delayed:
                        errors.append(
                            f"$.steps[{index}].after_cycles must be 0 for a combinational test"
                        )
            return
        # Clocked checks may observe intermediate transactions. Per-step and
        # per-check bounds are validated separately; the last drive is not a
        # lower bound on when earlier responses can be checked.

    def _validate_drives(
        self,
        drives: dict[str, Any],
        inputs: dict[str, dict[str, Any]],
        path: str,
        errors: list[str],
    ) -> None:
        for name, value in drives.items():
            reference = PORT_REFERENCE.fullmatch(str(name))
            port_name = reference.group(1) if reference else ""
            if port_name not in inputs:
                errors.append(f"{path}.{name} is not a DUT input port")
                continue
            selected = str(name) != port_name
            if selected and path == "$.defaults":
                errors.append(f"{path}.{name}: use whole-port defaults and put selections in steps")
                continue
            if not _is_safe_sv_value(value):
                errors.append(f"{path}.{name} is not a legal scalar SV value")
                continue
            self._validate_symbolic_value(value, f"{path}.{name}", errors)
            literal_width = _literal_width(str(value))
            port = inputs[port_name]
            port_width = _port_width(str(port.get("width") or ""), str(port.get("declaration") or ""))
            packed_selection = re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_$]*\[(\d+)(?::(\d+))?\]", str(name)
            )
            if packed_selection and not re.search(rf"\b{re.escape(port_name)}\s*\[", str(port.get("declaration") or "")):
                high = int(packed_selection.group(1))
                low = int(packed_selection.group(2) or packed_selection.group(1))
                bounds = re.fullmatch(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", str(port.get("width") or ""))
                if high < low or (bounds and not (
                    min(map(int, bounds.groups())) <= low <= high <= max(map(int, bounds.groups()))
                )):
                    errors.append(f"{path}.{name}: invalid or out-of-range bit slice")
                port_width = high - low + 1
            elif selected:
                # The compiler checks struct members and unpacked array
                # elements; their leaf width is absent from this declaration.
                port_width = None
            if literal_width and port_width and literal_width > port_width:
                errors.append(
                    f"{path}.{name} literal width {literal_width} exceeds port width {port_width}"
                )

    def _validate_step(
        self,
        step: Any,
        inputs: dict[str, dict[str, Any]],
        index: int,
        errors: list[str],
    ) -> None:
        path = f"$.steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{path} must be an object")
            return
        self._exact_keys(step, {"after_cycles", "drive", "hold_cycles"}, path, errors)
        drive = step.get("drive")
        if not isinstance(drive, dict):
            errors.append(f"{path}.drive must be an object")
        else:
            self._validate_drives(drive, inputs, f"{path}.drive", errors)
        for key in ("after_cycles", "hold_cycles"):
            try:
                value = int(step.get(key))
            except (TypeError, ValueError):
                errors.append(f"{path}.{key} must be an integer")
                continue
            if value < 0 or value > 200:
                errors.append(f"{path}.{key} must be in [0, 200]")

    def _validate_explicit_instruction_fields(
        self,
        steps: list[Any],
        inputs: dict[str, dict[str, Any]],
        errors: list[str],
    ) -> None:
        requirements = instruction_field_requirements(self.case.to_dict())
        if not requirements:
            return
        instruction_ports = {
            name: _port_width(str(port.get("width") or ""), str(port.get("declaration") or ""))
            for name, port in inputs.items()
            if "instr" in name.lower()
        }
        instruction_ports = {name: width for name, width in instruction_ports.items() if width and width >= 16}
        if not instruction_ports:
            return
        for requirement in requirements:
            high, low = requirement["high"], requirement["low"]
            expected = _sv_integer(requirement["value"])
            if expected is None:
                continue
            satisfied = False
            seen_instruction_drive = False
            for step in steps:
                drives = step.get("drive") if isinstance(step, dict) else None
                if not isinstance(drives, dict):
                    continue
                for port, width in instruction_ports.items():
                    if port in drives:
                        seen_instruction_drive = True
                        whole = _sv_integer(drives[port])
                        if whole is not None and ((whole >> low) & ((1 << (high - low + 1)) - 1)) == expected:
                            satisfied = True
                    for selection, raw in drives.items():
                        match = re.fullmatch(rf"{re.escape(port)}\[(\d+)(?::(\d+))?\]", selection)
                        if match:
                            seen_instruction_drive = True
                            sel_high = int(match.group(1))
                            sel_low = int(match.group(2) or match.group(1))
                            if (sel_high, sel_low) == (high, low) and _sv_integer(raw) == expected:
                                satisfied = True
            if seen_instruction_drive and not satisfied:
                errors.append(
                    f"instruction stimulus does not satisfy issue field {requirement['name']}={requirement['value']}"
                )

    def _validate_check(
        self,
        check: Any,
        outputs: dict[str, dict[str, Any]],
        index: int,
        check_ids: set[str],
        errors: list[str],
    ) -> None:
        path = f"$.checks[{index}]"
        keys = {"check_id", "signal", "op", "expected", "at_or_after_cycle", "message"}
        if not isinstance(check, dict):
            errors.append(f"{path} must be an object")
            return
        self._exact_keys(check, keys | ({"when", "occurrence"} & set(check)), path, errors)
        check_id = str(check.get("check_id") or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", check_id):
            errors.append(f"{path}.check_id must be a simple identifier")
        elif check_id == "simple_identifier":
            errors.append(f"{path}.check_id is an unfilled schema placeholder")
        elif check_id in check_ids:
            errors.append(f"duplicate check_id {check_id!r}")
        check_ids.add(check_id)
        signal = str(check.get("signal") or "")
        reference = PORT_REFERENCE.fullmatch(signal)
        output_name = reference.group(1) if reference else ""
        if output_name not in outputs:
            errors.append(f"{path}.signal {signal!r} is not a DUT output port")
        if signal != output_name and check.get("op") == "stable":
            errors.append(f"{path}.op 'stable' requires a whole output port")
        if check.get("op") not in {"==", "!=", "stable", "asserted", "deasserted"}:
            errors.append(f"{path}.op is unsupported")
        expected = check.get("expected")
        if not _is_safe_sv_value(expected):
            errors.append(f"{path}.expected is not a legal scalar SV value")
        else:
            self._validate_symbolic_value(expected, f"{path}.expected", errors)
        if re.search(r"'[sS]?[bBoOdDhH]?[0-9a-fA-F_]*[xXzZ?]", str(expected)):
            errors.append(f"{path}.expected cannot contain X, Z, or don't-care bits")
        output = outputs.get(output_name) or {}
        literal_width = _literal_width(str(expected))
        output_width = _port_width(str(output.get("width") or ""), str(output.get("declaration") or ""))
        if signal != output_name:
            output_width = None
        if check.get("op") in {"asserted", "deasserted"} and output_width not in {None, 1}:
            errors.append(
                f"{path}.op {check.get('op')!r} is only valid for a one-bit output"
            )
        if literal_width and output_width and literal_width > output_width:
            errors.append(
                f"{path}.expected width {literal_width} exceeds output width {output_width}"
            )
        try:
            cycle = int(check.get("at_or_after_cycle"))
            if cycle < 0 or cycle > 200:
                errors.append(f"{path}.at_or_after_cycle must be in [0, 200]")
        except (TypeError, ValueError):
            errors.append(f"{path}.at_or_after_cycle must be an integer")
        message = str(check.get("message") or "").strip()
        if not message:
            errors.append(f"{path}.message is required")
        elif message.lower() == "expected behavior stated by visible evidence":
            errors.append(f"{path}.message is an unfilled schema placeholder")

    def _validate_symbolic_value(
        self, value: Any, path: str, errors: list[str]
    ) -> None:
        """Reject identifiers invented outside the visible RTL context."""
        text = str(value).strip()
        match = re.fullmatch(
            r"(?:([A-Za-z_][A-Za-z0-9_$]*)::)?([A-Za-z_][A-Za-z0-9_$]*)",
            text,
        )
        if match and match.group(2) not in self.visible_symbols:
            suggestions = get_close_matches(
                match.group(2), sorted(self.visible_symbols), n=3, cutoff=0.72
            )
            hint = f"; closest visible symbols={suggestions}" if suggestions else ""
            errors.append(
                f"{path} symbolic value {text!r} is not declared or referenced in visible RTL; "
                f"use a visible symbol or an explicit literal{hint}"
            )

    def _validate_oracle(self, oracle: Any, errors: list[str]) -> None:
        if not isinstance(oracle, dict):
            errors.append("oracle must be an object")
            return
        optional = {"notes"} if "notes" in oracle else set()
        self._exact_keys(oracle, {"confidence", "evidence_refs"} | optional, "$.oracle", errors)
        if optional and not isinstance(oracle["notes"], str):
            errors.append("oracle.notes must be explanatory text")
        if oracle.get("confidence") not in {"high", "medium", "low"}:
            errors.append("oracle.confidence must be high, medium, or low")
        refs = oracle.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            errors.append("oracle.evidence_refs must be a non-empty list")
            return
        has_problem_evidence = False
        for index, ref in enumerate(refs):
            path = f"$.oracle.evidence_refs[{index}]"
            if not isinstance(ref, dict):
                errors.append(f"{path} must be an object")
                continue
            source = ref.get("source")
            if source == "problem_statement":
                self._exact_keys(ref, {"source", "text"}, path, errors)
                text = str(ref.get("text") or "")
                if not text.strip():
                    errors.append(f"{path}.text is required")
                elif _token_overlap(text, self.case.problem_statement) < 2:
                    errors.append(f"{path}.text is not traceable to the problem statement")
                has_problem_evidence = True
            elif source == "repository":
                self._exact_keys(ref, {"source", "path", "line"}, path, errors)
                raw_path = str(ref.get("path") or "")
                repo_path = (self.repo_root / raw_path).resolve()
                try:
                    repo_path.relative_to(self.repo_root.resolve())
                except ValueError:
                    errors.append(f"{path}.path escapes the base repository")
                    continue
                if not repo_path.is_file():
                    errors.append(f"{path}.path does not exist in the base repository")
                    continue
                try:
                    line = int(ref.get("line"))
                    total = len(repo_path.read_text(encoding="utf-8", errors="ignore").splitlines())
                    if line < 1 or line > max(total, 1):
                        errors.append(f"{path}.line is outside the referenced file")
                except (TypeError, ValueError):
                    errors.append(f"{path}.line must be an integer")
            else:
                errors.append(f"{path}.source must be problem_statement or repository")
        if not has_problem_evidence:
            errors.append("at least one oracle evidence_ref must come from problem_statement")

    @staticmethod
    def _exact_keys(value: dict[str, Any], expected: set[str] | frozenset[str], path: str, errors: list[str]) -> None:
        keys = set(value)
        extra = sorted(keys - set(expected))
        missing = sorted(set(expected) - keys)
        if extra or missing:
            errors.append(f"{path} keys mismatch: extra={extra}, missing={missing}")

    @staticmethod
    def _scan_forbidden(value: Any, path: str, errors: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_INPUT_KEYS:
                    errors.append(f"forbidden field at {path}.{key}")
                BehaviorSpecValidator._scan_forbidden(child, f"{path}.{key}", errors)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                BehaviorSpecValidator._scan_forbidden(child, f"{path}[{index}]", errors)


class OpenAICompatibleClient:
    def __init__(self, project_root: Path):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(project_root / ".env")
        config_path = project_root / "model_configs.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        profile = os.getenv("MODEL_PROFILE", "ds-v4-flash")
        model_config = config.get(profile, {}) if isinstance(config.get(profile), dict) else {}
        api_key = model_config.get("api_key") or os.getenv("BAILIAN_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = model_config.get("base_url") or os.getenv("OPENAI_COMPAT_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
        self.model = (
            model_config.get("model_name")
            or model_config.get("model")
            or os.getenv("MODEL_NAME", "deepseek-v4-flash")
        )
        if not api_key:
            raise RuntimeError("model API key is not configured")
        hostname = (urlparse(str(base_url)).hostname or "").lower()
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if os.getenv("KGVERI_MINITB_LOCAL_ONLY") == "1" and hostname not in local_hosts:
            raise RuntimeError(
                "KGVERI_MINITB_LOCAL_ONLY=1 requires a loopback model endpoint"
            )
        if hostname in local_hosts:
            for key in ("NO_PROXY", "no_proxy"):
                entries = [item.strip() for item in os.getenv(key, "").split(",") if item.strip()]
                for item in ("127.0.0.1", "localhost"):
                    if item not in entries:
                        entries.append(item)
                os.environ[key] = ",".join(entries)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)

    def complete_json(self, prompt: str) -> tuple[dict[str, Any], str]:
        self.last_usage = {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        raw = message.content or ""
        usage = getattr(response, "usage", None)
        self.last_usage = usage.model_dump() if usage is not None else {
            "estimated": True,
            "prompt_tokens": (len(prompt) + 3) // 4,
            "completion_tokens": (len(raw) + 3) // 4,
        }
        return parse_json_response(raw), raw


class BehaviorSpecGenerator:
    def __init__(
        self,
        validator: BehaviorSpecValidator,
        prompt_path: Path,
        client: OpenAICompatibleClient,
        attempt_writer: Callable[[dict[str, Any]], None],
        budget: RetryBudget | None = None,
    ):
        self.validator = validator
        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.client = client
        self.attempt_writer = attempt_writer
        self.budget = budget
        self.candidate_specs: list[dict[str, Any]] = []

    def _remember_candidate(self, spec: dict[str, Any]) -> None:
        key = json.dumps(spec, sort_keys=True)
        if all(json.dumps(item, sort_keys=True) != key for item in self.candidate_specs):
            self.candidate_specs.append(json.loads(json.dumps(spec)))

    def generate(self, context: dict[str, Any], output_path: Path) -> dict[str, Any]:
        previous: dict[str, Any] | None = None
        errors: list[str] = []
        for round_number in (1, 2, 3):
            compact = compact_generation_context(context)
            payload = {
                "context": compact,
                "previous_behavior_spec": previous,
                "validator_errors": errors,
            }
            if errors and previous is not None:
                prompt = _build_spec_correction_prompt(compact, previous, errors)
            else:
                prompt = self.prompt.replace(
                    "{{INPUT_JSON}}",
                    json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                )
            suggested = (
                payload["context"].get("generation_contract") or {}
            ).get("suggested_bug_class")
            if suggested == "event_trace":
                event_ports = (payload["context"].get("generation_contract") or {}).get(
                    "event_port_candidates", []
                )
                prompt += (
                    "\n\nEVENT_TRACE REQUIRED. Every event-qualified check adds exactly these fields: "
                    "\"when\": {\"REAL_EVENT_PORT\": \"1'b1\"}. Add occurrence only when the "
                    "issue distinguishes repeated events. "
                    f"Choose REAL_EVENT_PORT from {json.dumps(event_ports)} and replace the name. "
                    "The event predicate may be a DUT input or output; an input predicate may also "
                    "appear in step.drive. Check an associated data/state output, not the event "
                    "port itself."
                )
            if errors and previous is None:
                prompt += (
                    "\n\nCORRECTION REQUIRED. The previous object was rejected for these exact reasons: "
                    + json.dumps(errors, ensure_ascii=True)
                    + ". Return a changed object that fixes every error. Read generation_contract: "
                    "drive only allowed_input_ports and check only allowed_output_ports."
                )
            if self.budget:
                self.budget.consume("behavior_spec")
            try:
                candidate, raw = self.client.complete_json(prompt)
            except Exception as exc:
                if not isinstance(exc, (ValueError, json.JSONDecodeError)) and not type(exc).__module__.startswith("openai"):
                    raise
                errors = [f"model response is unavailable or not a JSON object: {exc}"]
                record = {
                    "stage": "behavior_spec",
                    "round": round_number,
                    "model": self.client.model,
                    "raw_output": "",
                    "status": "SPEC_INVALID",
                    "validator_errors": errors,
                    "usage": getattr(self.client, "last_usage", {}),
                }
                record.update(source="model_intent", generation_policy="kg-self-correction-v1", model_spec_used=False)
                self.attempt_writer(record)
                continue
            record = {
                "stage": "behavior_spec",
                "round": round_number,
                "model": self.client.model,
                "raw_output": raw,
                "prompt_chars": len(prompt),
                "usage": getattr(self.client, "last_usage", {}),
                "source": "model_intent",
                "generation_policy": "kg-self-correction-v1",
                "model_spec_used": False,
            }
            compact_candidate: dict[str, Any] | None = None
            try:
                if set(candidate) >= set(INTENT_KEYS) and set(candidate) - set(INTENT_KEYS) <= {
                    "title", "problem_statement"
                }:
                    candidate = {key: candidate[key] for key in INTENT_KEYS}
                if set(candidate) == set(SPEC_KEYS):
                    raise SpecValidationError([
                        "model must return the compact intent, not a complete BehaviorSpec"
                    ])
                if set(candidate) == {"select_target"}:
                    choice = candidate["select_target"]
                    if not isinstance(choice, dict) or set(choice) != {"file", "module"}:
                        raise SpecValidationError(["select_target requires file and module"])
                    selected = next((target for target in (context.get("candidate_targets") or [])[:12]
                                     if target.get("file") == choice["file"] and target.get("module") == choice["module"]), None)
                    if selected is None:
                        raise SpecValidationError(["select_target is not a supplied target option"])
                    context = {**context, "selected_target": selected}
                    record.update(status="TARGET_SELECTED", target=choice)
                    self.attempt_writer(record)
                    previous = None
                    errors = ["Target selected. Produce the complete behavior intent using this interface; do not select again."]
                    continue
                inferred = infer_intent_target(candidate, context)
                if inferred is not None:
                    candidate = inferred
                    record["target_inferred_from_ports"] = candidate["target"]
                compact_candidate = {
                    key: json.loads(json.dumps(candidate[key]))
                    for key in INTENT_KEYS
                    if key in candidate
                }
                candidate = materialize_behavior_intent(candidate, context)
                validated = self.validator.validate(candidate)
            except SpecValidationError as exc:
                errors = exc.errors
                previous = compact_candidate if compact_candidate is not None else candidate
                record["status"] = "SPEC_INVALID"
                record["validator_errors"] = errors
                self.attempt_writer(record)
                continue
            record["status"] = "VALID"
            record["model_spec_used"] = True
            record["execution_template"] = validated["template_id"]
            self.attempt_writer(record)
            self._remember_candidate(validated)
            context = {**context, "selected_target": next(
                t for t in context["candidate_targets"]
                if t["file"] == validated["target"]["file"] and t["module"] == validated["target"]["module"]
            )}
            reviewed = validated
            if validated["bug_class"] != "unsupported" and validated["oracle"]["confidence"] != "low":
                reviewed = self._self_correct(context, compact_candidate or candidate, validated)
            write_json(output_path, reviewed)
            return reviewed
        raise SpecValidationError(errors or ["model did not produce a valid BehaviorSpec"])

    def _self_correct(
        self,
        context: dict[str, Any],
        previous_intent: dict[str, Any],
        previous_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Run two small KG-grounded critique/revision iterations."""
        compact = compact_generation_context(context)
        evidence = compact.get("minitb_evidence") or {}
        current_intent = previous_intent
        current_spec = previous_spec
        for review_round in (1, 2):
            review_hints = evidence_review(current_spec, evidence)
            record = {
                "stage": "behavior_spec_review",
                "round": review_round,
                "model": self.client.model,
                "source": "kg_self_correction",
                "generation_policy": "kg-self-correction-v1",
                "model_spec_used": False,
            }
            try:
                if self.budget:
                    self.budget.consume("behavior_spec")
                prompt = _build_self_correction_prompt(
                    compact, current_intent, current_spec, review_hints
                )
                record["prompt_chars"] = len(prompt)
                response, raw = self.client.complete_json(prompt)
                record["raw_output"] = raw
                critique = response.get("critique") if isinstance(response, dict) else None
                verdict = response.get("verdict") if isinstance(response, dict) else None
                candidate = response.get("revised_intent") if isinstance(response, dict) else None
                if not isinstance(candidate, dict):
                    candidate = response
                if not isinstance(candidate, dict):
                    raise SpecValidationError(["self-correction response has no revised_intent object"])
                if set(candidate) == set(SPEC_KEYS):
                    reviewed = self.validator.validate(candidate)
                else:
                    if set(candidate) >= set(INTENT_KEYS) and set(candidate) - set(INTENT_KEYS) <= {
                        "title", "problem_statement"
                    }:
                        candidate = {key: candidate[key] for key in INTENT_KEYS}
                    if set(candidate) != set(INTENT_KEYS):
                        raise SpecValidationError([
                            "self-correction must return the same eight-key compact intent"
                        ])
                    reviewed = self.validator.validate(materialize_behavior_intent(candidate, context))
            except (BudgetExhausted, ValueError, json.JSONDecodeError, SpecValidationError) as exc:
                errors = exc.errors if isinstance(exc, SpecValidationError) else [str(exc)]
                record.update(
                    status="FALLBACK_TO_VALID_DRAFT",
                    validator_errors=errors,
                    usage=getattr(self.client, "last_usage", {}),
                )
                self.attempt_writer(record)
                continue
            except Exception as exc:
                if not type(exc).__module__.startswith("openai"):
                    raise
                record.update(
                    status="FALLBACK_TO_VALID_DRAFT",
                    validator_errors=[str(exc)],
                    usage=getattr(self.client, "last_usage", {}),
                )
                self.attempt_writer(record)
                continue

            remaining = evidence_review(reviewed, evidence)
            record.update(
                status="VALID",
                model_spec_used=True,
                changed=reviewed != current_spec,
                critique=critique,
                verdict=verdict,
                review_hints=review_hints,
                remaining_review_hints=remaining,
                usage=getattr(self.client, "last_usage", {}),
            )
            self.attempt_writer(record)
            self._remember_candidate(reviewed)
            current_spec = reviewed
            current_intent = candidate
        # Evidence review is a ranking signal, not a second schema validator.
        # Keeping the valid drafts lets the execution qualifier decide which
        # one actually reproduces the defect, while the tiny candidate beam
        # remains bounded to the initial draft and two revisions.
        unresolved = [hint for hint in evidence_review(current_spec, evidence) if _critical_review_hint(hint)]
        if unresolved:
            self.attempt_writer({
                "stage": "behavior_spec_review",
                "status": "VALID_WITH_EVIDENCE_WARNINGS",
                "remaining_review_hints": unresolved,
                "model_spec_used": True,
            })
        return current_spec

    def repair(self, context: dict[str, Any], previous: dict[str, Any], feedback: dict[str, Any], output_path: Path) -> dict[str, Any]:
        """One bounded correction from buggy-base feedback; no evaluation patches."""
        if self.budget:
            self.budget.consume("reproduction")
        payload = {
            "context": compact_generation_context(context),
            "previous_behavior_spec": previous,
            "base_execution_feedback": feedback,
            "correction_constraints": (
                "Correct stimulus encodings, timing or compilation using visible source and issue only. "
                "The supplied step start cycles are absolute simulation cycles after reset and must be used when aligning checks. "
                "Keep target, oracle references, check IDs, signals, operators and expected values unchanged. "
                "Do not remove checks. A compile failure or base pass does not justify changing the oracle."
            ),
        }
        prompt = self.prompt.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        prompt += (
            "\n\nEXECUTION CORRECTION REQUIRED. The previous test passed or failed to build on the "
            "defective base. Change its trigger drive or scheduling using visible RTL. Preserve the "
            "functional check and do not return the same stimulus."
        )
        record = {"stage": "reproduction", "round": 1, "model": self.client.model, "prompt_chars": len(prompt),
                  "source": "model_intent", "generation_policy": "kg-self-correction-v1", "model_spec_used": False}
        try:
            candidate, raw = self.client.complete_json(prompt)
            record["raw_output"] = raw
            candidate = materialize_behavior_intent(candidate, context, previous)
            result = self.validator.validate(candidate)
            if repair_contract(result) != repair_contract(previous):
                raise SpecValidationError(["execution correction changed the target or functional oracle"])
            if (
                result.get("steps") == previous.get("steps")
                and (result.get("timing") or {}).get("sample_phase")
                == (previous.get("timing") or {}).get("sample_phase")
            ):
                raise SpecValidationError(["execution correction repeated the same stimulus and sample phase"])
        except Exception as exc:
            if not isinstance(exc, (ValueError, SpecValidationError)) and not type(exc).__module__.startswith("openai"):
                raise
            record.update(status="SPEC_INVALID", validator_errors=[str(exc)], usage=getattr(self.client, "last_usage", {}))
            self.attempt_writer(record)
            raise SpecValidationError([str(exc)]) from exc
        record.update(status="VALID", model_spec_used=True, usage=getattr(self.client, "last_usage", {}))
        self.attempt_writer(record)
        write_json(output_path, result)
        return result


def _build_spec_correction_prompt(
    context: dict[str, Any], previous: dict[str, Any], errors: list[str]
) -> str:
    """Build a short second-round prompt centered on actionable constraints."""
    issue = context.get("issue") or {}
    target = (context.get("candidate_targets") or [{}])[0]
    contract = context.get("generation_contract") or {}
    event_rule = ""
    if contract.get("suggested_bug_class") == "event_trace":
        event_rule = (
            " EVENT_TRACE checks require when={REAL_EVENT_PORT: \"1'b1\"}; use occurrence only "
            "for repeated events. "
            f"event ports are {contract.get('event_port_candidates') or []}. Event ports are "
            "DUT inputs or outputs; input predicates may also appear in step.drive."
        )
    return (
        "Correct the rejected MiniTB intent. Return exactly one JSON object with the same eight "
        "top-level keys as the previous object and no prose. Change every field implicated by an "
        "error; do not repeat an invalid port, symbol, placeholder, or encoding. The renderer owns "
        "clock and reset. Drive at least one real non-clock input that activates the issue. Drive "
        "only allowed inputs and check only allowed outputs. Expected values must follow the issue, "
        "never the defective implementation."
        + event_rule
        + "\nISSUE:\n"
        + json.dumps(
            {"title": issue.get("title"), "problem_statement": issue.get("problem_statement")},
            ensure_ascii=True,
        )
        + "\nFIXED TEMPLATE CLASS:\n"
        + json.dumps(contract.get("suggested_bug_class"), ensure_ascii=True)
        + "\nALLOWED INPUTS:\n"
        + json.dumps(contract.get("allowed_input_ports") or [], ensure_ascii=True)
        + "\nALLOWED OUTPUTS:\n"
        + json.dumps(contract.get("allowed_output_ports") or [], ensure_ascii=True)
        + "\nISSUE FACTS:\n"
        + json.dumps(context.get("issue_facts") or {}, ensure_ascii=True)
        + "\nRANKED OBSERVABLE OUTPUTS:\n"
        + json.dumps(contract.get("ranked_observation_ports") or [], ensure_ascii=True)
        + "\nVISIBLE RTL:\n"
        + str(target.get("source") or "")[:7000]
        + "\nVISIBLE PACKAGE DEFINITIONS:\n"
        + str(context.get("package_context") or "")
        + "\nVALIDATOR ERRORS:\n"
        + json.dumps(errors, ensure_ascii=True)
        + "\nEDIT THIS REJECTED INTENT AND RETURN THE CORRECTED JSON ONLY:\n"
        + json.dumps(previous, ensure_ascii=True)
        + "\nCORRECTED JSON ONLY:\n"
    )


def _build_self_correction_prompt(
    context: dict[str, Any],
    previous_intent: dict[str, Any],
    previous_spec: dict[str, Any],
    review_hints: list[str],
) -> str:
    review_context = _compact_review_context(context)
    return (
        "Act as an independent RTL verification critic. Audit the draft against the issue and KG "
        "evidence, then return one JSON object with exactly: verdict, critique, revised_intent. "
        "verdict is accept or revise. critique is an object with concise trigger, timing, "
        "observation, and oracle findings. revised_intent must always contain the same eight keys "
        "as the supplied compact draft. Keep the localized target and template family. Ensure the "
        "stimulus reaches the target span, satisfies guards, and checks issue-required outputs. "
        "Re-derive every expected value independently from the issue or visible test evidence; KG "
        "and defective RTL establish structure, never the correct value. Recompute multi-bit "
        "encodings field by field. Keep only the primary issue property and directly necessary "
        "supporting checks; remove unrelated assertions. Fix every deterministic hint. Return "
        "Allowed check operators are ==, !=, stable, asserted, deasserted. At most 4 checks. "
        "Keep the response compact. JSON only.\nCONTEXT:\n"
        + json.dumps(review_context, ensure_ascii=True, separators=(",", ":"))
        + "\nDETERMINISTIC REVIEW HINTS:\n"
        + json.dumps(review_hints, ensure_ascii=True)
        + "\nMATERIALIZED DRAFT:\n"
        + json.dumps(previous_spec, ensure_ascii=True, separators=(",", ":"))
        + "\nCOMPACT DRAFT TO REVISE:\n"
        + json.dumps(previous_intent, ensure_ascii=True, separators=(",", ":"))
        + "\nREVIEW JSON ONLY:\n"
    )


def _compact_review_context(context: dict[str, Any]) -> dict[str, Any]:
    evidence = json.loads(json.dumps(context.get("minitb_evidence") or {}))
    for span in evidence.get("support_spans") or []:
        if isinstance(span, dict):
            span["source"] = str(span.get("source") or "")[:600]
    return {
        "issue": context.get("issue") or {},
        "package_context": context.get("package_context", ""),
        "candidate_targets": context.get("candidate_targets", []),
        "generation_contract": context.get("generation_contract") or {},
        "issue_facts": context.get("issue_facts") or {},
        "minitb_evidence": evidence,
        "adjacent_assets": [
            {"path": item.get("path"), "excerpt": str(item.get("excerpt") or "")[:600]}
            for item in (context.get("adjacent_assets") or [])[:2]
            if isinstance(item, dict)
        ],
    }


def _critical_review_hint(hint: str) -> bool:
    return (
        hint.startswith("no issue-named trigger input")
        or hint.startswith("no issue-named output")
        or "after the issue trigger is asserted" in hint
    )


def repair_contract(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": spec["target"],
        "oracle": spec["oracle"],
        "checks": sorted([
            {**{key: check[key] for key in ("check_id", "signal", "op", "expected")},
             **{key: check[key] for key in ("when", "occurrence") if key in check}}
            for check in spec["checks"]
        ], key=lambda check: check["check_id"]),
    }


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def load_and_validate_spec(
    path: Path, validator: BehaviorSpecValidator, output_path: Path | None = None
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    validated = validator.validate(value)
    if output_path:
        write_json(output_path, validated)
    return validated


def _is_safe_sv_value(value: Any) -> bool:
    return isinstance(value, (str, int)) and bool(SV_VALUE.fullmatch(str(value).strip()))


def _literal_width(value: str) -> int | None:
    match = re.match(r"(\d+)'", value)
    return int(match.group(1)) if match else None


def _sv_integer(value: Any) -> int | None:
    text = str(value).strip().replace("_", "")
    if text == "'0":
        return 0
    if text == "'1":
        return 1
    match = re.fullmatch(r"\d+'[sS]?([bBoOdDhH])([0-9a-fA-F]+)", text)
    if match:
        base = {"b": 2, "o": 8, "d": 10, "h": 16}[match.group(1).lower()]
        try:
            return int(match.group(2), base)
        except ValueError:
            return None
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", text):
        return int(text, 16)
    if re.fullmatch(r"0[bB][01]+", text):
        return int(text, 2)
    if re.fullmatch(r"-?\d+", text):
        return int(text, 10)
    return None


def _port_width(width: str, declaration: str = "") -> int | None:
    match = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", width)
    if not match:
        # Package/typedef widths cannot be inferred from an absent packed range.
        if "::" in declaration or re.search(r"\b[A-Za-z_]\w*_t\b", declaration):
            return None
        return 1 if not width.strip() else None
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def _token_overlap(left: str, right: str) -> int:
    left_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", left) if len(token) > 2}
    right_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", right) if len(token) > 2}
    return len(left_tokens & right_tokens)


def _looks_clock(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:clk|clock)(?:_|$)", name, re.IGNORECASE))


def _looks_reset(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:rst|reset)(?:_|$)", name, re.IGNORECASE))
