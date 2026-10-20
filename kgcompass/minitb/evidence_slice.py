from __future__ import annotations

import re
from typing import Any


def build_minitb_evidence_slice(
    kg: dict[str, Any], target: dict[str, Any], issue: dict[str, Any]
) -> dict[str, Any]:
    """Build a compact verification slice from exported localization data."""
    ports = [item for item in target.get("ports") or [] if isinstance(item, dict)]
    inputs = [str(item.get("name") or "") for item in ports if item.get("direction") == "input"]
    outputs = [str(item.get("name") or "") for item in ports if item.get("direction") == "output"]
    issue_text = " ".join(str(issue.get(key) or "") for key in ("title", "problem_statement"))
    issue_identifiers = set(
        re.findall(r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_$]*)", issue_text)
    )
    issue_inputs = [
        name for name in inputs
        if name in issue_identifiers or _identifier_matches_issue(name, issue_text, input_side=True)
    ]
    issue_outputs = [
        name for name in outputs
        if name in issue_identifiers or _identifier_matches_issue(name, issue_text)
    ]
    nonstructural_inputs = {
        name
        for name in inputs
        if not re.search(r"(?:^|_)(?:clk|clock|rst|reset)(?:_|$)", name, re.I)
    }

    related = kg.get("related_entities") or {}
    target_file = str(target.get("file") or "")
    target_module = str(target.get("module") or "")
    spans = [
        item
        for item in (related.get("patchable_spans") or [])
        if isinstance(item, dict)
        and str(item.get("module_name") or "") == target_module
        and _same_path(str(item.get("file_path") or ""), target_file)
    ]

    def span_priority(item: dict[str, Any]) -> tuple[int, float, int]:
        issue_hits = len(_span_signals(item) & issue_identifiers)
        score = float(item.get("score") or item.get("similarity") or 0.0)
        rank = int(item.get("global_rank") or item.get("mode_rank") or 10_000)
        return (-issue_hits, -score, rank)

    ranked_spans = sorted(spans, key=span_priority)
    selected_spans = list(ranked_spans[:4])
    # Add one enclosing/control span that connects issue-named inputs to other
    # controllable ports (for example request/grant before a response error).
    control_span = max(
        spans,
        key=lambda item: (
            len(set(item.get("reads_signals") or []) & set(issue_inputs)),
            len(_span_signals(item) & nonstructural_inputs),
            -int(item.get("global_rank") or 10_000),
        ),
        default=None,
    )
    if control_span is not None:
        selected_spans.append(control_span)
    # Preserve direct issue-output drivers even when their localization score
    # is lower because the defective implementation ignores an error input.
    for output in issue_outputs:
        driver = next(
            (
                item
                for item in ranked_spans
                if output
                in set(item.get("writes_signals") or [])
                | set(item.get("drives_signals") or [])
            ),
            None,
        )
        if driver is not None:
            selected_spans.append(driver)
    spans = []
    seen_spans: set[str] = set()
    for item in selected_spans + ranked_spans:
        key = str(item.get("name") or "")
        if key in seen_spans:
            continue
        seen_spans.add(key)
        spans.append(item)
        if len(spans) >= 8:
            break
    read_signals = _ordered_union(spans, "reads_signals")
    guard_signals = _ordered_union(spans, "guards_signals")
    written_signals = _ordered_union(spans, "writes_signals", "drives_signals")
    connected_signals = _ordered_union(spans, "connects_signals")
    input_support = set(read_signals) | set(guard_signals) | set(connected_signals)
    output_support = set(written_signals) | set(connected_signals)

    controllable = _dedupe(issue_inputs + [name for name in inputs if name in input_support])
    observable = _dedupe(issue_outputs + [name for name in outputs if name in output_support])
    clock = next(
        (name for name in inputs if re.search(r"(?:^|_)(?:clk|clock)(?:_|$)", name, re.I)), ""
    )
    reset = next(
        (name for name in inputs if re.search(r"(?:^|_)(?:rst|reset)(?:_|$)", name, re.I)), ""
    )
    support_spans = [
        {
            "name": span.get("name"),
            "kind": span.get("span_kind") or span.get("verilog_kind"),
            "lines": [span.get("start_line"), span.get("end_line")],
            "reads": span.get("reads_signals") or [],
            "guards": span.get("guards_signals") or [],
            "writes": _dedupe(
                list(span.get("writes_signals") or []) + list(span.get("drives_signals") or [])
            ),
            "source": str(span.get("source_code") or "")[:1200],
        }
        for span in spans
    ]
    return {
        "schema_version": "kgveri-minitb-evidence-v1",
        "available": bool(spans),
        "target": {"file": target_file, "module": target_module},
        "issue_named_inputs": issue_inputs,
        "issue_named_outputs": issue_outputs,
        "controllable_inputs": controllable,
        "required_guard_signals": [name for name in guard_signals if name in inputs],
        "observable_outputs": observable,
        "timing": {"clock": clock, "reset": reset},
        "support_spans": support_spans,
        "oracle_policy": (
            "RTL paths justify trigger and observation selection, but buggy RTL values are not "
            "oracle evidence. Hard expected values require explicit issue or repository-test evidence."
        ),
    }


def evidence_review(spec: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    """Produce review hints for one bounded model self-correction."""
    hints: list[str] = []
    driven = {
        str(name).split("[", 1)[0]
        for step in spec.get("steps") or []
        if isinstance(step, dict)
        for name in (step.get("drive") or {})
    }
    checked = {
        re.split(r"[\[.]", str(check.get("signal") or ""), maxsplit=1)[0]
        for check in spec.get("checks") or []
        if isinstance(check, dict)
    }
    structural = {
        (evidence.get("timing") or {}).get("clock"),
        (evidence.get("timing") or {}).get("reset"),
    }
    missing_inputs = [
        name
        for name in evidence.get("issue_named_inputs") or []
        if name not in driven and name not in structural
    ]
    missing_outputs = [
        name for name in evidence.get("issue_named_outputs") or [] if name not in checked
    ]
    issue_inputs = [
        name for name in evidence.get("issue_named_inputs") or [] if name not in structural
    ]
    issue_outputs = list(evidence.get("issue_named_outputs") or [])
    if issue_inputs and not (set(issue_inputs) & driven):
        hints.append(f"no issue-named trigger input is driven: {issue_inputs}")
    elif missing_inputs:
        hints.append(f"additional issue-named inputs are not driven: {missing_inputs}")
    if issue_outputs and not (set(issue_outputs) & checked):
        hints.append(f"no issue-named output is checked: {issue_outputs}")
    elif missing_outputs:
        hints.append(f"additional issue-named outputs are not checked: {missing_outputs}")
    unsupported_drives = [
        name
        for name in driven
        if evidence.get("controllable_inputs") and name not in evidence["controllable_inputs"]
    ]
    if unsupported_drives:
        hints.append(f"drives lack support from the localized slice: {unsupported_drives}")
    unsupported_checks = [
        name
        for name in checked
        if evidence.get("observable_outputs") and name not in evidence["observable_outputs"]
    ]
    if unsupported_checks:
        hints.append(f"checks lack support from the localized slice: {unsupported_checks}")
    trigger_inputs = list(evidence.get("issue_named_inputs") or [])
    if trigger_inputs:
        cycle = 0
        trigger_cycles: list[int] = []
        deassert_cycles: list[int] = []
        for step in spec.get("steps") or []:
            if not isinstance(step, dict):
                continue
            cycle += int(step.get("after_cycles") or 0)
            drive = step.get("drive") or {}
            if all(name in drive and _is_asserted(drive[name]) for name in trigger_inputs):
                trigger_cycles.append(cycle)
            if all(name in drive and not _is_asserted(drive[name]) for name in trigger_inputs):
                deassert_cycles.append(cycle)
            cycle += int(step.get("hold_cycles") or 0)
        if trigger_cycles:
            trigger_cycle = trigger_cycles[0]
            for check in spec.get("checks") or []:
                if not isinstance(check, dict):
                    continue
                check_cycle = int(check.get("at_or_after_cycle") or 0)
                if any(trigger_cycle < off_cycle <= check_cycle for off_cycle in deassert_cycles):
                    hints.append(
                        f"check {check.get('check_id')} samples at cycle {check_cycle}, after the "
                        f"issue trigger is asserted at cycle {trigger_cycle} and then deasserted; "
                        "sample the combinational response while the trigger is active"
                    )
    hints.append(
        "re-derive every multi-bit expected value field by field from the issue; do not copy the defective RTL"
    )
    return hints


def _same_path(left: str, right: str) -> bool:
    left = left.replace("\\", "/")
    right = right.replace("\\", "/")
    return bool(
        left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left))
    )


def _identifier_matches_issue(
    identifier: str, issue_text: str, *, input_side: bool = False
) -> bool:
    aliases = {"instr": "instruction", "err": "error", "intg": "integrity"}
    if input_side:
        aliases.update({"req": "request", "gnt": "grant"})
    generic = {"data", "signal", "input", "output", "lsu", "is"}
    words = set(re.findall(r"[a-z][a-z0-9]*", issue_text.lower()))
    tokens = [token for token in identifier.lower().split("_") if token not in {"i", "o", "q", "n"}]
    meaningful = [
        aliases.get(token, token)
        for token in tokens
        if (len(token) >= 4 or token in aliases) and token not in generic
    ]
    return any(token in words for token in meaningful)


def _span_signals(span: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for key in (
            "reads_signals",
            "writes_signals",
            "drives_signals",
            "connects_signals",
            "guards_signals",
        )
        for value in (span.get(key) or [])
    }


def _ordered_union(spans: list[dict[str, Any]], *keys: str) -> list[str]:
    return _dedupe(
        [str(value) for span in spans for key in keys for value in (span.get(key) or [])]
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_asserted(value: Any) -> bool:
    text = str(value).strip().lower().replace("_", "")
    return text not in {"0", "'0", "1'b0", "1'd0", "0x0", "0b0"}
