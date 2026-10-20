from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

try:
    from utils import context_entity_sort_key
except Exception:  # pragma: no cover - package-relative fallback.
    from .utils import context_entity_sort_key

try:
    from verilog_timing import normalize_verilog_related_entities
except Exception:  # pragma: no cover - package-relative fallback.
    from .verilog_timing import normalize_verilog_related_entities


def _compact_text(text: str, limit: int = 260) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _unique_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for item in items:
        marker = str(item)
        if marker in seen:
            continue
        ordered.append(item)
        seen.add(marker)
    return ordered


def _entity_role(entity: Dict[str, Any]) -> str:
    return str(
        entity.get("repair_role")
        or entity.get("entity_role")
        or entity.get("verilog_kind")
        or entity.get("type")
        or entity.get("label")
        or "unknown"
    ).strip().lower()


def _entity_name(entity: Dict[str, Any]) -> str:
    return str(entity.get("signature") or entity.get("name") or entity.get("signal_name") or "").strip()


def _entity_path_text(entity: Dict[str, Any], max_hops: int = 4) -> str:
    path = entity.get("path") or []
    if not path:
        return ""
    steps = []
    for step in path[:max_hops]:
        if not isinstance(step, dict):
            continue
        start = str(step.get("start_node") or "").strip()
        end = str(step.get("end_node") or "").strip()
        relation = str(step.get("type") or step.get("relation_kind") or "RELATED").strip()
        description = str(step.get("description") or "").strip()
        if start or end:
            if description:
                steps.append(f"{start} -[{relation}: {description}]-> {end}")
            else:
                steps.append(f"{start} -[{relation}]-> {end}")
        elif relation or description:
            steps.append(f"{relation}: {description}".strip(": "))
    return " | ".join(steps)


def _entity_summary(entity: Dict[str, Any], include_path: bool = True) -> str:
    path = str(entity.get("file_path") or "").replace("\\", "/")
    start_line = entity.get("start_line")
    end_line = entity.get("end_line")
    span = ""
    if start_line is not None or end_line is not None:
        span = f":{start_line or '?'}-{end_line or '?'}"
    name = _entity_name(entity)
    role = _entity_role(entity)
    summary = str(entity.get("semantic_summary") or entity.get("declaration") or entity.get("source_code") or "").replace("\n", " ")
    summary = _compact_text(summary, 220)
    text = f"- {path}{span} {name} [{role}] {summary}".strip()
    if include_path:
        path_text = _entity_path_text(entity)
        if path_text:
            text = f"{text}\n  path: {path_text}"
    return text


def _pick_group_entities(related: Dict[str, List[Dict[str, Any]]], group_names: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    picked = []
    seen = set()
    for group_name in group_names:
        for entity in related.get(group_name) or []:
            marker = (
                str(entity.get("name") or entity.get("signature") or ""),
                str(entity.get("file_path") or ""),
                str(entity.get("start_line") or ""),
                str(entity.get("end_line") or ""),
            )
            if marker in seen:
                continue
            picked.append(entity)
            seen.add(marker)
            if len(picked) >= limit:
                return picked
    return picked


def _trace_templates(related: Dict[str, List[Dict[str, Any]]], analysis: Any = None) -> List[str]:
    observed_signal = str(getattr(analysis, "observed_signal", "") or "").strip()
    expected_idle = str(getattr(analysis, "expected_idle", "") or "").strip()
    templates = []
    if related.get("config_registers") and related.get("top_level_wiring") and related.get("direct_drivers"):
        tail = f" -> {observed_signal}" if observed_signal else ""
        templates.append(f"Issue -> Config/Register Evidence -> Top-Level Wiring -> Direct Driver{tail}")
    if related.get("top_level_wiring") and related.get("direct_drivers"):
        tail = f" -> {observed_signal}" if observed_signal else ""
        templates.append(f"Issue -> Top-Level Wiring -> Direct Driver{tail}")
    if any((entity.get("label") == "Testbench" or _entity_role(entity) == "testbench") for entity in related.get("evidence_entities") or []):
        templates.append("Issue -> Testbench Oracle -> Failing Stimulus -> Observable Output")
    if any(_entity_role(entity) == "state" for entity in (related.get("evidence_entities") or []) + (related.get("rtl_entities") or [])):
        templates.append("Issue -> FSM State -> Transition -> Output")
    if any(_entity_role(entity) in {"branch", "condition"} for entity in (related.get("evidence_entities") or []) + (related.get("rtl_entities") or [])):
        templates.append("Issue -> Branch/Condition -> Signal Read/Write -> Observable Output")
    if expected_idle:
        templates.append(f"Expected idle after write: {expected_idle}")
    return _unique_keep_order(templates)


def _collect_counterevidence(related: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    counter = []
    for entity in (related.get("evidence_entities") or []) + (related.get("rtl_entities") or []):
        label = str(entity.get("label") or "").lower()
        role = _entity_role(entity)
        if label == "assertion" or role == "assertion":
            counter.append(_entity_summary(entity, include_path=False))
        elif label == "testbench" or role == "testbench":
            counter.append(_entity_summary(entity, include_path=False))
    return _unique_keep_order(counter)


def _collect_candidate_files(entities: Sequence[Dict[str, Any]]) -> List[str]:
    files = []
    for entity in entities:
        file_path = str(entity.get("file_path") or "").replace("\\", "/")
        if file_path:
            files.append(file_path)
    return _unique_keep_order(files)


def build_verilog_bug_kg_digest(
    related_entities: Dict[str, Sequence[Dict[str, Any]]],
    *,
    issue_text: str = "",
    analysis: Any = None,
    limit_per_group: int = 6,
    max_paths: int = 8,
) -> Dict[str, Any]:
    normalized = normalize_verilog_related_entities(related_entities or {})

    primary_targets = _pick_group_entities(
        normalized,
        ("direct_drivers", "top_level_wiring", "config_registers", "edit_targets", "methods"),
        max_paths,
    )
    evidence_entities = _pick_group_entities(
        normalized,
        ("evidence_entities", "rtl_entities"),
        max_paths,
    )
    issue_entities = list(normalized.get("issues") or [])

    trace_templates = _trace_templates(normalized, analysis=analysis)
    counterevidence = _collect_counterevidence(normalized)
    candidate_files = _collect_candidate_files(primary_targets + evidence_entities + issue_entities)

    path_hints = []
    for entity in primary_targets + evidence_entities:
        hint = _entity_path_text(entity)
        if hint:
            path_hints.append(hint)
    path_hints = _unique_keep_order(path_hints)[:max_paths]

    if not trace_templates:
        trace_templates = ["Issue -> Evidence Entities -> Editable RTL Targets"]

    observed_signal = str(getattr(analysis, "observed_signal", "") or "").strip()
    expected_idle = str(getattr(analysis, "expected_idle", "") or "").strip()
    failure_signature = str(getattr(analysis, "signature", "") or "").strip()
    failure_summary = str(getattr(analysis, "summary", "") or "").strip()

    text_sections = [
        "## Bug KG Digest",
        f"- issue: {_compact_text(issue_text, 500)}" if issue_text else "- issue: (not provided)",
        f"- failure_signature: {failure_signature}" if failure_signature else "- failure_signature: (not provided)",
        f"- failure_summary: {_compact_text(failure_summary, 500)}" if failure_summary else "",
        f"- observed_signal: {observed_signal}" if observed_signal else "",
        f"- expected_idle: {expected_idle}" if expected_idle else "",
        "- trace_templates:",
        *[f"  - {item}" for item in trace_templates],
        "- primary_edit_targets:",
        *[_entity_summary(entity) for entity in primary_targets[:limit_per_group]],
        "- evidence_entities:",
        *[_entity_summary(entity) for entity in evidence_entities[:limit_per_group]],
        "- counterevidence:",
        *[f"  - {item}" for item in counterevidence[:limit_per_group]],
        "- path_hints:",
        *[f"  - {item}" for item in path_hints[:max_paths]],
        f"- candidate_files: {', '.join(candidate_files[:12])}" if candidate_files else "",
        "",
        "Use this digest as the first tracing guide. Validation evidence remains the oracle, but the repair should follow the KG chain before touching RTL.",
    ]
    text = "\n".join(line for line in text_sections if line is not None and str(line).strip())

    return {
        "issue_text": _compact_text(issue_text, 1200),
        "failure_signature": failure_signature,
        "failure_summary": _compact_text(failure_summary, 1000),
        "observed_signal": observed_signal,
        "expected_idle": expected_idle,
        "trace_templates": trace_templates,
        "primary_edit_targets": primary_targets,
        "evidence_entities": evidence_entities,
        "counterevidence": counterevidence,
        "path_hints": path_hints,
        "candidate_files": candidate_files,
        "text": text,
        "normalized_related_entities": normalized,
    }
