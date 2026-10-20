from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence

try:
    from verilog_kg_digest import build_verilog_bug_kg_digest as _build_verilog_bug_kg_digest_base
except Exception:  # pragma: no cover - package-relative fallback.
    from .verilog_kg_digest import build_verilog_bug_kg_digest as _build_verilog_bug_kg_digest_base

try:
    from verilog_timing import normalize_verilog_related_entities
except Exception:  # pragma: no cover - package-relative fallback.
    from .verilog_timing import normalize_verilog_related_entities


def _compact_text(text: str, limit: int = 240) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _unique_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for item in items:
        marker = json_key(item)
        if marker in seen:
            continue
        ordered.append(item)
        seen.add(marker)
    return ordered


def json_key(item: Any) -> str:
    try:
        import json

        return json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(item)


def _string_list(items: Iterable[Any], limit: int = 8) -> str:
    values = [str(item) for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) > limit:
        values = values[:limit] + ["..."]
    return ", ".join(values)


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


def _entity_file(entity: Dict[str, Any]) -> str:
    return str(entity.get("file_path") or "").replace("\\", "/").strip()


def _normalize_repo_path(path: str, repo_root: str = "") -> str:
    path = str(path or "").replace("\\", "/").strip()
    if not path:
        return ""
    if repo_root:
        repo_root = os.path.abspath(repo_root).replace("\\", "/")
        abs_path = os.path.abspath(path).replace("\\", "/")
        if abs_path.startswith(repo_root.rstrip("/") + "/"):
            rel = os.path.relpath(abs_path, repo_root).replace("\\", "/")
            return rel
    for marker in ("workdirs/", "playground/", "verilog_repair_cases/"):
        idx = path.find(marker)
        if idx >= 0:
            return path[idx + len(marker):]
    return path.lstrip("./")


def _same_file(left: str, right: str) -> bool:
    left = str(left or "").replace("\\", "/").lower()
    right = str(right or "").replace("\\", "/").lower()
    if not left or not right:
        return False
    if left == right:
        return True
    return os.path.basename(left) == os.path.basename(right)


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
    path = _entity_file(entity)
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


def _entity_text_blob(entity: Dict[str, Any]) -> str:
    parts = [
        str(entity.get("name") or ""),
        str(entity.get("signature") or ""),
        str(entity.get("file_path") or ""),
        str(entity.get("semantic_summary") or ""),
        str(entity.get("declaration") or ""),
        str(entity.get("source_code") or ""),
        str(entity.get("timing_summary") or ""),
        _entity_path_text(entity),
    ]
    return " ".join(part for part in parts if part).lower()


def _signal_tokens(*texts: str) -> List[str]:
    tokens = []
    for text in texts:
        if not text:
            continue
        tokens.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", text))
    tokens = [token.lower() for token in tokens if len(token) > 1]
    return _unique_keep_order(tokens)


def _score_entity(
    entity: Dict[str, Any],
    *,
    failure_file_path: str = "",
    failure_line_number: int = 0,
    observed_signal: str = "",
    expected_idle: str = "",
    seed_tokens: Sequence[str] = (),
    timing_sensitive: bool = False,
) -> float:
    score = 0.0
    file_path = _entity_file(entity)
    role = _entity_role(entity)
    label = str(entity.get("label") or entity.get("type") or "").strip().lower()
    verilog_kind = str(entity.get("verilog_kind") or entity.get("rtl_kind") or "").strip().lower()
    text_blob = _entity_text_blob(entity)
    path_text = _entity_path_text(entity)
    timing_tags = entity.get("timing_tags") or []
    if isinstance(timing_tags, str):
        timing_tags = [tag.strip() for tag in timing_tags.split(",") if tag.strip()]

    if failure_file_path and _same_file(file_path, failure_file_path):
        score += 4.5
        try:
            start_line = int(entity.get("start_line") or 0)
            end_line = int(entity.get("end_line") or 0)
        except (TypeError, ValueError):
            start_line = end_line = 0
        if failure_line_number > 0 and start_line > 0 and end_line >= start_line:
            if start_line <= failure_line_number <= end_line:
                score += 3.0
            else:
                distance = min(abs(failure_line_number - start_line), abs(failure_line_number - end_line))
                score += max(0.0, 2.8 - min(distance, 40) / 16.0)
        else:
            score += 1.2

    if observed_signal and observed_signal.lower() in text_blob:
        score += 3.2
    if expected_idle and expected_idle.lower() in text_blob:
        score += 1.4

    matched_tokens = 0
    for token in seed_tokens:
        if token and token in text_blob:
            matched_tokens += 1
    score += min(matched_tokens, 6) * 0.45

    if role == "direct_driver":
        score += 3.2
    elif role == "top_level_wiring":
        score += 2.5
    elif role == "config_register":
        score += 2.0
    elif role == "side_evidence":
        score += 0.9
    elif role == "issue":
        score += 0.4

    if label in {"signal", "port", "parameter", "macro", "state", "branch", "condition", "generateblock", "conditionalcompilationscope", "testbench", "assertion"}:
        score += 1.4
    if verilog_kind in {"assign", "always", "always_ff", "always_comb", "always_latch", "function", "task", "instance", "module_body"}:
        score += 0.8

    if "immediate_observable" in timing_tags:
        score += 0.4
    if "registered_output" in timing_tags:
        score += 0.35
    if "edge_sensitive" in timing_tags:
        score += 0.2

    if timing_sensitive and role in {"direct_driver", "top_level_wiring", "config_register"}:
        score += 0.8
    if path_text:
        score += min(len(path_text.split("|")), 4) * 0.15
    return score


def _pick_top_entities(entities: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(list(entities or []), key=lambda item: (
        -float(item.get("_slice_score") or 0.0),
        str(item.get("file_path") or ""),
        int(item.get("start_line") or 0),
        str(item.get("name") or item.get("signature") or ""),
    ))
    deduped = []
    seen = set()
    for entity in ordered:
        marker = (
            str(entity.get("name") or entity.get("signature") or ""),
            str(entity.get("file_path") or ""),
            int(entity.get("start_line") or 0),
            int(entity.get("end_line") or 0),
            str(entity.get("verilog_kind") or entity.get("rtl_kind") or ""),
        )
        if marker in seen:
            continue
        deduped.append({k: v for k, v in entity.items() if k != "_slice_score"})
        seen.add(marker)
        if len(deduped) >= limit:
            break
    return deduped


def _collect_candidate_spans(entities: Sequence[Dict[str, Any]], failure_file_path: str, failure_line_number: int) -> List[Dict[str, Any]]:
    spans = []
    seen = set()
    for entity in entities:
        file_path = _normalize_repo_path(_entity_file(entity))
        try:
            start_line = int(entity.get("start_line") or 0)
            end_line = int(entity.get("end_line") or 0)
        except (TypeError, ValueError):
            start_line = end_line = 0
        if not file_path or start_line <= 0 or end_line <= 0:
            continue
        marker = (file_path, start_line, end_line)
        if marker in seen:
            continue
        seen.add(marker)
        spans.append({
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "name": _entity_name(entity),
            "role": _entity_role(entity),
            "reason": "selected_by_kg_slice",
        })

    if failure_file_path:
        failure_file_path = _normalize_repo_path(failure_file_path)
        if failure_file_path:
            marker = (failure_file_path, max(1, failure_line_number - 12), failure_line_number + 12)
            if marker not in seen:
                spans.append({
                    "file_path": failure_file_path,
                    "start_line": marker[1],
                    "end_line": marker[2],
                    "name": "validation_failure_anchor",
                    "role": "failure_anchor",
                    "reason": "validation_failure_window",
                })
    return spans


def _collect_fault_anchor_entities(
    entities: Sequence[Dict[str, Any]],
    failure_file_path: str,
    failure_line_number: int,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    failure_file_path = _normalize_repo_path(failure_file_path)
    if not failure_file_path:
        return []

    anchored = []
    for entity in entities:
        if not _same_file(_entity_file(entity), failure_file_path):
            continue
        try:
            start_line = int(entity.get("start_line") or 0)
            end_line = int(entity.get("end_line") or 0)
        except (TypeError, ValueError):
            start_line = end_line = 0
        if start_line <= 0 or end_line <= 0:
            continue
        if failure_line_number > 0 and start_line <= failure_line_number <= end_line:
            distance = 0
        elif failure_line_number > 0:
            distance = min(abs(failure_line_number - start_line), abs(failure_line_number - end_line))
        else:
            distance = max(0, start_line - 1)
        enriched = dict(entity)
        enriched["_anchor_distance"] = distance
        enriched["_anchor_reason"] = "contains_failure_line" if distance == 0 else "nearest_entity_to_failure_line"
        anchored.append(enriched)

    anchored.sort(
        key=lambda item: (
            int(item.get("_anchor_distance") or 0),
            int(item.get("start_line") or 0),
            int(item.get("end_line") or 0),
            str(item.get("file_path") or ""),
            str(item.get("name") or item.get("signature") or ""),
        )
    )

    deduped = []
    seen = set()
    for entity in anchored:
        marker = (
            str(entity.get("name") or entity.get("signature") or ""),
            str(entity.get("file_path") or ""),
            int(entity.get("start_line") or 0),
            int(entity.get("end_line") or 0),
            str(entity.get("verilog_kind") or entity.get("rtl_kind") or ""),
        )
        if marker in seen:
            continue
        deduped.append(entity)
        seen.add(marker)
        if len(deduped) >= limit:
            break
    return deduped


def _fault_anchor_spans(
    failure_file_path: str,
    failure_line_number: int,
    fault_anchor_entities: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    spans = []
    seen = set()
    for entity in fault_anchor_entities:
        file_path = _normalize_repo_path(_entity_file(entity))
        try:
            start_line = int(entity.get("start_line") or 0)
            end_line = int(entity.get("end_line") or 0)
        except (TypeError, ValueError):
            start_line = end_line = 0
        if not file_path or start_line <= 0 or end_line <= 0:
            continue
        marker = (file_path, start_line, end_line)
        if marker in seen:
            continue
        seen.add(marker)
        spans.append({
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "name": _entity_name(entity),
            "role": _entity_role(entity),
            "reason": entity.get("_anchor_reason") or "fault_anchor_entity",
        })

    if not spans and failure_file_path:
        failure_file_path = _normalize_repo_path(failure_file_path)
        if failure_file_path:
            if failure_line_number > 0:
                start_line = max(1, failure_line_number - 12)
                end_line = failure_line_number + 12
            else:
                start_line = 1
                end_line = 24
            spans.append({
                "file_path": failure_file_path,
                "start_line": start_line,
                "end_line": end_line,
                "name": "validation_failure_anchor",
                "role": "fault_anchor",
                "reason": "validation_failure_window",
            })
    return spans


def _render_paths(entities: Sequence[Dict[str, Any]], limit: int = 8) -> List[str]:
    items = []
    for entity in entities:
        path_text = _entity_path_text(entity)
        if path_text:
            items.append(path_text)
    return _unique_keep_order(items)[:limit]


def build_verilog_fault_neighborhood(
    related_entities: Dict[str, Sequence[Dict[str, Any]]],
    *,
    issue_text: str = "",
    analysis: Any = None,
    repo_root: str = "",
    limit_per_group: int = 6,
    max_paths: int = 8,
    max_nodes: int = 18,
    max_files: int = 5,
) -> Dict[str, Any]:
    normalized = normalize_verilog_related_entities(related_entities or {})
    digest = _build_verilog_bug_kg_digest_base(
        normalized,
        issue_text=issue_text,
        analysis=analysis,
        limit_per_group=limit_per_group,
        max_paths=max_paths,
    )

    failure_signature = str(getattr(analysis, "signature", "") or "").strip()
    failure_summary = str(getattr(analysis, "summary", "") or "").strip()
    failure_reason = str(getattr(analysis, "failure_reason", "") or "").strip()
    failure_file_path = _normalize_repo_path(getattr(analysis, "failure_file_path", "") or "", repo_root=repo_root)
    failure_line_number = int(getattr(analysis, "failure_line_number", 0) or 0)
    observed_signal = str(getattr(analysis, "observed_signal", "") or "").strip()
    expected_idle = str(getattr(analysis, "expected_idle", "") or "").strip()
    timing_signature = str(getattr(analysis, "timing_signature", "") or "").strip()
    timing_sensitive = bool(getattr(analysis, "timing_sensitive", False))

    seed_tokens = _signal_tokens(
        issue_text,
        failure_signature,
        failure_summary,
        failure_reason,
        observed_signal,
        expected_idle,
        str(getattr(analysis, "failure_step", "") or ""),
        str(getattr(analysis, "failure_kind", "") or ""),
    )

    all_entities: List[Dict[str, Any]] = []
    for group_name in (
        "patchable_spans",
        "direct_drivers",
        "top_level_wiring",
        "config_registers",
        "edit_targets",
        "evidence_entities",
        "rtl_entities",
        "methods",
        "issues",
    ):
        for entity in normalized.get(group_name) or []:
            if not isinstance(entity, dict):
                continue
            enriched = dict(entity)
            enriched["_group"] = group_name
            enriched["_slice_score"] = _score_entity(
                enriched,
                failure_file_path=failure_file_path,
                failure_line_number=failure_line_number,
                observed_signal=observed_signal,
                expected_idle=expected_idle,
                seed_tokens=seed_tokens,
                timing_sensitive=timing_sensitive,
            )
            all_entities.append(enriched)

    if failure_file_path:
        for entity in all_entities:
            if _same_file(_entity_file(entity), failure_file_path):
                entity["_slice_score"] += 0.6

    ranked_entities = sorted(
        all_entities,
        key=lambda item: (
            -float(item.get("_slice_score") or 0.0),
            str(item.get("file_path") or ""),
            int(item.get("start_line") or 0),
            str(item.get("name") or item.get("signature") or ""),
        ),
    )

    primary_targets = _pick_top_entities(
        [
            entity for entity in ranked_entities
            if str(entity.get("type") or entity.get("label") or "").lower() in {"patchable_span", "patchablespan"}
            or _entity_role(entity) in {"direct_driver", "top_level_wiring", "config_register"}
        ],
        max_nodes // 2,
    )
    if not primary_targets:
        primary_targets = _pick_top_entities(ranked_entities, max_nodes // 2)

    repair_anchor_entities = _pick_top_entities(
        [
            entity for entity in primary_targets
            if str(entity.get("label") or "").lower() not in {"testbench", "assertion"}
            and _entity_role(entity) not in {"testbench", "assertion"}
        ],
        max_nodes // 2,
    )
    if not repair_anchor_entities:
        repair_anchor_entities = _pick_top_entities(primary_targets, max_nodes // 2)

    evidence_entities = _pick_top_entities(
        [
            entity for entity in ranked_entities
            if str(entity.get("label") or "").lower() in {"signal", "port", "parameter", "macro", "state", "generateblock", "conditionalcompilationscope", "testbench", "assertion"}
        ],
        max_nodes,
    )
    if not evidence_entities:
        evidence_entities = _pick_top_entities(ranked_entities, max_nodes // 2)

    bridge_entities = _pick_top_entities(
        [
            entity for entity in ranked_entities
            if _entity_path_text(entity)
        ],
        max_nodes,
    )

    issue_entities = _pick_top_entities(normalized.get("issues") or [], 3)
    all_selected = _unique_keep_order(repair_anchor_entities + primary_targets + bridge_entities + evidence_entities + issue_entities)
    fault_anchor_entities = _collect_fault_anchor_entities(all_selected, failure_file_path, failure_line_number)
    if fault_anchor_entities:
        primary_targets = _unique_keep_order([*fault_anchor_entities, *primary_targets])[: max_nodes // 2]
    candidate_files = _unique_keep_order([
        _normalize_repo_path(_entity_file(entity), repo_root=repo_root)
        for entity in all_selected
        if _entity_file(entity)
    ])
    if failure_file_path:
        candidate_files = _unique_keep_order([failure_file_path, *candidate_files])
    candidate_files = [item for item in candidate_files if item][:max_files * 2]

    fault_anchor_spans = _fault_anchor_spans(failure_file_path, failure_line_number, fault_anchor_entities)
    candidate_spans = _unique_keep_order([
        *fault_anchor_spans,
        *_collect_candidate_spans(all_selected, failure_file_path, failure_line_number),
    ])
    path_hints = _render_paths(all_selected, limit=max_paths)
    critical_paths = path_hints[:max_paths]

    trace_templates = list(digest.get("trace_templates") or [])
    if failure_file_path and observed_signal:
        trace_templates.insert(0, f"Validation failure at {failure_file_path}:{failure_line_number or '?'} for {observed_signal}")
    elif failure_file_path:
        trace_templates.insert(0, f"Validation failure anchored in {failure_file_path}:{failure_line_number or '?'}")
    if not trace_templates:
        trace_templates = ["Issue -> Fault Anchor -> Direct Driver -> Observed Output"]

    critical_nodes = []
    for entity in fault_anchor_entities[:max_nodes]:
        critical_nodes.append({
            "category": "fault_anchor",
            **{k: v for k, v in entity.items() if k != "_slice_score"},
        })
    for entity in repair_anchor_entities[:max_nodes]:
        critical_nodes.append({
            "category": "repair_anchor",
            **{k: v for k, v in entity.items() if k != "_slice_score"},
        })
    for entity in primary_targets[:max_nodes]:
        critical_nodes.append({
            "category": "edit_target",
            **{k: v for k, v in entity.items() if k != "_slice_score"},
        })
    for entity in evidence_entities[:max_nodes]:
        critical_nodes.append({
            "category": "evidence",
            **{k: v for k, v in entity.items() if k != "_slice_score"},
        })
    for entity in issue_entities[:max_nodes]:
        critical_nodes.append({
            "category": "issue",
            **{k: v for k, v in entity.items() if k != "_slice_score"},
        })

    graph_edges = []
    if failure_file_path:
        for entity in primary_targets[:6] + evidence_entities[:6]:
            graph_edges.append({
                "source": failure_file_path,
                "target": _entity_name(entity),
                "type": "CONTAINS" if _same_file(_entity_file(entity), failure_file_path) else "RELATED",
                "relation_kind": "contains" if _same_file(_entity_file(entity), failure_file_path) else "related",
                "description": "fault anchor to candidate entity" if not _same_file(_entity_file(entity), failure_file_path) else "file contains candidate entity",
            })
    for entity in all_selected[:max_nodes]:
        for step in entity.get("path") or []:
            if not isinstance(step, dict):
                continue
            graph_edges.append({
                "source": str(step.get("start_node") or ""),
                "target": str(step.get("end_node") or ""),
                "type": str(step.get("type") or step.get("relation_kind") or "RELATED"),
                "relation_kind": str(step.get("relation_kind") or step.get("type") or "related"),
                "description": str(step.get("description") or ""),
            })

    counterevidence = []
    for entity in evidence_entities:
        role = _entity_role(entity)
        label = str(entity.get("label") or "").lower()
        if role == "testbench" or label == "testbench":
            counterevidence.append(_entity_summary(entity, include_path=False))
        elif role == "assertion" or label == "assertion":
            counterevidence.append(_entity_summary(entity, include_path=False))
    counterevidence = _unique_keep_order(counterevidence)[:limit_per_group]

    text_lines = [
        "## Bug KG Slice",
        f"- issue: {_compact_text(issue_text, 500)}" if issue_text else "- issue: (not provided)",
        f"- failure_signature: {failure_signature}" if failure_signature else "- failure_signature: (not provided)",
        f"- failure_summary: {_compact_text(failure_summary, 500)}" if failure_summary else "",
        f"- failure_file_path: {failure_file_path}" if failure_file_path else "",
        f"- failure_line_number: {failure_line_number}" if failure_line_number else "",
        f"- observed_signal: {observed_signal}" if observed_signal else "",
        f"- expected_idle: {expected_idle}" if expected_idle else "",
        f"- timing_signature: {timing_signature}" if timing_signature else "",
        "- trace_templates:",
        *[f"  - {item}" for item in trace_templates[:max_paths]],
        "- fault_anchor_entities:",
        *[_entity_summary(entity) for entity in fault_anchor_entities[:limit_per_group]],
        "- repair_anchor_entities:",
        *[_entity_summary(entity) for entity in repair_anchor_entities[:limit_per_group]],
        "- primary_edit_targets:",
        *[_entity_summary(entity) for entity in primary_targets[:limit_per_group]],
        "- bridge_entities:",
        *[_entity_summary(entity) for entity in bridge_entities[:limit_per_group]],
        "- evidence_entities:",
        *[_entity_summary(entity) for entity in evidence_entities[:limit_per_group]],
        "- counterevidence:",
        *[f"  - {item}" for item in counterevidence[:limit_per_group]],
        "- candidate_files:",
        *[f"  - {item}" for item in candidate_files[:max_files]],
        "- candidate_spans:",
        *[f"  - {item['file_path']}:{item['start_line']}-{item['end_line']} {item.get('name', '')} [{item.get('role', '')}] {item.get('reason', '')}" for item in candidate_spans[:max_files * 2]],
        "- critical_paths:",
        *[f"  - {item}" for item in critical_paths[:max_paths]],
        "",
        "Use this slice to follow the fault neighborhood first. Prefer the direct driver chain and the failure anchor file before broad repo context.",
    ]
    text = "\n".join(line for line in text_lines if line is not None and str(line).strip())

    return {
        "issue_text": _compact_text(issue_text, 1200),
        "failure_signature": failure_signature,
        "failure_summary": _compact_text(failure_summary, 1000),
        "failure_file_path": failure_file_path,
        "failure_line_number": failure_line_number,
        "observed_signal": observed_signal,
        "expected_idle": expected_idle,
        "timing_signature": timing_signature,
        "timing_sensitive": timing_sensitive,
        "trace_templates": trace_templates,
        "fault_anchor_entities": fault_anchor_entities,
        "fault_anchor_spans": fault_anchor_spans[:max_files * 2],
        "repair_anchor_entities": repair_anchor_entities,
        "primary_edit_targets": primary_targets,
        "bridge_entities": bridge_entities,
        "evidence_entities": evidence_entities,
        "counterevidence": counterevidence,
        "candidate_files": candidate_files,
        "candidate_spans": candidate_spans,
        "critical_paths": critical_paths,
        "critical_nodes": critical_nodes,
        "graph_edges": graph_edges,
        "text": text,
        "normalized_related_entities": normalized,
    }


# Compatibility alias for existing callers and docs.
build_verilog_bug_kg_digest = build_verilog_fault_neighborhood
