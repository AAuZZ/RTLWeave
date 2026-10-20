from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Sequence


VERILOG_TIMING_TAGS = {
    "edge_sensitive",
    "registered_output",
    "combinational_idle",
    "immediate_observable",
}

VERILOG_REPAIR_ROLE_ORDER = {
    "direct_driver": 0,
    "top_level_wiring": 1,
    "config_register": 2,
    "side_evidence": 3,
    "unknown": 4,
}


def _compact_text(text: str, limit: int = 220) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _unique_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        ordered.append(item)
        seen.add(marker)
    return ordered


def _basename(path: str) -> str:
    return os.path.basename((path or "").replace("\\", "/")).lower()


def classify_verilog_entity_timing(entity: Dict[str, Any]) -> Dict[str, Any]:
    verilog_kind = str(entity.get("verilog_kind") or entity.get("rtl_kind") or "").strip().lower()
    source = str(entity.get("source_code") or entity.get("semantic_summary") or entity.get("declaration") or "").lower()
    file_path = str(entity.get("file_path") or "")
    module_name = str(entity.get("module_name") or "")
    signal_name = str(entity.get("signal_name") or "")
    name = str(entity.get("name") or "")
    assign_target = str(entity.get("target_signal") or "").lower()
    basename = _basename(file_path)

    tags: List[str] = []
    role = str(entity.get("repair_role") or entity.get("timing_role") or "").strip().lower()

    if verilog_kind in {"always_ff", "always_latch", "always"}:
        tags.append("edge_sensitive")
    if verilog_kind in {"assign", "always_comb"}:
        tags.extend(["combinational_idle", "immediate_observable"])
    if verilog_kind == "instance":
        tags.append("registered_output")
    if verilog_kind == "module_body":
        if "top" in basename or "ctrl" in basename:
            tags.append("registered_output")
            tags.append("immediate_observable")
        if "regs" in basename:
            tags.append("edge_sensitive")

    if any(token in source for token in ("cpol", "cpha", "sck", "idle", "cs_n")):
        tags.append("registered_output")
    if "bus_write" in source or "write" in source or "observe" in source:
        tags.append("immediate_observable")
    if "assign" in source or verilog_kind == "assign":
        tags.append("combinational_idle")

    if not role:
        if verilog_kind == "instance":
            role = "top_level_wiring"
        elif "regs" in basename or "register" in basename or "csr" in basename:
            role = "config_register"
        elif verilog_kind in {"assign", "always", "always_ff", "always_comb", "always_latch", "function", "task", "module_body"}:
            role = "direct_driver"
        elif "top" in basename or "ctrl_top" in basename:
            role = "top_level_wiring"
        else:
            role = "side_evidence"

    if role == "direct_driver" and (assign_target or signal_name):
        if any(token in (assign_target + " " + signal_name).lower() for token in ("sck", "spi", "mosi", "miso", "clk")):
            tags.append("registered_output")
    if role == "config_register" and "bus_we" in source:
        tags.append("registered_output")
    if role == "top_level_wiring":
        tags.append("immediate_observable")

    tags = [tag for tag in _unique_keep_order(tags) if tag in VERILOG_TIMING_TAGS]
    if not tags:
        tags = ["immediate_observable"] if role != "side_evidence" else []

    base_priority = {
        "direct_driver": 1.0,
        "top_level_wiring": 0.84,
        "config_register": 0.78,
        "side_evidence": 0.55,
        "unknown": 0.45,
    }.get(role or "unknown", 0.45)
    tag_boost = 0.0
    if "edge_sensitive" in tags:
        tag_boost += 0.08
    if "registered_output" in tags:
        tag_boost += 0.10
    if "combinational_idle" in tags:
        tag_boost += 0.05
    if "immediate_observable" in tags:
        tag_boost += 0.12

    timing_summary_bits = [
        f"repair_role={role}",
        f"timing_tags={','.join(tags) if tags else 'none'}",
    ]
    if module_name:
        timing_summary_bits.append(f"module={module_name}")
    if signal_name:
        timing_summary_bits.append(f"signal={signal_name}")
    if file_path:
        timing_summary_bits.append(f"file={file_path}")

    return {
        "repair_role": role,
        "timing_tags": tags,
        "timing_priority": round(base_priority + tag_boost, 4),
        "timing_summary": "; ".join(timing_summary_bits),
    }


def annotate_verilog_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(entity)
    enriched.update(classify_verilog_entity_timing(enriched))
    return enriched


def sort_verilog_entities(entities: Sequence[Dict[str, Any]], fallback_group: str | None = None) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]):
        timing_priority = item.get("timing_priority")
        if timing_priority is None:
            timing_priority = classify_verilog_entity_timing(item)["timing_priority"]
        try:
            similarity = float(item.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        try:
            parse_confidence = float(item.get("parse_confidence") or 0.0)
        except (TypeError, ValueError):
            parse_confidence = 0.0
        try:
            distance = float(item.get("distance") or 0.0)
        except (TypeError, ValueError):
            distance = 0.0
        role = str(item.get("repair_role") or item.get("entity_role") or "").strip().lower() or "unknown"
        role_priority = VERILOG_REPAIR_ROLE_ORDER.get(role, VERILOG_REPAIR_ROLE_ORDER["unknown"])
        group_priority = 0
        if fallback_group in {"edit_targets", "direct_drivers"}:
            group_priority = 0
        elif fallback_group in {"wiring_targets", "top_level_wiring"}:
            group_priority = 1
        elif fallback_group in {"config_targets", "config_registers"}:
            group_priority = 2
        elif fallback_group in {"evidence_entities", "side_evidence"}:
            group_priority = 3
        return (
            group_priority,
            role_priority,
            -float(timing_priority),
            -similarity,
            -parse_confidence,
            distance,
            str(item.get("file_path") or ""),
            int(item.get("start_line") or 0),
            str(item.get("name") or item.get("signature") or ""),
        )

    return sorted(list(entities or []), key=key)


def sort_patchable_spans(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve the localization score as the primary PatchableSpan order."""
    def key(item: Dict[str, Any]):
        if item.get("global_rank") is not None:
            try:
                return (0, int(item.get("global_rank")), 0, "", 0, "")
            except (TypeError, ValueError):
                pass
        try:
            similarity = float(item.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        try:
            distance = float(item.get("distance") or 0.0)
        except (TypeError, ValueError):
            distance = 0.0
        return (
            1,
            -similarity,
            distance,
            str(item.get("file_path") or ""),
            int(item.get("start_line") or 0),
            str(item.get("name") or ""),
        )

    return sorted(list(entities or []), key=key)


def classify_verilog_location_groups(related_entities: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {
        "direct_drivers": [],
        "top_level_wiring": [],
        "config_registers": [],
        "side_evidence": [],
    }

    patchable = list(related_entities.get("patchable_spans") or [])
    if not patchable:
        patchable = [
            item for item in related_entities.get("edit_targets", []) or []
            if str(item.get("type") or item.get("label") or "").lower() in {"patchable_span", "patchablespan"}
        ]
    edit_groups = ("patchable_spans",) if patchable else ("edit_targets", "methods")

    for group_name in edit_groups:
        for entity in related_entities.get(group_name, []) or []:
            enriched = annotate_verilog_entity(entity)
            role = enriched.get("repair_role") or "side_evidence"
            if role == "direct_driver":
                buckets["direct_drivers"].append(enriched)
            elif role == "top_level_wiring":
                buckets["top_level_wiring"].append(enriched)
            elif role == "config_register":
                buckets["config_registers"].append(enriched)
            else:
                buckets["side_evidence"].append(enriched)

    for group_name in ("rtl_entities", "evidence_entities", "signals", "entities"):
        for entity in related_entities.get(group_name, []) or []:
            if str(entity.get("type") or entity.get("label") or "").lower() in {"patchable_span", "patchablespan"}:
                continue
            buckets["side_evidence"].append(annotate_verilog_entity(entity))

    for key_name in list(buckets):
        buckets[key_name] = sort_verilog_entities(buckets[key_name], fallback_group=key_name)

    buckets["edit_targets"] = (
        sort_patchable_spans([annotate_verilog_entity(item) for item in patchable])
        if patchable
        else sort_verilog_entities(
            buckets["direct_drivers"] + buckets["top_level_wiring"] + buckets["config_registers"],
            fallback_group="edit_targets",
        )
    )
    buckets["evidence_entities"] = sort_verilog_entities(buckets["side_evidence"], fallback_group="evidence_entities")
    return buckets


def _entity_identity(entity: Dict[str, Any]) -> tuple:
    return (
        str(entity.get("name") or entity.get("signature") or ""),
        str(entity.get("signature") or ""),
        str(entity.get("file_path") or ""),
        int(entity.get("start_line") or 0),
        int(entity.get("end_line") or 0),
        str(entity.get("verilog_kind") or entity.get("rtl_kind") or ""),
    )


def _dedupe_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for entity in entities or []:
        deduped[_entity_identity(entity)] = entity
    return list(deduped.values())


def normalize_verilog_related_entities(related_entities: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    normalized: Dict[str, List[Dict[str, Any]]] = dict(related_entities or {})

    methods = sort_verilog_entities(
        [annotate_verilog_entity(item) for item in normalized.get("methods", []) or []],
        fallback_group="methods",
    )
    rtl_entities = sort_verilog_entities(
        [annotate_verilog_entity(item) for item in normalized.get("rtl_entities", []) or []],
        fallback_group="rtl_entities",
    )
    classes = list(normalized.get("classes", []) or [])
    issues = list(normalized.get("issues", []) or [])

    patchable_spans = list(normalized.get("patchable_spans", []) or [])
    if not patchable_spans:
        patchable_spans = [
            item for item in normalized.get("edit_targets", []) or []
            if str(item.get("type") or item.get("label") or "").lower() in {"patchable_span", "patchablespan"}
        ]
    patchable_spans = sort_patchable_spans(
        [annotate_verilog_entity(item) for item in _dedupe_entities(patchable_spans)]
    )

    if patchable_spans:
        hierarchy = classify_verilog_location_groups({
            "patchable_spans": patchable_spans,
            "rtl_entities": rtl_entities,
        })
        normalized["patchable_spans"] = patchable_spans
        normalized["methods"] = methods
        normalized["rtl_entities"] = rtl_entities
        normalized["classes"] = classes
        normalized["issues"] = issues
        normalized["direct_drivers"] = hierarchy["direct_drivers"]
        normalized["top_level_wiring"] = hierarchy["top_level_wiring"]
        normalized["config_registers"] = hierarchy["config_registers"]
        normalized["side_evidence"] = hierarchy["side_evidence"]
        normalized["edit_targets"] = patchable_spans
        normalized["evidence_entities"] = _dedupe_entities(rtl_entities)
        return normalized

    base_verilog_entities = _dedupe_entities(
        list(methods)
        + list(rtl_entities)
        + [annotate_verilog_entity(item) for item in normalized.get("edit_targets", []) or []]
        + [annotate_verilog_entity(item) for item in normalized.get("evidence_entities", []) or []]
    )

    hierarchy = classify_verilog_location_groups({
        "methods": base_verilog_entities,
        "rtl_entities": base_verilog_entities,
    })

    normalized["methods"] = methods
    normalized["rtl_entities"] = rtl_entities
    normalized["classes"] = classes
    normalized["issues"] = issues
    normalized["direct_drivers"] = hierarchy["direct_drivers"]
    normalized["top_level_wiring"] = hierarchy["top_level_wiring"]
    normalized["config_registers"] = hierarchy["config_registers"]
    normalized["side_evidence"] = hierarchy["side_evidence"]
    normalized["edit_targets"] = hierarchy["edit_targets"]
    normalized["evidence_entities"] = hierarchy["evidence_entities"]
    return normalized


_WRITE_OBSERVE_RE = re.compile(
    r"if\s*\(\s*([A-Za-z_][\w$]*)\s*!==\s*(1'b[01])\s*\)",
    re.IGNORECASE,
)


def classify_timing_failure_excerpt(source_excerpt: str, failure_text: str = "") -> Dict[str, Any]:
    lines = [line.rstrip() for line in (source_excerpt or "").splitlines() if line.strip()]
    joined = "\n".join(lines)
    failure_text = failure_text or ""

    if not lines:
        return {
            "timing_signature": "",
            "timing_sensitive": False,
            "timing_hints": [],
            "stimulus_window": "",
            "observed_signal": "",
            "expected_idle": "",
        }

    hit_index = -1
    for idx, line in enumerate(lines):
        if "bus_write(" in line or "write(" in line:
            hit_index = idx
            break

    observed_signal = ""
    expected_idle = ""
    immediate_window = ""
    timing_hints: List[str] = []
    if hit_index >= 0:
        lookahead = lines[hit_index : min(len(lines), hit_index + 6)]
        immediate_window = "\n".join(lookahead)
        for candidate in lookahead[1:]:
            match = _WRITE_OBSERVE_RE.search(candidate)
            if match:
                observed_signal = match.group(1)
                expected_idle = match.group(2)
                timing_hints.append("immediate_observable")
                timing_hints.append("registered_output")
                if "sck" in observed_signal.lower():
                    timing_hints.append("edge_sensitive")
                break

    if observed_signal and (immediate_window or "FAIL:" in failure_text or "FATAL:" in failure_text):
        timing_signature = "immediate_post_write_idle_state"
        timing_sensitive = True
    else:
        timing_signature = ""
        timing_sensitive = False

    return {
        "timing_signature": timing_signature,
        "timing_sensitive": timing_sensitive,
        "timing_hints": _unique_keep_order([hint for hint in timing_hints if hint in VERILOG_TIMING_TAGS]),
        "stimulus_window": immediate_window or joined[:1200],
        "observed_signal": observed_signal,
        "expected_idle": expected_idle,
    }
