"""
Failure-conditioned KG reslicing for Debug phase.

Pipeline:
  error_feedback → extract failure signals → Neo4j backward trace →
  locate suspected broken chain → build debug_kg_digest → LLM diagnosis

This reinstates KG into the Debug loop so that every debug iteration
is guided by structured signal-flow analysis rather than raw LLM guessing.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    from config import (
        DECAY_FACTOR,
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        VECTOR_SIMILARITY_WEIGHT,
    )
    from knowledge_graph import KnowledgeGraph
    from rtl_causal_localization import RTLCausalLocalizer
    from term import warn as twarn
except Exception:  # pragma: no cover - package import path
    from .config import (
        DECAY_FACTOR,
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        VECTOR_SIMILARITY_WEIGHT,
    )
    from .knowledge_graph import KnowledgeGraph
    from .rtl_causal_localization import RTLCausalLocalizer
    from .term import warn as twarn


# ---------------------------------------------------------------------------
# 1.  Signal / module / file extraction from validation error output
# ---------------------------------------------------------------------------

_SIGNAL_PATTERN = re.compile(
    r"""
    (?:
        (?P<signal_word>
            signal|port|wire|reg|bus|flag|status|state|ack|req|done|err|rdy|valid
            |addr|data|id|sel|en|we|re|mask|cnt|ptr|fifo|dma|irq|intr|ctrl|cfg|cfg_
        )
        _?[A-Za-z0-9_]*
        |
        [A-Za-z_][A-Za-z0-9_]*          # generic Verilog identifier
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MODULE_PATTERN = re.compile(
    r"""
    \b(?:module|instance|instantiate)\s+
    (?P<module_name>[A-Za-z_][A-Za-z0-9_]+(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)
    (?!\s*\.\s*(?:sv|v|svh|vh)\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_FILE_PATTERN = re.compile(
    r"""
    (?:
        (?:^|\s|"|')
        (?P<file_path>
            (?:\.\.?/)?
            [A-Za-z0-9_./-]+
            \.(?:sv|v|svh|vh)
        )
        (?:\s|"|'|:|$)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_TEST_NAME_PATTERN = re.compile(
    r"""
    (?:TEST|test)\s*[:=]?\s*(?P<test_name>\w+)
    |
    (?P<fail_test>\w+)\s*\.{3,}\s*FAIL
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Low-information tokens commonly found in error output but useless as KG query targets.
# These are common English / Verilator / UVM words, not design signals.
_LOW_INFO_SIGNALS: Set[str] = {
    # English noise
    "test", "fail", "passed", "failed", "expected", "got",
    "time", "line", "block", "entry", "at", "of", "in", "on", "to", "by", "with",
    "from", "for", "the", "and", "or", "not", "is", "was", "are", "be",
    "value", "values", "bits", "width", "size", "count", "index",
    "assert", "assertion", "mismatch", "error", "warning", "info", "blocker",
    "hwe", "bench", "docker", "validation", "validator", "evidence",
    "hwe_bench_docker_final",
    "kgveri", "internal", "structured", "raw", "check", "status",
    "returncode", "output", "tail", "candidate", "patch",
    "minitb", "observed", "cycle",
    "tb_fail", "tb_pass", "sim_fail", "sim_pass", "hwe_bench_results_start",
    "hwe_bench_results_end",
    "module", "instance", "cell", "port", "signal", "wire", "reg",
    "parameter", "macro", "state", "type", "class", "method",
    "read", "write", "drive", "connect", "feed",
    "input", "output", "inout", "ref",
    "always", "always_ff", "always_comb", "initial",
    "posedge", "negedge", "edge",
    # environment
    "test_en", "test_en_i", "scan_enable", "scan_mode",
    "clk", "rst", "reset", "rst_n", "rst_ni", "clk_i", "rst_ni",
    "clk_ungated", "clk_gate_en", "dft_enable",
    "zero", "one", "ones", "zeros",
    "vdd", "vss", "gnd", "vcc",
    "tie_hi", "tie_lo", "tie_high", "tie_low",
    # time/unit
    "ns", "ps", "us", "ms", "hz", "mhz", "ghz",
    "true", "false", "high", "low", "ended",
    # testbench observation qualifiers; keep the design-bearing identifier
    # components (for example, ``cpuctrl`` in ``cpuctrl_after_first``).
    "after", "before", "first", "second", "last", "previous", "next",
    "current", "actual", "gold", "reference", "result",
}


def _extract_failure_signals(error_text: str) -> Dict[str, List[str]]:
    """Extract signal names, module names, file paths and test names from error output.

    Returns a dict with keys: signals, modules, files, tests.
    """
    signals: List[str] = []
    modules: List[str] = []
    files: List[str] = []
    tests: List[str] = []

    explicit_signals = [
        match.group(1).lower()
        for match in re.finditer(
            r"\b(?:signal|port|wire|reg)\s*[:=]\s*([A-Za-z_][A-Za-z0-9_$]*)",
            error_text or "",
            re.IGNORECASE,
        )
    ]
    if explicit_signals:
        signals.extend(dict.fromkeys(explicit_signals))
    else:
        for m in _SIGNAL_PATTERN.finditer(error_text):
            token = m.group().strip().lower()
            if token and len(token) > 2 and token not in _LOW_INFO_SIGNALS:
                if token not in signals:
                    signals.append(token)

    for m in _MODULE_PATTERN.finditer(error_text):
        name = m.group("module_name").strip()
        # skip file-like names ending with .sv/.v/.svh
        if re.search(r"\.(sv|v|svh|vh)$", name, re.IGNORECASE):
            continue
        if name and name not in modules:
            modules.append(name)

    for m in _FILE_PATTERN.finditer(error_text):
        path = m.group("file_path").strip()
        if path and path not in files:
            files.append(path)

    for m in _TEST_NAME_PATTERN.finditer(error_text):
        name = (m.group("test_name") or m.group("fail_test") or "").strip()
        if name and name not in tests:
            tests.append(name)

    return {
        "signals": signals[:20],
        "modules": modules[:10],
        "files": files[:10],
        "tests": tests[:10],
    }


def _signal_query_terms(signal_names: Sequence[str], limit: int = 20) -> List[str]:
    """Expand testbench observation names into conservative RTL lookup terms.

    Validators often expose aliases such as ``CPUCTRL_AFTER_FIRST`` rather than
    the repository identifier ``cpuctrl_q``.  Exact aliases remain the first
    lookup terms, followed by non-generic identifier components.  Short and
    observation-only components are discarded to avoid broad graph matches.
    """
    terms: List[str] = []
    for raw_name in signal_names:
        normalized = str(raw_name or "").strip().lower().strip("_$")
        if len(normalized) >= 4 and normalized not in _LOW_INFO_SIGNALS:
            terms.append(normalized)
        camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(raw_name or ""))
        for token in re.split(r"[^A-Za-z0-9$]+|_+", camel_split):
            token = token.lower().strip("_$")
            if len(token) < 4 or token in _LOW_INFO_SIGNALS:
                continue
            terms.append(token)
    return list(dict.fromkeys(terms))[:max(1, int(limit))]


# ---------------------------------------------------------------------------
# 2.  Neo4j backward trace from failure signals
# ---------------------------------------------------------------------------

_BACKTRACE_QUERY = """
// Given a set of candidate signal names, trace backwards through
// DRIVES/WRITES/CONNECTS/FEEDS/CLOCKED_BY/RESET_BY to find upstream driving logic.
// Also find methods that READ or WRITE the signal.
WITH $candidates AS candidates
UNWIND candidates AS cand
OPTIONAL MATCH (s)
WHERE (s:Signal OR s:Port OR s:ClockDomain OR s:ResetDomain) AND toLower(s.name) CONTAINS cand
WITH DISTINCT s
LIMIT 30

// Patchable spans are authoritative repair targets; Method is a legacy fallback.
OPTIONAL MATCH (w)-[:WRITES]->(s)
WHERE w:PatchableSpan OR w:Method
WITH s, collect(DISTINCT {
    name: w.name,
    file_path: w.file_path,
    start_line: w.start_line,
    end_line: w.end_line,
    source_code: w.source_code,
    signature: coalesce(w.signature, w.name),
    entity_type: CASE WHEN w:PatchableSpan THEN 'patchable_span' ELSE 'method' END,
    span_kind: w.span_kind
}) AS writers

// Patchable spans that DRIVE this signal (continuous assignment)
OPTIONAL MATCH (d)-[:DRIVES]->(s)
WHERE d:PatchableSpan OR d:Method
WITH s, writers, collect(DISTINCT {
    name: d.name,
    file_path: d.file_path,
    start_line: d.start_line,
    end_line: d.end_line,
    source_code: d.source_code,
    signature: coalesce(d.signature, d.name),
    entity_type: CASE WHEN d:PatchableSpan THEN 'patchable_span' ELSE 'method' END,
    span_kind: d.span_kind
}) AS drivers

// Patchable spans that READ this signal
OPTIONAL MATCH (r)-[:READS]->(s)
WHERE r:PatchableSpan OR r:Method
WITH s, writers, drivers, collect(DISTINCT {
    name: r.name,
    file_path: r.file_path,
    start_line: r.start_line,
    end_line: r.end_line,
    source_code: r.source_code,
    signature: coalesce(r.signature, r.name),
    entity_type: CASE WHEN r:PatchableSpan THEN 'patchable_span' ELSE 'method' END,
    span_kind: r.span_kind
}) AS readers

// Upstream signals/domains that feed into this one
OPTIONAL MATCH (up)-[:FEEDS|CONNECTS|CLOCKED_BY|RESET_BY]->(s)
WHERE up:Signal OR up:Port OR up:ClockDomain OR up:ResetDomain
WITH s, writers, drivers, readers, collect(DISTINCT up) AS upstream_signal_nodes, collect(DISTINCT {
    name: up.name,
    file_path: up.file_path,
    start_line: up.start_line,
    end_line: up.end_line,
    declaration: up.declaration,
    labels: labels(up)
}) AS upstream_signals

// Patchable spans that WRITE the upstream signals
OPTIONAL MATCH (up_s)<-[:WRITES|DRIVES|CLOCKED_BY|RESET_BY]-(up_w)
WHERE up_s IN upstream_signal_nodes AND (up_w:PatchableSpan OR up_w:Method)
WITH s, writers, drivers, readers, upstream_signals,
     collect(DISTINCT {
         name: up_w.name,
         file_path: up_w.file_path,
         start_line: up_w.start_line,
         end_line: up_w.end_line,
         signature: coalesce(up_w.signature, up_w.name),
         entity_type: CASE WHEN up_w:PatchableSpan THEN 'patchable_span' ELSE 'method' END,
         span_kind: up_w.span_kind,
         target_signal: up_s.name
     }) AS upstream_writers

RETURN {
    signal_name: s.name,
    file_path: s.file_path,
    start_line: s.start_line,
    end_line: s.end_line,
    declaration: s.declaration,
    module_name: s.module_name,
    writers: [w IN writers WHERE w.name IS NOT NULL],
    drivers: [d IN drivers WHERE d.name IS NOT NULL],
    readers: [r IN readers WHERE r.name IS NOT NULL],
    upstream_signals: [us IN upstream_signals WHERE us.name IS NOT NULL],
    upstream_writers: [uw IN upstream_writers WHERE uw.name IS NOT NULL]
} AS trace
ORDER BY size(writers) + size(drivers) + size(readers) DESC
LIMIT 15
"""


def _query_backward_trace(kg: KnowledgeGraph, signal_names: List[str]) -> List[Dict[str, Any]]:
    """Query Neo4j for bounded backward signal-flow traces.

    The original monolithic OPTIONAL-MATCH query multiplied writer, driver,
    reader, and upstream-writer rows before aggregation.  Dense RTL signals
    could therefore create a large intermediate Cartesian product.  Resolve
    the signal nodes first and query each bounded relation group separately.
    """
    candidates = _signal_query_terms(signal_names)
    if not candidates:
        return []

    with kg.driver.session() as session:
        signal_rows = list(session.run(
            """
            UNWIND $candidates AS cand
            MATCH (s)
            WHERE (s:Signal OR s:Port OR s:ClockDomain OR s:ResetDomain)
              AND toLower(coalesce(s.name, '')) CONTAINS cand
            WITH DISTINCT s
            RETURN id(s) AS sid, properties(s) AS props
            LIMIT 15
            """,
            candidates=candidates,
        ))
        traces: List[Dict[str, Any]] = []
        for signal_row in signal_rows:
            sid = signal_row.get("sid")
            props = dict(signal_row.get("props") or {})
            grouped: Dict[str, List[Dict[str, Any]]] = {
                "WRITES": [], "DRIVES": [], "READS": [],
            }
            for record in session.run(
                """
                MATCH (entity)-[rel]->(s)
                WHERE id(s) = $sid
                  AND type(rel) IN ['WRITES', 'DRIVES', 'READS']
                  AND (entity:PatchableSpan OR entity:Method)
                RETURN type(rel) AS relation, properties(entity) AS props,
                       labels(entity) AS labels
                LIMIT 48
                """,
                sid=sid,
            ):
                entity = dict(record.get("props") or {})
                labels = set(record.get("labels") or [])
                entity["signature"] = entity.get("signature") or entity.get("name")
                entity["entity_type"] = "patchable_span" if "PatchableSpan" in labels else "method"
                grouped.setdefault(str(record.get("relation") or ""), []).append(entity)

            upstream_nodes: List[Dict[str, Any]] = []
            upstream_ids: List[int] = []
            for record in session.run(
                """
                MATCH (up)-[rel]->(s)
                WHERE id(s) = $sid
                  AND type(rel) IN ['FEEDS', 'CONNECTS', 'CLOCKED_BY', 'RESET_BY']
                  AND (up:Signal OR up:Port OR up:ClockDomain OR up:ResetDomain)
                RETURN id(up) AS uid, properties(up) AS props,
                       labels(up) AS labels
                LIMIT 16
                """,
                sid=sid,
            ):
                upstream_ids.append(int(record.get("uid")))
                upstream = dict(record.get("props") or {})
                upstream["labels"] = list(record.get("labels") or [])
                upstream_nodes.append(upstream)

            upstream_writers: List[Dict[str, Any]] = []
            if upstream_ids:
                for record in session.run(
                    """
                    UNWIND $upstream_ids AS uid
                    MATCH (up) WHERE id(up) = uid
                    MATCH (writer)-[rel]->(up)
                    WHERE type(rel) IN ['WRITES', 'DRIVES', 'CLOCKED_BY', 'RESET_BY']
                      AND (writer:PatchableSpan OR writer:Method)
                    RETURN properties(writer) AS props, labels(writer) AS labels,
                           up.name AS target_signal
                    LIMIT 32
                    """,
                    upstream_ids=upstream_ids,
                ):
                    writer = dict(record.get("props") or {})
                    labels = set(record.get("labels") or [])
                    writer["signature"] = writer.get("signature") or writer.get("name")
                    writer["entity_type"] = "patchable_span" if "PatchableSpan" in labels else "method"
                    writer["target_signal"] = record.get("target_signal")
                    upstream_writers.append(writer)

            traces.append({
                "signal_name": props.get("name"),
                "file_path": props.get("file_path"),
                "start_line": props.get("start_line"),
                "end_line": props.get("end_line"),
                "declaration": props.get("declaration"),
                "module_name": props.get("module_name"),
                "writers": grouped.get("WRITES", []),
                "drivers": grouped.get("DRIVES", []),
                "readers": grouped.get("READS", []),
                "upstream_signals": upstream_nodes,
                "upstream_writers": upstream_writers,
            })
    return traces


# ---------------------------------------------------------------------------
# 3.  Build debug KG digest
# ---------------------------------------------------------------------------

def _entity_ref(e: Dict[str, Any]) -> str:
    """Short human-readable reference line for an entity."""
    fp = str(e.get("file_path") or "?").replace("\\", "/")
    sl = e.get("start_line") or "?"
    el = e.get("end_line") or "?"
    name = e.get("name") or e.get("signature") or "?"
    return f"{fp}:{sl}-{el} {name}"


def _deduplicate_by_key(items: Sequence[Dict[str, Any]], key: str = "name") -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    result: List[Dict[str, Any]] = []
    for item in items:
        k = str(item.get(key, ""))
        if k and k not in seen:
            seen.add(k)
            result.append(item)
    return result


def _format_path_hints(traces: List[Dict[str, Any]]) -> List[str]:
    """Build path_hint strings from backward trace results."""
    hints: List[str] = []
    for t in traces:
        signal = t.get("signal_name", "?")
        writers = t.get("writers") or []
        drivers = t.get("drivers") or []
        readers = t.get("readers") or []
        upstream = t.get("upstream_signals") or []
        upstream_writers = t.get("upstream_writers") or []

        for d in drivers[:2]:
            hints.append(f"{signal} ←[DRIVEN BY]← {d.get('name','?')} @ {d.get('file_path','?')}:{d.get('start_line','?')}")
        for w in writers[:2]:
            hints.append(f"{signal} ←[WRITTEN BY]← {w.get('name','?')} @ {w.get('file_path','?')}:{w.get('start_line','?')}")
        for uw in upstream_writers[:3]:
            hints.append(f"{uw.get('target_signal','?')} ←[WRITTEN BY]← {uw.get('name','?')} @ {uw.get('file_path','?')}:{uw.get('start_line','?')}")
        for us in upstream[:2]:
            hints.append(f"{us.get('name','?')} -[FEEDS/CONNECTS]-> {signal}")
        for r in readers[:2]:
            hints.append(f"{signal} -[READ BY]-> {r.get('name','?')} @ {r.get('file_path','?')}:{r.get('start_line','?')}")
    return hints


def _format_trace_templates(traces: List[Dict[str, Any]]) -> List[str]:
    """Build high-level trace templates."""
    templates: List[str] = []
    for t in traces[:8]:
        signal = t.get("signal_name", "?")
        mod = t.get("module_name") or "?"
        writers = t.get("writers") or []
        drivers = t.get("drivers") or []

        chain_parts = [f"failure({signal})"]
        if drivers:
            chain_parts.insert(0, f"driver({drivers[0].get('name','?')})")
        if writers:
            chain_parts.append(f"assigner({writers[0].get('name','?')})")
        upstream = t.get("upstream_signals") or []
        if upstream:
            chain_parts.append(f"upstream({upstream[0].get('name','?')})")
        if upstream_writers := t.get("upstream_writers"):
            chain_parts.append(f"upstream_writer({upstream_writers[0].get('name','?')})")

        templates.append(" → ".join(chain_parts))
    return templates


def _compact_inference_path_text(path: Sequence[Dict[str, Any]]) -> str:
    steps: List[str] = []
    for edge in path:
        start = str(edge.get("start_node") or "?")
        end = str(edge.get("end_node") or "?")
        relation = str(edge.get("type") or edge.get("relation_kind") or "RELATED")
        steps.append(f"{start} -[{relation}]-> {end}")
    return " | ".join(steps)


def _trace_to_issue_paths(traces: List[Dict[str, Any]], limit: int = 16) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for trace in traces:
        signal = str(trace.get("signal_name") or "").strip()
        signal_file = str(trace.get("file_path") or "").replace("\\", "/")
        signal_module = str(trace.get("module_name") or "")
        if not signal:
            continue
        candidates = []
        for group_name in ("drivers", "writers", "upstream_writers", "readers"):
            for entity in trace.get(group_name) or []:
                candidates.append((group_name, entity))
        for group_name, entity in candidates:
            if not isinstance(entity, dict):
                continue
            file_path = str(entity.get("file_path") or "").replace("\\", "/")
            target = str(entity.get("signature") or entity.get("name") or "").strip()
            if not file_path or not target:
                continue
            relation = {
                "drivers": "DRIVES",
                "writers": "WRITES",
                "upstream_writers": "WRITES",
                "readers": "READS",
            }.get(group_name, "RELATED")
            marker = f"{file_path}:{entity.get('start_line')}:{target}:{signal}:{relation}"
            if marker in seen:
                continue
            seen.add(marker)
            path = [
                {
                    "start_node": "Issue:root",
                    "start_labels": ["Issue"],
                    "end_node": f"FailureSignal:{signal}",
                    "end_labels": ["Signal"],
                    "type": "GROUNDS_TO",
                    "relation_kind": "grounding",
                    "description": "validation feedback grounds the failing RTL signal",
                    "contributes_to_causal_distance": False,
                },
                {
                    "start_node": f"FailureSignal:{signal}",
                    "start_labels": ["Signal"],
                    "end_node": target,
                    "end_labels": [str(entity.get("entity_type") or "PatchableSpan")],
                    "type": relation,
                    "relation_kind": relation.lower(),
                    "description": f"{target} is connected to failure signal {signal}",
                    "contributes_to_causal_distance": True,
                },
            ]
            rows.append({
                "group": "debug_trace",
                "source": "failure_conditioned_kg_trace",
                "target": target,
                "type": entity.get("entity_type") or "patchable_span",
                "file_path": file_path,
                "start_line": entity.get("start_line"),
                "end_line": entity.get("end_line"),
                "module_name": signal_module,
                "signal": signal,
                "path": path,
                "path_text": (
                    _compact_inference_path_text(path)
                    + f" @ {file_path}:{entity.get('start_line', '?')}"
                ),
            })
            if len(rows) >= limit:
                return rows
    return rows


def _reranked_to_issue_paths(spans: Sequence[Dict[str, Any]], limit: int = 16) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for span in spans[:limit]:
        file_path = str(span.get("file_path") or "").replace("\\", "/")
        target = str(span.get("name") or span.get("signature") or "")
        if not file_path or not target:
            continue
        mode = str(span.get("distance_mode") or "text_only")
        path = list(span.get("path") or [])
        if not path:
            path = [{
                "start_node": "Issue:root",
                "start_labels": ["Issue"],
                "end_node": target,
                "end_labels": ["PatchableSpan"],
                "type": "SEMANTIC_RETRIEVAL",
                "relation_kind": "text_only",
                "description": "failure-conditioned semantic fallback without a grounded RTL anchor",
                "contributes_to_causal_distance": False,
            }]
        rows.append({
            "group": "debug_rerank",
            "source": "failure_conditioned_rtl_causal_localization",
            "target": target,
            "type": "patchable_span",
            "file_path": file_path,
            "start_line": span.get("start_line"),
            "end_line": span.get("end_line"),
            "module_name": span.get("module_name") or "",
            "similarity": span.get("similarity"),
            "distance": span.get("distance"),
            "distance_mode": mode,
            "global_rank": span.get("global_rank"),
            "path": path,
            "path_text": (
                _compact_inference_path_text(path)
                + f" @ {file_path}:{span.get('start_line', '?')}-{span.get('end_line', '?')}"
            ),
        })
    return rows


def build_debug_kg_trace(
    error_feedback: str,
    kg: Optional[KnowledgeGraph] = None,
    instance_id: str = "",
    max_signals: int = 10,
    issue_text: str = "",
    issue_mentioned_files: Optional[Sequence[str]] = None,
    rerank_limit: int = 20,
    enable_rerank: bool = True,
) -> Dict[str, Any]:
    """Main entry point: from Docker error output → structured KG debug digest.

    Returns a dict with:
      - failure_signals (extracted signal names)
      - traces (Neo4j backward trace results)
      - path_hints (readable path strings)
      - trace_templates (high-level chain)
      - text (Markdown formatted digest for LLM consumption)
      - suggested_edit_targets (files/lines that appear in upstream writers)
    """
    extracted = _extract_failure_signals(error_feedback)
    candidates: List[str] = []
    traces: List[Dict[str, Any]] = []

    reranked_spans: List[Dict[str, Any]] = []
    rerank_metadata: Dict[str, Any] = {}
    if enable_rerank and kg and error_feedback:
        try:
            mode = os.getenv("KGVERI_LOCALIZATION_MODE", "rtl_causal_b2").strip().lower()
            if mode in {"kgcompass_b0", "b0", "legacy", "rtl_causal_b1", "b1"}:
                mode = "rtl_causal_b2"
            localizer = RTLCausalLocalizer(
                kg.driver,
                vector_weight=VECTOR_SIMILARITY_WEIGHT,
                decay_factor=DECAY_FACTOR,
                mode=mode,
            )
            query_embedding = kg._get_embedding("\n".join(
                item for item in (issue_text, error_feedback) if item
            ))
            reranked, rerank_metadata = localizer.localize(
                issue_text=issue_text,
                evidence_text=error_feedback,
                issue_mentioned_files=list(dict.fromkeys([
                    *(issue_mentioned_files or []),
                    *(extracted.get("files") or []),
                ])),
                limit=rerank_limit,
                max_cost=3.0,
                query_embedding=query_embedding,
            )
            reranked_spans = list(reranked.get("edit_targets") or [])
        except Exception as exc:
            twarn(f"Failure-conditioned RTL causal reranking failed: {exc}")

    mapped_signal_anchors: List[str] = []
    explicit_failure_signals = set(extracted["signals"])
    for anchor in rerank_metadata.get("anchors") or []:
        if (
            anchor.get("source") != "validation"
            or anchor.get("anchor_type") != "identifier"
            or anchor.get("match_mode") == "unmapped"
        ):
            continue
        labels = {
            label
            for node in (anchor.get("matched_nodes") or [])
            for label in (node.get("labels") or [])
        }
        if not labels & {"Signal", "Port", "Parameter", "State", "ClockDomain", "ResetDomain"}:
            continue
        identifier = str((anchor.get("normalized") or {}).get("exact_identifier") or "")
        if explicit_failure_signals and identifier not in explicit_failure_signals:
            continue
        if identifier and identifier not in mapped_signal_anchors:
            mapped_signal_anchors.append(identifier)
    candidates = (mapped_signal_anchors or extracted["signals"])[:max_signals]

    if kg and candidates:
        try:
            traces = _query_backward_trace(kg, candidates)
        except Exception as exc:
            twarn(f"Neo4j backward trace failed: {exc}")

    # Build path hints from traces
    path_hints = _format_path_hints(traces)
    trace_templates = _format_trace_templates(traces)
    updated_issue_paths = [
        *_trace_to_issue_paths(traces),
        *_reranked_to_issue_paths(reranked_spans),
    ][:24]

    # Collect suggested edit targets (the upstream writers look most like "where to fix")
    suggested_targets: List[Dict[str, Any]] = []
    for t in traces:
        for entity in (t.get("upstream_writers") or []) + (t.get("drivers") or []) + (t.get("writers") or []):
            if entity.get("file_path") and not str(entity.get("file_path","")).startswith("tb/"):
                suggested_targets.append(entity)
    suggested_targets.extend(reranked_spans[:12])
    suggested_targets = _deduplicate_by_key(suggested_targets, key="name")[:12]

    # Build Markdown text
    text_sections = [
        "## Debug KG Trace (failure-conditioned reslice)",
        "",
        "### Failure Signals Extracted",
        *([f"- {s}" for s in candidates] or ["  (none extracted)"]),
        "",
    ]

    if extracted["tests"]:
        text_sections += [
            "### Failing Tests",
            *[f"- {t}" for t in extracted["tests"]],
            "",
        ]

    if extracted["modules"]:
        text_sections += [
            "### Related Modules (from error output)",
            *[f"- {m}" for m in extracted["modules"]],
            "",
        ]

    if trace_templates:
        text_sections += [
            "### KG Backward Trace Chains",
            "(failure_signal ← upstream chain: each line traces backward from observed failure)",
            *[f"- {t}" for t in trace_templates],
            "",
        ]

    if reranked_spans:
        text_sections += [
            "### Failure-Conditioned RTL Causal Reranking",
            *[
                f"- rank={span.get('global_rank')} mode={span.get('distance_mode')} "
                f"score={float(span.get('score') or 0.0):.4f} {_entity_ref(span)}"
                for span in reranked_spans[:12]
            ],
            "",
        ]

    if path_hints:
        text_sections += [
            "### KG Signal-Flow Path Hints",
            *[f"- {h}" for h in path_hints[:16]],
            "",
        ]

    if suggested_targets:
        text_sections += [
            "### Suggested Edit Targets (from KG upstream trace)",
            *[f"- {_entity_ref(e)}" for e in suggested_targets],
            "",
        ]

    if not traces:
        text_sections += [
            "### Notes",
            "- No KG backward trace available (Neo4j query returned empty). "
            "Proceed with visible source context for diagnosis.",
            "",
        ]

    text_sections.append(
        "Use this KG trace to pinpoint the broken signal chain. "
        "The failure signals extracted from validation output are linked "
        "to upstream drivers/writers via the knowledge graph."
    )

    text = "\n".join(text_sections)

    return {
        "failure_signals": candidates,
        "modules": extracted["modules"],
        "tests": extracted["tests"],
        "traces": traces,
        "path_hints": path_hints,
        "trace_templates": trace_templates,
        "dynamic_anchor_to_patchable_span_paths": updated_issue_paths,
        # Backward-compatible artifact key used by existing repair runs.
        "updated_issue_to_entity_paths": updated_issue_paths,
        "suggested_edit_targets": suggested_targets,
        "reranked_spans": reranked_spans,
        "rerank_metadata": rerank_metadata,
        "text": text,
    }
