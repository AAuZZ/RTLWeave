from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

try:
    from repair_types import RepairPlan
    from issue_path_navigation import IssuePathNavigator
except Exception:  # pragma: no cover
    from .repair_types import RepairPlan
    from .issue_path_navigation import IssuePathNavigator


DEFAULT_CONTEXT_BUDGET = {
    "issue": 6000,
    "kg_digest": 6000,
    "repair_plan": 4000,
    "candidate_source": 70000,
    "failure_evidence": 8000,
    "minitb_summary": 4000,
    "kg_trace": 8000,
    "expanded_source": 50000,
}

DEFAULT_FORBIDDEN_PATTERNS = [
    r"(^|/)tb/",
    r"(^|/)test/",
    r"(^|/)tests/",
    r"(^|/)dv/",
    r"(^|/)sim/",
    r"(^|/)verif/",
    r"(^|/)formal/",
    r"(^|/)verification[^/]*/",
    r"(^|/)uvm[^/]*/",
    r"(^|/)uvmf[^/]*/",
    r"(^|/)environment_packages/",
    r"(^|/)vendor/.*(?:dv|test|uvm)",
    r"(^|/)benchmark",
    r"(^|/)benchmarks/",
    r"(^|/)autogen/",
    r"(^|/)generated/",
    r"(^|/)datasets/",
    r"(^|/)(?:hidden|held[_-]?out|gold(?:en)?|reference)[^/]*(?:test|evaluator|patch|artifact)",
    r"(^|/)(?:tb|test)[_-]?script(?:\.|/|$)",
    r"(^|/)(?:fix|test|reference|gold)[_-]?patch(?:\.|/|$)",
]


def _clean_file_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if text.startswith("./"):
        text = text[2:]
    if re.search(r"\.(?:sv|svh|v|vh)$", text, re.IGNORECASE):
        text = text.strip("'\"`")
        should_strip_root = text.startswith("/") or "validation_attempts/" in text or "workdirs/" in text or text.startswith("home/")
        if text.startswith("/"):
            text = text.lstrip("/")
        if should_strip_root:
            matches = [(idx, marker) for marker in ("src/", "hw/", "core/", "rtl/", "include/", "vendor/") if (idx := text.find(marker)) >= 0]
            if matches:
                idx, _ = min(matches, key=lambda item: item[0])
                text = text[idx:]
    return text


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        item = _clean_file_path(item)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _entity_ref(entity: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "file_path": _clean_file_path(entity.get("file_path")),
        "start_line": entity.get("start_line"),
        "end_line": entity.get("end_line"),
        "name": entity.get("name") or entity.get("signal_name") or "",
        "signature": entity.get("signature") or "",
        "kind": entity.get("repair_role") or entity.get("verilog_kind") or entity.get("type") or "",
        "score": entity.get("similarity") or entity.get("score") or 0,
        "reason": entity.get("semantic_summary") or entity.get("description") or "",
    }


def _collect_entities(related: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ordered_groups = [
        "repair_anchor_entities",
        "fault_anchor_entities",
        "patchable_spans",
        "edit_targets",
        "direct_drivers",
        "top_level_wiring",
        "config_registers",
        "methods",
        "evidence_entities",
        "rtl_entities",
    ]
    result: List[Dict[str, Any]] = []
    seen = set()
    for group_name in ordered_groups:
        for entity in related.get(group_name) or []:
            if not isinstance(entity, dict):
                continue
            fp = _clean_file_path(entity.get("file_path"))
            if not fp or not re.search(r"\.(?:sv|svh|v|vh)$", fp, re.IGNORECASE):
                continue
            key = (fp, entity.get("start_line"), entity.get("end_line"), entity.get("name"), entity.get("signature"))
            if key in seen:
                continue
            seen.add(key)
            result.append(_entity_ref(entity, group_name))
    return result


def _issue_hypothesis(problem_statement: str, edit_cone: List[Dict[str, Any]]) -> str:
    issue = re.sub(r"\s+", " ", problem_statement or "").strip()
    issue = issue[:300] if issue else "No issue text available."
    if edit_cone:
        top = edit_cone[0]
        target = top.get("signature") or top.get("name") or top.get("file_path")
        return f"KG localization points to {target}; repair the smallest RTL cone consistent with the issue: {issue}"
    return f"Repair the smallest RTL cone consistent with the issue: {issue}"


def build_repair_plan(
    *,
    locate_result: Dict[str, Any],
    prompt_context: Dict[str, Any],
    previous_diagnosis: Dict[str, Any] | None = None,
    previous_evidence: List[Dict[str, Any]] | None = None,
) -> RepairPlan:
    related = prompt_context.get("related_entities") or locate_result.get("related_entities", {}) or {}
    edit_cone = _collect_entities(related)[:24]

    issue_files = _dedupe(prompt_context.get("issue_mentioned_files") or [])
    issue_path_nav = IssuePathNavigator(prompt_context.get("issue_to_entity_paths") or [])
    issue_path_files = _dedupe(prompt_context.get("issue_path_files") or issue_path_nav.files())
    context_files = _dedupe(prompt_context.get("files") or [])
    cone_files = _dedupe(item.get("file_path") for item in edit_cone)

    diagnosis_files: List[str] = []
    if previous_diagnosis:
        for path in previous_diagnosis.get("need_files") or []:
            diagnosis_files.append(_clean_file_path(path))
        for item in previous_diagnosis.get("edit_candidates") or []:
            if isinstance(item, dict):
                diagnosis_files.append(_clean_file_path(item.get("file") or item.get("file_path")))

    evidence_files: List[str] = []
    for item in previous_evidence or []:
        if isinstance(item, dict):
            evidence_files.append(_clean_file_path(item.get("file") or item.get("file_path")))

    allowed_files = [
        path for path in _dedupe([*issue_files, *issue_path_files, *cone_files, *diagnosis_files, *evidence_files, *context_files])
        if path and not is_forbidden_file(path)
    ][:40]
    if not allowed_files:
        allowed_files = cone_files[:20]

    forbidden_files = [
        "tb/",
        "test/",
        "tests/",
        "dv/",
        "benchmarks/",
        "datasets/",
        "hidden benchmark evaluator artifacts and reference patches",
    ]

    hypothesis = _issue_hypothesis(prompt_context.get("problem_statement", ""), edit_cone)
    if previous_diagnosis and previous_diagnosis.get("failure_hypothesis"):
        hypothesis += f" Retry diagnosis: {str(previous_diagnosis.get('failure_hypothesis'))[:240]}"

    return RepairPlan(
        hypothesis=hypothesis,
        edit_cone=edit_cone,
        allowed_files=allowed_files,
        forbidden_files=forbidden_files,
        validator_profile={
            "mode": "layered_internal_validation",
            "stages": [
                "static_validation",
                "frozen_minitb",
                "repository_regression",
                "clean_replay",
            ],
            "success_condition": "all_required_stages_pass",
            "blockers": [
                "patch_apply",
                "forbidden_file_edit",
                "replace_format",
                "static_validation_failure",
                "frozen_minitb_failure",
                "repository_new_failure",
                "clean_replay_failure",
            ],
            "warnings": [
                "outside_kg_edit_cone",
                "validation_failure_signal_anchor",
                "repository_preexisting_failure",
                "kg_trace_incomplete",
            ],
            "functional_checks": [
                "immutable frozen Mini-Testbench",
                "repository-provided build or test command with baseline comparison",
                "failure evidence to KG backward trace",
                "fresh-checkout clean replay",
            ],
            "frozen_minitb": "issue_oracle_when_qualified_else_module_smoke",
            "hidden_benchmark_artifacts": "forbidden",
        },
        context_budget=dict(DEFAULT_CONTEXT_BUDGET),
    )


def repair_plan_to_prompt(plan: RepairPlan) -> str:
    payload = plan.to_dict()
    return json.dumps(payload, indent=2, ensure_ascii=False)


def is_forbidden_file(path: str) -> bool:
    normalized = _clean_file_path(path).lower()
    return any(re.search(pattern, normalized) for pattern in DEFAULT_FORBIDDEN_PATTERNS)
