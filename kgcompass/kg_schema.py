from __future__ import annotations

from typing import Dict, List


KG_SCHEMA_VERSION = "kgveri-v1.2"

CORE_NODE_LABELS: List[str] = [
    "Issue",
    "Method",
    "Class",
    "File",
    "Directory",
    "Commit",
]

HDL_ENTITY_LABELS: List[str] = [
    "PatchableSpan",
    "Signal",
    "Port",
    "Parameter",
    "Macro",
    "State",
    "Branch",
    "Condition",
    "GenerateBlock",
    "ConditionalCompilationScope",
    "Testbench",
    "Assertion",
    "Instance",
    "ClockDomain",
    "ResetDomain",
    "Typedef",
    "Import",
    "Include",
]

SEMANTIC_RELATION_TYPES: Dict[str, str] = {
    "contains": "CONTAINS",
    "mentions": "MENTIONS",
    "calls": "CALLS",
    "reads": "READS",
    "writes": "WRITES",
    "drives": "DRIVES",
    "feeds": "FEEDS",
    "connects": "CONNECTS",
    "instantiates": "INSTANTIATES",
    "defines": "DEFINES",
    "modifies": "MODIFIES",
    "guards": "GUARDS",
    "transitions_to": "TRANSITIONS_TO",
    "branches": "BRANCHES",
    "tests": "TESTS",
    "exercises": "EXERCISES",
    "clocked_by": "CLOCKED_BY",
    "reset_by": "RESET_BY",
    "declares": "DECLARES",
    "imports": "IMPORTS",
    "includes": "INCLUDES",
    "binds_port": "BINDS_PORT",
    "parameterizes": "PARAMETERIZES",
    "aliases": "ALIASES",
    "observes": "OBSERVES",
    "projects_to": "PROJECTS_TO",
    # Generic fallback for legacy HDL helper callers.
    "uses": "MENTIONS",
}


RELATION_REVERSE_DESCRIPTIONS: Dict[str, str] = {
    "contains method": "contained in method",
    "contained in method": "contains method",
    "contains class": "contained in class",
    "contained in class": "contains class",
    "contains file": "contained in file",
    "contained in file": "contains file",
    "points to issue": "referenced by issue",
    "referenced by issue": "points to issue",
    "calls method": "called by method",
    "called by method": "calls method",
    "reads signal": "read by method",
    "read by method": "reads signal",
    "writes signal": "written by method",
    "written by method": "writes signal",
    "drives signal": "driven by method",
    "driven by method": "drives signal",
    "feeds signal": "fed by signal",
    "fed by signal": "feeds signal",
    "connects signal": "connected by instance",
    "connected by instance": "connects signal",
    "instantiates module": "instantiated by method",
    "instantiated by method": "instantiates module",
    "contained in module": "contains rtl entity",
    "contains clock domain": "clock domain of module",
    "clock domain of module": "contains clock domain",
    "contains reset domain": "reset domain of module",
    "reset domain of module": "contains reset domain",
    "includes header": "included from file",
    "included from file": "includes header",
    "imports package symbol": "imported by file",
    "imported by file": "imports package symbol",
    "declares type": "declared in file",
    "declared in file": "declares type",
    "clocked by signal": "clocks logic",
    "clocks logic": "clocked by signal",
    "reset by signal": "resets logic",
    "resets logic": "reset by signal",
    "guards conditional compilation": "guarded by macro",
    "guarded by macro": "guards conditional compilation",
    "transitions to state": "transitioned from state",
    "transitioned from state": "transitions to state",
    "exercises module": "exercised by testbench",
    "exercised by testbench": "exercises module",
    "projects method to patchable span": "patchable projection of method",
    "patchable projection of method": "projects method to patchable span",
    "projects semantic entity to patch span": "patch span represents semantic entity",
    "patch span represents semantic entity": "projects semantic entity to patch span",
    "contains nested patch span": "nested in patch span",
    "nested in patch span": "contains nested patch span",
}


def node_projection_labels() -> List[str]:
    return [*CORE_NODE_LABELS, *HDL_ENTITY_LABELS]


def semantic_relation_values() -> List[str]:
    return sorted(set(SEMANTIC_RELATION_TYPES.values()))


def schema_summary() -> Dict[str, object]:
    return {
        "version": KG_SCHEMA_VERSION,
        "core_node_labels": list(CORE_NODE_LABELS),
        "hdl_entity_labels": list(HDL_ENTITY_LABELS),
        "semantic_relation_types": dict(SEMANTIC_RELATION_TYPES),
    }
