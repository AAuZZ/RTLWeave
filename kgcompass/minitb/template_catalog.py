"""Small, frozen MiniTB template taxonomy.

Templates are execution semantics, never issue- or repository-specific recipes.
Adding a template requires changing ``MAX_TEMPLATE_COUNT`` and its tests, making
accidental growth into one-template-per-case visible in review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MAX_TEMPLATE_COUNT = 5


@dataclass(frozen=True)
class TemplatePolicy:
    bug_class: str
    template_id: str
    functional: bool
    purpose: str


TEMPLATE_POLICIES = (
    TemplatePolicy("combinational", "COMB_DIRECTED", True, "bounded input/output mappings"),
    TemplatePolicy("clocked_state", "CLOCKED_SEQUENCE", True, "reset and bounded state transitions"),
    TemplatePolicy("protocol", "PROTOCOL_SEQUENCE", True, "request/response and handshake transactions"),
    TemplatePolicy("event_trace", "EVENT_TRACE", True, "ordered or repeated architectural events"),
    TemplatePolicy("unsupported", "MINIMAL_MODULE_SMOKE", False, "compile, elaborate, and execute only"),
)

if len(TEMPLATE_POLICIES) != MAX_TEMPLATE_COUNT:
    raise RuntimeError("MiniTB template count changed without updating the fixed catalog boundary")
if len({item.bug_class for item in TEMPLATE_POLICIES}) != len(TEMPLATE_POLICIES):
    raise RuntimeError("duplicate MiniTB bug class")
if len({item.template_id for item in TEMPLATE_POLICIES}) != len(TEMPLATE_POLICIES):
    raise RuntimeError("duplicate MiniTB template ID")

TEMPLATE_BY_CLASS = {item.bug_class: item.template_id for item in TEMPLATE_POLICIES}
FUNCTIONAL_TEMPLATE_IDS = frozenset(item.template_id for item in TEMPLATE_POLICIES if item.functional)


def classify_template(issue: dict[str, Any], target: dict[str, Any]) -> str:
    """Select an execution semantic from issue language and interface shape."""
    text = f"{issue.get('title', '')} {issue.get('problem_statement', '')}".lower()
    ports = [item for item in (target.get("ports") or []) if isinstance(item, dict)]
    inputs = [str(item.get("name") or "") for item in ports if item.get("direction") == "input"]
    has_clock = any(re.search(r"(?:^|_)(?:clk|clock)(?:_|$)", name, re.I) for name in inputs)
    if has_clock and re.search(
        r"\b(?:rvfi|retir(?:e|ed|ement|ing)|execution trace|"
        r"instruction sequence|following instruction|ordered event)\b", text
    ):
        return "event_trace"
    if has_clock and re.search(
        r"\b(?:csr|register|debug mode|counter|reset|interrupt|trap|privilege|"
        r"state machine|state transition|state update|state register)\b",
        text,
    ):
        return "clocked_state"
    if has_clock and re.search(
        r"\b(?:handshake|transaction|outstanding)\b", text
    ) or (has_clock and re.search(r"\brequest\b", text) and re.search(
        r"\b(?:grant|ready|valid|response)\b", text
    )):
        return "protocol"
    if has_clock and re.search(
        r"\b(?:sequential|clocked)\b",
        text,
    ):
        return "clocked_state"
    return "combinational"


def issue_facts(issue: dict[str, Any]) -> dict[str, Any]:
    """Extract visible evidence for prompting without synthesizing an oracle."""
    text = f"{issue.get('title', '')}\n{issue.get('problem_statement', '')}"
    quoted = list(dict.fromkeys(re.findall(r"`([^`\n]{1,80})`", text)))[:24]
    literals = list(dict.fromkeys(re.findall(
        r"(?<![A-Za-z0-9_])(?:\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|0[xX][0-9a-fA-F]+|0[bB][01]+)",
        text,
    )))[:24]
    requirements = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if re.search(r"\b(?:must|should|expected)\b", sentence, re.I):
            requirements.append(sentence.strip()[:500])
    result = {
        "quoted_terms": quoted,
        "numeric_literals": literals,
        "required_behavior_sentences": requirements[:8],
    }
    # Defined below; runtime lookup is safe after module initialization.
    result["explicit_instruction_fields"] = instruction_field_requirements(issue)
    return result


_TOKEN_ALIASES = {
    "instruction": "instr", "instructions": "instr", "indication": "indicator",
    "error": "err", "errors": "err", "request": "req", "response": "resp",
    "selection": "sel", "operand": "op", "write": "we", "enable": "en",
}


def _semantic_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", text.lower()):
        token = _TOKEN_ALIASES.get(token, token)
        if len(token) >= 2 and token not in {"the", "and", "must", "should", "output", "signal"}:
            tokens.add(token)
    return tokens


def rank_observation_ports(issue: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank output ports against required issue language using transparent tokens."""
    facts = issue_facts(issue)
    required = " ".join(facts["required_behavior_sentences"])
    text = required or f"{issue.get('title', '')} {issue.get('problem_statement', '')}"
    issue_tokens = _semantic_tokens(text)
    ranked = []
    for port in target.get("ports") or []:
        if not isinstance(port, dict) or port.get("direction") != "output":
            continue
        name = str(port.get("name") or "")
        base = re.sub(r"_(?:o|out)$", "", name, flags=re.I)
        tokens = _semantic_tokens(base.replace("_", " "))
        score = len(tokens & issue_tokens)
        if re.search(
            rf"(?<![A-Za-z0-9_$]){re.escape(base.lower())}(?![A-Za-z0-9_$])",
            text.lower(),
        ):
            score += 3
        if score:
            ranked.append({"port": name, "score": score, "matched_tokens": sorted(tokens & issue_tokens)})
    return sorted(ranked, key=lambda item: (-item["score"], item["port"]))


def instruction_field_requirements(issue: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only bit-field values explicitly stated by the issue."""
    text = f"{issue.get('title', '')}\n{issue.get('problem_statement', '')}"
    fields = []
    pattern = re.compile(
        r"(?:instr(?:uction)?)\s*\[\s*(\d+)\s*(?::\s*(\d+)\s*)?\]\s*=\s*"
        r"(\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|[01])",
        re.I,
    )
    for match in pattern.finditer(text):
        high, low = int(match.group(1)), int(match.group(2) or match.group(1))
        fields.append({"name": f"instr[{high}:{low}]", "high": high, "low": low, "value": match.group(3)})
    for name, high, low in (("funct3", 14, 12), ("funct7", 31, 25), ("opcode", 6, 0)):
        match = re.search(
            rf"\b{name}\b\s*=\s*(\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|0[xX][0-9a-fA-F]+|\d+)",
            text,
            re.I,
        )
        if match:
            fields.append({"name": name, "high": high, "low": low, "value": match.group(1)})
    unique = {}
    for item in fields:
        unique[(item["high"], item["low"])] = item
    return list(unique.values())
