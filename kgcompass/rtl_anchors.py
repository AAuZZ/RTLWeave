from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RTL_SUFFIX_ROLES = {
    "i": "input",
    "o": "output",
    "q": "registered",
    "d": "next_state",
    "n": "active_low",
    "ni": "active_low_input",
    "no": "active_low_output",
}

RTL_FILE_SUFFIXES = (".sv", ".svh", ".v", ".vh")

GENERIC_WORDS = {
    "about", "access", "actual", "after", "against", "allows", "also",
    "always", "another", "assert", "because", "before", "being", "between",
    "block", "case", "causes", "change", "check", "code", "condition",
    "configuration", "correct", "currently", "data", "description", "does",
    "during", "error", "expected", "fails", "failure", "false", "feature",
    "file", "from", "function", "hardware", "have", "incorrect", "input",
    "instruction", "into", "issue", "logic", "module", "must", "only",
    "output", "path", "problem", "register", "related", "result", "same",
    "should", "signal", "software", "state", "support", "test", "than",
    "that", "their", "then", "there", "these", "this", "through", "true",
    "value", "when", "where", "which", "while", "with", "without", "would",
}

PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:svh?|vh?))"
    r"(?::(?P<line>\d+))?",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*(?:::[A-Za-z_][A-Za-z0-9_$]*)*\b")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class RTLIdentifier:
    exact_identifier: str
    base_identifier: str
    base_tokens: Tuple[str, ...]
    role: str = ""

    @property
    def canonical_identifier(self) -> str:
        """Single role-insensitive form used by IDF coverage scoring."""
        return self.base_identifier or self.exact_identifier

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["base_tokens"] = list(self.base_tokens)
        result["canonical_identifier"] = self.canonical_identifier
        return result


@dataclass
class RTLAnchor:
    text: str
    anchor_type: str
    normalized: RTLIdentifier
    source: str = "issue"
    explicit: bool = False
    file_path: str = ""
    line: Optional[int] = None
    matched_nodes: List[Dict[str, Any]] = field(default_factory=list)
    match_mode: str = "unmapped"
    mapping_confidence: float = 0.0

    @property
    def key(self) -> Tuple[str, str, str, Optional[int]]:
        return (
            self.anchor_type,
            self.normalized.exact_identifier,
            self.file_path,
            self.line,
        )

    def to_dict(self, include_nodes: bool = True) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "anchor_type": self.anchor_type,
            "normalized": self.normalized.to_dict(),
            "source": self.source,
            "explicit": self.explicit,
            "file_path": self.file_path,
            "line": self.line,
            "match_mode": self.match_mode,
            "mapping_confidence": self.mapping_confidence,
        }
        if include_nodes:
            result["matched_nodes"] = list(self.matched_nodes)
        return result


def normalize_identifier(value: str) -> RTLIdentifier:
    exact = str(value or "").strip().strip("`'\"").lower()
    exact = re.sub(r"\[[^\]]+\]$", "", exact)
    exact = exact.split("::")[-1]
    exact = exact.lstrip(".")
    role = ""
    base = exact
    suffix_match = re.match(r"^(?P<base>.+)_(?P<suffix>ni|no|i|o|q|d|n)$", exact)
    if suffix_match and suffix_match.group("base"):
        base = suffix_match.group("base")
        role = RTL_SUFFIX_ROLES[suffix_match.group("suffix")]
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base).lower()
    tokens = tuple(
        token
        for token in re.split(r"[^a-z0-9]+", camel_split)
        if len(token) >= 2 and token not in RTL_SUFFIX_ROLES
    )
    return RTLIdentifier(exact, base, tokens, role)


def _looks_like_identifier(token: str, explicit: bool = False) -> bool:
    lowered = token.lower()
    if lowered in GENERIC_WORDS or len(lowered) < 3:
        return False
    if any(lowered.endswith(suffix) for suffix in RTL_FILE_SUFFIXES):
        return False
    if explicit:
        return True
    if re.fullmatch(r"[A-Z][a-z]+", token):
        return False
    return (
        "_" in token
        or "::" in token
        or any(char.isdigit() for char in token)
        or (any(char.islower() for char in token) and any(char.isupper() for char in token))
        or token.isupper()
    )


def extract_rtl_anchors(text: str, source: str = "issue", limit: int = 96) -> List[RTLAnchor]:
    text = str(text or "")
    anchors: List[RTLAnchor] = []
    occupied: List[Tuple[int, int]] = []

    for match in PATH_RE.finditer(text):
        path = match.group("path").replace("\\", "/")
        line = int(match.group("line")) if match.group("line") else None
        normalized = normalize_identifier(PurePosixPath(path).name)
        anchors.append(RTLAnchor(
            text=match.group(0),
            anchor_type="direct_location" if line else "file",
            normalized=normalized,
            source=source,
            file_path=path,
            line=line,
        ))
        occupied.append(match.span())

    explicit_tokens = set()
    for match in BACKTICK_RE.finditer(text):
        explicit_tokens.update(IDENTIFIER_RE.findall(match.group(1)))

    for token_match in IDENTIFIER_RE.finditer(text):
        if len(anchors) >= limit:
            break
        if any(start <= token_match.start() < end for start, end in occupied):
            continue
        token = token_match.group(0)
        explicit = token in explicit_tokens
        if not _looks_like_identifier(token, explicit=explicit):
            continue
        normalized = normalize_identifier(token)
        if not normalized.exact_identifier:
            continue
        anchors.append(RTLAnchor(
            text=token,
            anchor_type="identifier",
            normalized=normalized,
            source=source,
            explicit=explicit,
        ))

    deduped: List[RTLAnchor] = []
    seen = set()
    for anchor in anchors:
        if anchor.key in seen:
            continue
        seen.add(anchor.key)
        deduped.append(anchor)
    return deduped[:limit]


def anchor_payload(anchors: Sequence[RTLAnchor]) -> List[Dict[str, Any]]:
    return [anchor.to_dict() for anchor in anchors]


def classify_mapped_anchor(anchor: RTLAnchor) -> str:
    labels = {
        str(label)
        for node in anchor.matched_nodes
        for label in (node.get("labels") or [])
    }
    if anchor.anchor_type == "direct_location":
        return "direct_location"
    if anchor.anchor_type == "file":
        return "structural"
    if labels & {
        "Signal", "Port", "Parameter", "State", "ClockDomain", "ResetDomain", "Instance",
        "Branch", "Condition", "Macro",
    }:
        return "causal"
    if "File" in labels or "Class" in labels:
        return "structural"
    return "unmapped"
