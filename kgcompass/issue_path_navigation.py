from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


VERILOG_EXT_RE = re.compile(r"\.(?:sv|svh|v|vh)$", re.IGNORECASE)


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        value = str(item or "").replace("\\", "/").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip().strip("'\"`")
    if not text:
        return ""
    if text.startswith("./"):
        text = text[2:]
    if VERILOG_EXT_RE.search(text):
        if text.startswith("/"):
            text = text.lstrip("/")
        for marker in ("rtl/", "src/", "hw/", "include/", "vendor/"):
            idx = text.find(marker)
            if idx >= 0:
                return text[idx:]
        return text
    return ""


def _extract_paths_from_text(text: str) -> List[str]:
    return _dedupe(match.group(1) for match in re.finditer(r"([A-Za-z0-9_./-]+\.(?:sv|svh|v|vh))", text or "", re.IGNORECASE))


@dataclass
class IssuePathRecord:
    target: str = ""
    file_path: str = ""
    start_line: Any = None
    end_line: Any = None
    kind: str = ""
    group: str = ""
    module_name: str = ""
    signal: str = ""
    path_text: str = ""
    path: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "localization"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IssuePathNavigator:
    """Lightweight index over persisted Issue-to-PatchableSpan paths.

    This intentionally does not answer arbitrary graph traversal queries.
    It is a deterministic navigation layer for cached paths; dynamic
    cross-file signal propagation remains a Neo4j responsibility.
    """

    capabilities = {
        "cached_issue_paths": True,
        "cached_file_entity_lookup": True,
        "cross_file_signal_propagation": False,
        "requires_neo4j_for_dynamic_trace": True,
    }

    def __init__(self, paths: Sequence[Any] | None = None):
        self.records = [record for record in (self._coerce(item) for item in paths or []) if record]

    @classmethod
    def from_location_payload(cls, payload: Dict[str, Any]) -> "IssuePathNavigator":
        return cls((payload or {}).get("issue_to_entity_paths") or [])

    def merge(self, paths: Sequence[Any] | None) -> "IssuePathNavigator":
        return IssuePathNavigator([*self.to_dicts(), *(paths or [])])

    def files(self, limit: int | None = None) -> List[str]:
        files: List[str] = []
        for record in self.records:
            if record.file_path:
                files.append(record.file_path)
            for path in _extract_paths_from_text(record.path_text):
                files.append(normalize_path(path))
        result = _dedupe(path for path in files if path)
        return result[:limit] if limit else result

    def entities(self, limit: int | None = None) -> List[Dict[str, Any]]:
        rows = []
        for record in self.records:
            rows.append({
                "target": record.target,
                "file_path": record.file_path,
                "start_line": record.start_line,
                "end_line": record.end_line,
                "kind": record.kind,
                "group": record.group,
                "module_name": record.module_name,
                "signal": record.signal,
                "source": record.source,
            })
        return rows[:limit] if limit else rows

    def anchors_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        normalized = normalize_path(file_path)
        if not normalized:
            return []
        return [
            row
            for row in self.entities()
            if normalize_path(row.get("file_path")) == normalized
        ]

    def summarize(self, limit: int = 12) -> str:
        lines = [
            "SnapshotKG navigation index:",
            "- capabilities: cached issue paths and file/entity anchors; no arbitrary signal propagation",
            f"- files: {', '.join(self.files(limit=8)) or '(none)'}",
            "- paths:",
        ]
        for record in self.records[:limit]:
            location = record.file_path
            if location and (record.start_line or record.end_line):
                location = f"{location}:{record.start_line or '?'}-{record.end_line or '?'}"
            header = " ".join(
                part
                for part in [
                    record.target,
                    f"[{record.kind}]" if record.kind else "",
                    f"@ {location}" if location else "",
                ]
                if part
            )
            lines.append(f"  - {header or record.path_text or '(path)'}")
            if record.path_text and record.path_text != header:
                lines.append(f"    path: {record.path_text}")
        return "\n".join(lines)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "capabilities": dict(self.capabilities),
            "files": self.files(),
            "entities": self.entities(),
            "path_count": len(self.records),
        }

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def _coerce(self, item: Any) -> IssuePathRecord | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            paths = _extract_paths_from_text(text)
            return IssuePathRecord(
                file_path=normalize_path(paths[0]) if paths else "",
                path_text=text,
                source="path_text",
            )
        if not isinstance(item, dict):
            return None
        path_text = str(item.get("path_text") or "").strip()
        path = item.get("path") if isinstance(item.get("path"), list) else []
        file_path = normalize_path(item.get("file_path") or item.get("file") or item.get("target_file"))
        if not file_path:
            text_paths = _extract_paths_from_text(path_text)
            file_path = normalize_path(text_paths[0]) if text_paths else ""
        target = str(item.get("target") or item.get("name") or item.get("signature") or "").strip()
        return IssuePathRecord(
            target=target,
            file_path=file_path,
            start_line=item.get("start_line"),
            end_line=item.get("end_line"),
            kind=str(item.get("type") or item.get("kind") or item.get("verilog_kind") or "").strip(),
            group=str(item.get("group") or item.get("category") or "").strip(),
            module_name=str(item.get("module_name") or "").strip(),
            signal=str(item.get("signal") or item.get("signal_name") or "").strip(),
            path_text=path_text,
            path=path,
            source=str(item.get("source") or "localization"),
        )
