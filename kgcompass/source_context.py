from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


VERILOG_EXT_RE = re.compile(r"\.(?:sv|svh|v|vh)$", re.IGNORECASE)
SKIP_DIR_RE = re.compile(
    r"(^|/)(\.git|tb|test|tests|dv|sim|verif|formal|verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages|benchmark|benchmarks|out|build|bazel-out|autogen|generated)(/|$)",
    re.IGNORECASE,
)
VENDOR_TEST_RE = re.compile(r"(^|/)vendor/.*(?:dv|test|uvm)", re.IGNORECASE)


@dataclass
class SourceContextEntry:
    file_path: str
    mode: str
    shown_lines: str
    start_line: int
    end_line: int
    total_lines: int
    char_count: int
    reason: str = ""
    entity_count: int = 0
    anchors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceContextBundle:
    text: str
    manifest: Dict[str, Any]


@dataclass
class SourceContextState:
    """Repository-relative source files retained across repair attempts."""

    initial_files: List[str] = field(default_factory=list)
    initial_full_files: List[str] = field(default_factory=list)
    requested_files: List[str] = field(default_factory=list)
    rendered_expanded_files: List[str] = field(default_factory=list)

    @classmethod
    def from_initial_manifest(cls, manifest: Dict[str, Any] | None) -> "SourceContextState":
        entries = (manifest or {}).get("files") or []
        initial_files = _dedupe(entry.get("file_path") for entry in entries if entry.get("file_path"))
        initial_full_files = _dedupe(
            entry.get("file_path")
            for entry in entries
            if entry.get("file_path") and str(entry.get("mode") or "").startswith("full_file_")
        )
        return cls(initial_files=initial_files, initial_full_files=initial_full_files)

    def add_requested_files(self, files: Sequence[str]) -> None:
        self.requested_files = _dedupe([*self.requested_files, *(files or [])])

    def set_rendered_expanded_files(self, files: Sequence[str]) -> None:
        self.rendered_expanded_files = _dedupe(files or [])

    @property
    def expanded_files(self) -> List[str]:
        full_initial = set(self.initial_full_files)
        return [path for path in self.requested_files if path not in full_initial]

    @property
    def all_loaded_files(self) -> List[str]:
        return _dedupe([*self.initial_files, *self.rendered_expanded_files])

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "initial_files": list(self.initial_files),
            "initial_full_files": list(self.initial_full_files),
            "requested_files": list(self.requested_files),
            "expanded_files": list(self.expanded_files),
            "rendered_expanded_files": list(self.rendered_expanded_files),
            "all_loaded_files": list(self.all_loaded_files),
        }


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        item = str(item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def normalize_repo_relative_path(repo_path: str, file_path: Any) -> str:
    text = str(file_path or "").replace("\\", "/").strip().strip("'\"`")
    if not text or not VERILOG_EXT_RE.search(text):
        return ""
    if text.startswith("./"):
        text = text[2:]
    repo_abs = os.path.abspath(repo_path).replace("\\", "/").rstrip("/") if repo_path else ""
    if repo_abs and os.path.isabs(text):
        abs_text = os.path.abspath(text).replace("\\", "/")
        if abs_text.startswith(repo_abs + "/"):
            return os.path.relpath(abs_text, repo_abs).replace("\\", "/")
    if repo_abs and repo_abs in text:
        return text.split(repo_abs, 1)[1].lstrip("/")
    cleaned = text.lstrip("/")
    for marker in ("rtl/", "src/", "hw/", "include/", "vendor/"):
        idx = cleaned.find(marker)
        if idx >= 0:
            candidate = cleaned[idx:]
            if os.path.exists(os.path.join(repo_path, *candidate.split("/"))):
                return candidate
    return cleaned


def is_source_path_allowed(path: str) -> bool:
    lowered = str(path or "").replace("\\", "/").lower()
    return bool(path) and VERILOG_EXT_RE.search(path) and not (SKIP_DIR_RE.search(lowered) or VENDOR_TEST_RE.search(lowered))


def entity_anchor(entity: Dict[str, Any]) -> str:
    name = str(entity.get("signature") or entity.get("name") or entity.get("signal_name") or "").strip()
    start = entity.get("start_line")
    end = entity.get("end_line")
    role = str(entity.get("repair_role") or entity.get("verilog_kind") or entity.get("type") or "").strip()
    bits = []
    if name:
        bits.append(name)
    if start or end:
        bits.append(f"{start or '?'}-{end or '?'}")
    if role:
        bits.append(f"[{role}]")
    return " ".join(bits) or "entity"


def repo_file_index(repo_path: str, *, limit: int = 4000) -> List[Dict[str, Any]]:
    """Build a compact source-file index for diagnosis prompts.

    This is intentionally shallow: it gives the diagnosis agent a map of
    candidate RTL files without dumping their contents.
    """
    root = Path(repo_path)
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if len(rows) >= limit:
            break
        if not path.is_file() or not VERILOG_EXT_RE.search(path.name):
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not is_source_path_allowed(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        modules = re.findall(r"\b(?:module|interface|package)\s+([A-Za-z_][A-Za-z0-9_$]*)", text)
        rows.append({
            "file_path": rel,
            "line_count": text.count("\n") + (1 if text else 0),
            "modules": _dedupe(modules)[:6],
            "summary": _summarize_file(rel, modules),
        })
    return rows


def _summarize_file(rel: str, modules: Sequence[str]) -> str:
    base = os.path.splitext(os.path.basename(rel))[0]
    if modules:
        return f"defines {', '.join(modules[:4])}"
    if rel.endswith((".svh", ".vh")):
        return f"include/header likely defining macros or parameters for {base}"
    return f"RTL source {base}"


class SourceContextManager:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)

    def build_initial_context(
        self,
        *,
        related_entities: Dict[str, Sequence[Dict[str, Any]]],
        preferred_files: Sequence[str] = (),
        fault_anchor_entities: Sequence[Dict[str, Any]] = (),
        repair_anchor_entities: Sequence[Dict[str, Any]] = (),
        fault_anchor_spans: Sequence[Dict[str, Any]] = (),
        max_files: int = 8,
        full_file_limit: int = 2500,
        max_full_files: int = 4,
        window: int = 160,
        max_total_chars: int = 70000,
    ) -> SourceContextBundle:
        groups: List[Tuple[str, Dict[str, Any]]] = []
        for entity in repair_anchor_entities or []:
            groups.append(("repair_anchor", entity))
        for entity in fault_anchor_entities or []:
            groups.append(("fault_anchor", entity))
        for group_name in ("patchable_spans", "edit_targets", "direct_drivers", "top_level_wiring", "config_registers", "methods", "rtl_entities"):
            for entity in related_entities.get(group_name) or []:
                groups.append((group_name, entity))

        span_map: Dict[str, List[Tuple[int, int]]] = {}
        for span in fault_anchor_spans or []:
            if not isinstance(span, dict):
                continue
            file_path = normalize_repo_relative_path(self.repo_path, span.get("file_path") or span.get("file"))
            if not is_source_path_allowed(file_path):
                continue
            start, end = _line_pair(span)
            if start and end:
                span_map.setdefault(file_path, []).append((start, end))

        selected: Dict[str, Dict[str, Any]] = {}
        for raw in preferred_files or []:
            rel = normalize_repo_relative_path(self.repo_path, raw)
            if is_source_path_allowed(rel):
                selected.setdefault(rel, {"reason": "issue_mentioned", "entities": []})
        for rel in span_map:
            selected.setdefault(rel, {"reason": "fault_anchor_span", "entities": []})
        for reason, entity in groups:
            rel = normalize_repo_relative_path(self.repo_path, entity.get("file_path"))
            if not is_source_path_allowed(rel):
                continue
            info = selected.setdefault(rel, {"reason": reason, "entities": []})
            info["entities"].append(entity)
            if len(selected) >= max_files:
                break

        return self._render(
            selected,
            span_map,
            max_files=max_files,
            full_file_limit=full_file_limit,
            max_full_files=max_full_files,
            window=window,
            phase="initial",
            max_total_chars=max_total_chars,
        )

    def build_expanded_context(
        self,
        *,
        requested_files: Sequence[str],
        exclude_files: Sequence[str] = (),
        fallback_text: str = "",
        max_files: int = 10,
        full_file_limit: int = 3500,
        max_full_files: int = 6,
        window: int = 180,
        anchor_spans: Sequence[Dict[str, Any]] = (),
        max_total_chars: int = 50000,
    ) -> SourceContextBundle:
        selected: Dict[str, Dict[str, Any]] = {}
        span_map: Dict[str, List[Tuple[int, int]]] = {}
        for span in anchor_spans or []:
            if not isinstance(span, dict):
                continue
            rel = normalize_repo_relative_path(self.repo_path, span.get("file_path") or span.get("file"))
            if not is_source_path_allowed(rel):
                continue
            start, end = _line_pair(span)
            if start and end:
                span_map.setdefault(rel, []).append((start, end))
        excluded = {
            normalize_repo_relative_path(self.repo_path, raw)
            for raw in (exclude_files or [])
        }
        excluded.discard("")
        skipped_loaded: List[str] = []
        for raw in requested_files or []:
            rel = normalize_repo_relative_path(self.repo_path, raw)
            if rel in excluded:
                skipped_loaded.append(rel)
                continue
            if is_source_path_allowed(rel):
                selected.setdefault(rel, {"reason": "requested_by_diagnosis", "entities": []})
            if len(selected) >= max_files:
                break
        bundle = self._render(
            selected,
            span_map,
            max_files=max_files,
            full_file_limit=full_file_limit,
            max_full_files=max_full_files,
            window=window,
            phase="diagnosis_expansion",
            max_total_chars=max_total_chars,
        )
        if not bundle.text and fallback_text:
            bundle.text = fallback_text
            bundle.manifest["fallback_used"] = True
        bundle.manifest["requested_files"] = _dedupe(requested_files or [])
        bundle.manifest["excluded_already_full"] = _dedupe(skipped_loaded)
        return bundle

    def _render(
        self,
        selected: Dict[str, Dict[str, Any]],
        span_map: Dict[str, List[Tuple[int, int]]],
        *,
        max_files: int,
        full_file_limit: int,
        max_full_files: int,
        window: int,
        phase: str,
        max_total_chars: int,
    ) -> SourceContextBundle:
        sections: List[str] = []
        entries: List[SourceContextEntry] = []
        full_files_used = 0
        rendered_chars = 0
        for rel, info in list(selected.items())[:max_files]:
            remaining = max_total_chars - rendered_chars
            if remaining < 1200:
                break
            abs_path = os.path.join(self.repo_path, *rel.split("/"))
            if not os.path.exists(abs_path):
                continue
            try:
                lines = Path(abs_path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            if not lines:
                continue
            total = len(lines)
            reason = str(info.get("reason") or "candidate")
            entities = list(info.get("entities") or [])
            anchor_lines = list(span_map.get(rel) or [])
            for entity in entities:
                pair = _line_pair(entity)
                if pair[0] and pair[1]:
                    anchor_lines.append(pair)
            force_full = reason in {"issue_mentioned", "requested_by_diagnosis"} or rel in span_map
            if total <= full_file_limit and full_files_used < max_full_files:
                full_estimate = sum(len(str(i)) + 2 + len(lines[i - 1]) + 1 for i in range(1, total + 1))
                if anchor_lines and full_estimate > remaining:
                    anchor_start = min(a for a, _ in anchor_lines)
                    anchor_end = max(b for _, b in anchor_lines)
                    start = max(1, anchor_start - window)
                    end = min(total, anchor_end + window)
                    mode = f"window_{reason}_{window}"
                else:
                    start, end = 1, total
                    mode = "full_file_" + reason if force_full else "full_file_small"
                    full_files_used += 1
            else:
                if anchor_lines:
                    anchor_start = min(a for a, _ in anchor_lines)
                    anchor_end = max(b for _, b in anchor_lines)
                else:
                    anchor_start = anchor_end = 1
                start = max(1, anchor_start - window)
                end = min(total, anchor_end + window)
                mode = f"window_{reason}_{window}"

            start, end, excerpt = self._excerpt_with_budget(lines, start, end, remaining)
            if not excerpt.strip():
                continue
            anchors = [entity_anchor(entity) for entity in entities[:8]]
            sections.append("\n".join([
                f"### {rel}",
                f"- source_mode: {mode}",
                f"- shown_lines: {start}-{end}",
                f"- total_lines: {total}",
                f"- context_reason: {reason}",
                "- candidate_entities:",
                *[f"- {anchor}" for anchor in anchors],
                "```verilog",
                excerpt,
                "```",
            ]))
            rendered_chars += len(excerpt)
            entries.append(SourceContextEntry(
                file_path=rel,
                mode=mode,
                shown_lines=f"{start}-{end}",
                start_line=start,
                end_line=end,
                total_lines=total,
                char_count=len(excerpt),
                reason=reason,
                entity_count=len(entities),
                anchors=anchors,
            ))
        manifest = {
            "phase": phase,
            "repo_path": self.repo_path,
            "max_files": max_files,
            "full_file_limit": full_file_limit,
            "max_full_files": max_full_files,
            "window": window,
            "max_total_chars": max_total_chars,
            "total_rendered_chars": rendered_chars,
            "files": [entry.to_dict() for entry in entries],
            "file_index": repo_file_index(self.repo_path, limit=600),
        }
        return SourceContextBundle(text="\n\n".join(sections), manifest=manifest)

    def _excerpt_with_budget(self, lines: Sequence[str], start: int, end: int, budget: int) -> Tuple[int, int, str]:
        start = max(1, start)
        end = min(len(lines), end)
        if budget <= 0 or start > end:
            return start, end, ""
        selected: List[str] = []
        last_line = start - 1
        used = 0
        for line_no in range(start, end + 1):
            line = f"{line_no}: {lines[line_no - 1]}"
            added = len(line) + (1 if selected else 0)
            if selected and used + added > budget:
                break
            selected.append(line)
            used += added
            last_line = line_no
        return start, last_line, "\n".join(selected)


def _line_pair(item: Dict[str, Any]) -> Tuple[int, int]:
    try:
        start = int(item.get("start_line") or 0)
        end = int(item.get("end_line") or 0)
    except (TypeError, ValueError):
        return 0, 0
    if start <= 0 or end <= 0:
        return 0, 0
    if end < start:
        end = start
    return start, end
