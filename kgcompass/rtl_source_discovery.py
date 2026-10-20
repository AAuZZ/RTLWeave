from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from rtl_anchors import GENERIC_WORDS, RTLAnchor, extract_rtl_anchors
except Exception:  # pragma: no cover
    from .rtl_anchors import GENERIC_WORDS, RTLAnchor, extract_rtl_anchors


RTL_SUFFIXES = {".v", ".vh", ".sv", ".svh"}
SKIP_PARTS = {
    ".git", "build", "out", "dist", "generated", "node_modules", "vendor",
    "third_party", "test", "tests", "tb", "dv",
}


@dataclass(frozen=True)
class SourceDiscoveryResult:
    file_path: str
    explicit_path_match: bool
    explicit_anchor_matches: Tuple[str, ...]
    identifier_anchor_matches: Tuple[str, ...]
    path_token_matches: Tuple[str, ...]
    discovery_mode: str = "anchor_match"
    dependency_parent: str = ""
    dependency_relation: str = ""
    dependency_symbol: str = ""

    @property
    def rank_key(self) -> Tuple[int, int, int, int, str]:
        return (
            -int(self.explicit_path_match),
            -len(self.explicit_anchor_matches),
            -len(self.identifier_anchor_matches),
            -len(self.path_token_matches),
            self.file_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("explicit_anchor_matches", "identifier_anchor_matches", "path_token_matches"):
            result[key] = list(result[key])
        result["rank_key"] = list(self.rank_key)
        return result


def _source_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RTL_SUFFIXES:
            continue
        relative = path.relative_to(repo_root)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & SKIP_PARTS:
            continue
        yield path


def _contains_identifier(content: str, identifier: str) -> bool:
    if not identifier:
        return False
    pattern = r"(?<![A-Za-z0-9_$])" + re.escape(identifier) + r"(?![A-Za-z0-9_$])"
    return bool(re.search(pattern, content, re.IGNORECASE))


def _issue_path_tokens(issue_text: str) -> set:
    return {
        token.lower()
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", issue_text or "")
        if token.lower() not in GENERIC_WORDS and len(token) >= 4
    }


def discover_rtl_source_files(
    repo_root: str | Path,
    issue_text: str,
    limit: int = 8,
    dependency_depth: int = 1,
) -> List[SourceDiscoveryResult]:
    root = Path(repo_root).resolve()
    anchors = extract_rtl_anchors(issue_text)
    path_anchors = [anchor for anchor in anchors if anchor.file_path]
    explicit_identifiers = {
        anchor.normalized.exact_identifier
        for anchor in anchors
        if anchor.anchor_type == "identifier" and anchor.explicit
    }
    identifiers = {
        anchor.normalized.exact_identifier
        for anchor in anchors
        if anchor.anchor_type == "identifier" and anchor.normalized.exact_identifier
    }
    path_tokens = _issue_path_tokens(issue_text)

    results: List[SourceDiscoveryResult] = []
    source_contents: Dict[str, str] = {}
    definition_index: Dict[str, List[str]] = {}
    for source_path in _source_files(root):
        relative = source_path.relative_to(root).as_posix()
        try:
            content = source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        source_contents[relative] = content
        for _, symbol in re.findall(
            r"(?m)^\s*(module|interface|package)\s+([A-Za-z_][A-Za-z0-9_$]*)",
            content,
        ):
            definition_index.setdefault(symbol.lower(), []).append(relative)
        explicit_path_match = any(
            relative.lower() == anchor.file_path.lower()
            or relative.lower().endswith("/" + anchor.file_path.lower())
            or source_path.name.lower() == Path(anchor.file_path).name.lower()
            for anchor in path_anchors
        )
        explicit_hits = tuple(sorted(
            identifier for identifier in explicit_identifiers
            if _contains_identifier(content, identifier)
        ))
        identifier_hits = tuple(sorted(
            identifier for identifier in identifiers
            if identifier not in explicit_identifiers and _contains_identifier(content, identifier)
        ))
        relative_tokens = {
            token for token in re.split(r"[^a-z0-9]+", relative.lower()) if len(token) >= 3
        }
        path_hits = tuple(sorted(relative_tokens & path_tokens))
        if not explicit_path_match and not explicit_hits and not identifier_hits and not path_hits:
            continue
        results.append(SourceDiscoveryResult(
            file_path=relative,
            explicit_path_match=explicit_path_match,
            explicit_anchor_matches=explicit_hits,
            identifier_anchor_matches=identifier_hits,
            path_token_matches=path_hits,
        ))

    results.sort(key=lambda item: item.rank_key)
    limit = max(1, int(limit))
    if not results or dependency_depth <= 0:
        return results[:limit]

    # Interleave typed dependencies immediately after each high-ranked seed.
    # This lets a top-level module bring its implementation dependencies into a
    # fixed file budget without globally mixing unrelated lexical matches.
    ordered: List[SourceDiscoveryResult] = []
    seen = set()
    for seed in results:
        if len(ordered) >= limit:
            break
        if seed.file_path in seen:
            continue
        seen.add(seed.file_path)
        ordered.append(seed)
        current_level = [seed]
        for _ in range(dependency_depth):
            next_level: List[SourceDiscoveryResult] = []
            for parent in current_level:
                content = source_contents.get(parent.file_path, "")
                for relation, symbol in _typed_dependencies(content, definition_index):
                    for dependency_file in definition_index.get(symbol.lower(), []):
                        if dependency_file in seen or dependency_file == parent.file_path:
                            continue
                        seen.add(dependency_file)
                        dependency = SourceDiscoveryResult(
                            file_path=dependency_file,
                            explicit_path_match=False,
                            explicit_anchor_matches=(),
                            identifier_anchor_matches=(),
                            path_token_matches=(),
                            discovery_mode="typed_dependency",
                            dependency_parent=parent.file_path,
                            dependency_relation=relation,
                            dependency_symbol=symbol,
                        )
                        ordered.append(dependency)
                        next_level.append(dependency)
                        if len(ordered) >= limit:
                            return ordered[:limit]
            if not next_level:
                break
            current_level = next_level
    return ordered[:limit]


def _typed_dependencies(
    content: str,
    definition_index: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    dependencies: List[Tuple[str, str]] = []
    seen = set()
    known = set(definition_index)

    instance_pattern = re.compile(
        r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*"
        r"(?:#\s*\([^;]*?\)\s*)?"
        r"[A-Za-z_][A-Za-z0-9_$]*\s*\("
    )
    for match in instance_pattern.finditer(content or ""):
        symbol = match.group(1)
        key = ("instantiates", symbol.lower())
        if symbol.lower() not in known or key in seen:
            continue
        seen.add(key)
        dependencies.append(("instantiates", symbol))

    for symbol in re.findall(r"\bimport\s+([A-Za-z_][A-Za-z0-9_$]*)::", content or ""):
        key = ("imports", symbol.lower())
        if symbol.lower() not in known or key in seen:
            continue
        seen.add(key)
        dependencies.append(("imports", symbol))
    return dependencies
