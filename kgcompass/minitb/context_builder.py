from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .evidence_slice import build_minitb_evidence_slice
from .models import PortInfo, TargetContext, VisibleCase, write_json


RTL_SUFFIXES = {".v", ".sv", ".vh", ".svh"}
SKIP_DIRS = {
    ".git",
    "build",
    "out",
    "obj_dir",
    "vendor/riscv-tests",
    "vendor/riscv-dv",
}
SKIP_PARTS = {
    ".git",
    "build",
    "out",
    "obj_dir",
    "coverage",
    "uvmf_2022",
    "uvmf_template_output",
}
MAX_RTL_PATHS = 20_000
MAX_FALLBACK_CONTENT_FILES = 512
MAX_ADJACENT_FILES = 4_000
# Keep a small repository-level shortlist rather than forcing generation to
# the first three lexical/KG hits.  Several Ibex issues mention a pipeline
# symptom before the leaf block that exposes the useful oracle (for example,
# ibex_pmp for debug-mode PMP checks).  Eight module interfaces remain compact
# enough for target selection while avoiding that truncation.  Only the chosen
# target contributes full RTL source to the prompt; the rest are compact port
# summaries, so twelve remains a small bounded context.
MAX_CANDIDATE_TARGETS = 12
ASSET_PARTS = {"dv", "test", "tests", "sim", "examples", "example"}
PATH_PATTERN = re.compile(r"[A-Za-z0-9_./-]+\.(?:svh?|vh?)\b")
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*\b")
KG_FORBIDDEN_KEYS = frozenset(
    {
        "gold_files",
        "gold_regions",
        "fix_patch",
        "test_patch",
        "tb_script",
        "reproducer_path",
        "reproducer_signal",
    }
)
ISSUE_STOP_WORDS = frozenset(
    {
        "actual",
        "affected",
        "behavior",
        "core",
        "correctly",
        "current",
        "expected",
        "function",
        "instruction",
        "issue",
        "must",
        "should",
        "signal",
        "triggered",
        "when",
        "with",
    }
)


class ContextBuilder:
    def __init__(self, repo_root: Path, case: VisibleCase, kg_location: Path | None = None):
        self.repo_root = repo_root.resolve()
        self.case = case
        self.kg_location = kg_location.resolve() if kg_location else None
        self.kg: dict[str, Any] = {}
        if self.kg_location:
            with self.kg_location.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("KG location file must contain a JSON object")
            if value.get("instance_id") not in (None, "", case.instance_id):
                raise ValueError("KG location instance_id does not match the selected case")
            forbidden = sorted(_recursive_keys(value) & KG_FORBIDDEN_KEYS)
            if forbidden:
                raise ValueError(
                    "KG location contains offline evaluation fields and cannot be used online: "
                    + ", ".join(forbidden)
                )
            self.kg = value

    def build(self) -> dict[str, Any]:
        candidates = self._candidate_entities()
        targets: list[TargetContext] = []
        seen: set[tuple[str, str]] = set()
        for rank, candidate in enumerate(candidates, start=1):
            target = self._target_from_candidate(candidate, rank)
            if target is None:
                continue
            key = (target.file, target.module)
            if key in seen:
                continue
            seen.add(key)
            targets.append(target)
            if len(targets) >= MAX_CANDIDATE_TARGETS:
                break
        if not targets:
            targets = self._fallback_targets()
        if not targets:
            raise ValueError("no candidate RTL module could be derived from KG, issue text, or repository")

        primary = targets[0]
        assets = self._adjacent_assets(primary)
        paths = self._kg_paths()
        issue = {"title": self.case.title, "problem_statement": self.case.problem_statement}
        target_payloads = []
        for target in targets:
            payload = target.to_dict()
            payload["minitb_evidence"] = build_minitb_evidence_slice(self.kg, payload, issue)
            target_payloads.append(payload)
        evidence_slice = target_payloads[0]["minitb_evidence"]
        package_context = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")[:24000]
            for path in sorted((self.repo_root / "rtl").glob("*_pkg.sv"))[:3]
        )
        summary = {
            "schema_version": "kgveri-minitb-context-v1",
            "instance_id": self.case.instance_id,
            "repo": self.case.repo,
            "kg_available": bool(self.kg),
            "package_context": package_context,
            "kg_location": str(self.kg_location) if self.kg_location else "",
            "issue": {
                "title": self.case.title[:1000],
                "problem_statement": self.case.problem_statement[:6000],
            },
            "candidate_targets": target_payloads,
            "selected_target": target_payloads[0],
            "kg_paths": paths,
            "minitb_evidence": evidence_slice,
            "adjacent_assets": assets,
            "budgets": {
                "issue_chars": min(len(self.case.problem_statement), 6000),
                "kg_path_chars": len(json.dumps(paths, ensure_ascii=True)),
                "target_source_chars": len(primary.source),
                "adjacent_asset_chars": sum(len(item["excerpt"]) for item in assets),
            },
            "provenance": {
                "repository": str(self.repo_root),
                "base_commit": self.case.base_commit,
                "all_source_paths_are_repository_relative": True,
            },
        }
        return summary

    def write(self, path: Path) -> dict[str, Any]:
        summary = self.build()
        write_json(path, summary)
        return summary

    def _candidate_entities(self) -> list[dict[str, Any]]:
        if not self.kg:
            return []
        related = self.kg.get("related_entities") or {}
        groups = (
            "patchable_spans",
            "edit_targets",
            "methods",
            "classes",
            "rtl_entities",
            "evidence_entities",
        )
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        # Start from localized repair modules.  An issue may mention several
        # downstream files; putting those lexical anchors first can select a
        # leaf block whose inputs bypass the faulty upstream behavior.
        for group in groups:
            values = related.get(group) or []
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                file_path = str(value.get("file_path") or "")
                module = str(value.get("module_name") or "")
                if value.get("type") == "class" or value.get("verilog_kind") == "module":
                    module = module or str(value.get("name") or "")
                key = (file_path, module)
                if not file_path or key in seen:
                    continue
                seen.add(key)
                item = dict(value)
                item["_group"] = group
                candidates.append(item)
        # Explicit issue paths remain repository-level fallback boundaries
        # when localization has no candidate for that file.
        for file_path in self.kg.get("issue_mentioned_files") or []:
            if not isinstance(file_path, str) or not file_path:
                continue
            key = (file_path, "")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "file_path": file_path,
                    "module_name": "",
                    "name": file_path,
                    "similarity": "exact_issue_file_anchor",
                    "distance": 0,
                    "_group": "issue_mentioned_files",
                }
            )
        return candidates

    def _target_from_candidate(self, candidate: dict[str, Any], rank: int) -> TargetContext | None:
        path = self._resolve_source_path(str(candidate.get("file_path") or ""))
        if path is None or path.suffix.lower() not in RTL_SUFFIXES:
            return None
        module_hint = str(candidate.get("module_name") or "")
        if not module_hint and candidate.get("verilog_kind") == "module":
            module_hint = str(candidate.get("name") or "")
        parsed = self._parse_modules(path)
        if not parsed:
            return None
        module = next((item for item in parsed if item.module == module_hint), parsed[0])
        module.rank = rank
        module.reason = (
            f"KG group={candidate.get('_group', '')}; entity={candidate.get('name', '')}; "
            f"similarity={candidate.get('similarity', '')}; distance={candidate.get('distance', '')}"
        )
        return module

    def _fallback_targets(self) -> list[TargetContext]:
        explicit_paths = PATH_PATTERN.findall(self.case.problem_statement)
        candidates: list[Path] = []
        for raw in explicit_paths:
            path = self._resolve_source_path(raw)
            if path and path not in candidates:
                candidates.append(path)

        identifier_tokens = {
            token.lower()
            for token in IDENTIFIER_PATTERN.findall(self.case.problem_statement)
            if "_" in token and len(token) >= 5
        }
        issue_tokens = {
            token.lower()
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9]+", f"{self.case.title} {self.case.problem_statement}"
            )
            if len(token) >= 4 and token.lower() not in ISSUE_STOP_WORDS
        }
        rtl_files = list(self._rtl_files())
        path_scores: list[tuple[int, Path]] = []
        for path in rtl_files:
            if path in candidates:
                continue
            normalized_name = path.stem.lower()
            path_tokens = set(re.findall(r"[a-z0-9]+", normalized_name))
            score = 4 * sum(1 for token in issue_tokens if token in path_tokens)
            score += 2 * sum(1 for token in identifier_tokens if token in normalized_name)
            score += sum(1 for token in issue_tokens if token in normalized_name)
            path_scores.append((score, path))

        # File names and issue paths are cheap to inspect across a large
        # repository.  Source contents are only read for a bounded shortlist;
        # this keeps no-KG fallback usable on Caliptra's generated UVM tree.
        path_scores.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
        positive = [path for score, path in path_scores if score > 0]
        content_candidates = _dedupe_paths(
            [*positive[:128], *(path for _, path in path_scores[:MAX_FALLBACK_CONTENT_FILES])]
        )[:MAX_FALLBACK_CONTENT_FILES]
        scored: list[tuple[int, Path]] = []
        path_score_by_file = {path: score for score, path in path_scores}
        for path in content_candidates:
            score = path_score_by_file.get(path, 0)
            # File names alone are insufficient for cases such as CSR issues,
            # where the visible anchors are register/signal identifiers (mtvec,
            # mepc, data_rvalid_i) rather than the containing module name.
            source = path.read_text(encoding="utf-8", errors="ignore").lower()
            score += 3 * sum(1 for token in identifier_tokens if token in source)
            score += sum(
                1
                for token in issue_tokens
                if re.search(rf"(?<![a-z0-9_$]){re.escape(token)}(?![a-z0-9_$])", source)
            )
            if score:
                scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
        candidates.extend(path for _, path in scored[:12])

        targets: list[TargetContext] = []
        seen: set[tuple[str, str]] = set()
        for path in candidates:
            for target in self._parse_modules(path):
                key = (target.file, target.module)
                if key in seen:
                    continue
                seen.add(key)
                target.rank = len(targets) + 1
                target.reason = "issue filename/identifier fallback; kg_available=false"
                targets.append(target)
                if len(targets) >= MAX_CANDIDATE_TARGETS:
                    return targets
        return targets

    def _parse_modules(self, path: Path) -> list[TargetContext]:
        raw_source = path.read_text(encoding="utf-8", errors="ignore")
        # Match the MiniTB backend: Verilator defines VERILATOR while parsing.
        # A module hidden by that condition cannot be a usable DUT candidate.
        active_source = _active_preprocessor_text(
            raw_source, {"VERILATOR", *_local_feature_defines(raw_source)}
        )
        active_modules = set(
            re.findall(r"\bmodule\s+(?:automatic\s+)?([A-Za-z_][A-Za-z0-9_$]*)", active_source)
        )
        try:
            from kgcompass.verilog_ast import VerilogAstExtractor

            extractor = VerilogAstExtractor(repo_root=str(self.repo_root))
            classes = extractor.extract_classes(str(path))
            parsed: list[TargetContext] = []
            for item in classes:
                if item.get("verilog_kind") != "module":
                    continue
                if str(item.get("name") or "") not in active_modules:
                    continue
                ports: list[PortInfo] = []
                parameters: dict[str, str] = {}
                for entity in item.get("rtl_entities") or []:
                    kind = entity.get("verilog_kind")
                    if kind == "port":
                        ports.append(
                            PortInfo(
                                name=str(entity.get("signal_name") or str(entity.get("name", "")).split(".")[-1]),
                                direction=str(entity.get("direction") or ""),
                                width=str(entity.get("width") or ""),
                                declaration=str(entity.get("declaration") or entity.get("source_code") or ""),
                            )
                        )
                    # Only header/module parameters belong to the public DUT
                    # interface.  A localparam is scoped inside the module and
                    # must not be mirrored or overridden by the harness.
                    elif kind == "parameter":
                        name = str(entity.get("signal_name") or str(entity.get("name", "")).split(".")[-1])
                        parameters[name] = str(entity.get("declaration") or entity.get("source_code") or "")
                source = str(item.get("source_code") or "")[:45000]
                parsed.append(
                    TargetContext(
                        file=str(path.relative_to(self.repo_root)),
                        module=str(item.get("name") or ""),
                        source=source,
                        ports=_dedupe_ports(ports),
                        parameters=parameters,
                        imports=_imports(source),
                        includes=_includes(source),
                    )
                )
            if parsed:
                # The AST frontend may omit compile-visible ports guarded by
                # a feature macro. Merge the active ANSI interface recovered
                # by the lightweight parser while retaining richer AST types.
                regex_modules = {
                    item.module: item for item in self._parse_modules_regex(path, active_source)
                }
                for target in parsed:
                    regex_target = regex_modules.get(target.module)
                    if regex_target:
                        target.ports = _dedupe_ports([*regex_target.ports, *target.ports])
                return parsed
        except Exception:
            pass
        return self._parse_modules_regex(path, active_source)

    def _parse_modules_regex(self, path: Path, source: str | None = None) -> list[TargetContext]:
        source = source if source is not None else _active_preprocessor_text(
            path.read_text(encoding="utf-8", errors="ignore"), {"VERILATOR"}
        )
        modules: list[TargetContext] = []
        pattern = re.compile(
            r"\bmodule\s+(?:automatic\s+)?([A-Za-z_][A-Za-z0-9_$]*)\s*"
            r"(?:import\s+[^;]+;\s*)*"
            r"(?:#\s*\((.*?)\)\s*)?\((.*?)\)\s*;",
            re.DOTALL,
        )
        for match in pattern.finditer(source):
            name, parameter_text, port_text = match.groups()
            end_match = re.search(r"\bendmodule\b", source[match.end() :])
            end = match.end() + end_match.end() if end_match else len(source)
            module_source = source[match.start() : end]
            ports = _parse_ansi_ports(port_text)
            parameters = {
                item.group(1): item.group(2).strip()
                for item in re.finditer(
                    r"\bparameter(?:\s+\w+|\s*\[[^]]+\])*\s+([A-Za-z_]\w*)\s*=\s*([^,\n)]+)",
                    parameter_text or "",
                )
            }
            modules.append(
                TargetContext(
                    file=str(path.relative_to(self.repo_root)),
                    module=name,
                    source=module_source[:45000],
                    ports=ports,
                    parameters=parameters,
                    imports=_imports(module_source),
                    includes=_includes(module_source),
                )
            )
        return modules

    def _resolve_source_path(self, raw: str) -> Path | None:
        normalized = raw.replace("\\", "/").strip()
        direct = self.repo_root / normalized
        if direct.is_file():
            return direct.resolve()
        raw_path = Path(normalized)
        parts = raw_path.parts
        for index in range(len(parts)):
            candidate = self.repo_root.joinpath(*parts[index:])
            if candidate.is_file():
                return candidate.resolve()
        basename = raw_path.name
        if not basename:
            return None
        matches = [path for path in self._rtl_files() if path.name == basename]
        if len(matches) == 1:
            return matches[0].resolve()
        return None

    def _rtl_files(self) -> Iterable[Path]:
        emitted = 0
        roots = {
            "ibex": [self.repo_root / "rtl"],
            "cva6": [self.repo_root / "core"],
            "caliptra-rtl": [self.repo_root / "src"],
        }.get(self.case.repo, [self.repo_root])
        roots = [root for root in roots if root.is_dir()] or [self.repo_root]
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in RTL_SUFFIXES:
                    continue
                relative_path = path.relative_to(self.repo_root)
                relative = relative_path.as_posix()
                if set(relative_path.parts) & SKIP_PARTS:
                    continue
                if any(relative == skip or relative.startswith(skip + "/") for skip in SKIP_DIRS):
                    continue
                yield path
                emitted += 1
                if emitted >= MAX_RTL_PATHS:
                    return

    def _kg_paths(self) -> list[dict[str, Any]]:
        paths = self.kg.get("issue_to_entity_paths") or []
        compact: list[dict[str, Any]] = []
        used = 0
        for item in paths:
            if not isinstance(item, dict):
                continue
            entry = {
                key: item.get(key)
                for key in (
                    "group",
                    "target",
                    "type",
                    "file_path",
                    "start_line",
                    "end_line",
                    "module_name",
                    "similarity",
                    "distance",
                    "path_text",
                )
            }
            path = self._resolve_source_path(str(entry.get("file_path") or ""))
            if path:
                entry["file_path"] = str(path.relative_to(self.repo_root))
            size = len(json.dumps(entry, ensure_ascii=True))
            if used + size > 8000:
                break
            compact.append(entry)
            used += size
        return compact

    def _adjacent_assets(self, target: TargetContext) -> list[dict[str, Any]]:
        terms = [target.module]
        terms.extend(port.name for port in target.ports[:12])
        candidates: list[tuple[int, Path, str]] = []
        inspected = 0
        for path in self.repo_root.rglob("*"):
            if not path.is_file() or path.stat().st_size > 300_000:
                continue
            relative = path.relative_to(self.repo_root)
            if set(relative.parts) & SKIP_PARTS:
                continue
            inspected += 1
            if inspected > MAX_ADJACENT_FILES:
                break
            parts = set(relative.parts)
            is_manifest = path.suffix.lower() in {".core", ".f"} or path.name.lower().startswith("makefile")
            if not (parts & ASSET_PARTS or is_manifest):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = [(text.find(term), term) for term in terms if term and text.find(term) >= 0]
            if not hits:
                continue
            index, term = min(hits)
            score = sum(1 for candidate in terms if candidate and candidate in text)
            candidates.append((score, path, _window(text, index, 9000)))
        candidates.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
        result = []
        budget = 20000
        for score, path, excerpt in candidates[:2]:
            excerpt = excerpt[:budget]
            result.append(
                {
                    "path": str(path.relative_to(self.repo_root)),
                    "score": score,
                    "excerpt": excerpt,
                }
            )
            budget -= len(excerpt)
            if budget <= 0:
                break
        return result


def _parse_ansi_ports(text: str) -> list[PortInfo]:
    text = _active_preprocessor_text(text)
    # Strip comments before splitting: commas in prose are not port separators.
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)
    ports: list[PortInfo] = []
    current_direction = ""
    current_width = ""
    for fragment in _split_top_level(text):
        clean = re.sub(r"/\*.*?\*/|//[^\n]*", " ", fragment, flags=re.DOTALL).strip()
        direction_match = re.search(r"\b(input|output|inout)\b", clean)
        if direction_match:
            current_direction = direction_match.group(1)
            # Only names sharing a declaration inherit the packed width.
            current_width = ""
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_$]*)\s*((?:\[[^]]+\]\s*)*)(?:=\s*[^,]+)?$", clean)
        if name_match and direction_match:
            current_width = "".join(re.findall(r"\[[^]]+\]", clean[:name_match.start()]))
        if current_direction and name_match:
            ports.append(
                PortInfo(
                    name=name_match.group(1),
                    direction=current_direction,
                    width=current_width,
                    declaration=clean,
                )
            )
    return _dedupe_ports(ports)


def _active_preprocessor_text(text: str, predefined: set[str] | None = None) -> str:
    """Resolve the small `ifdef subset used by module port declarations."""
    defines = set(predefined or ())
    active = True
    stack: list[tuple[bool, bool]] = []
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        directive = re.match(r"\s*`(ifdef|ifndef|else|endif|define|undef)\b\s*([A-Za-z_]\w*)?", line)
        if not directive:
            if active:
                output.append(line)
            continue
        kind, name = directive.groups()
        if kind in {"ifdef", "ifndef"}:
            condition = (name in defines) if kind == "ifdef" else (name not in defines)
            stack.append((active, condition))
            active = active and condition
        elif kind == "else" and stack:
            parent, condition = stack[-1]
            stack[-1] = (parent, not condition)
            active = parent and not condition
        elif kind == "endif" and stack:
            parent, _ = stack.pop()
            active = parent
        elif active and name:
            if kind == "define":
                defines.add(name)
            elif kind == "undef":
                defines.discard(name)
    return "".join(output)


def _local_feature_defines(text: str) -> set[str]:
    """Find locally declared macros that positively guard source features."""
    defined = set(re.findall(r"(?m)^\s*`define\s+([A-Za-z_]\w*)\b", text))
    enabled = set(re.findall(r"(?m)^\s*`ifdef\s+([A-Za-z_]\w*)\b", text))
    return defined & enabled


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _dedupe_ports(ports: list[PortInfo]) -> list[PortInfo]:
    output: list[PortInfo] = []
    seen: set[str] = set()
    for port in ports:
        if not port.name or port.name in seen:
            continue
        seen.add(port.name)
        output.append(port)
    return output


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


def _imports(source: str) -> list[str]:
    return sorted(set(re.findall(r"\bimport\s+([A-Za-z_][A-Za-z0-9_$]*)::", source)))


def _includes(source: str) -> list[str]:
    return sorted(set(re.findall(r"`include\s+[\"<]([^\">]+)[\">]", source)))


def _window(text: str, index: int, limit: int) -> str:
    half = limit // 2
    start = max(0, index - half)
    end = min(len(text), index + half)
    return text[start:end]


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).strip().lower())
            keys.update(_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys
