from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from diagnostic_evidence import build_validation_report
    from repair_types import PatchDelta, RepairPlan, ValidationEvidence, ValidationReport
except Exception:  # pragma: no cover
    from .diagnostic_evidence import build_validation_report
    from .repair_types import PatchDelta, RepairPlan, ValidationEvidence, ValidationReport


VERILOG_EXTS = (".sv", ".v", ".svh", ".vh")
CONTEXT_LIMITED_PATTERNS = (
    r"Package/class\s+'[^']+'\s+not found",
    r"\bPKGNODECL\b",
    r"Cannot find include file",
    r"Can't find",
    r"\bnot found\b",
    r"Unknown module type",
    r"Unable to bind",
    r"Unable to find",
    r"Could not find",
    r"no source files",
    r"No top.level module",
    r"No top level module",
    r"Include file .* not found",
    r"import .*::\*",
)


def _clean(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        item = _clean(item)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _short(raw: str, limit: int = 5000) -> str:
    raw = raw or ""
    return raw if len(raw) <= limit else raw[:limit].rstrip() + "\n...[truncated]..."


def _run(cmd: Sequence[str], cwd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(cmd),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "", exc.stderr or "timeout")
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def _filter_iverilog_noise(raw_output: str) -> str:
    lines = []
    for line in (raw_output or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if re.search(r"No top.level module", s, re.IGNORECASE):
            continue
        if re.search(r"Module\s+'[^']+'\s+not found", s, re.IGNORECASE):
            continue
        if s.lower().startswith("warning:") or "sorry: constant selects" in s.lower():
            continue
        if s.startswith("Unsupported:"):
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def _filter_verilator_noise(raw_output: str) -> str:
    lines = []
    for line in (raw_output or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%Warning") or "Exiting due to" in s:
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def _is_context_limited_output(output: str) -> bool:
    text = output or ""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in CONTEXT_LIMITED_PATTERNS)


def _is_hard_error_line(line: str) -> bool:
    lowered = (line or "").lower()
    if not lowered or "warning" in lowered or "note:" in lowered:
        return False
    return "%error" in lowered or "syntax error" in lowered or bool(re.search(r"\berror\b", lowered))


def _tool_diagnostic_signatures(output: str, candidate_path: str) -> List[str]:
    normalized_path = str(Path(candidate_path).resolve()).replace("\\", "/")
    signatures: List[str] = []
    for raw in (output or "").replace("\\", "/").splitlines():
        line = raw.strip().replace(normalized_path, "<edited-file>")
        if not _is_hard_error_line(line) and not _is_context_limited_output(line):
            continue
        line = re.sub(r":\d+(?::\d+)?", ":<line>", line)
        line = re.sub(r"\s+", " ", line).strip().lower()
        if line and line not in signatures:
            signatures.append(line)
    return signatures


def _new_tool_diagnostics(
    candidate_output: str,
    baseline_output: str,
    *,
    candidate_path: str,
) -> List[str]:
    candidate = _tool_diagnostic_signatures(candidate_output, candidate_path)
    baseline = set(_tool_diagnostic_signatures(baseline_output, candidate_path))
    return [item for item in candidate if item not in baseline]


def _file_matches(file_path: str, candidates: Sequence[str]) -> bool:
    fp = _clean(file_path).lower()
    if not fp:
        return False
    for candidate in candidates:
        cand = _clean(candidate).lower()
        if fp == cand or fp.endswith("/" + cand) or cand.endswith("/" + fp):
            return True
    return False


def _line_in_changed_spans(file_path: str, line_no: int | None, changed_spans: Sequence[Dict[str, Any]], padding: int = 8) -> bool:
    if line_no is None:
        return False
    for span in changed_spans or []:
        span_file = _clean(str(span.get("file") or span.get("file_path") or ""))
        if not _file_matches(file_path, [span_file]):
            continue
        try:
            start_line = int(span.get("start_line") or 0)
            end_line = int(span.get("end_line") or 0)
        except (TypeError, ValueError):
            continue
        if start_line and end_line and start_line - padding <= line_no <= end_line + padding:
            return True
    return False


class BuildContext:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.include_dirs: List[str] = []
        self.defines: List[str] = []
        self.files: List[str] = []
        self.sources: List[str] = []
        self._discover()

    def _discover(self) -> None:
        self._parse_f_files()
        self._parse_core_files()
        self._parse_bender_yml()
        self._parse_compile_yml()
        self.files = _dedupe(self.files)
        self.include_dirs = _dedupe(self.include_dirs)
        self.defines = _dedupe(self.defines)

    def _add_file(self, value: str, base_dir: str) -> None:
        value = value.strip().strip("'\"")
        if not value or value.startswith(("+", "-", "#", "//")):
            return
        if not value.endswith(VERILOG_EXTS):
            return
        abs_path = os.path.abspath(os.path.join(base_dir, value))
        if os.path.exists(abs_path):
            self.files.append(os.path.relpath(abs_path, self.repo_path).replace("\\", "/"))

    def _parse_f_files(self) -> None:
        for path in Path(self.repo_path).rglob("*.f"):
            if _skip_path(path):
                continue
            self.sources.append(os.path.relpath(path, self.repo_path).replace("\\", "/"))
            base_dir = str(path.parent)
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.split("//", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("+incdir+"):
                    inc = line[len("+incdir+") :]
                    abs_inc = os.path.abspath(os.path.join(base_dir, inc))
                    if os.path.isdir(abs_inc):
                        self.include_dirs.append(os.path.relpath(abs_inc, self.repo_path).replace("\\", "/"))
                    continue
                if line.startswith("+define+"):
                    self.defines.append(line[len("+define+") :])
                    continue
                self._add_file(line, base_dir)

    def _parse_core_files(self) -> None:
        for path in Path(self.repo_path).rglob("*.core"):
            if _skip_path(path):
                continue
            self.sources.append(os.path.relpath(path, self.repo_path).replace("\\", "/"))
            base_dir = str(path.parent)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in re.finditer(r"^\s*-\s+([^:\n#]+?\.(?:sv|svh|v|vh))\b", text, flags=re.MULTILINE):
                self._add_file(match.group(1), base_dir)
            for match in re.finditer(r"^\s*-\s+([A-Za-z0-9_./$-]+)\s*:\s*\{\s*is_include_file", text, flags=re.MULTILINE):
                self._add_file(match.group(1), base_dir)

    def _parse_bender_yml(self) -> None:
        path = os.path.join(self.repo_path, "Bender.yml")
        if not os.path.exists(path):
            return
        self.sources.append("Bender.yml")
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for match in re.finditer(r"^\s*-\s+([^:\n#]+?\.(?:sv|svh|v|vh))\b", text, flags=re.MULTILINE):
            self._add_file(match.group(1), self.repo_path)
        for match in re.finditer(r"^\s*-\s+([A-Za-z0-9_./$-]+)\s*$", text, flags=re.MULTILINE):
            token = match.group(1)
            if token.endswith(VERILOG_EXTS):
                self._add_file(token, self.repo_path)
            elif os.path.isdir(os.path.join(self.repo_path, token)):
                self.include_dirs.append(token)

    def _parse_compile_yml(self) -> None:
        for path in Path(self.repo_path).rglob("compile.yml"):
            if _skip_path(path):
                continue
            self.sources.append(os.path.relpath(path, self.repo_path).replace("\\", "/"))
            base_dir = str(path.parent)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in re.finditer(r"(?P<path>[A-Za-z0-9_./$-]+\.(?:sv|svh|v|vh))", text):
                self._add_file(match.group("path"), base_dir)

    def command_files_for(self, edited_files: Sequence[str], max_files: int = 80) -> List[str]:
        selected: List[str] = []
        for edited in edited_files:
            edited = _clean(edited)
            if edited in self.files:
                idx = self.files.index(edited)
                selected.extend(self.files[max(0, idx - 20) : idx + 20])
            selected.append(edited)
        return _dedupe(selected)[:max_files]


def _skip_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & {".git", "build", "out", "bazel-out", "node_modules", "__pycache__"})


class VerilogValidationManager:
    def __init__(
        self,
        repo_path: str,
        related_entities: Dict[str, Any] | None = None,
        issue_text: str = "",
        artifact_dir: str = "",
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.related_entities = related_entities or {}
        self.issue_text = issue_text or ""
        self.artifact_dir = os.path.abspath(artifact_dir) if artifact_dir else ""
        self.build_context = BuildContext(self.repo_path)

    def validate(
        self,
        *,
        patch_text: str,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> ValidationReport:
        # Patch policy is enforced once, before this manager is entered.  The
        # required static stage has one narrow responsibility: run Icarus and
        # Verilator on each edited RTL file.  Issue-level behavior belongs to
        # the frozen MiniTB, and repository context belongs to the lightweight
        # repository regression stage.
        evidence = self._syntax_evidence(patch_delta.edited_files, patch_delta)
        syntax_blockers = [item for item in evidence if item.severity == "blocker"]
        return build_validation_report(
            evidence,
            metadata={
                "validation_layers": ["iverilog", "verilator"],
                "edited_rtl_files": [
                    path
                    for path in patch_delta.edited_files
                    if str(path).endswith(VERILOG_EXTS)
                ],
                "syntax_blockers": len(syntax_blockers),
                "validation_strength": (
                    "modified_file_syntax_lint_failed"
                    if syntax_blockers
                    else "modified_file_syntax_lint_passed"
                ),
            },
        )

    def _delta_gate_evidence(self, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        result: List[ValidationEvidence] = []
        for item in patch_delta.policy_violations:
            result.append(ValidationEvidence(
                type=str(item.get("type") or "PATCH_DELTA"),
                severity="blocker" if item.get("severity") == "blocker" else "warning",
                file=str(item.get("file") or ""),
                message=str(item.get("message") or ""),
                validator="patch_delta",
            ))
        return result

    def _syntax_evidence(self, edited_files: Sequence[str], patch_delta: PatchDelta) -> List[ValidationEvidence]:
        evidence: List[ValidationEvidence] = []
        for rel in edited_files:
            if not rel.endswith(VERILOG_EXTS):
                continue
            abs_path = os.path.join(self.repo_path, rel)
            if not os.path.exists(abs_path):
                evidence.append(ValidationEvidence(
                    type="FILE_MISSING",
                    severity="blocker",
                    file=rel,
                    message=f"Edited file does not exist after patch: {rel}",
                    validator="syntax",
                ))
                continue
            evidence.extend(self._run_iverilog_single_file(rel, abs_path, patch_delta))
            evidence.extend(self._run_verilator_single_file(rel, abs_path, patch_delta))
        return evidence

    def _run_iverilog_single_file(self, rel: str, abs_path: str, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        if not shutil.which("iverilog"):
            return [ValidationEvidence(
                type="TOOL_UNAVAILABLE",
                severity="warning",
                file=rel,
                message="iverilog not found; skipped single-file syntax check.",
                validator="iverilog",
            )]
        cmd = ["iverilog", "-g2012", "-t", "null", "-o", os.devnull]
        for inc in self.build_context.include_dirs[:20]:
            cmd.append(f"-I{os.path.join(self.repo_path, inc)}")
        cmd.append(abs_path)
        result = _run(cmd, cwd=self.repo_path, timeout=20)
        output = _filter_iverilog_noise((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode != 0:
            baseline_returncode, baseline_output = self._head_tool_output(
                rel,
                abs_path,
                cmd,
                timeout=20,
                output_filter=_filter_iverilog_noise,
            )
            new_diagnostics = _new_tool_diagnostics(
                output,
                baseline_output,
                candidate_path=abs_path,
            )
            if (
                baseline_returncode not in (None, 0)
                and baseline_returncode == result.returncode
                and not new_diagnostics
            ):
                return [ValidationEvidence(
                    type="IVERILOG_PREEXISTING_DIAGNOSTIC",
                    severity="info",
                    file=rel,
                    message="Single-file Icarus diagnostics are unchanged from git HEAD.",
                    raw_log=_short(output),
                    validator="iverilog",
                )]
            output = "\n".join(new_diagnostics) or output or "iverilog exited with an error"
            ev = self._parse_tool_output(output, rel, "IVERILOG_ERROR", "iverilog")
            if _is_context_limited_output(output) or self._source_line_context_limited(abs_path, output):
                relevant = self._first_relevant_tool_diagnostic(output, patch_delta)
                if relevant:
                    ev.severity = "blocker"
                    ev.file = relevant.get("file") or ev.file
                    ev.line = relevant.get("line") or ev.line
                    ev.message = relevant.get("message") or ev.message
                else:
                    ev.type = "IVERILOG_CONTEXT_INCOMPLETE"
                    ev.severity = "warning"
                    ev.message = f"Single-file iverilog lacks repo/package context for {rel}: {ev.message}"
            return [ev]
        return []

    def _run_verilator_single_file(self, rel: str, abs_path: str, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        if not shutil.which("verilator"):
            return [ValidationEvidence(
                type="TOOL_UNAVAILABLE",
                severity="warning",
                file=rel,
                message="verilator not found; skipped single-file lint.",
                validator="verilator",
            )]
        cmd = ["verilator", "--lint-only", "-sv", "-Wno-fatal", abs_path]
        for inc in self.build_context.include_dirs[:20]:
            cmd.insert(-1, f"-I{os.path.join(self.repo_path, inc)}")
        result = _run(cmd, cwd=self.repo_path, timeout=30)
        output = _filter_verilator_noise((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode != 0:
            baseline_returncode, baseline_output = self._head_tool_output(
                rel,
                abs_path,
                cmd,
                timeout=30,
                output_filter=_filter_verilator_noise,
            )
            new_diagnostics = _new_tool_diagnostics(
                output,
                baseline_output,
                candidate_path=abs_path,
            )
            if (
                baseline_returncode not in (None, 0)
                and baseline_returncode == result.returncode
                and not new_diagnostics
            ):
                return [ValidationEvidence(
                    type="VERILATOR_PREEXISTING_DIAGNOSTIC",
                    severity="info",
                    file=rel,
                    message="Single-file Verilator diagnostics are unchanged from git HEAD.",
                    raw_log=_short(output),
                    validator="verilator",
                )]
            output = "\n".join(new_diagnostics) or output or "verilator exited with an error"
            ev = self._parse_tool_output(output, rel, "VERILATOR_ERROR", "verilator")
            if _is_context_limited_output(output):
                relevant = self._first_relevant_tool_diagnostic(output, patch_delta)
                if relevant:
                    ev.severity = "blocker"
                    ev.file = relevant.get("file") or ev.file
                    ev.line = relevant.get("line") or ev.line
                    ev.message = relevant.get("message") or ev.message
                else:
                    ev.type = "VERILATOR_CONTEXT_INCOMPLETE"
                    ev.severity = "warning"
                    ev.message = f"Single-file Verilator lacks repo/package context for {rel}: {ev.message}"
            return [ev]
        return []

    def _head_tool_output(
        self,
        rel: str,
        candidate_path: str,
        command: Sequence[str],
        *,
        timeout: int,
        output_filter,
    ) -> tuple[int | None, str]:
        """Run the same single-file command on the immutable git-HEAD source."""

        original = subprocess.run(
            ["git", "-C", self.repo_path, "show", f"HEAD:{_clean(rel)}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if original.returncode != 0 or not original.stdout:
            return None, ""
        suffix = Path(candidate_path).suffix or ".sv"
        temporary = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=suffix,
                prefix=".kgveri_head_",
                dir=str(Path(candidate_path).parent),
                delete=False,
            ) as handle:
                handle.write(original.stdout)
                temporary = handle.name
            baseline_command = [
                temporary if item == candidate_path else item
                for item in command
            ]
            baseline = _run(baseline_command, cwd=self.repo_path, timeout=timeout)
            baseline_output = output_filter(
                (baseline.stdout or "") + "\n" + (baseline.stderr or "")
            )
            return baseline.returncode, baseline_output.replace(temporary, candidate_path)
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _repo_compile_evidence(self, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        edited_files = patch_delta.edited_files
        selected = self.build_context.command_files_for(edited_files)
        if not selected:
            return [ValidationEvidence(
                type="REPO_COMPILE_INCOMPLETE",
                severity="warning",
                message="No repo filelist/core/Bender/compile.yml context found for edited files.",
                validator="repo_compile",
            )]
        if not shutil.which("verilator"):
            return []
        files = [os.path.join(self.repo_path, f) for f in selected if os.path.exists(os.path.join(self.repo_path, f))]
        if not files:
            return []
        cmd = ["verilator", "--lint-only", "-sv", "-Wno-fatal"]
        for inc in self.build_context.include_dirs[:30]:
            cmd.append(f"-I{os.path.join(self.repo_path, inc)}")
        cmd.extend(files[:80])
        result = _run(cmd, cwd=self.repo_path, timeout=60)
        output = _filter_verilator_noise((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode != 0 and output:
            ev = self._parse_tool_output(output, "", "REPO_VERILATOR_ERROR", "repo_compile")
            relevant = self._first_relevant_repo_diagnostic(output, patch_delta)
            if relevant:
                ev.severity = "blocker"
                ev.file = relevant.get("file") or ev.file
                ev.line = relevant.get("line") or ev.line
                ev.message = relevant.get("message") or ev.message
            else:
                ev.type = "REPO_VERILATOR_CONTEXT_OR_UNRELATED"
                ev.severity = "warning"
                ev.message = (
                    "Repo-aware Verilator emitted only unrelated, pre-existing, or context-limited "
                    f"diagnostics for edited files {', '.join(edited_files[:4])}: {ev.message}"
                )
            return [ev]
        return []

    def _changed_span_structural_evidence(self, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        evidence: List[ValidationEvidence] = []
        added_by_file = self._added_lines_by_file(patch_delta.raw_diff or "")
        for rel, additions in added_by_file.items():
            abs_path = os.path.join(self.repo_path, rel)
            if not os.path.exists(abs_path):
                continue
            try:
                lines = Path(abs_path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, text in additions:
                if not self._looks_like_declaration(text):
                    continue
                reason = self._declaration_after_statement(lines, line_no)
                if not reason:
                    continue
                evidence.append(ValidationEvidence(
                    type="DECLARATION_AFTER_STATEMENT_IN_CHANGED_SPAN",
                    severity="blocker",
                    file=rel,
                    line=line_no,
                    message=(
                        "Changed span appears to insert a declaration after executable/procedural "
                        f"statements near line {line_no}: {text.strip()} ({reason})."
                    ),
                    raw_log="\n".join(lines[max(0, line_no - 8): min(len(lines), line_no + 6)]),
                    validator="static_verilog_structure",
                ))
        return evidence

    def _module_interface_evidence(self, patch_delta: PatchDelta) -> List[ValidationEvidence]:
        evidence: List[ValidationEvidence] = []
        edited = set(_clean(path) for path in patch_delta.edited_files or [])
        for rel in patch_delta.edited_files or []:
            rel = _clean(rel)
            if not rel.endswith(VERILOG_EXTS):
                continue
            abs_path = os.path.join(self.repo_path, rel)
            if not os.path.exists(abs_path):
                continue
            original = self._git_show_file(rel)
            if not original:
                continue
            try:
                patched = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            before = self._module_ports(original)
            after = self._module_ports(patched)
            for module_name in sorted(set(before) & set(after)):
                before_ports = before.get(module_name, set())
                after_ports = after.get(module_name, set())
                added = sorted(after_ports - before_ports)
                removed = sorted(before_ports - after_ports)
                if not added and not removed:
                    continue
                external_instances = [
                    path for path in self._find_module_instance_files(module_name)
                    if _clean(path) not in edited
                ]
                if external_instances:
                    evidence.append(ValidationEvidence(
                        type="MODULE_INTERFACE_CHANGED_WITH_UNUPDATED_INSTANCES",
                        severity="blocker",
                        file=rel,
                        module=module_name,
                        message=(
                            f"Patch changes public module interface for {module_name}: "
                            f"added ports={added or []}, removed ports={removed or []}. "
                            f"Unedited instance files include {', '.join(external_instances[:6])}."
                        ),
                        raw_log="\n".join(external_instances[:20]),
                        validator="static_verilog_interface",
                    ))
                else:
                    evidence.append(ValidationEvidence(
                        type="MODULE_INTERFACE_CHANGED",
                        severity="warning",
                        file=rel,
                        module=module_name,
                        message=(
                            f"Patch changes module interface for {module_name}: "
                            f"added ports={added or []}, removed ports={removed or []}."
                        ),
                        validator="static_verilog_interface",
                    ))
        return evidence

    def _git_show_file(self, rel: str) -> str:
        result = _run(["git", "show", f"HEAD:{rel}"], cwd=self.repo_path, timeout=10)
        return result.stdout if result.returncode == 0 else ""

    def _module_ports(self, text: str) -> Dict[str, set[str]]:
        text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
        text = re.sub(r"//.*", " ", text)
        ports: Dict[str, set[str]] = {}
        for match in re.finditer(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text):
            name = match.group(1)
            open_idx = text.find("(", match.end())
            if open_idx < 0:
                continue
            close_idx = self._find_balanced_module_header_end(text, open_idx)
            if close_idx < 0:
                continue
            header = text[open_idx + 1:close_idx]
            ports[name] = self._port_names_from_header(header)
        return ports

    def _find_balanced_module_header_end(self, text: str, open_idx: int) -> int:
        depth = 0
        for idx in range(open_idx, min(len(text), open_idx + 20000)):
            ch = text[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    semi_idx = idx + 1
                    while semi_idx < len(text) and text[semi_idx].isspace():
                        semi_idx += 1
                    if semi_idx < len(text) and text[semi_idx] == ";":
                        return idx
        return -1

    def _port_names_from_header(self, header: str) -> set[str]:
        names: set[str] = set()
        keywords = {
            "input", "output", "inout", "wire", "reg", "logic", "signed", "unsigned",
            "bit", "int", "integer", "parameter", "localparam",
        }
        for part in header.split(","):
            clean = re.sub(r"\[[^\]]+\]", " ", part)
            tokens = [
                token for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", clean)
                if token not in keywords
            ]
            if tokens:
                names.add(tokens[-1])
        return names

    def _find_module_instance_files(self, module_name: str) -> List[str]:
        pattern = re.compile(rf"\b{re.escape(module_name)}\s*(?:#\s*\([^;]*?\)\s*)?[A-Za-z_][A-Za-z0-9_$]*\s*\(", re.DOTALL)
        matches: List[str] = []
        for root, dirs, files in os.walk(self.repo_path):
            rel_root = os.path.relpath(root, self.repo_path).replace("\\", "/").lower()
            dirs[:] = [
                d for d in dirs
                if not re.search(
                    r"(^|/)(\.git|tb|test|tests|dv|sim|verif|formal|autogen|generated|out|build|bazel-out)(/|$)",
                    f"{rel_root}/{d}".lower(),
                )
            ]
            for filename in files:
                if not filename.endswith(VERILOG_EXTS):
                    continue
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, self.repo_path).replace("\\", "/")
                try:
                    content = Path(path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if pattern.search(content):
                    matches.append(rel)
        return _dedupe(matches)[:40]

    def _added_lines_by_file(self, raw_diff: str) -> Dict[str, List[tuple[int, str]]]:
        result: Dict[str, List[tuple[int, str]]] = {}
        current_file = ""
        new_line = 0
        for raw in (raw_diff or "").splitlines():
            if raw.startswith("diff --git "):
                current_file = ""
                continue
            if raw.startswith("+++ b/"):
                current_file = _clean(raw[len("+++ b/"):])
                result.setdefault(current_file, [])
                continue
            if raw.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,(\d+))?", raw)
                new_line = int(match.group(1)) if match else 0
                continue
            if not current_file or not new_line:
                continue
            if raw.startswith("+") and not raw.startswith("+++"):
                result.setdefault(current_file, []).append((new_line, raw[1:]))
                new_line += 1
            elif raw.startswith("-") and not raw.startswith("---"):
                continue
            else:
                new_line += 1
        return result

    def _looks_like_declaration(self, line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped or stripped.startswith(("//", "/*", "*")):
            return False
        if re.match(
            r"(?:localparam|parameter|typedef|import|logic|wire|reg|bit|byte|int|integer|genvar|"
            r"[A-Za-z_][A-Za-z0-9_$]*::[A-Za-z_][A-Za-z0-9_$]*)\b",
            stripped,
        ):
            return True
        if re.match(
            r"(?!assign\b|if\b|else\b|case\b|for\b|while\b|return\b|begin\b|end\b)"
            r"[A-Za-z_][A-Za-z0-9_$]*(?:_t)?\s+"
            r"(?:\[[^\]]+\]\s*)?[A-Za-z_][A-Za-z0-9_$]*(?:\s*(?:\[|,|;|=))",
            stripped,
        ):
            return True
        return False

    def _declaration_after_statement(self, lines: Sequence[str], line_no: int) -> str:
        if line_no <= 1 or line_no > len(lines):
            return ""
        current_indent = len(lines[line_no - 1]) - len(lines[line_no - 1].lstrip())
        seen_executable = ""
        for idx in range(line_no - 2, max(-1, line_no - 35), -1):
            raw = lines[idx]
            stripped = raw.strip()
            if not stripped or stripped.startswith(("//", "/*", "*")):
                continue
            if re.match(r"(?:module|interface|package|program|class|function|task)\b", stripped):
                return ""
            if re.match(r"(?:end|endcase|endfunction|endtask|endmodule|generate|endgenerate)\b", stripped):
                return ""
            if re.match(r"\}\s*[A-Za-z_][A-Za-z0-9_$]*\s*;", stripped):
                continue
            if re.match(r"\)\s*;", stripped):
                continue
            if re.match(r"(?:begin|case|casex|casez|if|else|for|foreach|while|repeat|unique|priority)\b", stripped):
                return seen_executable
            if self._looks_like_declaration(stripped):
                continue
            prev_indent = len(raw) - len(raw.lstrip())
            if prev_indent <= current_indent + 4 and (
                ";" in stripped
                or re.search(r"\b(?:if|case|for|while|return|break|continue)\b", stripped)
            ):
                seen_executable = stripped[:160]
                return seen_executable
        return ""

    def _parse_tool_output(self, output: str, fallback_file: str, ev_type: str, validator: str) -> ValidationEvidence:
        first = next((line for line in output.splitlines() if line.strip()), output.strip())
        file_path = fallback_file
        line_no = None
        match = re.search(r"(/?[A-Za-z0-9_./$-]+\.(?:sv|svh|v|vh)):(\d+)", output)
        if match:
            file_path = self._repo_relative_tool_path(match.group(1))
            try:
                line_no = int(match.group(2))
            except ValueError:
                line_no = None
        signal = ""
        sig_match = re.search(r"'([A-Za-z_][A-Za-z0-9_$]*)'", output)
        if sig_match:
            signal = sig_match.group(1)
        return ValidationEvidence(
            type=ev_type,
            severity="blocker",
            file=file_path,
            line=line_no,
            signal=signal,
            message=first[:500],
            raw_log=_short(output),
            validator=validator,
        )

    def _repo_relative_tool_path(self, path: str) -> str:
        text = str(path or "").replace("\\", "/").strip()
        if not text:
            return ""
        candidates = [text]
        if not text.startswith("/") and os.path.exists("/" + text):
            candidates.append("/" + text)
        repo_root = self.repo_path.replace("\\", "/").rstrip("/")
        for candidate in candidates:
            normalized = candidate.replace("\\", "/")
            if os.path.isabs(normalized):
                abs_candidate = os.path.abspath(normalized)
                if abs_candidate.startswith(self.repo_path + os.sep):
                    return os.path.relpath(abs_candidate, self.repo_path).replace("\\", "/")
            if repo_root and repo_root in normalized:
                return normalized.split(repo_root, 1)[1].lstrip("/")
        cleaned = _clean(text)
        if os.path.exists(os.path.join(self.repo_path, cleaned)):
            return cleaned
        for marker in ("/rtl/", "/src/", "/hw/", "/include/", "/vendor/"):
            idx = ("/" + cleaned).find(marker)
            if idx >= 0:
                return ("/" + cleaned)[idx + 1 :]
        return cleaned

    def _diagnostic_locations(self, output: str) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for line in (output or "").splitlines():
            for match in re.finditer(r"(/?[A-Za-z0-9_./$-]+\.(?:sv|svh|v|vh)):(\d+)", line):
                try:
                    line_no = int(match.group(2))
                except ValueError:
                    line_no = None
                diagnostics.append({
                    "file": self._repo_relative_tool_path(match.group(1)),
                    "line": line_no,
                    "message": line.strip()[:500],
                    "hard": _is_hard_error_line(line),
                })
        return diagnostics

    def _source_line_context_limited(self, abs_path: str, output: str) -> bool:
        try:
            lines = Path(abs_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return False
        for diag in self._diagnostic_locations(output):
            line_no = diag.get("line")
            if not isinstance(line_no, int) or line_no <= 0 or line_no > len(lines):
                continue
            source_line = lines[line_no - 1]
            if re.search(r"\bimport\s+[A-Za-z_][A-Za-z0-9_$]*::\*", source_line):
                return True
            if re.search(r"\b(?:package|typedef|interface)\b", source_line):
                return True
        return False

    def _first_relevant_tool_diagnostic(self, output: str, patch_delta: PatchDelta) -> Dict[str, Any] | None:
        for diag in self._diagnostic_locations(output):
            if not diag.get("hard"):
                continue
            file_path = str(diag.get("file") or "")
            if not _file_matches(file_path, patch_delta.edited_files):
                continue
            line_no = diag.get("line")
            if patch_delta.changed_spans and (
                line_no is None or not _line_in_changed_spans(file_path, line_no, patch_delta.changed_spans)
            ):
                continue
            return diag
        return None

    def _first_relevant_repo_diagnostic(self, output: str, patch_delta: PatchDelta) -> Dict[str, Any] | None:
        edited_files = patch_delta.edited_files
        for diag in self._diagnostic_locations(output):
            if not diag.get("hard"):
                continue
            file_path = str(diag.get("file") or "")
            if not _file_matches(file_path, edited_files):
                continue
            line_no = diag.get("line")
            if patch_delta.changed_spans:
                if line_no is None or not _line_in_changed_spans(file_path, line_no, patch_delta.changed_spans):
                    continue
            return diag
        return None
