from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, List, Sequence

try:
    from repair_planner import is_forbidden_file
    from repair_types import PatchDelta
except Exception:  # pragma: no cover
    from .repair_planner import is_forbidden_file
    from .repair_types import PatchDelta

try:
    from verilog_ast import VerilogAstExtractor, VerilogAstUnavailable
except Exception:  # pragma: no cover
    try:
        from .verilog_ast import VerilogAstExtractor, VerilogAstUnavailable
    except Exception:  # pragma: no cover
        VerilogAstExtractor = None
        VerilogAstUnavailable = RuntimeError


VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "assign", "always", "always_ff", "always_comb", "always_latch", "begin",
    "end", "if", "else", "case", "endcase", "for", "while", "generate",
    "endgenerate", "function", "endfunction", "task", "endtask", "parameter",
    "localparam", "typedef", "struct", "enum", "import", "package", "endpackage",
    "interface", "endinterface", "posedge", "negedge", "or", "and", "not",
    "signed", "unsigned", "bit", "byte", "int", "integer", "time", "return",
}
SYMBOL_NOISE_WORDS = {
    "hint", "keep", "requires", "valid", "decode", "decoder", "compressed",
    "instruction", "instructions", "legal", "illegal", "above", "below",
    "repair", "patch", "change", "changed", "remove", "removed", "added",
    "should", "must", "because", "preserve", "without", "with", "from",
    "else", "then", "when", "line", "lines", "block",
}


def _strip_verilog_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    return re.sub(r"//.*", " ", text)


def _is_symbol_token(token: str) -> bool:
    lowered = token.lower()
    if len(token) <= 2 or lowered in VERILOG_KEYWORDS or lowered in SYMBOL_NOISE_WORDS:
        return False
    if token.upper() == token and "_" not in token:
        return False
    if re.fullmatch(r"[bhodx]?[0-9a-fA-F_xzXZ]+", token):
        return False
    return True


def _clean(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = _clean(item)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_changed_spans(raw_diff: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    current_file = ""
    for line in raw_diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = _clean(line[len("+++ b/") :])
            continue
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if match and current_file:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            spans.append({
                "file": current_file,
                "start_line": start,
                "end_line": max(start, start + count - 1),
            })
    return spans


def _extract_patch_body_symbols(patch_text: str) -> List[str]:
    bodies = re.findall(r"\[REPLACE\](.*?)\[/REPLACE\]", patch_text or "", flags=re.DOTALL)
    tokens: List[str] = []
    for body in bodies:
        body = _strip_verilog_comments(body)
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", body):
            if not _is_symbol_token(token):
                continue
            tokens.append(token)
    return _dedupe(tokens)[:80]


def _extract_modules_from_files(repo_path: str, edited_files: Sequence[str]) -> List[str]:
    modules: List[str] = []
    for rel in edited_files:
        abs_path = os.path.join(repo_path, rel)
        if not os.path.exists(abs_path):
            continue
        ast_classes = _extract_ast_classes(abs_path)
        if ast_classes:
            for class_info in ast_classes:
                modules.append(str(class_info.get("name") or ""))
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        for match in re.finditer(r"\b(?:module|interface|package)\s+([A-Za-z_][A-Za-z0-9_$]*)", text):
            modules.append(match.group(1))
    return _dedupe(modules)[:40]


_AST_EXTRACTOR = None
_AST_DISABLED = False


def _extract_ast_classes(abs_path: str) -> List[Dict[str, Any]]:
    global _AST_EXTRACTOR, _AST_DISABLED
    if _AST_DISABLED or VerilogAstExtractor is None:
        return []
    try:
        if _AST_EXTRACTOR is None:
            _AST_EXTRACTOR = VerilogAstExtractor()
        return _AST_EXTRACTOR.extract_classes(abs_path) or []
    except (VerilogAstUnavailable, OSError, UnicodeError):
        _AST_DISABLED = True
    except Exception:
        return []
    return []


def _span_overlaps(start_a: int | None, end_a: int | None, start_b: int | None, end_b: int | None) -> bool:
    if not start_a or not end_a or not start_b or not end_b:
        return False
    return int(start_a) <= int(end_b) and int(start_b) <= int(end_a)


def _extract_changed_span_symbols(repo_path: str, changed_spans: Sequence[Dict[str, Any]]) -> List[str]:
    symbols: List[str] = []
    spans_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for span in changed_spans or []:
        rel = _clean(str(span.get("file") or ""))
        if rel:
            spans_by_file.setdefault(rel, []).append(span)

    for rel, spans in spans_by_file.items():
        abs_path = os.path.join(repo_path, rel)
        if not os.path.exists(abs_path):
            continue
        ast_classes = _extract_ast_classes(abs_path)
        if ast_classes:
            for class_info in ast_classes:
                class_start = class_info.get("start_line")
                class_end = class_info.get("end_line")
                for span in spans:
                    if _span_overlaps(class_start, class_end, span.get("start_line"), span.get("end_line")):
                        symbols.append(str(class_info.get("name") or ""))
                        for entity in class_info.get("rtl_entities") or []:
                            if _span_overlaps(entity.get("start_line"), entity.get("end_line"), span.get("start_line"), span.get("end_line")):
                                symbols.append(str(entity.get("signal_name") or entity.get("name") or ""))
                        for method in class_info.get("methods") or []:
                            if _span_overlaps(method.get("start_line"), method.get("end_line"), span.get("start_line"), span.get("end_line")):
                                symbols.append(str(method.get("target_signal") or method.get("name") or ""))
            continue
        try:
            lines = open(abs_path, "r", encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for span in spans:
            start = max(1, int(span.get("start_line") or 1) - 4)
            end = min(len(lines), int(span.get("end_line") or start) + 4)
            excerpt = _strip_verilog_comments("\n".join(lines[start - 1:end]))
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", excerpt):
                if _is_symbol_token(token):
                    symbols.append(token)
    return _dedupe(symbols)[:80]


def _allowed_file_match(file_path: str, allowed_files: Sequence[str]) -> bool:
    if not allowed_files:
        return True
    fp = _clean(file_path).lower()
    for allowed in allowed_files:
        cand = _clean(allowed).lower()
        if fp == cand or fp.endswith("/" + cand) or cand.endswith("/" + fp) or fp in cand or cand in fp:
            return True
    return False


def build_patch_delta(
    *,
    repo_path: str,
    patch_text: str,
    edited_files: Sequence[str],
    repair_plan: Any,
) -> PatchDelta:
    diff_result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--no-color"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw_diff = diff_result.stdout or ""
    files = _dedupe(edited_files)
    if not files:
        files = _dedupe(
            line[len("+++ b/") :]
            for line in raw_diff.splitlines()
            if line.startswith("+++ b/") and not line.startswith("+++ /dev/null")
        )

    changed_spans = _extract_changed_spans(raw_diff)
    allowed_files = getattr(repair_plan, "allowed_files", []) or []
    violations: List[Dict[str, Any]] = []

    if "[REPLACE]" in patch_text and "[/REPLACE]" not in patch_text:
        violations.append({
            "severity": "blocker",
            "type": "REPLACE_FORMAT",
            "message": "Patch contains [REPLACE] without matching [/REPLACE].",
        })

    for file_path in files:
        if is_forbidden_file(file_path):
            violations.append({
                "severity": "blocker",
                "type": "FORBIDDEN_FILE_EDIT",
                "file": file_path,
                "message": f"Patch edits forbidden validation/test/oracle path: {file_path}",
            })
        elif not _allowed_file_match(file_path, allowed_files):
            violations.append({
                "severity": "warning",
                "type": "OUTSIDE_KG_EDIT_CONE",
                "file": file_path,
                "message": f"Edited file is outside the KG repair plan allow-list: {file_path}",
            })

    if not raw_diff.strip():
        violations.append({
            "severity": "blocker",
            "type": "EMPTY_DIFF",
            "message": "Patch application produced no repository diff.",
        })

    return PatchDelta(
        edited_files=files,
        changed_spans=changed_spans,
        touched_symbols=_dedupe(_extract_patch_body_symbols(patch_text) + _extract_changed_span_symbols(repo_path, changed_spans)),
        touched_modules=_extract_modules_from_files(repo_path, files),
        raw_diff=raw_diff,
        policy_violations=violations,
    )
