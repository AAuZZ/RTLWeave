"""
KG signal-flow integrity check for internal verification.

Gate placed after iverilog compile-check and before patch output.
Verifies that the patch's structural changes are consistent with
the knowledge graph's known entity relationships.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

try:
    from term import warn as twarn
except Exception:  # pragma: no cover - package import path
    from .term import warn as twarn


VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "assign", "always", "always_ff", "always_comb", "always_latch", "begin",
    "end", "if", "else", "case", "endcase", "default", "for", "while",
    "repeat", "integer", "genvar", "posedge", "negedge", "or", "and", "not",
    "parameter", "localparam", "typedef", "generate", "endgenerate",
    "function", "endfunction", "task", "endtask", "initial", "forever",
    "unsigned", "signed", "bit", "byte", "shortint", "int", "longint",
    "time", "realtime", "shortreal", "real", "string", "event", "struct",
    "enum", "package", "endpackage", "import", "inside", "unique", "priority",
}

COMMENT_WORDS = {
    "hint", "keep", "requires", "valid", "decode", "decoder", "compressed",
    "instruction", "instr", "legal", "illegal", "else", "then", "when",
    "this", "that", "true", "false", "above", "below", "fix", "repair",
    "patch", "change", "changed", "remove", "removed", "add", "added",
    "should", "must", "may", "because", "preserve", "maintain", "without",
    "with", "from", "into", "only", "case", "block", "line", "lines",
}

INSTRUCTION_MNEMONICS = {
    "ebreak", "ecall", "jal", "jalr", "lui", "addi", "add", "sub", "xor",
    "or", "and", "sll", "srl", "sra", "beq", "bne", "blt", "bge", "lw",
    "sw", "nop", "mv", "li",
}


def _strip_verilog_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    return re.sub(r"//.*", " ", text)


def _is_probably_signal_token(token: str) -> bool:
    lowered = token.lower()
    if len(token) <= 2 or lowered in VERILOG_KEYWORDS:
        return False
    if lowered in COMMENT_WORDS or lowered in INSTRUCTION_MNEMONICS:
        return False
    if token.startswith("$"):
        return False
    if re.fullmatch(r"[bhodx]?[0-9a-fA-F_xzXZ]+", token):
        return False
    if re.fullmatch(r"h[0-9a-fA-F_]+", lowered):
        return False
    if token.upper() == token and "_" not in token:
        return False
    return True


def _extract_patch_signals(patch_text: str) -> List[str]:
    """Extract Verilog identifier-like tokens from a REPLACE patch body.

    Looks at lines between [REPLACE]..[/REPLACE] for signal-like tokens:
    lower_case_with_underscores, CamelCase, and common prefix patterns.
    """
    # Collect all text inside [REPLACE] blocks
    bodies = re.findall(
        r"\[REPLACE\](.*?)\[/REPLACE\]",
        patch_text,
        re.DOTALL,
    )
    all_tokens: List[str] = []
    for body in bodies:
        body = _strip_verilog_comments(body)
        # Extract likely signal identifiers: snake_case, camelCase, some_QUALIFIED
        tokens = re.findall(
            r"\b[a-z_][a-z0-9_]*\b", body, re.IGNORECASE
        )
        for t in tokens:
            if not _is_probably_signal_token(t):
                continue
            all_tokens.append(t)

    # deduplicate preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for t in all_tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:30]


def _collect_known_signals(related_entities: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    """Build a set of known signal/port names from KG related_entities."""
    known: Set[str] = set()
    for group_name, entities in (related_entities or {}).items():
        if group_name in {"issues", "issue", "classes", "methods"}:
            continue
        for e in entities:
            if not isinstance(e, dict):
                continue
            if not any(e.get(key) for key in ("signal_name", "direction", "width", "declaration", "module_name")):
                continue
            for key in ("signal_name", "name"):
                val = str(e.get(key) or "")
                if val:
                    # Split qualified names: hmac_core.hmac_idle → hmac_idle
                    for part in re.split(r"[\./]", val):
                        part = part.strip()
                        if part and len(part) > 2:
                            known.add(part.lower())
    return known


def _signal_groups_by_file(related_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Map file path → list of signal names known to belong to that file."""
    mapping: Dict[str, List[str]] = {}
    for group_name, entities in (related_entities or {}).items():
        for e in entities:
            if not isinstance(e, dict):
                continue
            fp = str(e.get("file_path") or "").replace("\\", "/")
            if not fp:
                continue
            for key in ("signal_name", "name"):
                val = str(e.get(key) or "")
                if val and len(val) > 2:
                    mapping.setdefault(fp, []).append(val.lower())
    return mapping


def check_patch_signal_flow(
    patch_text: str,
    edited_files: Sequence[str],
    related_entities: Dict[str, List[Dict[str, Any]]],
    candidate_files: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run KG signal-flow integrity checks on a candidate patch.

    Returns a dict:
      {
        "passed": bool,
        "issues": [{"severity": "error"|"warn", "message": "..."}],
        "feedback_text": str  <-- formatted for the Failure Analysis Agent
      }
    """
    issues: List[Dict[str, str]] = []

    # Determine if any edited file is a package/header — these files define
    # localparam/struct fields, not signals, so signal-existence checks
    # don't apply (the KG's signal entities only cover module-level signals).
    is_package_or_header = any(
        str(f or "").endswith(("_pkg.sv", "_pkg.v", ".svh", ".vh"))
        for f in (edited_files or [])
    )

    # ── 1. Edited-file relevance ──────────────────────────────────────
    candidate_set = set(
        str(f or "").replace("\\", "/").lower()
        for f in (candidate_files or [])
    )
    for f in edited_files:
        f_norm = str(f or "").replace("\\", "/").lower()
        # Exclude testbenches
        if f_norm.startswith("tb/") or f_norm.startswith("test/"):
            issues.append({
                "severity": "error",
                "message": f"Patch modifies testbench file '{f}' which should not be changed.",
            })
            continue
        # Check if the file appears in KG candidates (lenient match)
        if candidate_set and not any(
            f_norm in c or c in f_norm for c in candidate_set
        ):
            issues.append({
                "severity": "warn",
                "message": f"Edited file '{f}' is not among KG-suggested candidate files.",
            })

    # ── 2. Signal existence in KG ─────────────────────────────────────
    # Skip for package/header files — their localparam/struct field names
    # are NOT indexed as signal entities in the KG.
    patch_signals = _extract_patch_signals(patch_text)
    known_signals = _collect_known_signals(related_entities)
    if not is_package_or_header:
        unknown: List[str] = []
        if not known_signals:
            issues.append({
                "severity": "info",
                "message": "KG signal catalog is empty for this localization; unknown-signal checks were skipped.",
            })
        else:
            for sig in patch_signals:
                if sig.lower() not in known_signals:
                    unknown.append(sig)
        if unknown:
            issues.append({
                "severity": "warn",
                "message": f"Signals not found in KG: {', '.join(unknown[:8])}.",
            })

    # ── 3. Multi-driver check (same signal assigned in >1 always block) ─
    # Skip for package/header files (no always blocks in package definitions).
    if not is_package_or_header:
        bodies_with_lines = re.findall(
            r"- start_line: (\d+).*?\[REPLACE\](.*?)\[/REPLACE\]",
            patch_text,
            re.DOTALL,
        )
        signal_occurrences: Dict[str, List[str]] = {}
        for line_str, body in bodies_with_lines:
            for sig in re.findall(r"\b([a-z_][a-z0-9_]*)\b", body, re.IGNORECASE):
                sig_lower = sig.lower()
                if len(sig_lower) > 2 and sig_lower in known_signals:
                    if line_str not in signal_occurrences.setdefault(sig_lower, []):
                        signal_occurrences[sig_lower].append(line_str)
        multi_driven = {
            sig: lines for sig, lines in signal_occurrences.items()
            if len(lines) > 2
        }
        for sig, lines in multi_driven.items():
            issues.append({
                "severity": "warn",
                "message": f"'{sig}' appears in {len(lines)} separate REPLACE blocks ({', '.join(lines[:5])}) — potential multi-driver.",
            })

    # ── 4. Reader coverage warning ────────────────────────────────────
    # Skip for package/header files.
    if not is_package_or_header:
        file_signals = _signal_groups_by_file(related_entities)
        edited_set = set(
            str(f or "").replace("\\", "/").lower()
            for f in (edited_files or [])
        )
        for f, sigs in file_signals.items():
            f_lower = f.lower()
            if any(f_lower in e or e in f_lower for e in edited_set):
                continue
            for sig in patch_signals:
                if sig.lower() in sigs:
                    issues.append({
                        "severity": "info",
                        "message": f"Signal '{sig}' referenced in patch but its "
                                   f"declaration/module '{f}' is NOT in the edited file set.",
                    })

    # ── Build result ──────────────────────────────────────────────────
    passed = not any(i["severity"] == "error" for i in issues)

    feedback_lines = [
        "=== KG Signal Flow Integrity Check ===",
    ]
    if not issues:
        feedback_lines.append("✓ All checks passed.")
    else:
        has_error = any(i["severity"] == "error" for i in issues)
        feedback_lines.append(
            f"{'✗ BLOCKING' if has_error else '⚠ WARNINGS'} — "
            f"{len([i for i in issues if i['severity'] == 'error'])} errors, "
            f"{len([i for i in issues if i['severity'] in ('warn', 'info')])} warnings."
        )
        for issue in issues:
            prefix = {
                "error": "✗",
                "warn": "⚠",
                "info": "ℹ",
            }.get(issue["severity"], "?")
            feedback_lines.append(f"  {prefix} {issue['message']}")

    return {
        "passed": passed,
        "issues": issues,
        "feedback_text": "\n".join(feedback_lines),
    }
