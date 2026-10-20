from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from kg_signal_check import check_patch_signal_flow
    from repair_types import FunctionalCheckSpec, PatchDelta, RepairPlan, ReproTestResult, ValidationEvidence
except Exception:  # pragma: no cover
    from .kg_signal_check import check_patch_signal_flow
    from .repair_types import FunctionalCheckSpec, PatchDelta, RepairPlan, ReproTestResult, ValidationEvidence


VERILOG_EXTS = (".sv", ".v", ".svh", ".vh")
CSR_TERMS = {
    "mstatus", "misa", "mtvec", "mscratch", "mepc", "mcause", "mtval",
    "mhartid", "mcountinhibit", "mcycle", "mcycleh", "minstret", "minstreth",
    "dcsr", "dpc", "dscratch0", "dscratch1",
}
CSR_IMPLEMENTATION_FILES = (
    "rtl/ibex_cs_registers.sv",
    "rtl/cs_registers.sv",
)
ISSUE_SIGNAL_STOPWORDS = {
    "risc", "riscv", "risc-v", "ibex", "cva6", "caliptra", "rtl", "csr", "csrs",
    "cpu", "core", "module", "signal", "logic", "input", "output", "valid",
    "ready", "request", "response", "instruction", "compressed", "legal",
    "illegal", "debug", "exception", "interrupt",
}


def _clean(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


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


def _short(raw: str, limit: int = 5000) -> str:
    raw = raw or ""
    return raw if len(raw) <= limit else raw[:limit].rstrip() + "\n...[truncated]..."


def _run(
    cmd: Sequence[str],
    cwd: str,
    timeout: int = 30,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(cmd),
            cwd=cwd,
            input=input_text,
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


def _file_matches(file_path: str, candidates: Sequence[str]) -> bool:
    fp = _clean(file_path).lower()
    if not fp:
        return False
    for candidate in candidates:
        cand = _clean(candidate).lower()
        if fp == cand or fp.endswith("/" + cand) or cand.endswith("/" + fp):
            return True
    return False


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    return re.sub(r"//.*", " ", text)


def _split_port_list(port_text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(port_text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(port_text[start:idx].strip())
            start = idx + 1
    tail = port_text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _port_width(segment: str) -> str:
    matches = re.findall(r"\[[^\]]+\]", segment or "")
    return matches[-1].strip() if matches else ""


def _parse_ansi_ports(port_text: str) -> List[Dict[str, str]]:
    ports: List[Dict[str, str]] = []
    last_direction = ""
    last_width = ""
    for segment in _split_port_list(port_text):
        segment = re.sub(r"\s+", " ", segment.strip())
        segment = segment.strip(",")
        if not segment:
            continue
        direction_match = re.search(r"\b(input|output|inout)\b", segment)
        if direction_match:
            last_direction = direction_match.group(1)
            last_width = _port_width(segment)
        name_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*(?:=[^,]+)?$", segment)
        if not name_match or not last_direction:
            continue
        ports.append({
            "name": name_match.group(1),
            "direction": last_direction,
            "width": _port_width(segment) or last_width,
        })
    return ports


def _parse_modules(repo_path: str, rel_file: str) -> List[Dict[str, Any]]:
    abs_path = os.path.join(repo_path, rel_file)
    if not os.path.exists(abs_path):
        return []
    try:
        text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    modules: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        modules.append({
            "name": match.group(1),
            "file": rel_file,
            "ports": _parse_ansi_ports(match.group(2)),
            "text": text,
        })
    return modules


def _parse_package_name(repo_path: str, rel_file: str) -> str:
    abs_path = os.path.join(repo_path, rel_file)
    try:
        text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = re.search(r"\bpackage\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text)
    return match.group(1) if match else ""


def _changed_package_params(raw_diff: str) -> List[Dict[str, str]]:
    params: List[Dict[str, str]] = []
    current_file = ""
    for line in (raw_diff or "").splitlines():
        if line.startswith("+++ b/"):
            current_file = _clean(line[len("+++ b/") :])
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].strip()
        match = re.search(
            r"\b(?:localparam|parameter)\b(?:\s+(?:int|logic|bit|longint|shortint|byte|integer|time|real|string|[A-Za-z_][A-Za-z0-9_$:]*))?(?:\s+\[[^\]]+\])?\s+([A-Za-z_][A-Za-z0-9_$]*)\s*=\s*([^,;]+)",
            body,
        )
        if match:
            params.append({
                "file": current_file,
                "name": match.group(1),
                "value": match.group(2).strip(),
            })
    return params[:20]


def _port_named(module: Dict[str, Any], names: Sequence[str], direction: str = "") -> Dict[str, str] | None:
    wanted = {name.lower() for name in names}
    for port in module.get("ports") or []:
        pname = str(port.get("name") or "")
        if pname.lower() not in wanted:
            continue
        if direction and port.get("direction") != direction:
            continue
        return port
    return None


def _issue_hex_values(issue_text: str) -> List[str]:
    values: List[str] = []
    for match in re.finditer(r"\b0x([0-9a-fA-F_]+)\b", issue_text or ""):
        values.append(match.group(1).replace("_", ""))
    for match in re.finditer(r"\b\d+'\s*h([0-9a-fA-F_]+)\b", issue_text or "", flags=re.IGNORECASE):
        values.append(match.group(1).replace("_", ""))
    return _dedupe(values)


def _issue_signal_terms(issue_text: str) -> List[str]:
    terms: List[str] = []
    for raw in re.findall(r"`([^`]+)`", issue_text or ""):
        token = raw.strip()
        if re.fullmatch(r"0x[0-9a-fA-F_]+", token):
            continue
        for part in re.split(r"[^A-Za-z0-9_$]+", token):
            part = part.strip()
            if not part:
                continue
            terms.append(part)
    terms.extend(
        match.group(0)
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_$]*(?:_[ioqdn]|_ni|_no)\b", issue_text or "")
    )
    terms.extend(
        match.group(0)
        for match in re.finditer(r"\b[A-Z][A-Z0-9]+_[A-Z0-9_]+\b", issue_text or "")
    )
    result: List[str] = []
    for term in terms:
        stripped = term.strip()
        lowered = stripped.lower()
        if (
            len(stripped) <= 2
            or lowered in ISSUE_SIGNAL_STOPWORDS
            or lowered in CSR_TERMS
            or re.fullmatch(r"x\d+", lowered)
            or re.fullmatch(r"[0-9]+", lowered)
        ):
            continue
        if "_" not in stripped and not re.search(r"[a-z][A-Z]|[A-Z][a-z]", stripped):
            continue
        result.append(stripped)
    return _dedupe(result)[:24]


def _infer_illegal_expectation(issue_text: str) -> str:
    lowered = (issue_text or "").lower()
    not_illegal = bool(re.search(r"\b(?:not|never|must\s+not|should\s+not)\s+(?:be\s+)?illegal\b", lowered))
    legal_hint = bool(re.search(r"\b(?:legal|valid|accepted|accept)\b", lowered))
    should_illegal = bool(re.search(r"\b(?:should|must)\s+(?:be\s+)?illegal\b", lowered))
    reject_hint = bool(re.search(r"\b(?:reject|invalid)\b", lowered))
    if not_illegal or (legal_hint and not should_illegal and not reject_hint):
        return "1'b0"
    if should_illegal or reject_hint:
        return "1'b1"
    return ""


def _classify_testability(issue_text: str, patch_delta: PatchDelta, repair_plan: RepairPlan) -> Dict[str, Any]:
    lowered = (issue_text or "").lower()
    edited_files = list(patch_delta.edited_files or [])
    touched_modules = list(patch_delta.touched_modules or [])
    touched_symbols = list(patch_delta.touched_symbols or [])[:20]
    if re.search(r"\b(?:decode|decoder|instruction|instr|compressed|hint|illegal|legal)\b", lowered):
        return {
            "testability": "high",
            "bug_type": "decoder_or_instruction_constant",
            "recommended_check": "comb_module_proxy",
            "reason": "Issue mentions instruction/decode legality; a local combinational DUT check may reproduce it.",
            "edited_files": edited_files,
            "touched_modules": touched_modules,
            "touched_symbols": touched_symbols,
        }
    if re.search(r"\b(?:parameter|macro|config|configuration|width|localparam)\b", lowered):
        return {
            "testability": "high",
            "bug_type": "config_parameter",
            "recommended_check": "package_parameter_assertion",
            "reason": "Issue concerns package/config parameters, but a concrete expected value must be extractable before generating a runnable assertion.",
            "edited_files": edited_files,
            "touched_modules": touched_modules,
            "touched_symbols": touched_symbols,
        }
    if re.search(r"\b(?:csr|csrs|trap|pmp|mmu|tlb|exception|interrupt)\b", lowered):
        return {
            "testability": "medium",
            "bug_type": "architectural_state_or_protection",
            "recommended_check": "signal_obligation_or_cone_proxy",
            "reason": "Issue affects architectural state or protection paths; local executable proxy may require a larger integration cone.",
            "edited_files": edited_files,
            "touched_modules": touched_modules,
            "touched_symbols": touched_symbols,
        }
    if re.search(r"\b(?:valid|ready|handshake|axi|ahb|req|resp|ack|fifo|buffer|queue)\b", lowered):
        return {
            "testability": "medium",
            "bug_type": "interface_or_handshake",
            "recommended_check": "temporal_signal_proxy",
            "reason": "Issue mentions interface/handshake behavior; a future template can generate temporal checks when clock/reset and protocol endpoints are clear.",
            "edited_files": edited_files,
            "touched_modules": touched_modules,
            "touched_symbols": touched_symbols,
        }
    return {
        "testability": "low",
        "bug_type": "complex_or_unclassified",
        "recommended_check": "static_qualification_only",
        "reason": "No supported local proxy test template matched the issue.",
        "edited_files": edited_files,
        "touched_modules": touched_modules,
        "touched_symbols": touched_symbols,
    }


def _verilog_hex(width: int, hex_digits: str) -> str:
    digits = (hex_digits or "0").lower().lstrip("0") or "0"
    return f"{width}'h{digits}"


def _decl_for_port(port: Dict[str, str]) -> str:
    width = str(port.get("width") or "").strip()
    return f"logic {width} {port['name']};" if width else f"logic {port['name']};"


def _format_value_for_display(value: str) -> str:
    return str(value).replace("'", "\\'")


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._")[:120] or "proxy_check"


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _discover_import_package_files(repo_path: str, module_file: str) -> List[str]:
    abs_file = os.path.join(repo_path, module_file)
    try:
        text = Path(abs_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    packages = _dedupe(match.group(1) for match in re.finditer(r"\bimport\s+([A-Za-z_][A-Za-z0-9_$]*)::\*", text))
    if not packages:
        return []
    found: List[str] = []
    for path in Path(repo_path).rglob("*"):
        if not path.is_file() or path.suffix not in {".sv", ".v", ".svh", ".vh"}:
            continue
        try:
            rel = os.path.relpath(path, repo_path).replace("\\", "/")
            if rel == module_file or any(part in {".git", "build", "out", "__pycache__"} for part in Path(rel).parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for package in packages:
            if re.search(rf"\bpackage\s+{re.escape(package)}\b", text):
                found.append(rel)
                break
    return _dedupe(found)[:8]


def _write_proxy_tb(tb_path: str, spec: FunctionalCheckSpec, module: Dict[str, Any]) -> None:
    if spec.kind == "package_param":
        expectation = (spec.expectations or [{}])[0]
        target_signal = str(expectation.get("signal") or spec.target_signal)
        expected = str(expectation.get("value") or expectation.get("expected") or "")
        lines = [
            "`timescale 1ns/1ps",
            "module kgveri_proxy_tb;",
        ]
        if spec.target_module:
            lines.append(f"  import {spec.target_module}::*;")
        lines.extend([
            "  initial begin",
            f"    if ({target_signal} !== {expected}) begin",
            "      $display(\"KGVERI_SIGNAL_MISMATCH "
            f"check_id={spec.check_id} module={spec.target_module or 'package'} signal={target_signal} "
            f"expected={_format_value_for_display(expected)} observed=%0d\", {target_signal});",
            "      $fatal;",
            "    end",
            f"    $display(\"KGVERI_PROXY_PASS check_id={spec.check_id}\");",
            "    $finish;",
            "  end",
            "endmodule",
        ])
        Path(tb_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    ports = module.get("ports") or []
    stimulus = spec.stimulus or {}
    expectation = (spec.expectations or [{}])[0]
    target_signal = str(expectation.get("signal") or spec.target_signal)
    expected = str(expectation.get("value") or expectation.get("expected") or "")
    lines: List[str] = [
        "`timescale 1ns/1ps",
        "module kgveri_proxy_tb;",
    ]
    for port in ports:
        lines.append(f"  {_decl_for_port(port)}")
    lines.append("")
    lines.append(f"  {spec.target_module} dut (")
    conns = [f"    .{port['name']}({port['name']})" for port in ports]
    lines.append(",\n".join(conns))
    lines.append("  );")
    for port in ports:
        pname = port["name"]
        if port.get("direction") != "input":
            continue
        if re.search(r"\bclk|clock\b", pname, re.IGNORECASE):
            lines.append(f"  initial {pname} = 1'b0;")
            lines.append(f"  always #5 {pname} = ~{pname};")
    lines.append("  initial begin")
    for port in ports:
        pname = port["name"]
        if port.get("direction") != "input":
            continue
        if re.search(r"\bclk|clock\b", pname, re.IGNORECASE):
            continue
        if pname in stimulus:
            value = str(stimulus[pname])
        elif re.search(r"\brst_?n|reset_?n|rst_ni\b", pname, re.IGNORECASE):
            value = "1'b1"
        elif re.search(r"\brst|reset\b", pname, re.IGNORECASE):
            value = "1'b0"
        else:
            value = "'0"
        lines.append(f"    {pname} = {value};")
    lines.append("    #1;")
    lines.append(f"    if ({target_signal} !== {expected}) begin")
    lines.append(
        "      $display(\"KGVERI_SIGNAL_MISMATCH "
        f"check_id={spec.check_id} module={spec.target_module} signal={target_signal} "
        f"expected={_format_value_for_display(expected)} observed=%b cycle=0\", {target_signal});"
    )
    lines.append("      $fatal;")
    lines.append("    end")
    lines.append(f"    $display(\"KGVERI_PROXY_PASS check_id={spec.check_id}\");")
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    Path(tb_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _proxy_filelist(spec: FunctionalCheckSpec) -> List[str]:
    cone = spec.test_cone or {}
    files = list(cone.get("files") or [])
    if spec.target_file not in files:
        files.append(spec.target_file)
    return _dedupe(files)


def _write_proxy_run_script(run_path: str, spec: FunctionalCheckSpec) -> None:
    files = _proxy_filelist(spec)
    incdirs = list((spec.test_cone or {}).get("include_dirs") or [])
    defines = list((spec.test_cone or {}).get("defines") or [])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'REPO="${1:-.}"',
        'OUT_DIR="${2:-./out}"',
        "mkdir -p \"$OUT_DIR\"",
        "CMD=(iverilog -g2012 -o \"$OUT_DIR/proxy.vvp\")",
    ]
    for inc in incdirs:
        lines.append(f'CMD+=("-I$REPO/{inc}")')
    for define in defines:
        lines.append(f'CMD+=("-D{define}")')
    for rel in files:
        lines.append(f'CMD+=("$REPO/{rel}")')
    lines.extend([
        'CMD+=("$(dirname "$0")/proxy_tb.sv")',
        '"${CMD[@]}"',
        'vvp "$OUT_DIR/proxy.vvp"',
    ])
    Path(run_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(run_path, 0o755)


class VerilogFunctionalValidationLayer:
    def __init__(
        self,
        repo_path: str,
        *,
        related_entities: Dict[str, Any] | None = None,
        issue_text: str = "",
        artifact_dir: str = "",
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.related_entities = related_entities or {}
        self.issue_text = issue_text or ""
        self.artifact_dir = os.path.abspath(artifact_dir) if artifact_dir else ""
        if self.artifact_dir:
            os.makedirs(self.artifact_dir, exist_ok=True)

    def validate(
        self,
        *,
        patch_text: str,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[ValidationEvidence]:
        obligation_evidence: List[ValidationEvidence] = []
        obligation_evidence.extend(self._issue_obligation_evidence(patch_delta, repair_plan))
        obligation_evidence.extend(self._issue_signal_obligation_evidence(patch_delta, repair_plan))
        kg_evidence = self._kg_signal_evidence(patch_text, patch_delta, repair_plan)
        proxy_evidence = self._proxy_functional_evidence(patch_delta, repair_plan)
        if any(item.validator == "functional_proxy" and item.type == "VALID_PROXY_PASS" for item in proxy_evidence):
            for item in obligation_evidence:
                if item.type == "ISSUE_SIGNAL_OBLIGATION_MISSING":
                    item.severity = "warning"
                    item.message += " A valid executable proxy passed, so this generic signal-coverage gap is advisory."
        return [*obligation_evidence, *kg_evidence, *proxy_evidence]

    def skipped_due_to_syntax(self) -> ValidationEvidence:
        return ValidationEvidence(
            type="FUNCTIONAL_CHECK_SKIPPED",
            severity="info",
            message="Signal-level functional checks were skipped because syntax/elaboration blockers exist.",
            validator="functional_proxy",
        )

    def _issue_obligation_evidence(
        self,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[ValidationEvidence]:
        issue_text = "\n".join([
            self.issue_text or "",
            getattr(repair_plan, "hypothesis", "") or "",
        ])
        lowered_issue = issue_text.lower()
        if not lowered_issue:
            return []

        named_csrs = sorted({
            term for term in CSR_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered_issue)
        })
        csr_issue = bool(named_csrs) or bool(
            re.search(r"\b(?:csr|csrs|control\s+and\s+status|trap[- ]handling|trap csr)\b", lowered_issue)
        )
        if not csr_issue:
            return []

        diff_text = (patch_delta.raw_diff or "").lower()
        touched = " ".join(patch_delta.touched_symbols or []).lower()
        edited_files = patch_delta.edited_files or []
        evidence: List[ValidationEvidence] = []

        csr_file_exists = next(
            (path for path in CSR_IMPLEMENTATION_FILES if os.path.exists(os.path.join(self.repo_path, path))),
            "",
        )
        csr_file_touched = bool(csr_file_exists and _file_matches(csr_file_exists, edited_files))
        if csr_file_exists and not csr_file_touched:
            evidence.append(ValidationEvidence(
                type="SIGNAL_OBLIGATION_MISSING",
                severity="blocker",
                file=csr_file_exists,
                module=os.path.splitext(os.path.basename(csr_file_exists))[0],
                check_id="csr_implementation_file_touched",
                expected=f"edit {csr_file_exists}",
                observed=", ".join(edited_files) or "no edited RTL files",
                message=(
                    "Issue describes CSR/trap register behavior, but the patch does not edit the "
                    f"CSR implementation file {csr_file_exists}. Touching controller wiring alone is "
                    "insufficient for CSR read/write and trap-state obligations."
                ),
                validator="functional_obligation",
            ))

        for term in named_csrs:
            macro = f"csr_{term}".lower()
            if term in diff_text or macro in diff_text or term in touched:
                continue
            evidence.append(ValidationEvidence(
                type="SIGNAL_OBLIGATION_MISSING",
                severity="blocker",
                file=csr_file_exists or (edited_files[0] if edited_files else ""),
                module=os.path.splitext(os.path.basename(csr_file_exists or (edited_files[0] if edited_files else "")))[0],
                signal=term,
                check_id=f"csr_obligation_{term}",
                expected=f"diff covers {term} CSR decode/read/write/save behavior",
                observed=f"{term} absent from patch delta",
                message=(
                    f"Patch does not cover issue-mentioned CSR obligation '{term}'. "
                    "The repair should update the CSR decode/read/write/save path, or explicitly "
                    "show why the obligation is already satisfied."
                ),
                validator="functional_obligation",
            ))

        if "mtval" in named_csrs and "mtval" in diff_text and csr_file_exists and not csr_file_touched:
            evidence.append(ValidationEvidence(
                type="SIGNAL_OBLIGATION_MISSING",
                severity="blocker",
                file=csr_file_exists,
                module=os.path.splitext(os.path.basename(csr_file_exists))[0],
                signal="mtval",
                check_id="csr_obligation_mtval_storage",
                expected="mtval storage/readback path updated in CSR module",
                observed="mtval appears only outside CSR implementation file",
                message=(
                    "Patch mentions mtval but does not edit the CSR storage/readback path. "
                    "A valid mtval repair needs storage or readback logic in the CSR module, not only a controller save signal."
                ),
                validator="functional_obligation",
            ))

        if not evidence:
            covered = ", ".join(named_csrs) if named_csrs else "CSR/trap behavior"
            evidence.append(ValidationEvidence(
                type="SIGNAL_OBLIGATION_SATISFIED",
                severity="info",
                file=csr_file_exists or (edited_files[0] if edited_files else ""),
                module=os.path.splitext(os.path.basename(csr_file_exists or (edited_files[0] if edited_files else "")))[0],
                check_id="csr_obligation_coverage",
                expected=f"patch covers {covered}",
                observed=", ".join(patch_delta.edited_files or []) or "patch delta inspected",
                message="Issue-level CSR/trap obligations are represented in the patch delta.",
                validator="functional_obligation",
            ))

        return evidence

    def _issue_signal_obligation_evidence(
        self,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[ValidationEvidence]:
        issue_text = "\n".join([
            self.issue_text or "",
            getattr(repair_plan, "hypothesis", "") or "",
        ])
        terms = _issue_signal_terms(issue_text)
        if not terms:
            return []
        diff_text = (patch_delta.raw_diff or "").lower()
        touched_text = " ".join(patch_delta.touched_symbols or []).lower()
        covered = [
            term for term in terms
            if term.lower() in diff_text or term.lower() in touched_text
        ]
        target_file = (patch_delta.edited_files or [""])[0]
        target_module = (patch_delta.touched_modules or [""])[0]
        if not covered:
            return [ValidationEvidence(
                type="ISSUE_SIGNAL_OBLIGATION_MISSING",
                severity="blocker",
                file=target_file,
                module=target_module,
                signal=terms[0],
                check_id="issue_signal_obligation",
                expected="patch references at least one issue-mentioned signal or architectural constant",
                observed="none of " + ", ".join(terms[:8]) + " appeared in the patch delta",
                message=(
                    "Issue names concrete signal/constant anchors, but the patch delta does not cover any of them. "
                    "This is a weak functional repair signal and should be diagnosed through KG traceback."
                ),
                validator="functional_obligation",
            )]

        evidence: List[ValidationEvidence] = [ValidationEvidence(
            type="ISSUE_SIGNAL_OBLIGATION_SATISFIED",
            severity="info",
            file=target_file,
            module=target_module,
            signal=covered[0],
            check_id="issue_signal_obligation",
            expected="patch references issue-mentioned signal or architectural constant anchors",
            observed=", ".join(covered[:8]),
            message="Patch delta covers at least one issue-mentioned signal or constant anchor.",
            validator="functional_obligation",
        )]
        for term in [term for term in terms if term not in covered][:6]:
            evidence.append(ValidationEvidence(
                type="ISSUE_SIGNAL_OBLIGATION_PARTIAL",
                severity="warning",
                file=target_file,
                module=target_module,
                signal=term,
                check_id=f"issue_signal_obligation_{term}",
                expected=f"patch may need to address issue anchor {term}",
                observed=f"{term} not found in patch delta",
                message="Issue-mentioned signal/constant is not directly visible in the patch delta.",
                validator="functional_obligation",
            ))
        return evidence

    def _kg_signal_evidence(
        self,
        patch_text: str,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[ValidationEvidence]:
        kg_result = check_patch_signal_flow(
            patch_text=patch_text,
            edited_files=patch_delta.edited_files,
            related_entities=self.related_entities,
            candidate_files=list(repair_plan.allowed_files or []),
        )
        evidence: List[ValidationEvidence] = []
        for issue in kg_result.get("issues") or []:
            message = str(issue.get("message") or "")
            sev = str(issue.get("severity") or "warn")
            signal_matches = re.findall(r"'([A-Za-z_][A-Za-z0-9_$]*)'", message)
            if "Signals not found in KG:" in message:
                tail = message.split("Signals not found in KG:", 1)[1].split(".", 1)[0]
                signal_matches.extend(sig.strip() for sig in tail.split(",") if sig.strip())
            signals = _dedupe(signal_matches) or [""]
            for signal in signals[:8]:
                evidence.append(ValidationEvidence(
                    type="KG_SIGNAL_FLOW",
                    severity="blocker" if sev == "error" else "warning",
                    signal=signal,
                    check_id=f"kg_signal_{signal}" if signal else "kg_signal_flow",
                    message=message,
                    validator="functional_kg_signal",
                ))
        return evidence

    def _proxy_functional_evidence(
        self,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[ValidationEvidence]:
        testability = _classify_testability(
            self.issue_text or getattr(repair_plan, "hypothesis", "") or "",
            patch_delta,
            repair_plan,
        )
        if self.artifact_dir:
            _write_json(os.path.join(self.artifact_dir, "testability.json"), testability)
        specs = self._plan_proxy_checks(patch_delta, repair_plan)
        if not specs:
            if self.artifact_dir:
                _write_json(os.path.join(self.artifact_dir, "repro_index.json"), {
                    "status": "SKIPPED_NOT_APPLICABLE",
                    "testability": testability,
                    "checks": [],
                })
            return [ValidationEvidence(
                type="FUNCTIONAL_PROXY_SKIPPED",
                severity="info",
                check_id="testability_analysis",
                expected=testability.get("recommended_check", ""),
                observed="no executable proxy generated",
                message="No supported signal-level proxy functional check matched this issue and patch cone.",
                validator="functional_proxy",
            )]
        if self.artifact_dir:
            _write_json(os.path.join(self.artifact_dir, "repro_index.json"), {
                "status": "PLANNED",
                "testability": testability,
                "checks": [spec.to_dict() for spec in specs],
            })
        evidence: List[ValidationEvidence] = []
        base_repo = self._make_base_repo(patch_delta.raw_diff)
        if not base_repo:
            return [ValidationEvidence(
                type="FUNCTIONAL_PROXY_SKIPPED",
                severity="warning",
                message="Unable to construct a base-revision worktree by reversing the candidate diff; proxy checks skipped.",
                validator="functional_proxy",
            )]
        try:
            for spec in specs[:3]:
                evidence.extend(self._run_base_patch_proxy_pair(spec, base_repo))
        finally:
            shutil.rmtree(os.path.dirname(base_repo), ignore_errors=True)
        return evidence

    def _plan_proxy_checks(
        self,
        patch_delta: PatchDelta,
        repair_plan: RepairPlan,
    ) -> List[FunctionalCheckSpec]:
        issue_text = self.issue_text or getattr(repair_plan, "hypothesis", "") or ""
        lowered = issue_text.lower()
        package_specs = self._plan_package_param_checks(issue_text, patch_delta)
        if package_specs:
            return package_specs
        if not re.search(r"\b(?:decode|decoder|instruction|instr|compressed|hint|illegal|legal)\b", lowered):
            return []
        hex_values = _issue_hex_values(issue_text)
        expected_illegal = _infer_illegal_expectation(issue_text)
        if not hex_values or not expected_illegal:
            return []
        files = _dedupe([*list(patch_delta.edited_files or []), *list(repair_plan.allowed_files or [])])
        for rel in files:
            if not rel.endswith(VERILOG_EXTS):
                continue
            for module in _parse_modules(self.repo_path, rel):
                instr_port = _port_named(module, ("instr_i", "instruction_i", "insn_i", "instr"), "input")
                illegal_port = _port_named(module, ("illegal_instr_o", "illegal_insn_o", "illegal_o"), "output")
                if not instr_port or not illegal_port:
                    continue
                width_match = re.search(r"\[(\d+)\s*:\s*(\d+)\]", instr_port.get("width") or "")
                width = 32
                if width_match:
                    high = int(width_match.group(1))
                    low = int(width_match.group(2))
                    width = abs(high - low) + 1
                value = _verilog_hex(width, hex_values[0])
                package_files = _discover_import_package_files(self.repo_path, rel)
                include_dirs = [
                    inc for inc in ("rtl", "src", "hw", "include")
                    if os.path.isdir(os.path.join(self.repo_path, inc))
                ]
                return [FunctionalCheckSpec(
                    check_id=f"decode_{module['name']}_{hex_values[0].lower()}_{illegal_port['name']}",
                    kind="comb_module",
                    target_file=rel,
                    target_module=module["name"],
                    target_signal=illegal_port["name"],
                    stimulus={instr_port["name"]: value},
                    expectations=[{
                        "signal": illegal_port["name"],
                        "op": "==",
                        "value": expected_illegal,
                    }],
                    requires_base_fail=True,
                    source="issue_constant_decoder_proxy",
                    confidence=0.75,
                    testability="high",
                    test_cone={
                        "files": _dedupe([*package_files, rel]),
                        "include_dirs": include_dirs,
                        "defines": [],
                        "ports": module.get("ports") or [],
                        "reason": (
                            "Issue contains an instruction constant and legal/illegal decoder language; "
                            "KG/RTL interface exposes an instruction input and illegal output."
                        ),
                    },
                    kg_anchors={
                        "files": [rel],
                        "modules": [module["name"]],
                        "signals": [instr_port["name"], illegal_port["name"]],
                        "stimulus_constants": [hex_values[0]],
                    },
                )]
        return []

    def _plan_package_param_checks(
        self,
        issue_text: str,
        patch_delta: PatchDelta,
    ) -> List[FunctionalCheckSpec]:
        lowered = (issue_text or "").lower()
        if not re.search(r"\b(?:parameter|localparam|macro|config|configuration|width)\b", lowered):
            return []
        changed = _changed_package_params(patch_delta.raw_diff)
        if not changed:
            return []
        issue_tokens = {token.lower() for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", issue_text or "")}
        selected = [
            item for item in changed
            if item.get("name", "").lower() in issue_tokens
        ] or changed[:3]
        specs: List[FunctionalCheckSpec] = []
        for item in selected[:3]:
            rel = item.get("file") or ""
            name = item.get("name") or ""
            value = item.get("value") or ""
            if not rel or not name or not value or not rel.endswith(VERILOG_EXTS):
                continue
            package_name = _parse_package_name(self.repo_path, rel)
            if not package_name:
                continue
            include_dirs = [
                inc for inc in ("rtl", "src", "hw", "include", "core/include")
                if os.path.isdir(os.path.join(self.repo_path, inc))
            ]
            specs.append(FunctionalCheckSpec(
                check_id=f"package_param_{package_name}_{name}",
                kind="package_param",
                target_file=rel,
                target_module=package_name,
                target_signal=name,
                expectations=[{
                    "signal": name,
                    "op": "==",
                    "value": value,
                }],
                requires_base_fail=True,
                source="patch_delta_package_parameter_proxy",
                confidence=0.55,
                testability="high",
                test_cone={
                    "files": [rel],
                    "include_dirs": include_dirs,
                    "defines": [],
                    "ports": [],
                    "reason": (
                        "Issue is configuration/parameter-related and the patch changes a package parameter; "
                        "the proxy asserts that the patched public package value is observable."
                    ),
                },
                kg_anchors={
                    "files": [rel],
                    "modules": [package_name],
                    "signals": [name],
                },
            ))
        return specs

    def _make_base_repo(self, raw_diff: str) -> str:
        if not (raw_diff or "").strip():
            return ""
        tmp = tempfile.mkdtemp(prefix="kgveri_proxy_base_")
        base_repo = os.path.join(tmp, "repo")
        shutil.copytree(self.repo_path, base_repo, symlinks=True)
        result = _run(["git", "apply", "-R", "--whitespace=nowarn", "-"], cwd=base_repo, input_text=raw_diff, timeout=20)
        if result.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return ""
        return base_repo

    def _run_base_patch_proxy_pair(self, spec: FunctionalCheckSpec, base_repo: str) -> List[ValidationEvidence]:
        case_dir = self._artifact_case_dir(spec)
        base = self._run_proxy_check(spec, base_repo, artifact_case_dir=case_dir, label="base")
        patched = self._run_proxy_check(spec, self.repo_path, artifact_case_dir=case_dir, label="patched")
        base_signal = base.get("signal") or spec.target_signal
        base_observed = str(base.get("observed") or "")
        patched_observed = str(patched.get("observed") or "")
        expected = str((spec.expectations or [{}])[0].get("value") or "")
        result_status = ""
        result_message = ""
        if base["status"] == "compile_fail":
            result_status = "INVALID_COMPILE_FAIL"
            result_message = "Proxy functional check could not compile on the base revision."
            evidence = [ValidationEvidence(
                type="INVALID_PROXY_COMPILE_FAIL",
                severity="warning",
                file=spec.target_file,
                module=spec.target_module,
                signal=base_signal,
                check_id=spec.check_id,
                expected=expected,
                observed=base_observed,
                stimulus=spec.stimulus,
                raw_log=_short(base.get("raw_log", "")),
                message=result_message + " It is not a valid oracle.",
                validator="functional_proxy",
            )]
        elif base["status"] == "pass":
            result_status = "INVALID_BASE_PASS"
            result_message = "Proxy functional check passed on the base revision."
            evidence = [ValidationEvidence(
                type="INVALID_PROXY_BASE_PASS",
                severity="warning",
                file=spec.target_file,
                module=spec.target_module,
                signal=base_signal,
                check_id=spec.check_id,
                expected=expected,
                observed=expected,
                stimulus=spec.stimulus,
                message=result_message + " It does not reproduce the issue and is ignored as proof.",
                validator="functional_proxy",
            )]
        elif patched["status"] == "compile_fail":
            result_status = "PATCH_COMPILE_FAIL"
            result_message = "Proxy functional check compiles on base but fails to compile after the patch."
            evidence = [ValidationEvidence(
                type="PATCH_PROXY_COMPILE_FAIL",
                severity="blocker",
                file=spec.target_file,
                module=spec.target_module,
                signal=base_signal,
                check_id=spec.check_id,
                expected=expected,
                observed=patched_observed,
                stimulus=spec.stimulus,
                raw_log=_short(patched.get("raw_log", "")),
                message=result_message,
                validator="functional_proxy",
            )]
        elif patched["status"] == "pass":
            result_status = "VALID_PROXY_PASS"
            result_message = "Proxy functional check is valid: base fails and patched revision passes."
            evidence = [ValidationEvidence(
                type="VALID_PROXY_PASS",
                severity="info",
                file=spec.target_file,
                module=spec.target_module,
                signal=base_signal,
                check_id=spec.check_id,
                expected=expected,
                observed=expected,
                stimulus=spec.stimulus,
                message=result_message,
                validator="functional_proxy",
            )]
        else:
            result_status = "VALID_PROXY_FAIL"
            result_message = "Proxy functional check still fails after the patch; expected and observed signal values differ."
            evidence = [ValidationEvidence(
                type="SIGNAL_MISMATCH",
                severity="blocker",
                file=spec.target_file,
                module=spec.target_module,
                signal=patched.get("signal") or base_signal,
                check_id=spec.check_id,
                expected=expected,
                observed=patched_observed or "unknown",
                cycle=0,
                stimulus=spec.stimulus,
                raw_log=_short(patched.get("raw_log", "")),
                message=result_message,
                validator="functional_proxy",
            )]
        if case_dir:
            self._write_repro_result(
                case_dir,
                ReproTestResult(
                    check_id=spec.check_id,
                    status=result_status,
                    base_status=base.get("status", ""),
                    patched_status=patched.get("status", ""),
                    artifact_dir=case_dir,
                    message=result_message,
                    spec=spec.to_dict(),
                ),
            )
        return evidence

    def _artifact_case_dir(self, spec: FunctionalCheckSpec) -> str:
        if not self.artifact_dir:
            return ""
        path = os.path.join(self.artifact_dir, _safe_name(spec.check_id))
        os.makedirs(path, exist_ok=True)
        return path

    def _materialize_proxy_artifacts(
        self,
        spec: FunctionalCheckSpec,
        module: Dict[str, Any],
        artifact_case_dir: str,
    ) -> None:
        if not artifact_case_dir:
            return
        _write_json(os.path.join(artifact_case_dir, "spec.json"), spec.to_dict())
        _write_proxy_tb(os.path.join(artifact_case_dir, "proxy_tb.sv"), spec, module)
        _write_proxy_run_script(os.path.join(artifact_case_dir, "run.sh"), spec)
        with open(os.path.join(artifact_case_dir, "proxy_filelist.txt"), "w", encoding="utf-8") as f:
            for rel in _proxy_filelist(spec):
                f.write(rel + "\n")

    def _write_repro_result(self, artifact_case_dir: str, result: ReproTestResult) -> None:
        _write_json(os.path.join(artifact_case_dir, "result.json"), result.to_dict())

    def _run_proxy_check(
        self,
        spec: FunctionalCheckSpec,
        repo_path: str,
        *,
        artifact_case_dir: str = "",
        label: str = "",
    ) -> Dict[str, str]:
        if not shutil.which("iverilog") or not shutil.which("vvp"):
            return {"status": "compile_fail", "raw_log": "iverilog or vvp not found"}
        if spec.kind == "package_param":
            module = {"name": spec.target_module, "ports": []}
            if spec.target_module and not _parse_package_name(repo_path, spec.target_file):
                return {"status": "compile_fail", "raw_log": f"package {spec.target_module} not found"}
        else:
            modules = _parse_modules(repo_path, spec.target_file)
            module = next((item for item in modules if item["name"] == spec.target_module), None)
            if not module:
                return {"status": "compile_fail", "raw_log": f"module {spec.target_module} not found"}
        self._materialize_proxy_artifacts(spec, module, artifact_case_dir)
        with tempfile.TemporaryDirectory(prefix="kgveri_proxy_tb_") as tmp:
            tb_path = os.path.join(artifact_case_dir, "proxy_tb.sv") if artifact_case_dir else os.path.join(tmp, "kgveri_proxy_tb.sv")
            out_path = os.path.join(tmp, "kgveri_proxy_tb.vvp")
            if not artifact_case_dir:
                _write_proxy_tb(tb_path, spec, module)
            files = _proxy_filelist(spec)
            if not files:
                files = [*_discover_import_package_files(repo_path, spec.target_file), spec.target_file]
            cmd = ["iverilog", "-g2012", "-o", out_path]
            for inc in (spec.test_cone or {}).get("include_dirs") or ("rtl", "src", "hw", "include"):
                inc_path = os.path.join(repo_path, inc)
                if os.path.isdir(inc_path):
                    cmd.append(f"-I{inc_path}")
            for define in (spec.test_cone or {}).get("defines") or []:
                cmd.append(f"-D{define}")
            cmd.extend(os.path.join(repo_path, rel) for rel in files)
            cmd.append(tb_path)
            compile_result = _run(cmd, cwd=repo_path, timeout=30)
            compile_log = (compile_result.stdout or "") + "\n" + (compile_result.stderr or "")
            if artifact_case_dir and label:
                Path(os.path.join(artifact_case_dir, f"{label}.compile.log")).write_text(compile_log, encoding="utf-8")
            if compile_result.returncode != 0:
                return {"status": "compile_fail", "raw_log": compile_log}
            run_result = _run(["vvp", out_path], cwd=repo_path, timeout=15)
            raw_log = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
            if artifact_case_dir and label:
                Path(os.path.join(artifact_case_dir, f"{label}.run.log")).write_text(raw_log, encoding="utf-8")
            if "KGVERI_PROXY_PASS" in raw_log and run_result.returncode == 0:
                return {"status": "pass", "raw_log": raw_log}
            mismatch = re.search(
                r"KGVERI_SIGNAL_MISMATCH\s+check_id=(\S+)\s+module=(\S+)\s+signal=(\S+)\s+expected=(\S+)\s+observed=([^\s]+)",
                raw_log,
            )
            if mismatch:
                return {
                    "status": "signal_mismatch",
                    "signal": mismatch.group(3),
                    "expected": mismatch.group(4),
                    "observed": mismatch.group(5),
                    "raw_log": raw_log,
                }
            return {"status": "signal_mismatch", "raw_log": raw_log}
