from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence

try:
    from repair_types import ValidationEvidence, ValidationReport, evidence_from_dict
except Exception:  # pragma: no cover
    from .repair_types import ValidationEvidence, ValidationReport, evidence_from_dict


def evidence_to_dicts(evidence: Iterable[ValidationEvidence]) -> List[Dict[str, Any]]:
    return [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in evidence]


def normalize_evidence(items: Iterable[Any]) -> List[ValidationEvidence]:
    result: List[ValidationEvidence] = []
    for item in items:
        if isinstance(item, ValidationEvidence):
            result.append(item)
        elif isinstance(item, dict):
            result.append(evidence_from_dict(item))
    return result


def evidence_feedback_text(evidence: Sequence[ValidationEvidence], limit: int = 8000) -> str:
    lines = ["=== KGVeri Internal Validation Evidence ==="]
    if not evidence:
        lines.append("No validation evidence.")
    for idx, item in enumerate(evidence, start=1):
        loc = ""
        if item.file:
            loc = item.file
            if item.line:
                loc += f":{item.line}"
        signal = f" signal={item.signal}" if item.signal else ""
        module = f" module={item.module}" if item.module else ""
        check = f" check={item.check_id}" if item.check_id else ""
        comparison = ""
        if item.expected or item.observed:
            comparison = f" expected={item.expected or '?'} observed={item.observed or '?'}"
        stimulus = ""
        if item.stimulus:
            pairs = ", ".join(f"{k}={v}" for k, v in list(item.stimulus.items())[:6])
            stimulus = f" stimulus=({pairs})"
        lines.append(
            f"{idx}. [{item.severity.upper()}][{item.validator or item.type}] "
            f"{loc}{module}{signal}{check}{comparison}{stimulus} {item.message}".strip()
        )
        if item.raw_log:
            raw = item.raw_log.strip()
            if raw:
                lines.append("   raw: " + raw.replace("\n", "\n        ")[:1200])
    text = "\n".join(lines)
    return text[:limit].rstrip() + ("\n...[truncated]..." if len(text) > limit else "")


def evidence_localization_text(evidence: Sequence[Any], limit: int = 12000) -> str:
    """Project structured evidence to deterministic localization anchors.

    Raw logs remain available to the diagnosis prompt but are intentionally
    excluded here so compiler prose and command lines cannot become KG anchors.
    """
    rows: List[str] = []
    for raw_item in evidence or []:
        item = raw_item.to_dict() if hasattr(raw_item, "to_dict") else dict(raw_item or {})
        location = str(item.get("file") or "")
        if location and item.get("line") is not None:
            location += f":{item.get('line')}"
        fields = [location]
        for key in ("module", "signal", "check_id", "expected", "observed"):
            value = str(item.get(key) or "").strip()
            if value:
                label = "check" if key == "check_id" else key
                fields.append(f"{label}={value}")
        message = str(item.get("message") or "").strip()
        if message and not any(fields):
            fields.append(message)
        row = " ".join(field for field in fields if field)
        if row and row not in rows:
            rows.append(row)
    text = "\n".join(rows)
    return text[:limit]


def build_validation_report(
    evidence: Sequence[ValidationEvidence],
    metadata: Dict[str, Any] | None = None,
) -> ValidationReport:
    evidence = list(evidence)
    blocker_count = sum(1 for item in evidence if item.severity == "blocker")
    warning_count = sum(1 for item in evidence if item.severity in {"warning", "info"})
    return ValidationReport(
        passed=blocker_count == 0,
        evidence=evidence,
        feedback_text=evidence_feedback_text(evidence),
        blocker_count=blocker_count,
        warning_count=warning_count,
        metadata=metadata or {},
    )


ANCHOR_STOPWORDS = {
    "home", "caoms", "tmp", "workdirs", "validation_attempts", "attempt",
    "kgcompass", "patch", "diff", "repo", "compile", "verilator", "iverilog",
    "warning", "error", "blocker", "context", "incomplete", "single", "file",
    "files", "line", "lines", "raw", "message", "edited", "module", "signal",
    "diagnostic", "diagnostics", "unrelated", "pre", "existing",
    "tb_fail", "tb_pass", "sim_fail", "sim_pass", "hwe_bench_results_start",
    "hwe_bench_results_end",
}


def _clean_file(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    text = re.sub(r"^/+", "", text)
    for marker in ("rtl/", "src/", "hw/", "include/", "vendor/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    return text


def _anchor_token(token: str) -> bool:
    token = str(token or "").strip()
    lowered = token.lower()
    if len(token) <= 2 or lowered in ANCHOR_STOPWORDS:
        return False
    if token.upper() == token and "_" not in token:
        return False
    if re.fullmatch(r"[0-9a-fA-F_xzXZ]+", token):
        return False
    return lowered.endswith(("_i", "_o", "_q", "_d", "_n")) or "_" in token


def anchors_from_evidence(
    evidence: Sequence[ValidationEvidence],
    patch_delta: Dict[str, Any] | None = None,
) -> Dict[str, List[str]]:
    signals: List[str] = []
    modules: List[str] = []
    files: List[str] = []
    tests: List[str] = []
    patch_delta = patch_delta or {}
    for token in patch_delta.get("touched_symbols") or []:
        if _anchor_token(str(token)):
            signals.append(str(token))
    for token in patch_delta.get("touched_modules") or []:
        token = str(token or "").strip()
        if token and token.lower() not in ANCHOR_STOPWORDS:
            modules.append(token)
    for span in patch_delta.get("changed_spans") or []:
        if isinstance(span, dict):
            file_path = _clean_file(str(span.get("file") or span.get("file_path") or ""))
            if file_path:
                files.append(file_path)
    for item in evidence:
        if item.signal:
            signals.append(item.signal)
        if item.module:
            modules.append(item.module)
        if item.file:
            files.append(_clean_file(item.file))
        if item.check_id:
            tests.append(str(item.check_id))
        for key in (item.stimulus or {}).keys():
            if _anchor_token(str(key)):
                signals.append(str(key))
        for text in (item.message or "", item.raw_log or ""):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("TEST:") and "HWE_BENCH_RESULTS" not in line
            )
            sanitized_message = re.sub(r"/[A-Za-z0-9_./$-]+", " ", text)
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", sanitized_message):
                if item.check_id and token == str(item.check_id):
                    continue
                if _anchor_token(token):
                    signals.append(token)
    return {
        "signals": _dedupe(signals)[:20],
        "modules": _dedupe(modules)[:10],
        "files": _dedupe(files)[:10],
        "tests": _dedupe(tests)[:10],
    }


def write_evidence_trace(path: str, report: ValidationReport, extra: Dict[str, Any] | None = None) -> None:
    patch_delta = (extra or {}).get("patch_delta") if isinstance(extra, dict) else None
    payload = {
        "report": report.to_dict(),
        "anchors": anchors_from_evidence(report.evidence, patch_delta=patch_delta),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
