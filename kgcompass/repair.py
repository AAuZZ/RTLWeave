from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import openai

from benchmark import get_target_sample
from config import (
    BAILIAN_API_KEY,
    DEEPSEEK_BASE_URL,
    MAX_INPUT_LENGTH,
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
)
from repair_loop import (
    PatchApplicationResult,
    PromptBuilder,
    select_patch_structured_evidence,
    select_patch_validation_output,
)
from utils import (
    context_entity_sort_key,
    format_entity_content,
    parse_diff_edit_commands_strict,
    split_edit_multifile_commands,
    check_syntax,
)
from knowledge_graph import KnowledgeGraph
from kg_debug_trace import build_debug_kg_trace
from hwe_bench_validator import HweBenchValidator, hwe_result_to_validation_report
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from term import Stage, stage as tstage, ok as tok, fail as tfail, rule as trule
try:
    from verilog_kg_digest import build_verilog_bug_kg_digest
except Exception:
    from .verilog_kg_digest import build_verilog_bug_kg_digest
try:
    from verilog_kg_slice import build_verilog_fault_neighborhood
except Exception:
    from .verilog_kg_slice import build_verilog_fault_neighborhood
try:
    from verilog_timing import normalize_verilog_related_entities
except Exception:
    from .verilog_timing import normalize_verilog_related_entities
try:
    from diagnostic_evidence import build_validation_report, evidence_localization_text, write_evidence_trace
    from issue_path_navigation import IssuePathNavigator
    from patch_delta import build_patch_delta
    from repair_planner import build_repair_plan, is_forbidden_file, repair_plan_to_prompt
    from repair_types import ValidationEvidence
    from source_context import SourceContextManager, SourceContextState, normalize_repo_relative_path
    from minitb.visible_input import select_case
    from validation_stack import (
        VALIDATION_STACK_VERSION,
        LightweightValidationStack,
        merge_validation_reports,
    )
except Exception:
    from .diagnostic_evidence import build_validation_report, evidence_localization_text, write_evidence_trace
    from .issue_path_navigation import IssuePathNavigator
    from .patch_delta import build_patch_delta
    from .repair_planner import build_repair_plan, is_forbidden_file, repair_plan_to_prompt
    from .repair_types import ValidationEvidence
    from .source_context import SourceContextManager, SourceContextState, normalize_repo_relative_path
    from .minitb.visible_input import select_case
    from .validation_stack import (
        VALIDATION_STACK_VERSION,
        LightweightValidationStack,
        merge_validation_reports,
    )


def _compact_text(text: str, limit: int = 1200) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _unique_keep_order(items):
    seen = set()
    ordered = []
    for item in items:
        marker = str(item)
        if marker in seen:
            continue
        ordered.append(item)
        seen.add(marker)
    return ordered


def _collect_all_entity_files(related_entities):
    """Collect all unique file_path values from a related_entities dict."""
    files: list[str] = []
    for group_name, entities in (related_entities or {}).items():
        for e in (entities or []):
            fp = str((e or {}).get("file_path") or "").replace("\\", "/")
            if fp and not fp.startswith("tb/") and re.search(r"\.(?:sv|v|svh|vh)$", fp, re.IGNORECASE):
                if fp not in files:
                    files.append(fp)
    return files


def _issue_path_file(row):
    if not isinstance(row, dict):
        return ""
    for key in ("file_path", "file", "target_file"):
        value = str(row.get(key) or "").replace("\\", "/").strip()
        if value and re.search(r"\.(?:sv|v|svh|vh)$", value, re.IGNORECASE):
            return value
    return ""


def _merge_issue_to_entity_paths(*groups, limit=32):
    rows = []
    seen = set()
    for group in groups:
        for row in group or []:
            if isinstance(row, str):
                marker = row.strip()
                if not marker or marker in seen:
                    continue
                seen.add(marker)
                rows.append({"path_text": marker})
            elif isinstance(row, dict):
                path_text = str(row.get("path_text") or row.get("path") or "").strip()
                target = str(row.get("target") or row.get("name") or row.get("signature") or "").strip()
                file_path = _issue_path_file(row)
                marker = json.dumps(
                    {
                        "target": target,
                        "file_path": file_path,
                        "start_line": row.get("start_line"),
                        "path_text": path_text,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def _format_issue_to_entity_paths(paths, limit=16):
    lines = []
    for row in (paths or [])[:limit]:
        if isinstance(row, str):
            text = row.strip()
            if text:
                lines.append(f"- {text}")
            continue
        if not isinstance(row, dict):
            continue
        target = row.get("target") or row.get("name") or row.get("signature") or ""
        file_path = _issue_path_file(row)
        start = row.get("start_line")
        end = row.get("end_line")
        span = f":{start or '?'}-{end or '?'}" if file_path and (start or end) else ""
        group = row.get("group") or row.get("category") or ""
        kind = row.get("type") or row.get("kind") or ""
        path_text = row.get("path_text") or ""
        if not path_text and isinstance(row.get("path"), list):
            pieces = []
            for step in row.get("path")[:5]:
                if not isinstance(step, dict):
                    continue
                start_node = str(step.get("start_node") or "").strip()
                end_node = str(step.get("end_node") or "").strip()
                rel = str(step.get("type") or step.get("relation_kind") or "RELATED").strip()
                if start_node or end_node:
                    pieces.append(f"{start_node} -[{rel}]-> {end_node}")
            path_text = " | ".join(pieces)
        header = " ".join(
            part
            for part in [
                str(target).strip(),
                f"[{kind}]" if kind else "",
                f"({group})" if group else "",
            ]
            if part
        )
        location = f"{file_path}{span}" if file_path else ""
        if header and location:
            lines.append(f"- {header} @ {location}")
        elif header:
            lines.append(f"- {header}")
        elif location:
            lines.append(f"- {location}")
        if path_text:
            lines.append(f"  path: {path_text}")
    return "\n".join(lines)



_PROGRESS_LOG_PATH = os.getenv("KGCOMPASS_REPAIR_PROGRESS_LOG", "")

# Stage mapping from old string tags to term.Stage
_STAGE_MAP = {
    "start": Stage.REPAIR,
    "attempt": Stage.REPAIR,
    "workspace": Stage.INFO,
    "llm": Stage.LLM,
    "llm-preview": Stage.INFO,
    "patch": Stage.PATCH,
    "validation": Stage.VALIDATE,
    "debug": Stage.DEBUG,
    "debug-diagnose": Stage.DEBUG,
    "debug-context": Stage.DEBUG,
    "success": Stage.OK,
    "finish": Stage.INFO,
    "generation": Stage.REPAIR,
    "error": Stage.FAIL,
    "repair-plan": Stage.REPAIR,
    "delta-gate": Stage.VALIDATE,
    "internal-validation": Stage.VALIDATE,
    "evidence": Stage.DEBUG,
    "debug-trace": Stage.DEBUG,
    "failure-analysis": Stage.DEBUG,
}


def _log(stage: str, message: str = ""):
    s = _STAGE_MAP.get(stage, Stage.INFO)
    tstage(s, message)
    if _PROGRESS_LOG_PATH:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(_PROGRESS_LOG_PATH)), exist_ok=True)
            with open(_PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{stage}] {message}\n")
        except OSError:
            pass


def _load_frozen_minitb_binding(instance_id: str, run_root: str):
    project_root = Path(__file__).resolve().parents[1]
    visible_cases = Path(
        os.getenv(
            "KGVERI_MINITB_VISIBLE_CASES",
            str(project_root / "experiments" / "minitb" / "visible_cases.json"),
        )
    ).resolve()
    case_dir = Path(
        os.getenv(
            "KGVERI_MINITB_CASE_DIR",
            str(Path(run_root) / "minitb" / instance_id),
        )
    ).resolve()
    case = select_case(visible_cases, instance_id)
    frozen_dir = case_dir / "frozen"
    if not (frozen_dir / "sha256.json").is_file():
        raise ValueError(f"frozen Mini-TB or module-smoke assets are unavailable: {frozen_dir}")
    return case, frozen_dir, visible_cases, case_dir


def _final_response_content(message_or_text):
    if message_or_text is None:
        return ""
    if isinstance(message_or_text, str):
        text = message_or_text
    else:
        content = getattr(message_or_text, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(getattr(item, "text", "") or item))
            text = "\n".join(part for part in parts if part)
        else:
            text = str(content or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*(?:final answer|final)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class CodeRepair:
    def __init__(self):
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.model = MODEL_NAME
        self.MAX_INPUT_LENGTH = int(MAX_INPUT_LENGTH)
        self.client = openai.OpenAI(
            api_key=BAILIAN_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=float(os.getenv("KGVERI_LLM_TIMEOUT", "360")),
            max_retries=1,
        )
        self.prompt_builder = PromptBuilder(max_input_length=self.MAX_INPUT_LENGTH)
        self._kg = None  # lazy KG connection for debug trace
        self._kg_database_name = "debug_trace"
        self._llm_usage_events = []
        self._usage_ledger_path = ""
        self._current_attempt = 0
        self.debug_policy = self._normalize_debug_policy(
            os.getenv("KGVERI_DEBUG_POLICY", "diagnosis_kg")
        )
        self.internal_validator = os.getenv(
            "KGVERI_INTERNAL_VALIDATOR", "layered"
        ).strip().lower()

    @staticmethod
    def _normalize_debug_policy(value):
        aliases = {
            "raw": "raw_feedback",
            "raw_feedback": "raw_feedback",
            "diagnosis": "diagnosis",
            "diagnosis_only": "diagnosis",
            "full": "diagnosis_kg",
            "diagnosis_kg": "diagnosis_kg",
        }
        policy = aliases.get(str(value or "").strip().lower())
        if not policy:
            raise ValueError(
                "KGVERI_DEBUG_POLICY must be raw_feedback, diagnosis, or diagnosis_kg"
            )
        return policy

    @staticmethod
    def _estimate_tokens(text):
        """Conservative tokenizer-free estimate used only when providers omit usage."""
        return int(math.ceil(len(str(text or "")) / 4.0))

    def _usage_summary(self):
        prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in self._llm_usage_events)
        completion_tokens = sum(int(item.get("completion_tokens") or 0) for item in self._llm_usage_events)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "api_calls": len(self._llm_usage_events),
            "contains_estimates": any(bool(item.get("estimated")) for item in self._llm_usage_events),
            "estimation_method": "ceil(UTF-8 text characters / 4)" if any(
                bool(item.get("estimated")) for item in self._llm_usage_events
            ) else "provider-reported usage",
        }

    def _write_usage_ledger(self):
        if not self._usage_ledger_path:
            return
        self._write_json(
            self._usage_ledger_path,
            {
                "schema_version": "kgveri-llm-usage-v1",
                "model": self.model,
                "debug_policy": self.debug_policy,
                "summary": self._usage_summary(),
                "events": self._llm_usage_events,
            },
        )

    def _record_llm_usage(self, prompt, completion, call_kind, usage=None, *, shared_seed=False):
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        exact = prompt_tokens is not None and completion_tokens is not None
        if not exact:
            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(completion)
        event = {
            "attempt": self._current_attempt,
            "call_kind": call_kind,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
            "estimated": not exact,
            "shared_seed": bool(shared_seed),
        }
        self._llm_usage_events.append(event)
        self._write_usage_ledger()
        return event

    def _get_kg(self) -> KnowledgeGraph | None:
        if self._kg is None:
            try:
                self._kg = KnowledgeGraph(
                    NEO4J_URI,
                    NEO4J_USER,
                    NEO4J_PASSWORD,
                    self._kg_database_name,
                )
            except Exception as exc:
                _log("debug-trace", f"KG connection failed: {exc}")
                self._kg = None
        return self._kg

    def get_completion(self, prompt, stream=False, call_kind="unspecified"):
        messages = [{"role": "user", "content": prompt}]
        if stream and os.getenv("REPAIR_STREAM", "1").lower() in {"0", "false", "no"}:
            stream = False
        request = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(os.getenv("KGVERI_LLM_MAX_TOKENS", str(MAX_TOKENS))),
            stream=stream,
        )
        if os.getenv("KGVERI_DISABLE_THINKING", "0").lower() in {"1", "true", "yes"}:
            request["extra_body"] = {"thinking": {"type": "disabled"}}

        # Transport failures are infrastructure noise rather than repair attempts.
        # Retry the identical request without recording tokens until a provider
        # response is received; semantic/format failures remain part of the run.
        max_api_attempts = max(1, int(os.getenv("KGVERI_LLM_API_ATTEMPTS", "3")))
        for api_attempt in range(1, max_api_attempts + 1):
            try:
                response = self.client.chat.completions.create(**request)
                if stream:
                    return response
                content = _final_response_content(response.choices[0].message)
                self._record_llm_usage(
                    prompt,
                    content,
                    call_kind,
                    getattr(response, "usage", None),
                )
                return content
            except Exception as exc:
                print(
                    "An error occurred while calling the LLM API "
                    f"(attempt {api_attempt}/{max_api_attempts}): {exc}"
                )
                if api_attempt < max_api_attempts:
                    time.sleep(min(2 ** (api_attempt - 1), 8))
        return None

    def adjust_command_indentation(self, command, indent_change):
        search_replace = command["command"].split("\n=======\n")
        search_part = search_replace[0].split("<<<<<<< SEARCH")[1].strip("\n")
        replace_part = search_replace[1].split(">>>>>>> REPLACE")[0].strip("\n")

        def adjust_lines(text):
            lines = text.splitlines()
            if indent_change < 0:
                width = abs(indent_change)
                return "\n".join(
                    line[width:] if line.startswith(" " * width) else line
                    for line in lines
                )
            return "\n".join(" " * indent_change + line for line in lines)

        adjusted_search = adjust_lines(search_part)
        adjusted_replace = adjust_lines(replace_part)

        return {
            "command": f"<<<<<<< SEARCH\n{adjusted_search}\n=======\n{adjusted_replace}\n>>>>>>> REPLACE",
            "start_line": command["start_line"],
            "end_line": command["end_line"],
        }

    def _language_prompt_parts(self, language):
        if language == "verilog":
            return {
                "language_name": "Verilog/SystemVerilog",
                "fence_language": "verilog",
                "example_file": "src/rtl/spi_flash_irq.sv",
                "example_start_line": 42,
                "example_end_line": 57,
                "example_search": "    always @(posedge clk or negedge reset_n) begin\n        if (!reset_n) begin\n            irq_sticky <= 1'b0;\n            status_fifo_nonempty <= 1'b0;\n        end else begin\n            if (clear_irq) begin\n                irq_sticky <= 1'b0;\n            end else if (rx_overrun || fifo_overflow) begin\n                irq_sticky <= 1'b1;\n            end\n\n            status_fifo_nonempty <= fifo_nonempty;\n        end\n    end",
                "example_replace": "    always @(posedge clk or negedge reset_n) begin\n        if (!reset_n) begin\n            irq_sticky <= 1'b0;\n            status_fifo_nonempty <= 1'b0;\n        end else begin\n            if (clear_irq) begin\n                irq_sticky <= 1'b0;\n                status_fifo_nonempty <= 1'b0;\n            end else begin\n                if (rx_overrun || fifo_overflow) begin\n                    irq_sticky <= 1'b1;\n                end\n                if (fifo_nonempty) begin\n                    status_fifo_nonempty <= 1'b1;\n                end\n            end\n        end\n    end",
                "use_replace_only": True,
                "language_notes": (
                    "For Verilog/SystemVerilog, preserve module ports, output kinds, signal widths, nonblocking assignments, "
                    "reset behavior, clocked vs combinational block style, and cross-module interfaces. "
                    "Prefer editing the smallest RTL unit that directly implements the failing behavior. "
                    "Treat evidence entities as constraints and do not edit them unless the issue clearly points there. "
                    "Use explicit KG relations such as READS, WRITES, DRIVES, CONNECTS, INSTANTIATES, GUARDS, "
                    "TRANSITIONS_TO, TESTS, and EXERCISES to reason about signal flow and state behavior before editing. "
                    "If the bug is a wiring issue, edit the instance connection. If it is a sticky-state or latch issue, "
                    "edit the smallest always block in place and keep the existing interface unchanged. "
                    "If a targeted testbench checks an idle output immediately after a register write, make the idle value "
                    "visible without waiting for a later transaction edge."
                ),
            }

        return {
            "language_name": "python",
            "fence_language": "python",
            "example_file": "django/core/management/commands/migrate.py",
            "example_start_line": 120,
            "example_end_line": 122,
            "example_search": "    def my_method(self):\n        result = 1 + 1\n        return result",
            "example_replace": "    def my_method(self):\n        result = 1 + 2  # Fixed the calculation\n        return result",
            "language_notes": "",
        }

    def _extract_edit_blocks(self, raw_output_text, language):
        preferred = {"python": ["python"], "verilog": ["verilog", "systemverilog", "sv"]}.get(language, [language])
        blocks = []
        fence_pattern = re.compile(r"```([A-Za-z0-9_-]*)\n(.*?)\n```", re.DOTALL)
        for match in fence_pattern.finditer(raw_output_text):
            fence_lang, block = match.group(1), match.group(2)
            if fence_lang and fence_lang.lower() not in preferred:
                continue
            if "### " not in block:
                prefix = raw_output_text[:match.start()]
                header_match = re.search(
                    r"(###\s+[^\n]+\n(?:-\s+start_line\s*:\s*\d+\n)?(?:-\s+end_line\s*:\s*\d+\n)?)\s*$",
                    prefix,
                )
                if header_match:
                    block = header_match.group(1) + block
            blocks.append(block)
        end_marker = '[/REPLACE]'
        blocks = [
            b for b in blocks
            if ('<<<<<<< SEARCH' in b or '[REPLACE]' in b or '<<<<<<< REPLACE' in b)
            and (end_marker in b or '>>>>>>> REPLACE' in b)
        ]
        return blocks or [raw_output_text]

    def _candidate_is_valid(self, code, language):
        if not code.strip():
            return False
        if language == "python":
            return check_syntax(code)
        return True

    def _load_benchmark_item(self, instance_id):
        benchmark_name = os.getenv("BENCHMARK_NAME", "verilog-local")
        if benchmark_name in {"local", "verilog-local"}:
            return get_target_sample(instance_id, benchmark_name=benchmark_name)
        return None

    def _resolve_repo_path(self, playground_dir, repo_identifier, default_root=None):
        if playground_dir is None:
            playground_dir = default_root or os.path.join(os.getcwd(), "playground")
        if repo_identifier is None:
            raise ValueError("repo_identifier is required when playground_dir is not an explicit repo root")
        return os.path.abspath(os.path.join(playground_dir, repo_identifier))

    def _clone_base_repo(self, source_repo_path):
        source_repo_path = os.path.abspath(source_repo_path)
        workdir = tempfile.mkdtemp(prefix="kgcompass_patch_", dir=os.path.dirname(source_repo_path))
        clone_path = os.path.join(workdir, os.path.basename(source_repo_path))
        subprocess.run(["git", "clone", "--quiet", source_repo_path, clone_path], check=True)
        return os.path.abspath(workdir), os.path.abspath(clone_path)

    def _clone_repo_to_attempt_root(self, source_repo_path, attempt_root):
        source_repo_path = os.path.abspath(source_repo_path)
        os.makedirs(attempt_root, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="kgcompass_patch_", dir=attempt_root)
        clone_path = os.path.join(workdir, os.path.basename(source_repo_path))
        subprocess.run(["git", "clone", "--quiet", source_repo_path, clone_path], check=True)
        return os.path.abspath(workdir), os.path.abspath(clone_path)

    def _write_json(self, path, payload):
        if os.path.basename(path) == "repair_summary.json" and isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("debug_policy", self.debug_policy)
            payload.setdefault("internal_validator", self.internal_validator)
            payload.setdefault("llm_usage", self._usage_summary())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

    def _group_entities_for_prompt(self, related_entities, language="python"):
        grouped = []
        if not related_entities:
            return grouped

        if language == "verilog":
            group_order = [
                ("fault_anchor_entities", "Fault Anchor Entities"),
                ("repair_anchor_entities", "Repair Anchor Entities"),
                ("patchable_spans", "Ranked Patchable RTL Spans"),
                ("edit_targets", "Editable RTL Targets"),
                ("direct_drivers", "Direct Drivers"),
                ("top_level_wiring", "Top-Level Wiring"),
                ("config_registers", "Config Registers"),
                ("evidence_entities", "RTL Evidence Entities"),
                ("methods", "Legacy Methods"),
                ("rtl_entities", "Legacy RTL Entities"),
                ("issues", "Related Issues"),
            ]
        else:
            group_order = [
                ("methods", "Relevant Methods"),
                ("classes", "Relevant Classes"),
                ("issues", "Related Issues"),
            ]

        for group_name, _label in group_order:
            if group_name == "edit_targets" and related_entities.get("patchable_spans"):
                continue
            entities = related_entities.get(group_name) or []
            if not entities:
                continue
            deduped = {}
            for entity in entities:
                key = (
                    entity.get("name"),
                    entity.get("signature"),
                    entity.get("file_path"),
                    entity.get("start_line"),
                    entity.get("end_line"),
                )
                deduped[key] = entity
            sorted_entities = sorted(deduped.values(), key=lambda item: context_entity_sort_key(item, group_name))
            grouped.append((group_name, sorted_entities))
        return grouped

    def _truncate_entity_groups(self, grouped_entities, language="python"):
        if language == "verilog":
            limits = {
                "fault_anchor_entities": 6,
                "repair_anchor_entities": 6,
                "patchable_spans": 10,
                "edit_targets": 8,
                "direct_drivers": 6,
                "top_level_wiring": 6,
                "config_registers": 4,
            "evidence_entities": 10,
            "methods": 8,
            "rtl_entities": 10,
            "issues": 4,
            }
        else:
            limits = {
                "methods": 10,
                "classes": 6,
                "issues": 4,
            }
        truncated = []
        for group_name, entities in grouped_entities:
            limit = limits.get(group_name, 8)
            truncated.append((group_name, entities[:limit]))
        return truncated

    def _render_grouped_entities(self, grouped_entities, header_map, show_path_groups=None):
        sections = []
        show_path_groups = set(show_path_groups or [])
        for group_name, entities in grouped_entities:
            if not entities:
                continue
            sections.append(f"## {header_map.get(group_name, group_name)}")
            for entity in entities:
                sections.append(format_entity_content(entity, show_path=group_name in show_path_groups).rstrip())
        return "\n".join(sections)

    def _line_numbered_excerpt(self, lines, start_line, end_line):
        excerpt = []
        for line_no in range(start_line, end_line + 1):
            excerpt.append(f"{line_no}: {lines[line_no - 1]}")
        return "\n".join(excerpt)

    def _normalize_repo_relative_verilog_path(self, repo_path, file_path):
        normalized = str(file_path or "").replace("\\", "/").strip().strip("'\"`")
        if not normalized or not re.search(r"\.(?:sv|v|svh|vh)$", normalized, re.IGNORECASE):
            return ""
        if normalized.startswith("./"):
            normalized = normalized[2:]
        repo_root = os.path.abspath(repo_path).replace("\\", "/").rstrip("/") if repo_path else ""
        candidates = [normalized]
        if not os.path.isabs(normalized) and os.path.exists("/" + normalized):
            candidates.append("/" + normalized)
        for candidate in candidates:
            candidate_norm = candidate.replace("\\", "/")
            if repo_root and repo_root in candidate_norm:
                return candidate_norm.split(repo_root, 1)[1].lstrip("/")
            if repo_path and os.path.isabs(candidate_norm):
                abs_candidate = os.path.abspath(candidate_norm)
                repo_abs = os.path.abspath(repo_path)
                if abs_candidate.startswith(repo_abs + os.sep):
                    return os.path.relpath(abs_candidate, repo_abs).replace("\\", "/")
        cleaned = normalized.lstrip("/")
        if repo_path and os.path.exists(os.path.join(repo_path, *cleaned.split("/"))):
            return cleaned
        return cleaned

    def _resolve_patch_target_path(self, repo_path, edited_file, language):
        edited_file = str(edited_file or "").replace("\\", "/").strip().lstrip("./")
        if not edited_file:
            return edited_file, ""
        direct_path = os.path.join(repo_path, edited_file)
        if os.path.exists(direct_path):
            return edited_file, ""
        if language != "verilog" or not re.search(r"\.(?:sv|svh|v|vh)$", edited_file, re.IGNORECASE):
            return edited_file, ""

        basename = os.path.basename(edited_file)
        if not basename:
            return edited_file, ""

        candidates = []
        for root, dirs, files in os.walk(repo_path):
            rel_root = os.path.relpath(root, repo_path).replace("\\", "/")
            dirs[:] = [
                d for d in dirs
                if not re.search(
                    r"(^|/)(\.git|build|out|bazel-out|tb|test|tests|sim|verif|formal|verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages|benchmark|benchmarks|autogen|generated)(/|$)",
                    f"{rel_root}/{d}".lower(),
                )
            ]
            if basename not in files:
                continue
            rel = os.path.relpath(os.path.join(root, basename), repo_path).replace("\\", "/")
            if is_forbidden_file(rel):
                continue
            candidates.append(rel)

        suffix_matches = [
            rel for rel in candidates
            if rel.endswith("/" + edited_file) or rel.endswith(edited_file)
        ]
        if len(suffix_matches) == 1:
            resolved = suffix_matches[0]
            return resolved, f"Normalized patch target path {edited_file} -> {resolved} by unique suffix match."
        if len(candidates) == 1:
            resolved = candidates[0]
            return resolved, f"Normalized patch target path {edited_file} -> {resolved} by unique basename match."
        return edited_file, ""

    def _extract_verilog_issue_file_hints(self, problem_statement, repo_path="", related=None, limit=8):
        text = problem_statement or ""
        hints = []
        seen = set()

        def add_hint(file_path):
            normalized = self._normalize_repo_relative_verilog_path(repo_path, file_path)
            if not normalized or normalized in seen:
                return
            lowered = normalized.lower()
            if (
                re.search(r"(^|/)(tb|test|tests|dv|sim|verif|formal|benchmark|benchmarks|out|build|autogen|generated)(/|$)", lowered)
                or re.search(r"(^|/)(verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages)(/|$)", lowered)
                or re.search(r"(^|/)vendor/.*(?:dv|test|uvm)", lowered)
            ):
                return
            seen.add(normalized)
            hints.append(normalized)

        related_files = []
        for group_entities in (related or {}).values():
            for entity in group_entities or []:
                file_path = str(entity.get("file_path") or "").replace("\\", "/").strip()
                if file_path:
                    related_files.append(file_path)
        related_files = _unique_keep_order(related_files)

        file_patterns = [
            r"`([^`]+\.(?:sv|svh|v|vh))`",
            r"(?<![\w./-])([A-Za-z0-9_./-]+\.(?:sv|svh|v|vh))\b",
        ]
        raw_hints = []
        for pattern in file_patterns:
            raw_hints.extend(match.group(1).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))

        module_like_tokens = []
        for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{4,})\b", text):
            token = token.strip()
            if not token or token.lower() in {
                "module",
                "signal",
                "output",
                "input",
                "always",
                "assign",
                "reset",
                "clock",
                "error",
                "issue",
                "logic",
                "state",
                "register",
            }:
                continue
            if "_" not in token and not any(ch.isdigit() for ch in token):
                continue
            module_like_tokens.append(token)
        module_like_tokens = _unique_keep_order(module_like_tokens)[:12]

        def repo_file_candidates():
            if not repo_path or not os.path.isdir(repo_path):
                return []
            candidates = []
            for root, _, files in os.walk(repo_path):
                rel_root = os.path.relpath(root, repo_path).replace("\\", "/")
                if (
                    re.search(r"(^|/)(\.git|tb|test|tests|dv|sim|verif|formal|benchmark|benchmarks|out|build|bazel-out|autogen|generated)(/|$)", rel_root.lower())
                    or re.search(r"(^|/)(verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages)(/|$)", rel_root.lower())
                    or re.search(r"(^|/)vendor/.*(?:dv|test|uvm)", rel_root.lower())
                ):
                    continue
                for filename in files:
                    if not re.search(r"\.(?:sv|svh|v|vh)$", filename, re.IGNORECASE):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, filename), repo_path).replace("\\", "/")
                    candidates.append(rel_path)
            return _unique_keep_order(candidates)

        repo_candidates = repo_file_candidates()
        issue_tokens = []
        stopwords = {
            "module", "signal", "output", "input", "always", "assign", "reset", "clock",
            "error", "issue", "logic", "state", "register", "instruction", "instructions",
            "should", "would", "could", "there", "their", "when", "then", "with", "from",
            "this", "that", "have", "has", "into", "only", "also", "fail", "fails",
            "failing", "failure", "expected", "actual", "test", "case", "repo", "verilog",
        }
        for raw in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", text):
            token = raw.lower()
            if token in stopwords:
                continue
            issue_tokens.append(token)
        issue_tokens = _unique_keep_order(issue_tokens)[:24]
        csr_issue = bool(re.search(r"\b(?:csr|csrs|control\s+and\s+status|trap)\b", text, flags=re.IGNORECASE))
        csr_terms = {
            "mstatus", "misa", "mtvec", "mscratch", "mepc", "mcause", "mtval",
            "mhartid", "mcountinhibit", "mcycle", "minstret", "dcsr", "dpc",
            "dscratch0", "dscratch1",
        }

        for raw_hint in raw_hints:
            normalized = raw_hint.replace("\\", "/").strip("`'\" ")
            if not normalized:
                continue
            if re.search(r"\.(?:sv|svh|v|vh)$", normalized, re.IGNORECASE):
                if repo_path:
                    abs_hint = os.path.abspath(os.path.join(repo_path, *normalized.split("/")))
                    if os.path.exists(abs_hint):
                        add_hint(os.path.relpath(abs_hint, repo_path).replace("\\", "/"))
                        continue
                basename = os.path.splitext(os.path.basename(normalized))[0].lower()
                for candidate in [*related_files, *repo_candidates]:
                    candidate_base = os.path.splitext(os.path.basename(candidate))[0].lower()
                    if candidate_base == basename:
                        add_hint(candidate)
                        break
                else:
                    add_hint(normalized)
                continue

            token_lower = normalized.lower()
            for candidate in [*related_files, *repo_candidates]:
                candidate_base = os.path.splitext(os.path.basename(candidate))[0].lower()
                candidate_path = candidate.lower()
                if (
                    candidate_base == token_lower
                    or token_lower in candidate_base
                    or candidate_base in token_lower
                    or token_lower in candidate_path
                ):
                    add_hint(candidate)
                    break

        for token in module_like_tokens:
            token_lower = token.lower()
            for candidate in [*related_files, *repo_candidates]:
                candidate_base = os.path.splitext(os.path.basename(candidate))[0].lower()
                candidate_path = candidate.lower()
                if (
                    candidate_base == token_lower
                    or token_lower in candidate_base
                    or candidate_base in token_lower
                    or token_lower in candidate_path
                ):
                    add_hint(candidate)
                    break
            if len(hints) >= limit:
                break

        if len(hints) < limit and repo_candidates and issue_tokens:
            ranked = []
            for candidate in repo_candidates:
                path_lower = candidate.lower()
                base_lower = os.path.splitext(os.path.basename(candidate))[0].lower()
                split_tokens = set(re.split(r"[^a-z0-9]+", path_lower))
                score = 0
                for token in issue_tokens:
                    if token in split_tokens:
                        score += 4
                    elif token in base_lower:
                        score += 3
                    elif token in path_lower:
                        score += 1
                    if csr_issue and base_lower in {"ibex_cs_registers", "cs_registers"}:
                        if token in {"csr", "csrs", "register", "registers"} or token in csr_terms:
                            score += 8
                        elif token in {"trap", "machine", "privileged"}:
                            score += 3
                for first, second in zip(issue_tokens, issue_tokens[1:]):
                    if first in base_lower and second in base_lower:
                        score += 3
                if csr_issue and base_lower in {"ibex_cs_registers", "cs_registers"}:
                    score += 10
                if "rtl/" in path_lower:
                    score += 1
                if score:
                    ranked.append((score, candidate))
            for _, candidate in sorted(ranked, key=lambda item: (-item[0], len(item[1]), item[1]))[:limit]:
                add_hint(candidate)
                if len(hints) >= limit:
                    break

        return hints[:limit]

    def _collect_candidate_source_context(
        self,
        repo_path,
        related,
        language="python",
        max_files=8,
        full_file_limit=2500,
        window=160,
        fault_anchor_entities=None,
        repair_anchor_entities=None,
        fault_anchor_spans=None,
        preferred_files=None,
        return_bundle=False,
    ):
        if language != "verilog" or not repo_path:
            return {"text": "", "manifest": {}} if return_bundle else ""

        manager = SourceContextManager(repo_path)
        bundle = manager.build_initial_context(
            related_entities=related,
            preferred_files=preferred_files or [],
            fault_anchor_entities=fault_anchor_entities or [],
            repair_anchor_entities=repair_anchor_entities or [],
            fault_anchor_spans=fault_anchor_spans or [],
            max_files=max_files,
            full_file_limit=full_file_limit,
            max_full_files=4,
            window=window,
        )
        if return_bundle:
            return {"text": bundle.text, "manifest": bundle.manifest}
        return bundle.text

    def _extract_json_object(self, text):
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _collect_debug_expanded_context(
        self,
        repo_path,
        diagnosis,
        fallback_context="",
        failure_file_path="",
        anchor_spans=None,
        max_files=10,
        full_file_limit=3500,
        window=180,
        return_bundle=False,
        source_context_state=None,
        max_total_chars=50000,
        prefer_kg_targets=False,
    ):
        if not repo_path:
            payload = {"text": fallback_context, "manifest": {"fallback_used": True}}
            return payload if return_bundle else fallback_context

        requested = []
        failure_file_path = normalize_repo_relative_path(repo_path, failure_file_path)
        if failure_file_path:
            requested.append(failure_file_path)
        if prefer_kg_targets:
            for path in diagnosis.get("kg_suggested_files") or []:
                path = normalize_repo_relative_path(repo_path, path)
                if path:
                    requested.append(path)
        for path in diagnosis.get("need_files") or []:
            path = normalize_repo_relative_path(repo_path, path)
            if path:
                requested.append(path)
        for item in diagnosis.get("edit_candidates") or []:
            if isinstance(item, dict):
                path = normalize_repo_relative_path(repo_path, item.get("file") or item.get("file_path"))
                if path:
                    requested.append(path)
        if not prefer_kg_targets:
            for path in diagnosis.get("kg_suggested_files") or []:
                path = normalize_repo_relative_path(repo_path, path)
                if path:
                    requested.append(path)
        state = source_context_state or SourceContextState()
        state.add_requested_files(_unique_keep_order(requested))
        manager = SourceContextManager(repo_path)
        persistent_max_files = max(max_files, len(state.expanded_files), 20)
        bundle = manager.build_expanded_context(
            requested_files=state.expanded_files,
            exclude_files=state.initial_full_files,
            fallback_text="",
            max_files=persistent_max_files,
            full_file_limit=full_file_limit,
            max_full_files=persistent_max_files,
            window=window,
            anchor_spans=anchor_spans or [],
            max_total_chars=max_total_chars,
        )
        state.set_rendered_expanded_files(
            item.get("file_path")
            for item in (bundle.manifest.get("files") or [])
            if item.get("file_path")
        )
        bundle.manifest["persistent_state"] = state.to_manifest()
        if return_bundle:
            return {"text": bundle.text, "manifest": bundle.manifest}
        return bundle.text

    def _evidence_anchor_spans(self, evidence, context_lines=80):
        spans = []
        for item in evidence or []:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file") or item.get("file_path")
            line = item.get("line")
            try:
                line_no = int(line)
            except (TypeError, ValueError):
                continue
            if not file_path or line_no <= 0:
                continue
            spans.append({
                "file": file_path,
                "start_line": max(1, line_no - context_lines),
                "end_line": line_no + context_lines,
                "reason": item.get("type") or "validation_evidence",
            })
        return spans[:12]

    def _format_debug_kg_context(self, related, analysis=None, limit_per_group=6):
        if not related:
            return ""
        digest = build_verilog_fault_neighborhood(
            related,
            issue_text="",
            analysis=analysis,
            repo_root="",
            limit_per_group=limit_per_group,
            max_paths=10,
        )
        return digest.get("text", "")

    def _format_source_context_manifest_for_prompt(self, manifest, max_files=16, max_index=80):
        if not isinstance(manifest, dict):
            return ""
        lines = []
        shown = manifest.get("files") or []
        if shown:
            lines.append("Shown source files:")
            for item in shown[:max_files]:
                lines.append(
                    "- {file} mode={mode} lines={lines} reason={reason}".format(
                        file=item.get("file_path", ""),
                        mode=item.get("mode", ""),
                        lines=item.get("shown_lines", ""),
                        reason=item.get("reason", ""),
                    )
                )
        persistent = manifest.get("persistent_state") or {}
        loaded_files = persistent.get("all_loaded_files") or []
        requested_files = persistent.get("requested_files") or []
        if loaded_files:
            lines.append("Persistent source context state:")
            lines.append(f"- loaded_files={', '.join(loaded_files[:max_files])}")
            if requested_files:
                lines.append(f"- requested_in_prior_attempts={', '.join(requested_files[:max_files])}")
            lines.append("- Do not request a loaded file again unless a different code region is required.")
        nav = manifest.get("issue_path_navigation") or {}
        if nav:
            lines.append("SnapshotKG issue-path navigation:")
            capabilities = nav.get("capabilities") or {}
            lines.append(f"- cached_file_entity_lookup={bool(capabilities.get('cached_file_entity_lookup'))}")
            lines.append(f"- cross_file_signal_propagation={bool(capabilities.get('cross_file_signal_propagation'))}")
            files = nav.get("files") or []
            if files:
                lines.append(f"- issue_path_files={', '.join(files[:max_files])}")
        index = manifest.get("file_index") or []
        if index:
            lines.append("Candidate repository RTL file index (request unseen files from this index only when current snippets are insufficient):")
            for item in index[:max_index]:
                modules = ", ".join(item.get("modules") or [])
                detail = f" modules={modules}" if modules else ""
                lines.append(
                    f"- {item.get('file_path', '')} lines={item.get('line_count', '?')}{detail} summary={item.get('summary', '')}"
                )
        return "\n".join(lines)

    def _run_failure_analysis_and_expand(
        self,
        *,
        problem_statement,
        candidate_source_context,
        error_feedback,
        candidate_repo_path,
        raw_validation_feedback="",
        source_context_manifest=None,
        persistent_source_context="",
        source_context_state=None,
        previous_evidence=None,
        instance_id="",
        issue_mentioned_files=None,
        enable_kg_backtracking=True,
    ):
        """Run Failure Analysis Agent + KG re-slice + context expansion.

        Returns (diagnosis, expanded_context, debug_kg_trace_text, expanded_manifest, updated_issue_paths, debug_trace_payload).
        """
        # ── Step 1: KG re-slice (failure-conditioned backward trace) ──
        debug_kg_trace_text = ""
        debug_trace_payload = {}
        if enable_kg_backtracking and error_feedback:
            kg = self._get_kg()
            if kg:
                try:
                    localization_feedback = evidence_localization_text(previous_evidence or []) or error_feedback
                    debug_trace = build_debug_kg_trace(
                        localization_feedback, kg=kg,
                        instance_id=instance_id,
                        max_signals=10,
                        issue_text=problem_statement,
                        issue_mentioned_files=issue_mentioned_files or [],
                        # Debugging follows observed RTL signals to their
                        # writers/drivers. It does not rerank the initial
                        # Issue--PatchableSpan localization result.
                        enable_rerank=False,
                    )
                    debug_trace_payload = debug_trace
                    debug_kg_trace_text = debug_trace.get("text", "")
                    _log("debug-trace", f"KG trace built: {len(debug_kg_trace_text)} chars")
                except Exception as exc:
                    _log("debug-trace", f"KG trace build failed: {exc}")

        # ── Step 2: Failure Analysis Agent (structured diagnosis) ─────
        diagnosis_manifest = dict(source_context_manifest or {})
        if source_context_state is not None:
            diagnosis_manifest["persistent_state"] = source_context_state.to_manifest()
        diagnosis_prompt = self.prompt_builder.build_debug_diagnosis_prompt(
            problem_statement=problem_statement,
            candidate_source_context=candidate_source_context,
            error_feedback=error_feedback,
            raw_validation_output=raw_validation_feedback,
            debug_kg_trace=debug_kg_trace_text,
            source_context_manifest=self._format_source_context_manifest_for_prompt(diagnosis_manifest),
            persistent_source_context=persistent_source_context,
            max_files=20,
        )
        _log("failure-analysis", "requesting structured diagnosis from LLM")
        diagnosis_text = self.get_completion(
            diagnosis_prompt,
            stream=False,
            call_kind="failure_diagnosis",
        ) or ""
        diagnosis = self._extract_json_object(diagnosis_text)
        if not diagnosis:
            _log("failure-analysis", "diagnosis JSON parse failed; using fallback diagnosis")
            diagnosis = {
                "failure_hypothesis": error_feedback.strip()[:200] if error_feedback else "debug diagnosis unavailable",
                "failure_class": "unknown",
                "retry_action": "expand_context",
                "need_files": [],
                "edit_candidates": [],
                "do_not_change": ["testbench", "unrelated RTL"],
            }
        else:
            requested = diagnosis.get("need_files") or []
            hypothesis = str(diagnosis.get("failure_hypothesis") or "")
            _log("failure-analysis", f"hypothesis={_compact_text(hypothesis, 180)}")
            _log("failure-analysis", f"requested_files={requested[:5]}")

        kg_suggested_files = []
        for item in debug_trace_payload.get("suggested_edit_targets") or []:
            if isinstance(item, dict) and item.get("file_path"):
                kg_suggested_files.append(item.get("file_path"))
        if kg_suggested_files:
            diagnosis["kg_suggested_files"] = _unique_keep_order([
                *list(diagnosis.get("kg_suggested_files") or []),
                *kg_suggested_files,
            ])[:12]
        updated_issue_paths = (
            debug_trace_payload.get("dynamic_anchor_to_patchable_span_paths")
            or debug_trace_payload.get("updated_issue_to_entity_paths")
            or []
        )
        if updated_issue_paths:
            diagnosis["kg_updated_issue_paths"] = updated_issue_paths[:12]

        # ── Step 3: context expansion ─────────────────────────────────
        expanded_bundle = self._collect_debug_expanded_context(
            candidate_repo_path,
            diagnosis,
            fallback_context=candidate_source_context,
            failure_file_path="",
            anchor_spans=self._evidence_anchor_spans(previous_evidence or []),
            return_bundle=True,
            source_context_state=source_context_state,
            max_total_chars=30000 if enable_kg_backtracking else 50000,
            prefer_kg_targets=enable_kg_backtracking,
        )
        expanded_context = expanded_bundle.get("text", "") if isinstance(expanded_bundle, dict) else str(expanded_bundle or "")
        expanded_manifest = expanded_bundle.get("manifest", {}) if isinstance(expanded_bundle, dict) else {}
        _log("failure-analysis", f"expanded_context_chars={len(expanded_context or '')}")

        return diagnosis, expanded_context, debug_kg_trace_text, expanded_manifest, updated_issue_paths, debug_trace_payload

    def _collect_prompt_context(self, locate_result, verification_cfg, language, repo_path=None):
        related = locate_result.get("related_entities", {}) if locate_result else {}
        if language == "verilog":
            related = normalize_verilog_related_entities(related)
        issues = related.get("issues") or []
        sorted_issues = sorted(issues, key=lambda x: x.get("similarity", 0), reverse=True)

        problem_statement = ""
        if sorted_issues:
            problem_statement += f"### {sorted_issues[0].get('title', 'N/A')}\n{sorted_issues[0].get('content', '')}"
            if len(sorted_issues) > 1 and sorted_issues[1].get("similarity", 0) > 0.1:
                problem_statement += f"\n\n### {sorted_issues[1].get('title', 'N/A')}\n{sorted_issues[1].get('content', '')}"
        if not problem_statement:
            problem_statement = locate_result.get("issue", "No issue description provided.") if locate_result else "No issue description provided."

        edit_targets = related.get("edit_targets") or related.get("direct_drivers") or related.get("methods") or []
        evidence_entities = related.get("evidence_entities") or related.get("rtl_entities") or []

        edit_lines = []
        for entity in edit_targets[:8]:
            edit_lines.append(
                f"- {entity.get('file_path', '')}:{entity.get('start_line')}-{entity.get('end_line')} "
                f"{entity.get('signature') or entity.get('name', '')} "
                f"[{entity.get('verilog_kind') or entity.get('type') or 'unknown'}]"
            )

        evidence_lines = []
        for entity in evidence_entities[:10]:
            evidence_lines.append(
                f"- {entity.get('file_path', '')}:{entity.get('start_line')}-{entity.get('end_line')} "
                f"{entity.get('signature') or entity.get('name', '')} "
                f"[{entity.get('verilog_kind') or entity.get('type') or 'unknown'}]"
            )

        issue_lines = []
        for issue in sorted_issues[:3]:
            title = issue.get("title", "N/A")
            content = _compact_text(issue.get("content", ""), 500)
            issue_lines.append(f"- {title}: {content}")

        files = []
        edit_target_kinds = []
        evidence_kinds = []
        for entity in edit_targets + evidence_entities + issues:
            file_path = entity.get("file_path")
            if file_path:
                files.append(file_path)
            kind = str(entity.get("verilog_kind") or entity.get("type") or "").strip()
            if entity in edit_targets and kind:
                edit_target_kinds.append(kind)
            if entity in evidence_entities and kind:
                evidence_kinds.append(kind)
        files = _unique_keep_order(files)
        if language == "verilog":
            files = _unique_keep_order(
                self._normalize_repo_relative_verilog_path(repo_path or "", file_path)
                for file_path in files
            )
            files = [file_path for file_path in files if file_path and not is_forbidden_file(file_path)]
        edit_target_kinds = _unique_keep_order([kind.lower() for kind in edit_target_kinds if kind])
        evidence_kinds = _unique_keep_order([kind.lower() for kind in evidence_kinds if kind])

        localized_summary = []
        if issue_lines:
            localized_summary.append("Related issues:")
            localized_summary.extend(issue_lines)
        if edit_lines:
            localized_summary.append("Primary edit targets:")
            localized_summary.extend(edit_lines)
        if language == "verilog":
            for label, group_name in [
                ("Direct drivers:", "direct_drivers"),
                ("Top-level wiring:", "top_level_wiring"),
                ("Config registers:", "config_registers"),
                ("Repair anchors:", "repair_anchor_entities"),
            ]:
                group_entities = related.get(group_name) or []
                if not group_entities:
                    continue
                localized_summary.append(label)
                for entity in group_entities[:6]:
                    localized_summary.append(
                        f"- {entity.get('file_path', '')}:{entity.get('start_line')}-{entity.get('end_line')} "
                        f"{entity.get('signature') or entity.get('name', '')} "
                        f"[{entity.get('repair_role') or entity.get('verilog_kind') or 'unknown'}]"
                    )
        issue_mentioned_files = []
        if language == "verilog":
            issue_mentioned_files = self._extract_verilog_issue_file_hints(
                problem_statement,
                repo_path=repo_path or "",
                related=related,
            )
            if issue_mentioned_files:
                localized_summary.append(f"Issue-mentioned files: {', '.join(issue_mentioned_files[:8])}")
                files = _unique_keep_order([*issue_mentioned_files, *files])
                files = [file_path for file_path in files if file_path and not is_forbidden_file(file_path)]
        if evidence_lines:
            localized_summary.append("Evidence entities:")
            localized_summary.extend(evidence_lines)
        if files:
            localized_summary.append(f"Candidate files: {', '.join(files[:8])}")
        localization_summary = "\n".join(localized_summary)
        bug_kg_digest = {}
        if language == "verilog":
            bug_kg_digest = build_verilog_fault_neighborhood(
                related,
                issue_text=problem_statement,
                analysis=None,
                repo_root=repo_path or "",
            )
        issue_to_entity_paths = _merge_issue_to_entity_paths(
            locate_result.get("issue_to_entity_paths") or [],
            bug_kg_digest.get("critical_paths", []) if bug_kg_digest else [],
            bug_kg_digest.get("path_hints", []) if bug_kg_digest else [],
        )
        issue_path_nav = IssuePathNavigator(issue_to_entity_paths)
        issue_path_files = [path for path in issue_path_nav.files() if path and not is_forbidden_file(path)]
        if issue_path_files:
            files = _unique_keep_order([*files, *issue_path_files])

        fault_anchor_entities = bug_kg_digest.get("fault_anchor_entities") or []
        repair_anchor_entities = bug_kg_digest.get("repair_anchor_entities") or []
        fault_anchor_spans = bug_kg_digest.get("fault_anchor_spans") or []

        grouped_entities = self._truncate_entity_groups(self._group_entities_for_prompt(related, language=language), language=language)
        if language == "verilog" and (fault_anchor_entities or repair_anchor_entities or fault_anchor_spans):
            prompt_related = dict(related)
            prompt_related["fault_anchor_entities"] = fault_anchor_entities
            prompt_related["repair_anchor_entities"] = repair_anchor_entities
            prompt_related["edit_targets"] = prompt_related.get("edit_targets") or prompt_related.get("direct_drivers") or []
            grouped_entities = self._truncate_entity_groups(self._group_entities_for_prompt(prompt_related, language=language), language=language)
        header_map = {
            "fault_anchor_entities": "Fault Anchor Entities",
            "repair_anchor_entities": "Repair Anchor Entities",
            "patchable_spans": "Ranked Patchable RTL Spans",
            "methods": "Relevant Methods",
            "classes": "Relevant Classes",
            "edit_targets": "Editable RTL Targets",
            "direct_drivers": "Direct Drivers",
            "top_level_wiring": "Top-Level Wiring",
            "config_registers": "Config Registers",
            "edit_targets": "Editable RTL Targets",
            "rtl_entities": "Legacy RTL Entities",
            "evidence_entities": "RTL Evidence Entities",
            "issues": "Related Issues",
        }
        current_edit_targets = self._render_grouped_entities(
            [(name, entities) for name, entities in grouped_entities if name in {"fault_anchor_entities", "repair_anchor_entities", "patchable_spans", "edit_targets"}],
            header_map,
            show_path_groups={"fault_anchor_entities", "repair_anchor_entities", "patchable_spans", "edit_targets"},
        )
        if not current_edit_targets:
            current_edit_targets = self._render_grouped_entities(
                [(name, entities) for name, entities in grouped_entities if name in {"methods"}],
                header_map,
                show_path_groups={"methods"},
            )
        evidence_entities_text = self._render_grouped_entities(
            [(name, entities) for name, entities in grouped_entities if name in {"evidence_entities", "rtl_entities"}],
            header_map,
            show_path_groups={"evidence_entities", "rtl_entities"},
        )
        source_context_bundle = self._collect_candidate_source_context(
            repo_path,
            related,
            language=language,
            preferred_files=_unique_keep_order([*issue_mentioned_files, *issue_path_files]),
            return_bundle=True,
        )
        candidate_source_context = source_context_bundle.get("text", "") if isinstance(source_context_bundle, dict) else str(source_context_bundle or "")
        source_context_manifest = source_context_bundle.get("manifest", {}) if isinstance(source_context_bundle, dict) else {}
        if language == "verilog" and (fault_anchor_entities or repair_anchor_entities or fault_anchor_spans):
            source_context_bundle = self._collect_candidate_source_context(
                repo_path,
                related,
                language=language,
                fault_anchor_entities=fault_anchor_entities,
                repair_anchor_entities=repair_anchor_entities,
                fault_anchor_spans=fault_anchor_spans,
                preferred_files=_unique_keep_order([*issue_mentioned_files, *issue_path_files]),
                return_bundle=True,
            )
            candidate_source_context = source_context_bundle.get("text", "") or candidate_source_context
            source_context_manifest = source_context_bundle.get("manifest", {}) or source_context_manifest
        if isinstance(source_context_manifest, dict):
            source_context_manifest["issue_path_navigation"] = issue_path_nav.to_manifest()

        tests = []
        targeted = verification_cfg.get("targeted") or {}
        if targeted.get("name"):
            tests.append(targeted["name"])
        for section_name in ("regression", "coverage"):
            section = verification_cfg.get(section_name) or []
            if isinstance(section, dict):
                section = [section]
            for entry in section:
                if entry.get("name"):
                    tests.append(entry["name"])
        tests = _unique_keep_order(tests)

        hard_constraints = [
            "Only edit the cloned worktree under workdirs/.",
            "Never modify verilog_repair_cases/.",
            "Keep the change scope minimal and explain the failing RTL behavior directly.",
            "Preserve smoke-test behavior and avoid unrelated control-flow changes.",
        ]

        return {
            "problem_statement": problem_statement,
            "localization_summary": localization_summary,
            "current_edit_targets": current_edit_targets,
            "evidence_entities": evidence_entities_text,
            "candidate_source_context": candidate_source_context,
            "fault_anchor_entities": fault_anchor_entities,
            "repair_anchor_entities": repair_anchor_entities,
            "fault_anchor_spans": fault_anchor_spans,
            "files": files,
            "issue_mentioned_files": issue_mentioned_files,
            "tests": tests,
            "edit_target_kinds": edit_target_kinds,
            "evidence_kinds": evidence_kinds,
            "hard_constraints": hard_constraints,
            "related_entities": related,
            "bug_kg_digest": bug_kg_digest,
            "bug_kg_digest_text": bug_kg_digest.get("text", "") if bug_kg_digest else "",
            "issue_to_entity_paths": issue_to_entity_paths,
            "issue_to_entity_paths_text": _format_issue_to_entity_paths(issue_to_entity_paths),
            "issue_path_files": issue_path_files,
            "issue_path_navigation": issue_path_nav.to_manifest(),
            "source_context_manifest": source_context_manifest,
            "kg_schema": locate_result.get("kg_schema", {}),
        }

    def post_process_and_apply_patch(self, instance_id, raw_output_path, locations_dir, playground_dir=None, repo_identifier=None, repo_path=None, language="python"):
        if repo_path is None:
            repo_path = self._resolve_repo_path(playground_dir, repo_identifier, default_root=os.path.dirname(locations_dir))
        repo_path = os.path.abspath(repo_path)

        if not os.path.isdir(repo_path):
            return PatchApplicationResult(
                applied=False,
                status="repo_missing",
                reason=f"Repository path not found at {repo_path}.",
                raw_patch_path=raw_output_path,
                raw_patch_excerpt="",
            )

        with open(raw_output_path, "r", encoding="utf-8") as f:
            raw_output_text = f.read()

        raw_excerpt = _compact_text(raw_output_text, 2000)

        blocks = self._extract_edit_blocks(raw_output_text, language)
        file_to_commands = split_edit_multifile_commands(blocks)

        if not file_to_commands:
            return PatchApplicationResult(
                applied=False,
                status="no_edit_commands",
                reason="No edit commands found in LLM output.",
                raw_patch_path=raw_output_path,
                raw_patch_excerpt=raw_excerpt,
                notes=["no_edit_commands"],
            )

        applied_any = False
        failed_any = False
        edited_files = []
        diff_paths = []
        notes = []
        for file_path_str, edit_commands in file_to_commands.items():
            try:
                edited_file = ast.literal_eval(file_path_str)
            except Exception:
                notes.append(f"Could not parse file path: {file_path_str}")
                failed_any = True
                continue

            if edited_file.startswith(".."):
                edited_file = "/".join(edited_file.split("/")[2:])
            edited_file, path_note = self._resolve_patch_target_path(repo_path, edited_file, language)
            if path_note:
                notes.append(path_note)

            full_file_path = os.path.join(repo_path, edited_file)
            if not os.path.exists(full_file_path):
                notes.append(f"File to be edited not found: {full_file_path}")
                failed_any = True
                continue

            with open(full_file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            require_line_range = language == "verilog"
            missing_line_ranges = [
                cmd for cmd in edit_commands
                if cmd.get("start_line") is None or cmd.get("end_line") is None
            ]
            if require_line_range and missing_line_ranges:
                notes.append(f"Missing start_line/end_line metadata for {edited_file}.")
                failed_any = True
                continue

            new_content = parse_diff_edit_commands_strict(
                edit_commands,
                original_content,
                require_line_range=require_line_range,
            )

            if new_content == original_content or not self._candidate_is_valid(new_content, language):
                if not require_line_range:
                    for indent_change in [-4, 4, -8, 8]:
                        adjusted_commands = [self.adjust_command_indentation(cmd, indent_change) for cmd in edit_commands]
                        adjusted_content = parse_diff_edit_commands_strict(adjusted_commands, original_content)
                        if adjusted_content != original_content and self._candidate_is_valid(adjusted_content, language):
                            new_content = adjusted_content
                            break

            if new_content == original_content or not self._candidate_is_valid(new_content, language):
                if not require_line_range:
                    new_content = parse_diff_edit_commands_strict(edit_commands, original_content, only_one_replace=True)
                    if new_content == original_content or not self._candidate_is_valid(new_content, language):
                        for indent_change in [-4, 4, -8, 8]:
                            adjusted_commands = [self.adjust_command_indentation(cmd, indent_change) for cmd in edit_commands]
                            adjusted_content = parse_diff_edit_commands_strict(adjusted_commands, original_content, only_one_replace=True)
                            if adjusted_content != original_content and self._candidate_is_valid(adjusted_content, language):
                                new_content = adjusted_content
                                break

            if new_content == original_content or not self._candidate_is_valid(new_content, language):
                if require_line_range:
                    notes.append(f"Failed to apply exact line-range patch for {edited_file}.")
                else:
                    notes.append(f"Failed to generate a valid patch for {edited_file}.")
                failed_any = True
                continue

            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            diff = difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{edited_file}",
                tofile=f"b/{edited_file}",
            )
            patch_content = "".join(diff)

            diff_patch_dir = os.path.join(os.path.dirname(raw_output_path), "diff_patches")
            os.makedirs(diff_patch_dir, exist_ok=True)
            sanitized_file_path = edited_file.replace("/", "_")
            diff_file_path = os.path.join(diff_patch_dir, f"{instance_id}_{sanitized_file_path}.diff")
            abs_diff_file_path = os.path.abspath(diff_file_path)
            with open(abs_diff_file_path, "w", encoding="utf-8") as f:
                f.write(patch_content)

            notes.append(f"Applied patch for {edited_file}.")

            applied_any = True
            edited_files.append(edited_file)
            diff_paths.append(abs_diff_file_path)

        if applied_any and not failed_any:
            return PatchApplicationResult(
                applied=True,
                status="applied",
                reason="patch applied",
                edited_files=_unique_keep_order(edited_files),
                diff_paths=_unique_keep_order(diff_paths),
                notes=_unique_keep_order(notes),
                raw_patch_path=raw_output_path,
                raw_patch_excerpt=raw_excerpt,
            )

        status = "patch_apply_failed" if failed_any else "no_changes"
        reason = "; ".join(notes) if notes else "Failed to generate a valid patch."
        return PatchApplicationResult(
            applied=False,
            status=status,
            reason=reason,
            edited_files=_unique_keep_order(edited_files),
            diff_paths=_unique_keep_order(diff_paths),
            notes=_unique_keep_order(notes),
            raw_patch_path=raw_output_path,
            raw_patch_excerpt=raw_excerpt,
        )

    def _compile_check_repo(self, repo_path, files_to_check=None):
        """Compile specific files with iverilog for syntax checking.

        For each file, first compiles the ORIGINAL (git HEAD) version.
        If the original has pre-existing errors (e.g., base commit bug),
        the file is skipped entirely — those errors are not the patch's
        responsibility.  Only files where the original compiles cleanly
        are re-compiled after patching to detect newly-introduced errors.

        Non-real errors are also filtered out:
          - "No top level module" — standalone package/header
          - "Module '<name>' not found" — vendor IP not in iverilog path
          - "Unsupported: ..." / "warning:" — iverilog feature gaps

        Returns:
            (passed: bool, output: str)  — passed=True means no errors.
        """
        if not files_to_check:
            return True, ""

        abs_files = []
        for f in files_to_check:
            abs_path = os.path.abspath(os.path.join(repo_path, f))
            if os.path.exists(abs_path) and abs_path.endswith((".sv", ".v", ".svh", ".vh")):
                abs_files.append((f, abs_path))

        if not abs_files:
            return True, ""

        total_errors: list[str] = []
        skip_count = 0

        for relative_f, abs_f in abs_files:
            # ── Step 1: compile ORIGINAL (git HEAD) version ────────
            import tempfile
            try:
                original_content = subprocess.run(
                    ["git", "-C", repo_path, "show", f"HEAD:{relative_f}"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
            except Exception:
                original_content = ""

            if not original_content.strip():
                # can't get original — compile patched version as-is
                pass
            else:
                # Write original to temp file and compile it
                with tempfile.NamedTemporaryFile(
                    suffix=".sv", mode="w", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                    tmp.write(original_content)
                try:
                    orig_result = subprocess.run(
                        ["iverilog", "-g2012", "-t", "null", "-o", os.devnull, tmp_path],
                        capture_output=True, text=True, timeout=15,
                    )
                    orig_output = (orig_result.stdout + "\n" + orig_result.stderr).strip()
                except subprocess.TimeoutExpired:
                    orig_output = ""
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                # If original has real errors → skip compile check for this file
                if orig_output:
                    orig_real = self._filter_iverilog_noise(orig_output)
                    if orig_real.strip():
                        _log("compile-skip",
                             f"'{relative_f}' has pre-existing errors — skipping compile check")
                        skip_count += 1
                        continue

            # ── Step 2: compile PATCHED version ───────────────────
            cmd = ["iverilog", "-g2012", "-t", "null", "-o", os.devnull, abs_f]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                continue
            output = (result.stdout + "\n" + result.stderr).strip()
            real_errors = self._filter_iverilog_noise(output)
            if real_errors:
                total_errors.append(real_errors)

        if skip_count > 0:
            _log("compile-skip", f"skipped {skip_count} file(s) with pre-existing errors")

        result_output = "\n".join(total_errors) if total_errors else ""
        return len(total_errors) == 0, result_output

    def _filter_iverilog_noise(self, raw_output: str) -> str:
        """Remove non-real-error lines from iverilog output."""
        lines = []
        for line in raw_output.splitlines():
            s = line.strip()
            if re.search(r"No top.level module", s, re.IGNORECASE):
                continue
            if re.search(r"Module\s+'[^']+'\s+not found", s, re.IGNORECASE):
                continue
            if s.startswith("Unsupported:") or s.startswith("warning:"):
                continue
            lines.append(s)
        return "\n".join(lines).strip()

    def process_instance(self, instance_id, locations_dir, output_dir, playground_dir=None, repo_identifier=None):
        trule(f"REPAIR: {instance_id}")
        _log("start", f"instance={instance_id}")
        # ``fl.py`` stores each case in an instance-scoped Neo4j database.
        # Debug backtracking must query that exact graph rather than a shared
        # residual database from an earlier run.
        self._kg_database_name = instance_id

        location_file = os.path.join(locations_dir, f"{instance_id}.json")
        if not os.path.exists(location_file):
            tfail(f"location file not found: {location_file}")
            return

        with open(location_file, "r", encoding="utf-8") as f:
            locate_result = json.load(f)
        language = str(locate_result.get("language", "python")).lower()
        max_attempts = int(os.getenv("VERILOG_REPAIR_MAX_ATTEMPTS", "10"))
        benchmark_sample = self._load_benchmark_item(instance_id) or {}

        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.abspath(output_dir)
        run_root = os.path.abspath(os.path.dirname(output_dir))
        self._usage_ledger_path = os.path.join(run_root, "llm_usage.json")
        resume_existing = os.getenv("KGVERI_RESUME_EXISTING", "0").strip() == "1"
        self._llm_usage_events = []
        if resume_existing and os.path.isfile(self._usage_ledger_path):
            try:
                usage_doc = json.loads(Path(self._usage_ledger_path).read_text(encoding="utf-8"))
                self._llm_usage_events = list(usage_doc.get("events") or [])
            except Exception as exc:
                raise ValueError(f"cannot resume malformed usage ledger: {exc}") from exc
        self._write_usage_ledger()

        base_repo_path = self._resolve_repo_path(
            playground_dir,
            repo_identifier,
            default_root=os.path.join(os.path.dirname(locations_dir), "playground"),
        )

        prompt_context = self._collect_prompt_context(locate_result, {}, language, repo_path=base_repo_path)
        problem_statement = prompt_context["problem_statement"].replace("\r", "")
        localization_summary = prompt_context["localization_summary"].replace("\r", "")
        candidate_source_context = prompt_context.get("candidate_source_context", "")
        source_context_manifest = prompt_context.get("source_context_manifest", {})
        bug_kg_digest_text = prompt_context.get("bug_kg_digest_text", "")
        current_issue_to_entity_paths = list(prompt_context.get("issue_to_entity_paths", []) or [])
        issue_to_entity_paths_text = _format_issue_to_entity_paths(current_issue_to_entity_paths)
        hard_constraints = prompt_context["hard_constraints"]
        language_prompt_parts = self._language_prompt_parts(language)
        related_entities = prompt_context.get("related_entities") or locate_result.get("related_entities", {}) or {}

        validation_stack = None
        hwe_validator = None
        hwe_pr_number = None
        minitb_summary_text = ""
        if language == "verilog":
            repo_kind = str(benchmark_sample.get("repo") or "").strip()
            base_commit = str(benchmark_sample.get("base_commit") or "").strip()
            try:
                if repo_kind not in {"ibex", "cva6", "caliptra-rtl"}:
                    raise ValueError(
                        f"lightweight research flow supports only ibex/cva6/caliptra-rtl, got {repo_kind!r}"
                    )
                if not base_commit:
                    raise ValueError("benchmark base_commit is required for clean replay")
                if self.internal_validator == "hwe_final":
                    if repo_kind != "ibex":
                        raise ValueError("the HWE-final ablation runner currently supports Ibex only")
                    match = re.search(r"[_-]pr(\d+)", instance_id)
                    if not match:
                        raise ValueError(f"cannot extract PR number from {instance_id}")
                    hwe_pr_number = int(match.group(1))
                    hwe_validator = HweBenchValidator(repo=repo_kind)
                    if not hwe_validator.load_tb_script(hwe_pr_number):
                        raise ValueError(f"HWE-Bench final test is unavailable for PR {hwe_pr_number}")
                    self._write_json(
                        os.path.join(run_root, "internal_validation_manifest.json"),
                        {
                            "schema_version": "kgveri-hwe-final-ablation-v1",
                            "repo": repo_kind,
                            "base_commit": base_commit,
                            "pr_number": hwe_pr_number,
                            "debug_policy": self.debug_policy,
                            "oracle_assisted_ablation": True,
                            "held_out_benchmark_evaluator_used": True,
                        },
                    )
                    _log(
                        "validation",
                        f"oracle-assisted ablation ready: repo={repo_kind} HWE finaltest PR={hwe_pr_number}",
                    )
                elif self.internal_validator == "layered":
                    minitb_case, frozen_dir, visible_cases_path, minitb_case_dir = _load_frozen_minitb_binding(
                        instance_id, run_root
                    )
                    validation_stack = LightweightValidationStack(
                        repo_kind=repo_kind,
                        base_repo=base_repo_path,
                        base_commit=base_commit,
                        minitb_case=minitb_case,
                        frozen_dir=frozen_dir,
                        artifact_root=run_root,
                        related_entities=related_entities,
                        issue_text=problem_statement,
                        minitb_timeout=int(os.getenv("KGVERI_MINITB_TIMEOUT", "300")),
                        regression_timeout=int(os.getenv("KGVERI_REGRESSION_TIMEOUT", "180")),
                    )
                    baseline = validation_stack.profile_baseline()
                    if baseline.infra_error or baseline.timed_out:
                        _log("validation", "repository baseline infrastructure failure; retrying once")
                        baseline = validation_stack.profile_baseline(force=True)
                    minitb_summary_text = json.dumps(
                        validation_stack.frozen_summary,
                        indent=2,
                        ensure_ascii=False,
                    )
                    self._write_json(
                        os.path.join(run_root, "internal_validation_manifest.json"),
                        {
                            "schema_version": VALIDATION_STACK_VERSION,
                            "repo": repo_kind,
                            "base_commit": base_commit,
                            "visible_cases": str(visible_cases_path),
                            "minitb_case_dir": str(minitb_case_dir),
                            "frozen_minitb_summary": validation_stack.frozen_summary,
                            "repository_baseline": {
                                "profile_name": baseline.profile_name,
                                "command_id": baseline.command_id,
                                "returncode": baseline.returncode,
                                "passed": baseline.passed,
                                "infra_error": baseline.infra_error,
                                "failure_signatures": baseline.failure_signatures,
                                "artifact_dir": os.path.join(run_root, "repository_baseline"),
                            },
                            "held_out_benchmark_evaluator_used": False,
                        },
                    )
                    if baseline.infra_error or baseline.timed_out:
                        raise RuntimeError(
                            baseline.infra_error
                            or "repository baseline profiling timed out"
                        )
                    _log(
                        "validation",
                        f"internal stack ready: repo={repo_kind} minitb=frozen regression={baseline.profile_name}",
                    )
                else:
                    raise ValueError(
                        "KGVERI_INTERNAL_VALIDATOR must be layered or hwe_final"
                    )
            except Exception as exc:
                report = build_validation_report(
                    [
                        ValidationEvidence(
                            type="INTERNAL_VALIDATION_SETUP_FAILED",
                            severity="blocker",
                            message=str(exc),
                            raw_log=repr(exc),
                            validator="validation_setup",
                        )
                    ],
                    metadata={"validation_stack_version": VALIDATION_STACK_VERSION},
                )
                self._write_json(os.path.join(run_root, "repair_summary.json"), {
                    "instance_id": instance_id,
                    "validated": False,
                    "attempts": 0,
                    "last_feedback": report.feedback_text,
                    "validation_report": report.to_dict(),
                    "validation_stack_version": VALIDATION_STACK_VERSION,
                    "clean_replay": {"passed": False, "reason": "setup_failed"},
                })
                _log("validation", f"setup failed: {exc}")
                return

        attempts_root = os.path.join(run_root, "validation_attempts")
        os.makedirs(attempts_root, exist_ok=True)

        error_feedback = ""
        raw_validation_feedback = ""
        previous_patch = ""
        diagnosis = None
        expanded_context = ""
        expanded_context_manifest = {}
        debug_kg_trace_text = ""
        debug_trace_payload = {}
        debug_updated_issue_paths = []
        previous_evidence = []
        source_context_state = SourceContextState.from_initial_manifest(source_context_manifest)
        start_attempt = 1

        # Controlled ablations branch from one shared failed first attempt.  The
        # seed contains only artifacts produced by this runner; no reference fix
        # or hidden test source is added to either agent prompt.
        shared_seed_root = os.getenv("KGVERI_SHARED_SEED_DIR", "").strip()
        if shared_seed_root and not resume_existing:
            seed_root = os.path.abspath(shared_seed_root)
            seed_attempt = os.path.join(seed_root, "validation_attempts", "attempt_01")
            seed_report_path = os.path.join(seed_attempt, "validation_report.json")
            seed_diff_path = os.path.join(seed_attempt, "diff.patch")
            seed_raw_patch_path = os.path.join(seed_attempt, "raw.patch")
            seed_hwe_path = os.path.join(seed_attempt, "hwe_bench_validation.json")
            # A full-dataset paired run must also preserve first attempts that
            # fail before dynamic validation (for example, patch-format or
            # exact-range application failures). The validation report is the
            # only mandatory artifact; diff/finaltest evidence is used when
            # the candidate reached those stages.
            required = [seed_report_path]
            missing = [path for path in required if not os.path.isfile(path)]
            if missing:
                raise ValueError(f"shared seed is incomplete: {missing}")
            seed_report = json.loads(Path(seed_report_path).read_text(encoding="utf-8"))
            if bool(seed_report.get("passed")):
                raise ValueError("shared seed already passes; a failed first attempt is required")
            seed_hwe = (
                json.loads(Path(seed_hwe_path).read_text(encoding="utf-8"))
                if os.path.isfile(seed_hwe_path)
                else {}
            )
            local_seed_attempt = os.path.join(attempts_root, "attempt_01")
            if os.path.abspath(seed_attempt) != os.path.abspath(local_seed_attempt):
                if os.path.exists(local_seed_attempt):
                    shutil.rmtree(local_seed_attempt)
                shutil.copytree(seed_attempt, local_seed_attempt)
            error_feedback = str(seed_report.get("feedback_text") or "")
            raw_validation_feedback = str(seed_hwe.get("output") or error_feedback)
            previous_evidence = list(seed_report.get("evidence") or [])
            seed_patch_path = (
                seed_diff_path
                if os.path.isfile(seed_diff_path)
                else seed_raw_patch_path
                if os.path.isfile(seed_raw_patch_path)
                else ""
            )
            previous_patch = (
                Path(seed_patch_path).read_text(encoding="utf-8")[:12000]
                if seed_patch_path
                else ""
            )
            seed_usage_path = os.path.join(seed_root, "llm_usage.json")
            if os.path.isfile(seed_usage_path):
                seed_usage = json.loads(Path(seed_usage_path).read_text(encoding="utf-8"))
                for item in seed_usage.get("events") or []:
                    copied = dict(item)
                    copied["shared_seed"] = True
                    self._llm_usage_events.append(copied)
                self._write_usage_ledger()
            seed_hash_path = seed_patch_path or seed_report_path
            seed_sha256 = __import__("hashlib").sha256(
                Path(seed_hash_path).read_bytes()
            ).hexdigest()
            self._write_json(
                os.path.join(run_root, "shared_seed.json"),
                {
                    "seed_root": seed_root,
                    "seed_attempt": seed_attempt,
                    "seed_sha256": seed_sha256,
                    # Retained for compatibility with existing experiment
                    # summaries; for pre-validation failures this hashes the
                    # raw patch (or, if absent, the validation report).
                    "seed_diff_sha256": seed_sha256,
                    "seed_artifact": os.path.basename(seed_hash_path),
                    "seed_reached_hwe_finaltest": os.path.isfile(seed_hwe_path),
                    "seed_failure_evidence": len(previous_evidence),
                },
            )
            start_attempt = 2
            _log("attempt", f"loaded shared failed attempt from {seed_root}")

        # A selective experiment continuation must retain all paid-for calls
        # and resume from the last completed validation attempt.  This path is
        # deliberately opt-in so ordinary repair runs keep their clean-start
        # behavior.
        if resume_existing:
            completed = []
            for candidate in Path(attempts_root).glob("attempt_*"):
                match = re.fullmatch(r"attempt_(\d+)", candidate.name)
                if match and (candidate / "validation_report.json").is_file():
                    completed.append((int(match.group(1)), candidate))
            if not completed:
                raise ValueError(f"cannot resume without a completed attempt in {attempts_root}")
            last_attempt, last_attempt_dir = max(completed, key=lambda item: item[0])
            if last_attempt >= max_attempts:
                _log("attempt", f"resume is already complete at attempt={last_attempt}/{max_attempts}")
                return

            last_report = json.loads(
                (last_attempt_dir / "validation_report.json").read_text(encoding="utf-8")
            )
            if bool(last_report.get("passed")):
                raise ValueError("cannot resume a branch whose last validation attempt passed")
            last_hwe_path = last_attempt_dir / "hwe_bench_validation.json"
            last_hwe = (
                json.loads(last_hwe_path.read_text(encoding="utf-8"))
                if last_hwe_path.is_file()
                else {}
            )
            error_feedback = str(last_report.get("feedback_text") or "")
            raw_validation_feedback = str(last_hwe.get("output") or error_feedback)
            previous_evidence = list(last_report.get("evidence") or [])
            last_patch_path = (
                last_attempt_dir / "diff.patch"
                if (last_attempt_dir / "diff.patch").is_file()
                else last_attempt_dir / "raw.patch"
            )
            previous_patch = (
                last_patch_path.read_text(encoding="utf-8")[:12000]
                if last_patch_path.is_file()
                else ""
            )

            last_manifest_path = last_attempt_dir / "source_context_manifest.json"
            if last_manifest_path.is_file():
                last_manifest = json.loads(last_manifest_path.read_text(encoding="utf-8"))
                persistent = last_manifest.get("persistent_state") or {}
                source_context_state = SourceContextState(
                    initial_files=list(persistent.get("initial_files") or []),
                    initial_full_files=list(persistent.get("initial_full_files") or []),
                    requested_files=list(persistent.get("requested_files") or []),
                    rendered_expanded_files=list(persistent.get("rendered_expanded_files") or []),
                )
                restored_paths = list(last_manifest.get("issue_to_entity_paths") or [])
                if restored_paths:
                    current_issue_to_entity_paths = restored_paths
                    issue_to_entity_paths_text = _format_issue_to_entity_paths(
                        current_issue_to_entity_paths
                    )
                    prompt_context["issue_to_entity_paths"] = current_issue_to_entity_paths
                    prompt_context["issue_to_entity_paths_text"] = issue_to_entity_paths_text

            start_attempt = last_attempt + 1
            self._write_json(
                os.path.join(run_root, "resume_record.json"),
                {
                    "schema_version": "kgveri-selective-resume-v1",
                    "resumed_after_attempt": last_attempt,
                    "next_attempt": start_attempt,
                    "max_attempts": max_attempts,
                    "preserved_usage_events": len(self._llm_usage_events),
                    "debug_policy": self.debug_policy,
                },
            )
            _log(
                "attempt",
                f"resuming existing branch at attempt={start_attempt}/{max_attempts}; "
                f"preserved_api_calls={len(self._llm_usage_events)}",
            )

        for attempt_index in range(start_attempt, max_attempts + 1):
            self._current_attempt = attempt_index
            _log("attempt", f"attempt={attempt_index}/{max_attempts}")
            attempt_dir = os.path.join(attempts_root, f"attempt_{attempt_index:02d}")
            os.makedirs(attempt_dir, exist_ok=True)

            candidate_repo_workdir, candidate_repo_path = self._clone_repo_to_attempt_root(base_repo_path, attempts_root)
            _log("workspace", f"candidate_repo={candidate_repo_path}")

            try:
                if attempt_index > 1:
                    diagnosis = None
                    expanded_context = ""
                    expanded_context_manifest = {}
                    debug_kg_trace_text = ""
                    debug_trace_payload = {}
                    debug_updated_issue_paths = []
                    if self.debug_policy == "raw_feedback":
                        _log("failure-analysis", "policy=raw_feedback; skipping diagnosis and KG backtracking")
                    else:
                        use_kg = self.debug_policy == "diagnosis_kg"
                        _log(
                            "failure-analysis",
                            f"policy={self.debug_policy}; running diagnosis agent"
                            + (" with KG backtracking" if use_kg else " without KG backtracking"),
                        )
                        try:
                            diagnosis, expanded_context, debug_kg_trace_text, expanded_context_manifest, debug_updated_issue_paths, debug_trace_payload = self._run_failure_analysis_and_expand(
                            problem_statement=problem_statement,
                            candidate_source_context=candidate_source_context,
                            error_feedback=error_feedback,
                            raw_validation_feedback=raw_validation_feedback,
                            candidate_repo_path=candidate_repo_path,
                            source_context_manifest=source_context_manifest,
                            persistent_source_context=expanded_context,
                            source_context_state=source_context_state,
                            previous_evidence=previous_evidence,
                            instance_id=instance_id,
                            issue_mentioned_files=locate_result.get("issue_mentioned_files") or [
                                item.get("file_path")
                                for item in (locate_result.get("rtl_source_discovery") or [])
                                if item.get("file_path")
                            ],
                            enable_kg_backtracking=use_kg,
                        )
                            self._write_json(os.path.join(attempt_dir, "failure_analysis.json"), diagnosis or {})
                            self._write_json(os.path.join(attempt_dir, "debug_kg_trace.json"), debug_trace_payload or {})
                            if debug_kg_trace_text:
                                with open(os.path.join(attempt_dir, "debug_kg_trace.md"), "w", encoding="utf-8") as f:
                                    f.write(debug_kg_trace_text)
                            if debug_updated_issue_paths:
                                current_issue_to_entity_paths = _merge_issue_to_entity_paths(
                                    current_issue_to_entity_paths,
                                    debug_updated_issue_paths,
                                    limit=48,
                                )
                                current_issue_path_nav = IssuePathNavigator(current_issue_to_entity_paths)
                                issue_to_entity_paths_text = _format_issue_to_entity_paths(current_issue_to_entity_paths, limit=20)
                                prompt_context["issue_to_entity_paths"] = current_issue_to_entity_paths
                                prompt_context["issue_to_entity_paths_text"] = issue_to_entity_paths_text
                                prompt_context["issue_path_files"] = current_issue_path_nav.files()
                                prompt_context["issue_path_navigation"] = current_issue_path_nav.to_manifest()
                        except Exception as exc:
                            _log("failure-analysis", f"failure analysis failed: {exc}")
                            diagnosis = None
                            debug_kg_trace_text = ""
                            debug_trace_payload = {}
                            debug_updated_issue_paths = []

                repair_plan = build_repair_plan(
                    locate_result=locate_result,
                    prompt_context=prompt_context,
                    previous_diagnosis=diagnosis,
                    previous_evidence=previous_evidence,
                )
                repair_plan_text = repair_plan_to_prompt(repair_plan)
                self._write_json(os.path.join(attempt_dir, "repair_plan.json"), repair_plan.to_dict())
                self._write_json(os.path.join(attempt_dir, "source_context_manifest.json"), {
                    "initial": source_context_manifest,
                    "expanded": expanded_context_manifest,
                    "persistent_state": source_context_state.to_manifest(),
                    "issue_to_entity_paths": current_issue_to_entity_paths,
                    "issue_path_navigation": IssuePathNavigator(current_issue_to_entity_paths).to_manifest(),
                    "debug_updated_issue_to_entity_paths": debug_updated_issue_paths,
                    "kg_schema": prompt_context.get("kg_schema", {}),
                })
                _log("repair-plan", f"allowed_files={len(repair_plan.allowed_files)} edit_cone={len(repair_plan.edit_cone)}")

                patch_raw_validation_output = select_patch_validation_output(
                    self.debug_policy,
                    raw_validation_feedback,
                )
                patch_structured_evidence = select_patch_structured_evidence(
                    self.debug_policy,
                    error_feedback,
                )
                prompt = self.prompt_builder.build_generation_prompt(
                    problem_statement=problem_statement,
                    candidate_source_context=candidate_source_context,
                    hard_constraints=hard_constraints,
                    language_prompt_parts=language_prompt_parts,
                    attempt_index=attempt_index,
                    max_attempts=max_attempts,
                    error_feedback=error_feedback,
                    bug_kg_digest_text=bug_kg_digest_text,
                    issue_to_entity_paths_text=issue_to_entity_paths_text,
                    previous_patch=previous_patch,
                    diagnosis=diagnosis,
                    expanded_source_context=expanded_context,
                    debug_kg_trace=debug_kg_trace_text,
                    repair_plan_text=repair_plan_text,
                    validation_evidence=patch_structured_evidence,
                    raw_validation_output=patch_raw_validation_output,
                    minitb_summary=minitb_summary_text,
                )
                self._write_json(os.path.join(attempt_dir, "prompt_manifest.json"), {
                    "prompt_chars": len(prompt),
                    "problem_statement_chars": len(problem_statement),
                    "kg_digest_chars": len(bug_kg_digest_text),
                    "issue_to_entity_paths_chars": len(issue_to_entity_paths_text or ""),
                    "candidate_source_chars": len(candidate_source_context),
                    "expanded_context_chars": len(expanded_context or ""),
                    "debug_kg_trace_chars": len(debug_kg_trace_text or ""),
                    "repair_plan_chars": len(repair_plan_text),
                    "frozen_minitb_summary_chars": len(minitb_summary_text),
                    "source_context_files": len((source_context_manifest or {}).get("files") or []),
                    "expanded_context_files": len((expanded_context_manifest or {}).get("files") or []),
                    "persistent_requested_files": len(source_context_state.requested_files),
                    "persistent_loaded_files": len(source_context_state.all_loaded_files),
                    "persistent_source_files": source_context_state.all_loaded_files,
                    "raw_validation_chars": len(patch_raw_validation_output),
                    "diagnosis_raw_validation_chars": (
                        len(raw_validation_feedback or "")
                        if attempt_index > 1 and self.debug_policy != "raw_feedback"
                        else 0
                    ),
                    "structured_validation_chars": len(patch_structured_evidence),
                    "diagnosis_structured_validation_chars": (
                        len(error_feedback or "")
                        if attempt_index > 1 and self.debug_policy != "raw_feedback"
                        else 0
                    ),
                    "issue_to_entity_paths": len(current_issue_to_entity_paths or []),
                    "issue_path_files": len(IssuePathNavigator(current_issue_to_entity_paths).files()),
                    "debug_updated_issue_to_entity_paths": len(debug_updated_issue_paths or []),
                    "max_attempts": max_attempts,
                    "attempt_index": attempt_index,
                    "debug_policy": self.debug_policy,
                    "validation_backend": self.internal_validator,
                    "validation_stages": (
                        ["hwe_finaltest"]
                        if self.internal_validator == "hwe_final"
                        else [
                            "static",
                            "frozen_minitb",
                            "repository_regression",
                            "clean_replay",
                        ]
                    ),
                    "held_out_benchmark_evaluator_used": self.internal_validator == "hwe_final",
                    "oracle_assisted_ablation": self.internal_validator == "hwe_final",
                    "kg_schema_version": (prompt_context.get("kg_schema") or {}).get("version", ""),
                })

                usage_count_before = len(self._llm_usage_events)
                stream = self.get_completion(
                    prompt,
                    stream=True,
                    call_kind="patch_generation",
                )
                if not stream:
                    evidence = [ValidationEvidence(
                        type="LLM_NO_RESPONSE",
                        severity="blocker",
                        message="LLM did not return a patch response.",
                        validator="llm",
                    )]
                    report = build_validation_report(evidence)
                    self._write_json(os.path.join(attempt_dir, "validation_report.json"), report.to_dict())
                    write_evidence_trace(os.path.join(attempt_dir, "evidence_trace.json"), report)
                    error_feedback = report.feedback_text
                    raw_validation_feedback = report.feedback_text
                    previous_evidence = [item.to_dict() for item in report.evidence]
                    _log("llm", "failed to get a valid response from model")
                    continue

                raw_patch_path = os.path.join(attempt_dir, "raw.patch")
                _log("llm", f"writing raw patch output to {raw_patch_path}")
                if isinstance(stream, str):
                    final_text = _final_response_content(stream)
                else:
                    raw_chunks = []
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None) or ""
                        reasoning_content = getattr(delta, "reasoning_content", None) or ""
                        if reasoning_content and not content:
                            continue
                        raw_chunks.append(content)
                    final_text = _final_response_content("".join(raw_chunks))
                if len(self._llm_usage_events) == usage_count_before:
                    # Streaming providers do not expose usage consistently.  The
                    # ablation runner disables streaming to obtain exact counts;
                    # ordinary runs retain this auditable fallback estimate.
                    self._record_llm_usage(
                        prompt,
                        final_text,
                        "patch_generation",
                        usage=None,
                    )
                with open(raw_patch_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                _log("llm", f"received_patch_chars={len(final_text)}")
                preview = _compact_text(final_text, 900)
                _log("llm-preview", preview.replace("\n", "\\n"))
                print("----- patch preview begin -----")
                print(preview)
                print("----- patch preview end -----")

                patch_result = self.post_process_and_apply_patch(
                    instance_id,
                    raw_patch_path,
                    locations_dir,
                    repo_path=candidate_repo_path,
                    language=language,
                )
                _log("patch", f"status={patch_result.status} applied={patch_result.applied} files={patch_result.edited_files}")

                if not patch_result.applied:
                    evidence = [ValidationEvidence(
                        type="PATCH_APPLY_FAILED",
                        severity="blocker",
                        message=f"Patch application failed: {patch_result.reason}",
                        raw_log="\n".join(patch_result.notes or []),
                        validator="patch_apply",
                    )]
                    report = build_validation_report(evidence)
                    self._write_json(os.path.join(attempt_dir, "validation_report.json"), report.to_dict())
                    write_evidence_trace(os.path.join(attempt_dir, "evidence_trace.json"), report)
                    error_feedback = report.feedback_text
                    raw_validation_feedback = report.feedback_text
                    previous_evidence = [item.to_dict() for item in report.evidence]
                    _log("patch", f"reason={_compact_text(patch_result.reason, 300)}")
                    continue

                patch_delta = build_patch_delta(
                    repo_path=candidate_repo_path,
                    patch_text=final_text,
                    edited_files=patch_result.edited_files or [],
                    repair_plan=repair_plan,
                )
                diff_path = os.path.join(attempt_dir, "diff.patch")
                with open(diff_path, "w", encoding="utf-8") as f:
                    f.write(patch_delta.raw_diff or "")
                self._write_json(os.path.join(attempt_dir, "patch_delta.json"), patch_delta.to_dict())
                _log("delta-gate", f"files={patch_delta.edited_files} blockers={len([v for v in patch_delta.policy_violations if v.get('severity') == 'blocker'])}")

                delta_evidence = [
                    ValidationEvidence(
                        type=str(v.get("type") or "PATCH_DELTA_POLICY"),
                        severity=str(v.get("severity") or "warning"),
                        message=str(v.get("message") or "Patch delta policy violation."),
                        file=str(v.get("file") or v.get("file_path") or ""),
                        validator="patch_delta",
                    )
                    for v in (patch_delta.policy_violations or [])
                ]
                delta_blockers = [item for item in delta_evidence if item.severity == "blocker"]
                stack_result = None
                validation_route = {}
                validation_raw_feedback = ""
                if delta_blockers:
                    validation_report = build_validation_report(
                        delta_evidence,
                        metadata={
                            "validator": "patch_delta",
                            "validation_backend": VALIDATION_STACK_VERSION,
                            "validation_stage": "patch_delta",
                        },
                    )
                elif language == "verilog" and self.internal_validator == "hwe_final":
                    _log("internal-validation", "running HWE-Bench finaltest as the ablation oracle")
                    hwe_result = hwe_validator.validate_patch(
                        hwe_pr_number,
                        patch_delta.raw_diff or "",
                        base_commit=base_commit,
                        # Some HWE-Bench images do not contain newer Ibex
                        # base commits.  Mount the disposable per-attempt
                        # clone so Docker validates the exact repository
                        # revision and source tree used by the repair agent.
                        source_repo_path=candidate_repo_path,
                    )
                    self._write_json(
                        os.path.join(attempt_dir, "hwe_bench_validation.json"),
                        hwe_result,
                    )
                    validation_report = hwe_result_to_validation_report(hwe_result)
                    validation_raw_feedback = str(hwe_result.get("output") or validation_report.feedback_text)
                    hwe_output = str(hwe_result.get("output") or "")
                    infra_error = bool(
                        hwe_result.get("error")
                        and not hwe_result.get("tests")
                        and "CANDIDATE_PATCH_APPLY_FAILED" not in hwe_output
                    )
                    validation_route = {
                        "failure_class": "infrastructure" if infra_error else "oracle_failure",
                        "stage": "hwe_finaltest",
                        "oracle_assisted_ablation": True,
                    }
                elif language == "verilog":
                    _log(
                        "internal-validation",
                        "running static -> frozen Mini-TB/module smoke -> repository regression",
                    )
                    stack_result = validation_stack.validate_candidate(
                        candidate_repo=candidate_repo_path,
                        patch_text=final_text,
                        patch_delta=patch_delta,
                        repair_plan=repair_plan,
                        artifact_dir=attempt_dir,
                        validation_id=f"attempt-{attempt_index:02d}",
                    )
                    validation_report = stack_result.report
                    validation_route = stack_result.route
                    if validation_route.get("failure_class") == "infrastructure":
                        _log("internal-validation", "infrastructure failure; retrying validators once on the same patch")
                        self._write_json(
                            os.path.join(attempt_dir, "validation_report_before_infra_retry.json"),
                            validation_report.to_dict(),
                        )
                        stack_result = validation_stack.validate_candidate(
                            candidate_repo=candidate_repo_path,
                            patch_text=final_text,
                            patch_delta=patch_delta,
                            repair_plan=repair_plan,
                            artifact_dir=os.path.join(attempt_dir, "infra_retry"),
                            validation_id=f"attempt-{attempt_index:02d}-infra-retry",
                        )
                        validation_report = stack_result.report
                        validation_route = stack_result.route
                else:
                    validation_report = build_validation_report([])
                self._write_json(os.path.join(attempt_dir, "validation_report.json"), validation_report.to_dict())
                write_evidence_trace(
                    os.path.join(attempt_dir, "evidence_trace.json"),
                    validation_report,
                    extra={
                        "patch_delta": patch_delta.to_dict(),
                        "validation_route": validation_route,
                        "stage_reports": (
                            {
                                name: report.to_dict() if report is not None else None
                                for name, report in stack_result.reports.items()
                            }
                            if stack_result is not None
                            else {}
                        ),
                    },
                )
                _log(
                    "internal-validation",
                    f"backend={validation_report.metadata.get('validation_backend', 'none')} "
                    f"passed={validation_report.passed} blockers={validation_report.blocker_count} warnings={validation_report.warning_count}",
                )

                if not validation_report.passed:
                    route_prefix = (
                        "failure_route=" + json.dumps(validation_route, ensure_ascii=False) + "\n"
                        if validation_route
                        else ""
                    )
                    error_feedback = route_prefix + validation_report.feedback_text
                    raw_validation_feedback = (
                        validation_raw_feedback
                        or (stack_result.raw_feedback if stack_result is not None else "")
                        or validation_report.feedback_text
                    )
                    previous_evidence = [item.to_dict() for item in validation_report.evidence]
                    previous_patch = (patch_delta.raw_diff or "")[:12000]
                    _log("evidence", _compact_text(error_feedback, 500).replace("\n", "\\n"))
                    if validation_route.get("failure_class") == "infrastructure":
                        self._write_json(os.path.join(run_root, "repair_summary.json"), {
                            "instance_id": instance_id,
                            "validated": False,
                            "attempts": attempt_index,
                            "last_feedback": error_feedback,
                            "validation_report": validation_report.to_dict(),
                            "failure_route": validation_route,
                            "validation_backend": "layered_internal_validation",
                            "validation_stack_version": VALIDATION_STACK_VERSION,
                            "clean_replay": {"passed": False, "reason": "infrastructure"},
                            "held_out_benchmark_evaluator_used": False,
                        })
                        _log("finish", "persistent validator infrastructure failure; patch attempt preserved")
                        return
                    continue

                if self.internal_validator == "hwe_final":
                    final_patch_path = os.path.join(output_dir, f"{instance_id}.patch")
                    final_diff_path = os.path.join(output_dir, f"{instance_id}.diff")
                    shutil.copyfile(raw_patch_path, final_patch_path)
                    shutil.copyfile(diff_path, final_diff_path)
                    selection = {
                        "schema_version": "kgveri-single-patch-selection-v1",
                        "selection_policy": "first candidate to pass the HWE finaltest ablation oracle",
                        "selected_attempt": attempt_index,
                        "fully_validated": True,
                    }
                    self._write_json(os.path.join(run_root, "selection_record.json"), selection)
                    self._write_json(os.path.join(run_root, "repair_summary.json"), {
                        "instance_id": instance_id,
                        "validated": True,
                        "candidate_available": True,
                        "target_behavior_validated": True,
                        "validation_level": "oracle_assisted_hwe_finaltest",
                        "attempts": attempt_index,
                        "final_patch": final_patch_path,
                        "final_diff": final_diff_path,
                        "validation_report": validation_report.to_dict(),
                        "patch_delta": patch_delta.to_dict(),
                        "selection": selection,
                        "validation_backend": "hwe_finaltest_ablation",
                        "held_out_benchmark_evaluator_used": True,
                        "oracle_assisted_ablation": True,
                        "shared_seed_used": bool(shared_seed_root),
                    })
                    tok("HWE finaltest ablation oracle PASSED")
                    _log("success", f"output_patch={final_patch_path}")
                    return

                _log("internal-validation", "candidate passed; starting exact-base clean replay")
                clean_replay_dir = os.path.join(
                    run_root,
                    "final_replay",
                    f"attempt_{attempt_index:02d}",
                )
                clean_result = validation_stack.clean_replay(
                    diff_text=patch_delta.raw_diff or "",
                    patch_text=final_text,
                    patch_delta=patch_delta,
                    repair_plan=repair_plan,
                    artifact_dir=clean_replay_dir,
                )
                if clean_result.route.get("failure_class") == "infrastructure":
                    _log("internal-validation", "clean replay infrastructure failure; retrying once")
                    clean_result = validation_stack.clean_replay(
                        diff_text=patch_delta.raw_diff or "",
                        patch_text=final_text,
                        patch_delta=patch_delta,
                        repair_plan=repair_plan,
                        artifact_dir=os.path.join(
                            run_root,
                            "final_replay",
                            f"attempt_{attempt_index:02d}_infra_retry",
                        ),
                    )
                if not clean_result.report.passed:
                    validation_report = merge_validation_reports(
                        {
                            "candidate_validation": validation_report,
                            "clean_replay": clean_result.report,
                        },
                        required=("candidate_validation", "clean_replay"),
                        metadata={"validation_backend": VALIDATION_STACK_VERSION},
                    )
                    self._write_json(
                        os.path.join(attempt_dir, "validation_report.json"),
                        validation_report.to_dict(),
                    )
                    write_evidence_trace(
                        os.path.join(attempt_dir, "evidence_trace.json"),
                        validation_report,
                        extra={
                            "patch_delta": patch_delta.to_dict(),
                            "clean_replay": clean_result.to_dict(),
                        },
                    )
                    error_feedback = (
                        "failure_route="
                        + json.dumps(clean_result.route, ensure_ascii=False)
                        + "\n"
                        + validation_report.feedback_text
                    )
                    raw_validation_feedback = clean_result.raw_feedback or validation_report.feedback_text
                    previous_evidence = [item.to_dict() for item in validation_report.evidence]
                    previous_patch = (patch_delta.raw_diff or "")[:12000]
                    _log("evidence", _compact_text(error_feedback, 500).replace("\n", "\\n"))
                    if clean_result.route.get("failure_class") == "infrastructure":
                        self._write_json(os.path.join(run_root, "repair_summary.json"), {
                            "instance_id": instance_id,
                            "validated": False,
                            "attempts": attempt_index,
                            "last_feedback": error_feedback,
                            "validation_report": validation_report.to_dict(),
                            "failure_route": clean_result.route,
                            "validation_backend": "layered_internal_validation",
                            "validation_stack_version": VALIDATION_STACK_VERSION,
                            "clean_replay": clean_result.report.to_dict(),
                            "held_out_benchmark_evaluator_used": False,
                        })
                        _log("finish", "persistent clean-replay infrastructure failure")
                        return
                    continue

                final_patch_path = os.path.join(output_dir, f"{instance_id}.patch")
                final_diff_path = os.path.join(output_dir, f"{instance_id}.diff")
                shutil.copyfile(raw_patch_path, final_patch_path)
                shutil.copyfile(diff_path, final_diff_path)
                changed_lines = sum(
                    1
                    for line in (patch_delta.raw_diff or "").splitlines()
                    if (line.startswith("+") and not line.startswith("+++"))
                    or (line.startswith("-") and not line.startswith("---"))
                )
                if not validation_stack.target_behavior_validated:
                    self._write_json(os.path.join(run_root, "repair_summary.json"), {
                        "instance_id": instance_id,
                        "validated": False,
                        "candidate_available": True,
                        "target_behavior_validated": False,
                        "validation_level": "module_smoke_and_lightweight_regression",
                        "attempts": attempt_index,
                        "final_patch": final_patch_path,
                        "final_diff": final_diff_path,
                        "validation_report": validation_report.to_dict(),
                        "patch_delta": patch_delta.to_dict(),
                        "clean_replay": clean_result.report.to_dict(),
                        "validation_backend": "layered_internal_validation",
                        "validation_stack_version": VALIDATION_STACK_VERSION,
                        "held_out_benchmark_evaluator_used": False,
                    })
                    tinfo(
                        "static checks, module smoke, lightweight repository regression, "
                        "and clean replay passed; candidate lacks issue-specific behavior validation"
                    )
                    _log("partial-success", f"output_candidate={final_patch_path}")
                    return
                selection = {
                    "schema_version": "kgveri-single-patch-selection-v1",
                    "selection_policy": "first sequential patch to pass all required validation gates",
                    "selected_attempt": attempt_index,
                    "fully_validated": True,
                    "warning_count": validation_report.warning_count,
                    "edited_files": len(patch_delta.edited_files),
                    "changed_lines": changed_lines,
                }
                self._write_json(os.path.join(run_root, "selection_record.json"), selection)
                self._write_json(os.path.join(run_root, "repair_summary.json"), {
                    "instance_id": instance_id,
                    "validated": True,
                    "candidate_available": True,
                    "target_behavior_validated": True,
                    "validation_level": "issue_minitb_and_lightweight_regression",
                    "attempts": attempt_index,
                    "final_patch": final_patch_path,
                    "final_diff": final_diff_path,
                    "validation_report": validation_report.to_dict(),
                    "patch_delta": patch_delta.to_dict(),
                    "clean_replay": clean_result.report.to_dict(),
                    "selection": selection,
                    "validation_backend": "layered_internal_validation",
                    "validation_stack_version": VALIDATION_STACK_VERSION,
                    "held_out_benchmark_evaluator_used": False,
                })
                tok("layered validation and clean replay PASSED")
                _log("success", f"output_patch={final_patch_path}")
                return
            finally:
                if candidate_repo_workdir and os.path.isdir(candidate_repo_workdir):
                    shutil.rmtree(candidate_repo_workdir, ignore_errors=True)

        self._write_json(os.path.join(run_root, "repair_summary.json"), {
            "instance_id": instance_id,
            "validated": False,
            "attempts": max_attempts,
            "last_feedback": error_feedback,
            "validation_backend": (
                "hwe_finaltest_ablation"
                if self.internal_validator == "hwe_final"
                else "layered_internal_validation"
            ),
            "validation_stack_version": (
                "kgveri-hwe-final-ablation-v1"
                if self.internal_validator == "hwe_final"
                else VALIDATION_STACK_VERSION
            ),
            "clean_replay": {"passed": False, "reason": "no_candidate_passed"},
            "held_out_benchmark_evaluator_used": self.internal_validator == "hwe_final",
            "oracle_assisted_ablation": self.internal_validator == "hwe_final",
            "shared_seed_used": bool(shared_seed_root),
        })
        _log("finish", "all repair attempts failed validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Repair Script")
    parser.add_argument("locations_dir", type=str, help="Directory containing KG localization JSON files.")
    parser.add_argument("--instance_id", required=True, type=str, help="The specific instance ID to process.")
    parser.add_argument(
        "--playground_dir",
        type=str,
        default=None,
        help="Root directory where repositories are located (default: sibling 'playground' of locations_dir).",
    )
    parser.add_argument(
        "--repo_identifier",
        type=str,
        default=None,
        help="Repository directory name inside playground (e.g., 'astropy__astropy'). If omitted, it will be derived from instance_id.",
    )

    args = parser.parse_args()
    patch_dir = os.path.join(os.path.dirname(args.locations_dir), "patches")
    repairer = CodeRepair()
    repairer.process_instance(
        instance_id=args.instance_id,
        locations_dir=args.locations_dir,
        output_dir=patch_dir,
        playground_dir=args.playground_dir,
        repo_identifier=args.repo_identifier,
    )
