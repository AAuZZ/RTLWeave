from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


VALIDATION_PROTOCOL = (
    "Candidate patches are validated in order by static checks, the immutable frozen "
    "Mini-Testbench, repository-owned regression tests, and a clean replay from the exact "
    "base revision. Passing an earlier layer is not sufficient: all required layers must pass."
)

HIDDEN_ARTIFACT_BOUNDARY = (
    "Never request, quote, infer, or modify hidden benchmark evaluator artifacts, reference "
    "patches, gold files, or held-out expected results."
)

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def select_patch_validation_output(
    debug_policy: str,
    raw_validation_output: str,
) -> str:
    """Route full validator logs only to the direct-feedback ablation.

    Diagnosis-based policies consume the full log in the diagnosis prompt and
    expose only normalized evidence plus the diagnosis result to the patch
    generator.  This keeps the three debugging policies mutually distinct.
    """
    if str(debug_policy or "").strip().lower() == "raw_feedback":
        return str(raw_validation_output or "")
    return ""


def select_patch_structured_evidence(
    debug_policy: str,
    structured_evidence: str,
) -> str:
    """Hide system-normalized evidence from the direct-feedback baseline."""
    if str(debug_policy or "").strip().lower() == "raw_feedback":
        return ""
    return str(structured_evidence or "")

def _dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if marker in seen:
            continue
        ordered.append(item)
        seen.add(marker)
    return ordered


def _compact_text(text: str, limit: int = 1200) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _string_list(items: Iterable[Any], limit: int = 8) -> str:
    values = [str(item) for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) > limit:
        values = values[:limit] + ["..."]
    return ", ".join(values)


# ---------------------------------------------------------------------------
# patch-application result (kept as-is, used by repair.py)
# ---------------------------------------------------------------------------

@dataclass
class PatchApplicationResult:
    applied: bool
    status: str
    reason: str
    edited_files: List[str] = field(default_factory=list)
    diff_paths: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raw_patch_path: str = ""
    raw_patch_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# PromptBuilder  ---  KGVeri V1 repair plan + evidence feedback prompts
# ---------------------------------------------------------------------------

class PromptBuilder:
    def __init__(self, language: str = "verilog", max_input_length: int = 65536) -> None:
        self.language = language
        self.max_input_length = int(max_input_length)

    # -- formatting helpers --------------------------------------------------

    def _section(self, title: str, body: str) -> str:
        body = (body or "").strip()
        if not body:
            return ""
        return f"## {title}\n{body}\n"

    def _bullet_section(self, title: str, items: Sequence[str]) -> str:
        lines = [item.strip() for item in items if str(item).strip()]
        if not lines:
            return ""
        return self._section(title, "\n".join(f"- {line}" for line in lines))

    def _limit(self, text: str, limit: int) -> str:
        return _compact_text(text or "", limit)

    def _fit_prompt(self, sections: Sequence[str]) -> str:
        prompt = "\n".join(section for section in sections if section).strip()
        if len(prompt) <= self.max_input_length:
            return prompt
        trimmed = list(sections)
        for idx in range(len(trimmed) - 2, 0, -1):
            if len("\n".join(s for s in trimmed if s).strip()) <= self.max_input_length:
                break
            trimmed[idx] = _compact_text(trimmed[idx], 8000)
        prompt = "\n".join(s for s in trimmed if s).strip()
        if len(prompt) <= self.max_input_length:
            return prompt
        return prompt[: self.max_input_length - 20].rstrip() + "\n...[truncated]..."

    # -- edit-format block ---------------------------------------------------

    def _format_language_block(self, language_prompt_parts: Dict[str, str]) -> str:
        language_name = language_prompt_parts.get("language_name", "repository language")
        fence_language = language_prompt_parts.get("fence_language", "text")
        use_replace_only = bool(language_prompt_parts.get("use_replace_only"))
        example_file = language_prompt_parts.get("example_file", "path/to/example")
        example_start_line = int(language_prompt_parts.get("example_start_line", 1) or 1)
        example_end_line = int(language_prompt_parts.get("example_end_line", example_start_line) or example_start_line)
        example_search = language_prompt_parts.get("example_search", "")
        example_replace = language_prompt_parts.get("example_replace", "")
        language_notes = language_prompt_parts.get("language_notes", "")
        if use_replace_only:
            return f"""
Return ONLY line-range replacements. Every block uses `[REPLACE]` to start
and `[/REPLACE]` to end — these delimiters MUST appear in matched pairs.
NEVER put `[REPLACE]` inside a block; it only marks the start of a NEW block.

Format per block:
  ### file/path.sv
  - start_line: N
  - end_line: M
  [REPLACE]
  replacement code (NO [REPLACE] inside)
  [/REPLACE]

Multi-block example (two files, three blocks total):

```{fence_language}
### rtl/bus.sv
- start_line: 42
- end_line: 44
[REPLACE]
  assign bus_req = bus_gnt ? 1'b0 : pending_req;
  assign bus_addr = addr_queue[0];
[/REPLACE]

### rtl/ctrl.sv
- start_line: 88
- end_line: 90
[REPLACE]
  if (state == FETCH && !stall) begin
    next_state = DECODE;
  end
[/REPLACE]

### rtl/ctrl.sv
- start_line: 155
- end_line: 157
[REPLACE]
  assign valid_o = instr_valid & ~kill;
  assign ready_o = ~stall;
[/REPLACE]
```

Important:
- Every `[REPLACE]` MUST be followed by exactly one `[/REPLACE]`.
- For Verilog, only REPLACE — never SEARCH.  There is NO `=======` separator.
- One block per (file, start_line, end_line).
- `[REPLACE]` contains the NEW code ONLY. Never put the original code inside it.

DO NOT output a `=======` line — this is REPLACE-only, not SEARCH/REPLACE:
  WRONG:  [REPLACE]\n  assign x = old;\n=======\n  assign x = new;\n[/REPLACE]
  RIGHT:  [REPLACE]\n  assign x = new;\n[/REPLACE]

Do NOT wrap your answer in Markdown ``` fences. Output raw text directly.
{language_notes}
"""

    # -- prompt builders -----------------------------------------------------

    def build_repair_plan_prompt(
        self,
        *,
        problem_statement: str,
        localization_summary: str = "",
        bug_kg_digest_text: str = "",
        evidence_feedback: str = "",
    ) -> str:
        sections = [
            self._section("Agent Role", "RTL Repair Planner. Produce a graph-grounded repair hypothesis, not a patch."),
            self._section("Issue Summary", self._limit(problem_statement, 6000)),
        ]
        if localization_summary:
            sections.append(self._section("KG Localization Summary", self._limit(localization_summary, 4000)))
        if bug_kg_digest_text:
            sections.append(self._section("Bug Knowledge Graph Digest", self._limit(bug_kg_digest_text, 6000)))
        if evidence_feedback:
            sections.append(self._section("Previous Validation Evidence", self._limit(evidence_feedback, 8000)))
        schema = """
Return only JSON:
{
  "hypothesis": "one concise graph-grounded bug hypothesis",
  "edit_cone": [{"file_path": "rtl/file.sv", "reason": "why this cone matters"}],
  "allowed_files": ["rtl/file.sv"],
  "forbidden_files": ["tb/", "tests/"],
  "validator_profile": {
    "stages": ["static_validation", "frozen_minitb", "repository_regression", "clean_replay"]
  }
}
"""
        sections.append(self._section("Internal Validation Protocol", VALIDATION_PROTOCOL))
        sections.append(self._section("Benchmark Boundary", HIDDEN_ARTIFACT_BOUNDARY))
        sections.append(self._section("Output Contract", schema.strip()))
        return self._fit_prompt(sections)

    def build_generation_prompt(
        self,
        *,
        problem_statement: str,
        candidate_source_context: str,
        hard_constraints: Sequence[str],
        language_prompt_parts: Dict[str, str],
        attempt_index: int,
        max_attempts: int,
        error_feedback: str = "",
        bug_kg_digest_text: str = "",
        issue_to_entity_paths_text: str = "",
        previous_patch: str = "",
        diagnosis: Optional[Dict[str, Any]] = None,
        expanded_source_context: str = "",
        debug_kg_trace: str = "",
        repair_plan_text: str = "",
        validation_evidence: str = "",
        raw_validation_output: str = "",
        minitb_summary: str = "",
    ) -> str:
        """Unified patch-generation prompt used for both initial generation
        and subsequent repair iterations.

        On the first call (attempt_index==1) only problem_statement, KG
        context and format rules are included.  On retries the following
        extra sections are appended so the model sees what was tried, why
        it failed, and what additional context to use:

          - Previous Patch           (the .diff that failed validation)
          - Raw Validator Outputs    (direct-feedback ablation only)
          - Frozen Mini-Testbench Summary (target behavior and activated checks)
          - Structured Evidence      (compact fields used by KG)
          - Failure Analysis         (structured JSON from diagnosis agent)
          - KG Re-Slice Trace        (failure-conditioned backward trace)
          - Expanded Source Context   (files the diagnosis agent requested)
        """
        sections = [
            self._section("Agent Role", "RTL Repair Agent. Produce the smallest possible REPLACE-only patch."),
            self._section("Attempt", f"- attempt: {attempt_index}/{max_attempts}"),
            self._section("Issue Summary", self._limit(problem_statement, 6000)),
        ]
        sections.append(self._section("Internal Validation Protocol", VALIDATION_PROTOCOL))
        if minitb_summary:
            sections.append(self._section("Frozen Mini-Testbench Summary", self._limit(minitb_summary, 4000)))
        # The caller supplies this section only for the direct-feedback
        # ablation. Diagnosis-based policies route the log to the diagnosis
        # prompt and give this agent the normalized result instead.
        if raw_validation_output:
            sections.append(self._section("Raw Validator Outputs", raw_validation_output))
        if bug_kg_digest_text:
            sections.append(self._section("Bug Knowledge Graph Digest", self._limit(bug_kg_digest_text, 6000)))
        if issue_to_entity_paths_text:
            sections.append(self._section(
                "Issue-to-PatchableSpan KG Inference Paths",
                self._limit(issue_to_entity_paths_text, 6000),
            ))
        if repair_plan_text:
            sections.append(self._section("Graph-Grounded Repair Plan", self._limit(repair_plan_text, 4000)))
        if candidate_source_context:
            sections.append(self._section("Candidate RTL Source Files", self._limit(candidate_source_context, 70000)))

        # ── retry-only sections (from previous attempt) ──────────────────
        if previous_patch:
            sections.append(self._section("Previous Patch (failed validation)", previous_patch.strip()[:8000]))

        if not raw_validation_output and error_feedback and not validation_evidence:
            sections.append(self._section("Validation Failure Feedback", error_feedback))

        if validation_evidence:
            sections.append(self._section("Structured Validation Evidence", self._limit(validation_evidence, 8000)))

        if diagnosis is not None:
            sections.append(self._section("Failure Analysis (diagnosis agent)", json.dumps(diagnosis, indent=2, ensure_ascii=False)[:8000]))

        if debug_kg_trace:
            sections.append(self._section("KG Re-Slice Trace (failure-conditioned)", self._limit(debug_kg_trace, 8000)))

        if expanded_source_context:
            sections.append(self._section("Expanded Source Context", self._limit(expanded_source_context, 50000)))

        constraints = [
            "Edit only files allowed by the Graph-Grounded Repair Plan and shown in Candidate RTL Source Files or Expanded Source Context.",
            "Prefer the smallest complete RTL block that fixes the issue.",
            "For Verilog/SystemVerilog, output line-range REPLACE blocks only; do not output SEARCH.",
            "Do NOT modify the frozen Mini-Testbench, repository test files, or test expectations.",
            "Use validator outputs only as failure evidence; distinguish RTL failures from infrastructure failures.",
            HIDDEN_ARTIFACT_BOUNDARY,
        ]
        sections.append(self._bullet_section("Patch Rules", constraints))
        sections.append(self._format_language_block(language_prompt_parts))
        return self._fit_prompt(sections)

    def build_debug_diagnosis_prompt(
        self,
        *,
        problem_statement: str,
        candidate_source_context: str,
        error_feedback: str = "",
        raw_validation_output: str = "",
        debug_kg_trace: str = "",
        source_context_manifest: str = "",
        persistent_source_context: str = "",
        max_files: int = 5,
    ) -> str:
        sections = [
            self._section(
                "Agent Role",
                "Diagnosis Agent. Classify the validation failure, use KG signal traces to localize "
                "an RTL failure when present, and request only the source files needed next. Do NOT output a patch.",
            ),
            self._section("Issue Summary", self._limit(problem_statement, 6000)),
            self._section("Internal Validation Protocol", VALIDATION_PROTOCOL),
        ]
        if raw_validation_output:
            sections.append(self._section("Raw Validator Outputs", raw_validation_output))
        sections.append(self._section(
            "Structured Validation Evidence",
            self._limit(error_feedback.strip(), 8000) if error_feedback else "No structured failure details available.",
        ))
        if debug_kg_trace:
            sections.append(self._section("KG Backward Trace (failure-conditioned)", self._limit(debug_kg_trace, 8000)))
        if source_context_manifest:
            sections.append(self._section("Source Context Manifest", self._limit(source_context_manifest, 8000)))
        if candidate_source_context:
            sections.append(self._section("Initial Source Context", self._limit(candidate_source_context, 70000)))
        if persistent_source_context:
            sections.append(self._section("Persistent Expanded Source Context", persistent_source_context))

        schema = f"""
Return only JSON. Do not wrap it in Markdown fences.
{{
  "failure_hypothesis": "one concise sentence",
  "failure_class": "patch_format | rtl_build | rtl_behavior | regression | infrastructure | unknown",
  "retry_action": "revise_patch | expand_context | retry_validator | stop",
  "need_files": ["rtl/file1.sv"],
  "edit_candidates": [
    {{"file": "rtl/file.sv", "reason": "why this file may need editing"}}
  ],
  "do_not_change": ["testbench", "unrelated RTL"]
}}
	Rules:
	- Request at most {max_files} RTL/include files.
	- Base requests on failed signals/modules/files from structured evidence, validator outputs, and the KG Backward Trace.
	- Use revise_patch only for patch or RTL failures; use retry_validator for infrastructure failures.
	- Prefer files already visible in the source context.
	- If the visible snippets are insufficient, use KG suggested targets and the repository RTL file index to request the smallest additional file set.
	- Do not request or modify the frozen Mini-Testbench or repository test files.
	- {HIDDEN_ARTIFACT_BOUNDARY}
	"""
        sections.append(self._section("Diagnosis Output Contract", schema.strip()))
        return self._fit_prompt(sections)

    # build_debug_patch_prompt has been removed — its functionality is now
    # provided by build_generation_prompt with retry-only sections.
