from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .behavior_spec import OpenAICompatibleClient
from .models import HarnessValidationError, RetryBudget, write_json


HARNESS_KEYS = frozenset(
    {
        "declarations",
        "dut_parameters",
        "port_connections",
        "initial_assignments",
        "pre_stimulus_body",
        "stimulus_body",
        "check_body",
    }
)
DANGEROUS = re.compile(
    r"(?:\$system|\$fopen|\$fwrite|\$readmem|DPI-C|curl\b|wget\b|/home/|"
    r"tb_script|test_patch|fix_patch|candidate\.patch|forever\b|while\s*\(\s*1\s*\))",
    re.IGNORECASE,
)


class HarnessFillValidator:
    def __init__(self, expected: dict[str, str]):
        self.expected = expected

    def validate(self, value: dict[str, Any]) -> dict[str, str]:
        if not isinstance(value, dict):
            raise HarnessValidationError("HarnessFill must be a JSON object")
        keys = set(value)
        if keys != HARNESS_KEYS:
            raise HarnessValidationError(
                f"HarnessFill keys mismatch: extra={sorted(keys - HARNESS_KEYS)}, "
                f"missing={sorted(HARNESS_KEYS - keys)}"
            )
        normalized: dict[str, str] = {}
        for key in HARNESS_KEYS:
            content = value.get(key)
            if not isinstance(content, str):
                raise HarnessValidationError(f"HarnessFill.{key} must be a string")
            if DANGEROUS.search(content):
                raise HarnessValidationError(f"HarnessFill.{key} contains a forbidden construct")
            if _semantic_normalize(content) != _semantic_normalize(self.expected[key]):
                raise HarnessValidationError(
                    f"HarnessFill.{key} changed deterministic interface, stimulus, or oracle semantics"
                )
            normalized[key] = self.expected[key]
        return normalized


class HarnessFillGenerator:
    def __init__(
        self,
        prompt_path: Path,
        client: OpenAICompatibleClient,
        attempt_writer: Callable[[dict[str, Any]], None],
        budget: RetryBudget | None = None,
    ):
        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.client = client
        self.attempt_writer = attempt_writer
        self.budget = budget

    def generate(
        self,
        spec: dict[str, Any],
        target: dict[str, Any],
        expected: dict[str, str],
    ) -> dict[str, str]:
        validator = HarnessFillValidator(expected)
        previous: dict[str, Any] | None = None
        error = ""
        for round_number in (1, 2):
            payload = {
                "behavior_spec": spec,
                "target_interface": {
                    "file": target.get("file"),
                    "module": target.get("module"),
                    "ports": target.get("ports"),
                    "parameters": target.get("parameters"),
                },
                "required_fragments": expected,
                "previous_harness_fill": previous,
                "validator_error": error,
            }
            prompt = self.prompt.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=True, indent=2))
            if self.budget:
                self.budget.consume("harness_fill")
            try:
                candidate, raw = self.client.complete_json(prompt)
            except Exception as exc:
                if not isinstance(exc, (ValueError, json.JSONDecodeError)) and not type(exc).__module__.startswith("openai"):
                    raise
                error = f"model response is unavailable or not a JSON object: {exc}"
                self.attempt_writer({
                    "stage": "harness_fill",
                    "round": round_number,
                    "model": self.client.model,
                    "raw_output": "",
                    "status": "INVALID",
                    "validator_error": error,
                })
                continue
            record = {
                "stage": "harness_fill",
                "round": round_number,
                "model": self.client.model,
                "raw_output": raw,
            }
            try:
                result = validator.validate(candidate)
            except HarnessValidationError as exc:
                error = str(exc)
                previous = candidate
                record["status"] = "INVALID"
                record["validator_error"] = error
                self.attempt_writer(record)
                continue
            record["status"] = "VALID"
            self.attempt_writer(record)
            return result
        raise HarnessValidationError(error or "model did not produce a valid HarnessFill")


class TemplateRenderer:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def expected_fill(self, spec: dict[str, Any], target: dict[str, Any]) -> dict[str, str]:
        checked_names = [check["signal"] for check in spec["checks"]]
        checked_outputs = {_port_base(name) for name in checked_names}
        checked_outputs.update(
            _port_base(name) for check in spec["checks"] for name in check.get("when", {})
        )
        observed_names = list(
            dict.fromkeys([*checked_names, *(spec.get("observations") or [])])
        )[:8]
        observed_outputs = {_port_base(name) for name in observed_names}
        # Bind only outputs used by checks or the bounded execution trace.
        # Unrelated DUT outputs remain disconnected.
        ports = [port for port in target.get("ports") or []
                 if port.get("direction") != "output"
                 or port["name"] in checked_outputs | observed_outputs]
        timing = spec["timing"]
        clock = timing["clock_port"]
        reset = timing["reset_port"]
        declarations: list[str] = []
        for package in target.get("imports") or []:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", str(package)):
                declarations.append(f"  import {package}::*;")
        parameter_overrides = spec["target"]["parameters"]
        # A DUT parameter is not visible in the surrounding harness even when
        # it determines a port width. Mirror effective values locally so
        # declarations such as logic [BusWidth-1:0] remain self-contained.
        for name, raw_default in (target.get("parameters") or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", str(name)):
                continue
            value = parameter_overrides.get(name, _parameter_default(raw_default))
            prefix = str(raw_default).split("=", 1)[0].split()
            if value and "type" not in prefix:
                declarations.append(f"  localparam {name} = {value};")
        for port in ports:
            declarations.append(f"  {_signal_type(port)} dut_{port['name']}{_unpacked_dims(port)};")
        declarations.append("  bit kgveri_started;")
        declarations.append(f"  bit [{max(len(spec['checks']), 1) - 1}:0] kgveri_checked;")
        for index, check in enumerate(spec["checks"]):
            if "when" in check:
                declarations.append(f"  integer kgveri_events_{index};")
        for check in spec["checks"]:
            if check["op"] == "stable":
                port = next(item for item in ports if item["name"] == _port_base(check["signal"]))
                declarations.append(
                    f"  {_signal_type(port)} kgveri_prev_{check['check_id']}{_unpacked_dims(port)};"
                )

        parameter_items = [
            f".{name}({value})" for name, value in sorted(spec["target"]["parameters"].items())
        ]
        dut_parameters = ""
        if parameter_items:
            dut_parameters = "#(\n    " + ",\n    ".join(parameter_items) + "\n  ) "

        port_connections = ",\n    ".join(
            f".{port.get('physical_name', port['name'])}(dut_{port['name']})"
            for port in ports
        )

        assignments: list[str] = []
        defaults = spec["defaults"]
        for port in ports:
            if port.get("direction") != "input":
                continue
            name = port["name"]
            if name == clock:
                assignments.append(f"    dut_{name} = clk_i;")
            elif name == reset:
                reset_expr = "rst_ni" if timing["reset_active"] == "low" else "~rst_ni"
                assignments.append(f"    dut_{name} = {reset_expr};")
            else:
                default_value = defaults.get(name, "'0")
                assignments.append(f"    dut_{name} = {_assignment_value(port, default_value)};")
        check_blocks = _check_blocks(spec)
        drive_lines: list[str] = []
        schedule_cycle = 0
        for step in spec["steps"]:
            schedule_cycle += int(step["after_cycles"])
            if step["drive"]:
                drive_lines.append(f"    if (cycle_i >= 32'd{schedule_cycle}) begin")
                # Apply broad assignments before narrower overrides, regardless
                # of JSON/key order (e.g. word, funct7 slice, then one bit).
                for name, value in sorted(step["drive"].items(), key=lambda item: _drive_order(item[0])):
                    base = _port_base(name)
                    port = next((p for p in ports if p["name"] == base), {})
                    if name != base:
                        port = {}
                    drive_lines.append(f"      dut_{name} = {_assignment_value(port, value)};")
                drive_lines.append("    end")
            schedule_cycle += int(step["hold_cycles"])

        port_names = {port["name"] for port in ports}
        stimulus_lines = [
            f'      $display("KGVERI_MINITB_OBSERVE signal={name} cycle=%0d value=%0h", cycle_i, dut_{name});'
            for name in observed_names
            if _port_base(name) in port_names
        ]
        stimulus_lines.extend(_due_checks(check_blocks, "      "))
        for check in spec["checks"]:
            if check["op"] == "stable":
                stimulus_lines.append(
                    f"      kgveri_prev_{check['check_id']} = dut_{check['signal']};"
                )

        sample_phase = spec.get("timing", {}).get("sample_phase", "post_edge")
        pre_stimulus_body = "\n".join(stimulus_lines) if sample_phase == "pre_edge" else ""
        post_stimulus_body = "\n".join(stimulus_lines) if sample_phase != "pre_edge" else ""

        check_body = "\n".join(block["code"] for block in check_blocks)
        return {
            "declarations": "\n".join(declarations),
            "dut_parameters": dut_parameters,
            "port_connections": port_connections,
            "initial_assignments": "\n".join([*assignments, *drive_lines]),
            "pre_stimulus_body": pre_stimulus_body,
            "stimulus_body": post_stimulus_body,
            "check_body": check_body,
        }

    def render(
        self,
        spec: dict[str, Any],
        target: dict[str, Any],
        fill: dict[str, str],
        destination: Path,
    ) -> dict[str, Path]:
        template_name = {
            "COMB_DIRECTED": "comb_directed.sv.tpl",
            "CLOCKED_SEQUENCE": "clocked_sequence.sv.tpl",
            "PROTOCOL_SEQUENCE": "protocol_sequence.sv.tpl",
            "EVENT_TRACE": "event_trace.sv.tpl",
            "MINIMAL_MODULE_SMOKE": "minimal_module_smoke.sv.tpl",
        }[spec["template_id"]]
        template = (self.templates_dir / template_name).read_text(encoding="utf-8")
        replacements = {
            "{{INSTANCE_ID}}": spec["instance_id"],
            "{{TEMPLATE_ID}}": spec["template_id"],
            "{{TARGET_MODULE}}": target["module"],
            "{{DECLARATIONS}}": fill["declarations"],
            "{{DUT_PARAMETERS}}": fill["dut_parameters"],
            "{{PORT_CONNECTIONS}}": fill["port_connections"],
            "{{INITIAL_ASSIGNMENTS}}": fill["initial_assignments"],
            "{{PRE_STIMULUS_BODY}}": fill["pre_stimulus_body"],
            "{{STIMULUS_BODY}}": fill["stimulus_body"],
            "{{CHECK_TASKS}}": fill["check_body"],
            "{{TIMEOUT_CYCLES}}": str(spec["timing"]["timeout_cycles"]),
            "{{EVENT_RESET}}": "\n".join(
                f"      kgveri_events_{index} = 0;"
                for index, check in enumerate(spec["checks"]) if "when" in check
            ),
        }
        for key, value in replacements.items():
            template = template.replace(key, value)
        if "{{" in template or "}}" in template:
            raise HarnessValidationError("unresolved template placeholder")
        destination.mkdir(parents=True, exist_ok=True)
        top_path = destination / "minitb_top.sv"
        top_path.write_text(template, encoding="utf-8")
        sim_main = (self.templates_dir / "sim_main.cpp").read_text(encoding="utf-8")
        sim_main = sim_main.replace(
            "{{RESET_CYCLES}}", str(max(1, int(spec["timing"]["reset_cycles"])))
        ).replace("{{TIMEOUT_CYCLES}}", str(int(spec["timing"]["timeout_cycles"])))
        if "{{" in sim_main or "}}" in sim_main:
            raise HarnessValidationError("unresolved C++ driver placeholder")
        (destination / "sim_main.cpp").write_text(sim_main, encoding="utf-8")
        iverilog_driver = (self.templates_dir / "sim_main_iverilog.sv.tpl").read_text(
            encoding="utf-8"
        )
        iverilog_driver = iverilog_driver.replace(
            "{{RESET_CYCLES}}", str(max(1, int(spec["timing"]["reset_cycles"])))
        ).replace("{{TIMEOUT_CYCLES}}", str(int(spec["timing"]["timeout_cycles"])))
        if "{{" in iverilog_driver or "}}" in iverilog_driver:
            raise HarnessValidationError("unresolved Icarus driver placeholder")
        (destination / "sim_main_iverilog.sv").write_text(
            iverilog_driver, encoding="utf-8"
        )
        write_json(destination / "harness_fill.json", fill)
        return {
            "top": top_path,
            "driver": destination / "sim_main.cpp",
            "iverilog_driver": destination / "sim_main_iverilog.sv",
        }


def build_smoke_spec(instance_id: str, target: dict[str, Any]) -> dict[str, Any]:
    ports = target.get("ports") or []
    clock = next((item["name"] for item in ports if item["direction"] == "input" and _looks_clock(item["name"])), "")
    reset = next((item["name"] for item in ports if item["direction"] == "input" and _looks_reset(item["name"])), "")
    reset_active = "low" if reset.endswith(("_ni", "_n")) else ("high" if reset else "none")
    output = next((item["name"] for item in ports if item["direction"] == "output"), "")
    checks = []
    if output:
        checks = [
            {
                "check_id": "module_smoke_observation",
                "signal": output,
                "op": "==",
                "expected": "'0",
                "at_or_after_cycle": 1,
                "message": "non-functional smoke observation",
            }
        ]
    return {
        "schema_version": "kgveri-minitb-v1",
        "instance_id": instance_id,
        "bug_class": "unsupported",
        "target": {"file": target["file"], "module": target["module"], "parameters": {}},
        "timing": {
            "clock_port": clock,
            "reset_port": reset,
            "reset_active": reset_active,
            "reset_cycles": 2 if reset else 0,
            "timeout_cycles": 10,
        },
        "defaults": {},
        "steps": [{"after_cycles": 0, "drive": {}, "hold_cycles": 2}],
        "checks": checks,
        "observations": [output] if output else [],
        "oracle": {"confidence": "low", "evidence_refs": []},
        "template_id": "MINIMAL_MODULE_SMOKE",
        "assumptions": ["This is an elaboration and simulation smoke test, not a functional oracle."],
    }


def _check_blocks(spec: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for index, check in enumerate(spec["checks"]):
        signal = f"dut_{check['signal']}"
        expected = check["expected"]
        op = check["op"]
        if op == "==":
            failure = f"!({signal} === {expected})"
        elif op == "!=":
            failure = f"({signal} === {expected})"
        elif op == "asserted":
            failure = f"!({signal} === 1'b1)"
            expected = "1'b1"
        elif op == "deasserted":
            failure = f"!({signal} === 1'b0)"
            expected = "1'b0"
        else:
            failure = f"!({signal} === kgveri_prev_{check['check_id']})"
            expected = f"kgveri_prev_{check['check_id']}"
        code = "\n".join(
            [
                f"  task automatic kgveri_check_{index};",
                "    begin",
                f'      $display("KGVERI_MINITB_CHECK_START check={check["check_id"]} module={spec["target"]["module"]} signal={check["signal"]}");',
                f"      if ({failure}) begin",
                f'        $display("KGVERI_MINITB_FAIL check={check["check_id"]} module={spec["target"]["module"]} signal={check["signal"]} cycle=%0d expected=%0h observed=%0h", cycle_i, {expected}, {signal});',
                '        $display("KGVERI_MINITB_END status=FAIL");',
                "        pass_o = 1'b0;",
                "        done_o = 1'b1;",
                "      end else begin",
                f'        $display("KGVERI_MINITB_PASS check={check["check_id"]}");',
                "      end",
                f"      kgveri_checked[{index}] = 1'b1;",
                "    end",
                "  endtask",
            ]
        )
        blocks.append(
            {
                "index": index,
                "cycle": int(check["at_or_after_cycle"]),
                "task": f"kgveri_check_{index}",
                "code": code,
                "when": check.get("when"),
                "occurrence": check.get("occurrence", 1),
            }
        )
    return blocks


def _due_checks(blocks: list[dict[str, Any]], indent: str) -> list[str]:
    lines = []
    for block in blocks:
        if block.get("when"):
            condition = " && ".join(f"(dut_{name} === {value})" for name, value in block["when"].items())
            lines.extend([
                f"{indent}if (!done_o && !kgveri_checked[{block['index']}] && cycle_i >= 32'd{block['cycle']} && {condition}) begin",
                f"{indent}  kgveri_events_{block['index']} = kgveri_events_{block['index']} + 1;",
                f"{indent}  if (kgveri_events_{block['index']} == {block['occurrence']}) {block['task']}();",
                f"{indent}end",
            ])
            continue
        lines.append(
            f"{indent}if (!done_o && !kgveri_checked[{block['index']}] && cycle_i >= 32'd{block['cycle']}) {block['task']}();"
        )
    return lines


def _drive_order(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"([^[]+)\[(\d+)(?::(\d+))?\]", name)
    if not match:
        return name, -1_000_000
    return match.group(1), -(int(match.group(2)) - int(match.group(3) or match.group(2)) + 1)


def _port_base(reference: str) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)", str(reference))
    return match.group(1) if match else str(reference)


def _decl_width(port: dict[str, Any]) -> str:
    width = str(port.get("width") or "").strip()
    if re.fullmatch(r"\[[^;]+\]", width):
        return width + " "
    return ""


def _parameter_default(value: Any) -> str:
    text = str(value or "").strip().rstrip(",")
    if "=" in text:
        text = text.split("=", 1)[1].strip().rstrip(",")
    return text


def _unpacked_dims(port: dict[str, Any]) -> str:
    name = re.escape(str(port.get("name") or ""))
    match = re.search(rf"\b{name}\s*((?:\[[^]]+\]\s*)*)$", str(port.get("declaration") or "").strip())
    return " " + match.group(1).strip() if match and match.group(1).strip() else ""


def _assignment_value(port: dict[str, Any], value: Any) -> str:
    # Assignment patterns initialize every unpacked element, including structs.
    result = str(value)
    for _ in re.findall(r"\[[^]]+\]", _unpacked_dims(port)):
        result = "'{default: " + result + "}"
    return result


def _signal_type(port: dict[str, Any]) -> str:
    width = _decl_width(port).strip()
    declaration = str(port.get("declaration") or "")
    name = re.escape(str(port.get("name") or ""))
    type_match = re.search(
        rf"\b(?:input|output|inout)\b\s+(.+?)\s+{name}\s*(?:\[[^]]+\]\s*)*(?:=[^,;]+)?$",
        declaration.strip(),
    )
    if type_match:
        declared_type = re.sub(r"\b(?:wire|reg)\b", "", type_match.group(1)).strip()
        if "::" in declared_type:
            return declared_type
    return f"logic {width}".strip()


def _semantic_normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _looks_clock(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:clk|clock)(?:_|$)", name, re.IGNORECASE))


def _looks_reset(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:rst|reset)(?:_|$)", name, re.IGNORECASE))
