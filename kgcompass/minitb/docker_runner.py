from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .ibex_resolver import SourceResolution
from .models import Marker, QualificationResult, RunResult, VisibleCase, write_json


MARKER_PATTERN = re.compile(r"^KGVERI_MINITB_([A-Z_]+)(?:\s+(.*))?$")
FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
INFRA_PATTERNS = re.compile(
    r"(?:Cannot connect to the Docker daemon|pull access denied|manifest unknown|No such image|"
    r"permission denied while trying to connect|docker: command not found|"
    r"No such file or directory[^\n]*docker|docker[^\n]*No such file or directory|"
    r"MINITB_TOOLCHAIN_UNAVAILABLE)",
    re.IGNORECASE,
)

# Use the image's prepared toolchain directly.  The three research images do
# not expose a uniform PATH (CVA6, in particular, has no `verilator` on PATH).
VERILATOR_BY_REPO = {
    "ibex": "/usr/bin/verilator",
    "cva6": "/tools/verilator-v5.008/bin/verilator",
    "caliptra-rtl": "/opt/verilator/bin/verilator",
}


def toolchain_probe(repo: str) -> str:
    preferred = shlex.quote(VERILATOR_BY_REPO[repo])
    return f'''VERILATOR_BIN={preferred}
if [[ ! -x "$VERILATOR_BIN" ]]; then
  VERILATOR_BIN="$(command -v verilator || true)"
fi
if [[ ! -x "$VERILATOR_BIN" ]]; then
  for candidate in /usr/local/bin/verilator /opt/verilator/bin/verilator /tools/verilator*/bin/verilator /tools/verilator/*/bin/verilator /opt/micromamba/envs/*/bin/verilator; do
    if [[ -x "$candidate" ]]; then VERILATOR_BIN="$candidate"; break; fi
  done
fi
IVERILOG_BIN="$(command -v iverilog || true)"
VVP_BIN="$(command -v vvp || true)"
if [[ -x "$VERILATOR_BIN" ]] && command -v make >/dev/null && command -v g++ >/dev/null; then
  MINITB_BACKEND=verilator
elif [[ -x "$IVERILOG_BIN" ]] && [[ -x "$VVP_BIN" ]]; then
  MINITB_BACKEND=iverilog
else
  echo "MINITB_TOOLCHAIN_UNAVAILABLE: Verilator+make+g++ or Icarus+vvp is required" >&2
  exit 127
fi
'''


class MiniTBDockerRunner:
    def __init__(self, case: VisibleCase, repo_root: Path, case_output: Path, timeout: int = 300):
        self.case = case
        self.repo_root = repo_root.resolve()
        self.case_output = case_output.resolve()
        self.timeout = timeout

    def prepare_run_script(self, asset_dir: Path, resolution: SourceResolution) -> Path:
        asset_dir = asset_dir.resolve()
        asset_dir.relative_to(self.case_output)
        source_args = " ".join(shlex.quote(f"/repo/{path}") for path in resolution.sources)
        include_args = " ".join(shlex.quote(f"-I/repo/{path}") for path in resolution.include_dirs)
        define_args = " ".join(shlex.quote(f"-D{value}") for value in resolution.defines)
        # Old Ibex generic SRAM models export the same DPI function from
        # differently parameterized instances, which Verilator 4.x emits as
        # conflicting C++ typedefs. MiniTBs do not use memory backdoor DPI.
        undefine_args = "-UVERILATOR" if any(
            Path(path).name == "prim_generic_ram_1p.sv" for path in resolution.sources
        ) else ""
        script = f"""#!/bin/bash
set -euo pipefail
MODE="${{1:?build or run}}"
BUILD_DIR="${{2:-/work/build}}"
ASSET_DIR="$(cd "$(dirname "$0")" && pwd)"
{toolchain_probe(self.case.repo)}
if [[ "$MODE" == build ]]; then
  mkdir -p "$BUILD_DIR"
  find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
  if [[ "$MINITB_BACKEND" == verilator ]]; then
  "$VERILATOR_BIN" --version
  TIMING_ARGS=()
  if "$VERILATOR_BIN" --timing --version >/dev/null 2>&1; then
    TIMING_ARGS+=(--timing)
  fi
  "$VERILATOR_BIN" --cc --exe --build "${{TIMING_ARGS[@]}}" -Wno-fatal \\
    --top-module kgveri_minitb_top \\
    -Mdir "$BUILD_DIR/obj_dir" -o kgveri_minitb \\
    {include_args} {define_args} {undefine_args} {source_args} \\
    "$ASSET_DIR/minitb_top.sv" "$ASSET_DIR/sim_main.cpp"
  else
    "$IVERILOG_BIN" -V
    "$IVERILOG_BIN" -g2012 -s kgveri_iverilog_driver -o "$BUILD_DIR/kgveri_minitb.vvp" {include_args} {define_args} {source_args} "$ASSET_DIR/minitb_top.sv" "$ASSET_DIR/sim_main_iverilog.sv"
  fi
elif [[ "$MODE" == run ]]; then
  if [[ "$MINITB_BACKEND" == verilator ]]; then
    "$BUILD_DIR/obj_dir/kgveri_minitb"
  else
    "$VVP_BIN" "$BUILD_DIR/kgveri_minitb.vvp"
  fi
else
  echo "unknown mode: $MODE" >&2
  exit 64
fi
"""
        path = asset_dir / "run.sh"
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        return path

    def build(self, asset_dir: Path, build_dir: Path) -> RunResult:
        return self._docker_phase("build", asset_dir, build_dir)

    def preflight(self) -> RunResult:
        command = [
            "docker", "run", "--rm", "--pull", "never", "--network", "none",
            self.case.docker_image, "bash", "-c",
            toolchain_probe(self.case.repo) + '''
if [[ "$MINITB_BACKEND" == verilator ]]; then
  "$VERILATOR_BIN" --version
else
  "$IVERILOG_BIN" -V
fi''',
        ]
        return self._execute(command, "preflight")

    def run(self, asset_dir: Path, build_dir: Path) -> RunResult:
        return self._docker_phase("run", asset_dir, build_dir)

    def _docker_phase(self, phase: str, asset_dir: Path, build_dir: Path) -> RunResult:
        build_dir.resolve().mkdir(parents=True, exist_ok=True)
        command = self.docker_command(phase, asset_dir, build_dir)
        return self._execute(command, phase)

    def _execute(self, command: list[str], phase: str) -> RunResult:
        env = dict(os.environ)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            return RunResult(
                phase=phase,
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(
                phase=phase,
                command=command,
                returncode=-1,
                stdout=_decode_timeout(exc.stdout),
                stderr=_decode_timeout(exc.stderr),
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return RunResult(
                phase=phase,
                command=command,
                returncode=-1,
                stdout="",
                stderr=str(exc),
            )

    def docker_command(self, phase: str, asset_dir: Path, build_dir: Path) -> list[str]:
        if phase not in {"build", "run"}:
            raise ValueError(f"unsupported Docker phase: {phase}")
        resolved_asset = asset_dir.resolve()
        resolved_build = build_dir.resolve()
        resolved_asset.relative_to(self.case_output)
        resolved_build.relative_to(self.case_output)
        return [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "-v",
            f"{self.repo_root}:/repo:ro",
            "-v",
            f"{resolved_asset}:/work/assets:ro",
            "-v",
            f"{resolved_build}:/work/build",
            self.case.docker_image,
            "bash",
            "/work/assets/run.sh",
            phase,
            "/work/build",
        ]


class QualificationEngine:
    def __init__(self, runner: MiniTBDockerRunner):
        self.runner = runner

    def qualify(
        self,
        asset_dir: Path,
        build_dir: Path,
        spec: dict[str, Any],
        expected_failure: bool = True,
        repeat_count: int = 2,
    ) -> QualificationResult:
        build = self.runner.build(asset_dir, build_dir)
        if build.returncode != 0:
            combined = build.stdout + "\n" + build.stderr
            status = (
                "INFRA_ERROR"
                if build.timed_out or INFRA_PATTERNS.search(combined)
                else "TB_BUILD_FAIL"
            )
            return QualificationResult(status=status, reason="Docker build failed", build=build)

        runs: list[RunResult] = []
        all_markers: list[Marker] = []
        classifications: list[tuple[str, str]] = []
        for _ in range(repeat_count):
            run = self.runner.run(asset_dir, build_dir)
            runs.append(run)
            markers = parse_markers(run.stdout + "\n" + run.stderr)
            all_markers.extend(markers)
            classifications.append(classify_run(run, markers, spec, expected_failure))

        statuses = [item[0] for item in classifications]
        reasons = [item[1] for item in classifications]
        signatures = [_failure_signature(parse_markers(run.stdout + "\n" + run.stderr)) for run in runs]
        repeatable = len(set(statuses)) == 1 and len(set(signatures)) == 1
        status = statuses[0] if repeatable else "IRRELEVANT_FAIL"
        reason = reasons[0] if repeatable else f"non-repeatable classifications: {statuses}"
        return QualificationResult(
            status=status,
            reason=reason,
            build=build,
            runs=runs,
            markers=all_markers,
            repeatable=repeatable,
        )


def parse_markers(output: str) -> list[Marker]:
    markers: list[Marker] = []
    for line in output.splitlines():
        match = MARKER_PATTERN.match(line.strip())
        if not match:
            continue
        fields = {key: value for key, value in FIELD_PATTERN.findall(match.group(2) or "")}
        markers.append(Marker(kind=match.group(1), fields=fields, raw=line.strip()))
    return markers


def classify_run(
    run: RunResult,
    markers: list[Marker],
    spec: dict[str, Any],
    expected_failure: bool,
) -> tuple[str, str]:
    combined = run.stdout + "\n" + run.stderr
    if run.timed_out or any(marker.kind == "TIMEOUT" for marker in markers):
        return "IRRELEVANT_FAIL", "simulation timed out"
    if INFRA_PATTERNS.search(combined):
        return "INFRA_ERROR", "Docker infrastructure failure"
    if spec.get("template_id") == "MINIMAL_MODULE_SMOKE":
        if run.returncode == 0 and any(
            marker.kind == "END" and marker.fields.get("status") == "MODULE_SMOKE_ONLY"
            for marker in markers
        ):
            return "MODULE_SMOKE_ONLY", "module elaborated and completed the smoke run"
        return "UNSUPPORTED", "module smoke did not complete"

    failures = [marker for marker in markers if marker.kind == "FAIL"]
    starts = [marker for marker in markers if marker.kind == "CHECK_START"]
    expected_checks = {check["check_id"]: check for check in spec.get("checks") or []}
    if failures:
        for marker in failures:
            check_id = marker.fields.get("check")
            check = expected_checks.get(check_id)
            if not check:
                return "IRRELEVANT_FAIL", f"failure came from unknown check {check_id!r}"
            if marker.fields.get("module") != spec["target"]["module"]:
                return "IRRELEVANT_FAIL", "failure module does not match BehaviorSpec"
            if marker.fields.get("signal") != check["signal"]:
                return "IRRELEVANT_FAIL", "failure signal does not match BehaviorSpec"
            observed = marker.fields.get("observed", "")
            if observed and re.fullmatch(r"[xXzZ?]+", observed):
                return "IRRELEVANT_FAIL", "observed value is X/Z-only"
            marker_expected = marker.fields.get("expected", "")
            # A stable check compares against the value sampled in the prior
            # cycle. Its marker therefore carries a dynamic reference value,
            # rather than the schema's literal expected field.
            if check.get("op") != "stable" and not _marker_expected_matches(
                check.get("expected"), marker_expected
            ):
                return "IRRELEVANT_FAIL", "failure expected value does not match BehaviorSpec"
        activated = {marker.fields.get("check") for marker in starts}
        if not starts or any(marker.fields.get("check") not in activated for marker in failures):
            return "IRRELEVANT_FAIL", "failure marker appeared without check activation"
        kinds = [marker.kind for marker in markers]
        if "START" not in kinds or kinds.index("START") > kinds.index("CHECK_START"):
            return "IRRELEVANT_FAIL", "target check ran before reset/start activation"
        if expected_failure:
            if run.returncode == 0:
                return "IRRELEVANT_FAIL", "target failure did not produce a failing exit code"
            return "QUALIFIED_MINITB", "target check failed with matching signal-level feedback"
        return "PATCH_REJECTED", "frozen target check still fails after patch"

    passed_checks = {marker.fields.get("check") for marker in markers if marker.kind == "PASS"}
    if run.returncode == 0 and expected_checks and set(expected_checks).issubset(passed_checks):
        if expected_failure:
            return "BASE_PASS", "all target checks passed on the buggy base"
        return "PATCH_ACCEPTED", "all frozen target checks passed after patch"
    return "IRRELEVANT_FAIL", "simulation ended without a complete target check result"


def save_run_logs(output_dir: Path, result: QualificationResult, prefix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.build:
        _write_log(output_dir / f"{prefix}build.log", result.build)
    for index, run in enumerate(result.runs, start=1):
        _write_log(output_dir / f"{prefix}run_{index:02d}.log", run)


def _write_log(path: Path, result: RunResult) -> None:
    command = " ".join(shlex.quote(item) for item in result.command)
    path.write_text(
        f"COMMAND: {command}\nRETURN_CODE: {result.returncode}\nTIMED_OUT: {result.timed_out}\n\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n",
        encoding="utf-8",
    )


def _failure_signature(markers: list[Marker]) -> tuple:
    return tuple(
        (
            marker.kind,
            marker.fields.get("check"),
            marker.fields.get("module"),
            marker.fields.get("signal"),
            marker.fields.get("expected"),
            marker.fields.get("observed"),
        )
        for marker in markers
        if marker.kind in {"FAIL", "PASS", "TIMEOUT", "END"}
    )


def _decode_timeout(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _marker_expected_matches(spec_value: Any, marker_value: str) -> bool:
    parsed = _parse_sv_integer(spec_value)
    if not marker_value or not re.fullmatch(r"[0-9a-fA-F]+", marker_value):
        return False
    if parsed is None:
        # Safe symbolic constants (typically package enum members) are resolved
        # by the SV compiler before the generated harness prints the marker.
        # The marker is emitted from the same frozen check, so its numeric form
        # cannot be compared textually with the schema identifier.
        return bool(
            re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_$]*(?:::[A-Za-z_][A-Za-z0-9_$]*)?",
                str(spec_value).strip(),
            )
        )
    return parsed == int(marker_value, 16)


def _parse_sv_integer(value: Any) -> int | None:
    text = str(value).strip().replace("_", "")
    if re.fullmatch(r"-?\d+", text):
        return int(text, 10)
    match = re.fullmatch(r"\d+'[sS]?([bBoOdDhH])([0-9a-fA-F]+)", text)
    if not match:
        if text == "'0":
            return 0
        if text == "'1":
            return 1
        return None
    base = {"b": 2, "o": 8, "d": 10, "h": 16}[match.group(1).lower()]
    try:
        return int(match.group(2), base)
    except ValueError:
        return None
