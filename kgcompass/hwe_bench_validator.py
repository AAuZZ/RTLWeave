# Copyright (c) 2024 ByteDance Ltd.
# HWE-Bench Docker-based patch validator.

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional

try:
    from diagnostic_evidence import build_validation_report
    from repair_types import ValidationEvidence, ValidationReport
except Exception:  # pragma: no cover
    from .diagnostic_evidence import build_validation_report
    from .repair_types import ValidationEvidence, ValidationReport


# Paths to HWE-Bench datasets
HWE_BENCH_DATASETS = {
    "ibex": "/home/caoms/hwe-bench/datasets/lowRISC__ibex.jsonl",
    "cva6": "/home/caoms/hwe-bench/datasets/openhwgroup__cva6.jsonl",
    "caliptra-rtl": "/home/caoms/hwe-bench/datasets/chipsalliance__caliptra-rtl.jsonl",
}

# Docker image prefixes per repo
IMAGE_PREFIXES = {
    "ibex": "ghcr.io/pku-liang/lowrisc_m_ibex",
    "cva6": "ghcr.io/pku-liang/openhwgroup_m_cva6",
    "caliptra-rtl": "ghcr.io/pku-liang/chipsalliance_m_caliptra-rtl",
}

# Timeout for Docker validation (seconds)
DOCKER_TIMEOUT = 1200


class HweBenchValidator:
    """Validate candidate Verilog patches using HWE-Bench Docker containers."""

    def __init__(self, repo: str = "ibex", dataset_path: Optional[str] = None):
        self._repo = repo
        self._dataset_path = dataset_path or HWE_BENCH_DATASETS.get(repo, HWE_BENCH_DATASETS["ibex"])
        self._image_prefix = IMAGE_PREFIXES.get(repo, IMAGE_PREFIXES["ibex"])
        self._cache: Dict[int, str] = {}  # pr_number -> tb_script

    def load_tb_script(self, pr_number: int) -> Optional[str]:
        """Load tb_script for a given PR from the HWE-Bench dataset.

        Results are cached in-memory since the dataset is static.
        """
        if pr_number in self._cache:
            return self._cache[pr_number]

        if not os.path.exists(self._dataset_path):
            self._cache[pr_number] = ""
            return ""

        try:
            with open(self._dataset_path) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("number") == pr_number:
                        tb = d.get("tb_script", "")
                        self._cache[pr_number] = tb
                        return tb
        except (json.JSONDecodeError, OSError):
            pass

        self._cache[pr_number] = ""
        return ""

    def validate_patch(self, pr_number: int, candidate_diff: str, base_commit: Optional[str] = None,
                       source_repo_path: Optional[str] = None) -> dict:
        """Validate a candidate patch using HWE-Bench Docker.

        Args:
            pr_number: The ibex PR number (e.g., 48, 167).
            candidate_diff: Standard unified diff of the candidate fix.
            base_commit: Optional base commit SHA to checkout before applying
                         the patch.  If omitted the validator expects a
                         ``{repo}_base_commit.txt`` file inside the image.
            source_repo_path: Optional local repo path to mount into Docker,
                              overriding the embedded repo.  Ensures the LLM
                              sees the same file content as Docker sees.

        Returns:
            dict with keys:
                - passed (bool): Whether all tests passed.
                - tests (dict): Test name -> PASS/FAIL mapping.
                - returncode (int): Docker exit code.
                - error (str): Error message if something went wrong.
                - output (str): Full Docker stdout+stderr for debugging.
        """
        tb_script = self.load_tb_script(pr_number)
        if not tb_script:
            return {
                "passed": False,
                "tests": {},
                "returncode": -1,
                "error": f"No tb_script found for PR #{pr_number} in HWE-Bench dataset",
                "output": "",
                "repo": self._repo,
                "pr_number": pr_number,
                "image": "",
            }

        image_tag = f"{self._image_prefix}:pr-{pr_number}"

        # Build the validation script that runs inside Docker
        fix_run_script = _build_fix_run_script(self._repo)

        try:
            with tempfile.TemporaryDirectory(prefix="hwe_valid_") as tmpdir:
                files_to_mount = [
                    ("fix-run.sh", fix_run_script),
                    ("tb_script.sh", tb_script),
                    ("candidate.patch", candidate_diff),
                ]

                # If base_commit is provided, write it to a file and mount
                # it so the fix-run.sh script can read it.  This is needed
                # because the caliptra-rtl and cva6 Docker images are NOT
                # shipped with the prepopulated ``{repo}_base_commit.txt``
                # file that fix-run.sh expects.
                base_commit_mount = ""
                if base_commit:
                    base_commit_path = os.path.join(tmpdir, f"{self._repo}_base_commit.txt")
                    with open(base_commit_path, "w", encoding="utf-8") as f:
                        f.write(base_commit.strip())
                    base_commit_mount = f"{base_commit_path}:/home/{self._repo}_base_commit.txt:ro"

                for name, content in files_to_mount:
                    path = os.path.join(tmpdir, name)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    os.chmod(path, 0o755 if name.endswith(".sh") else 0o644)

                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{tmpdir}/fix-run.sh:/home/fix-run.sh:ro",
                    "-v", f"{tmpdir}/tb_script.sh:/home/tb_script.sh:ro",
                    "-v", f"{tmpdir}/candidate.patch:/home/candidate.patch:ro",
                ]
                if base_commit_mount:
                    cmd.extend(["-v", base_commit_mount])
                # Mount the local repo to override the Docker embedded repo,
                # ensuring file content matches what the LLM saw.
                if source_repo_path:
                    home_dir = f"/home/{self._repo}"
                    cmd.extend(["-v", f"{source_repo_path}:{home_dir}"])
                    # Case repositories are cloned with Git alternates to
                    # avoid duplicating the full Ibex object database.  The
                    # absolute alternate path is otherwise invisible inside
                    # Docker, making valid base commits look missing.  Mount
                    # each referenced object directory at the same path.
                    for object_dir in _git_alternate_object_dirs(source_repo_path):
                        cmd.extend(["-v", f"{object_dir}:{object_dir}:ro"])
                cmd += [image_tag, "bash", "/home/fix-run.sh"]

                result = _run_docker(cmd)

                output = result.stdout + "\n" + result.stderr
                tests = _parse_test_markers(output)
                passed = bool(tests) and all(v == "PASS" for v in tests.values())

                return {
                    "passed": passed,
                    "tests": tests,
                    "returncode": result.returncode,
                    "error": "" if tests else "No HWE_BENCH_RESULTS markers found in Docker output",
                    "output": output,
                    "repo": self._repo,
                    "pr_number": pr_number,
                    "image": image_tag,
                }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "tests": {},
                "returncode": -1,
                "error": f"Docker validation timed out after {DOCKER_TIMEOUT}s",
                "output": "",
                "repo": self._repo,
                "pr_number": pr_number,
                "image": image_tag,
            }
        except FileNotFoundError:
            return {
                "passed": False,
                "tests": {},
                "returncode": -1,
                "error": "docker command not found; is Docker installed and accessible?",
                "output": "",
                "repo": self._repo,
                "pr_number": pr_number,
                "image": image_tag,
            }
        except Exception as e:
            return {
                "passed": False,
                "tests": {},
                "returncode": -1,
                "error": f"Docker validation error: {e}",
                "output": "",
                "repo": self._repo,
                "pr_number": pr_number,
                "image": image_tag,
            }


def _run_docker(cmd: list) -> subprocess.CompletedProcess:
    """Run a Docker command, falling back to sg docker if permission denied."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT
    )
    if "permission denied" not in (result.stderr or "").lower():
        return result

    # Fallback: use sg docker for group-based access
    docker_cmd_str = " ".join(cmd)  # pragma: allowlist secret
    sg_cmd = ["sg", "docker", "-c", docker_cmd_str]
    return subprocess.run(
        sg_cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT
    )


def _git_alternate_object_dirs(repo_path: str) -> List[str]:
    """Return existing object directories referenced by a repository clone."""
    dotgit = os.path.join(os.path.abspath(repo_path), ".git")
    git_dir = dotgit
    if os.path.isfile(dotgit):
        try:
            marker = open(dotgit, encoding="utf-8").read().strip()
        except OSError:
            return []
        if not marker.lower().startswith("gitdir:"):
            return []
        git_dir = marker.split(":", 1)[1].strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.abspath(os.path.join(repo_path, git_dir))

    alternates_file = os.path.join(git_dir, "objects", "info", "alternates")
    try:
        entries = open(alternates_file, encoding="utf-8").read().splitlines()
    except OSError:
        return []

    object_root = os.path.join(git_dir, "objects")
    result = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        path = entry if os.path.isabs(entry) else os.path.abspath(os.path.join(object_root, entry))
        if os.path.isdir(path) and path not in result:
            result.append(path)
    return result


def _build_fix_run_script(repo: str) -> str:
    """Build the fix-run.sh script that validates a candidate patch inside Docker."""
    home_dir = f"/home/{repo}"
    base_file = f"{home_dir}_base_commit.txt"
    return f"""#!/bin/bash
set -e

# Suppress "dubious ownership" errors when the repo is mounted from
# the host filesystem with a different UID than the Docker user.
/usr/bin/git config --global --add safe.directory {home_dir} 2>/dev/null || true

cd {home_dir}
/usr/bin/git reset --hard
/usr/bin/git clean -fdx

BASE_FILE="{base_file}"
if [[ -f "$BASE_FILE" ]]; then
  /usr/bin/git checkout "$(cat "$BASE_FILE")"
else
  echo "[ERROR] Missing $BASE_FILE (image not prepared?)"
  exit 1
fi

if [[ -s /home/candidate.patch ]]; then
  if ! /usr/bin/git apply --whitespace=nowarn /home/candidate.patch 2>/tmp/patch_error.log; then
    echo "CANDIDATE_PATCH_APPLY_FAILED"
    cat /tmp/patch_error.log
    exit 1
  fi
else
  echo "[ERROR] /home/candidate.patch missing or empty"
  exit 1
fi

if [[ -s /home/tb_script.sh ]]; then
  bash /home/tb_script.sh
  exit $?
fi

echo "[ERROR] /home/tb_script.sh missing or empty"
exit 1
"""


def _parse_test_markers(output: str) -> Dict[str, str]:
    """Parse HWE_BENCH_RESULTS markers from Docker output.

    Expected format:
        HWE_BENCH_RESULTS_START
        TEST: <name> ... <PASS|FAIL|SKIP>
        HWE_BENCH_RESULTS_END
    """
    tests: Dict[str, str] = {}
    if "HWE_BENCH_RESULTS_START" not in output:
        return tests

    # HWE-Bench test labels are not guaranteed to be a single token (for
    # example, ``csr_irq_flush pending_irq_taken``).  Anchor the marker to one
    # line and use the ellipsis as the name/status delimiter.
    for match in re.finditer(
        r"^\s*TEST:\s*(.+?)\s*\.{3}\s*(PASS|FAIL|SKIP)\s*$",
        output,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        tests[match.group(1).strip()] = match.group(2).upper()

    return tests


_IDENTIFIER_STOPWORDS = {
    "test", "fail", "failed", "failure", "pass", "passed", "error", "warning",
    "expected", "observed", "actual", "got", "saw", "cycle", "time", "line",
    "module", "input", "output", "wire", "logic", "fatal", "display", "finish",
    "build", "compile", "verilator", "docker", "candidate", "patch", "home",
    "root", "obj_dir", "make", "entering", "leaving", "directory", "timeout",
    "tb_fail", "tb_pass", "sim_fail", "sim_pass", "hwe_bench_results_start",
    "hwe_bench_results_end",
    "true", "false", "high", "low", "null", "none", "start", "end",
    "selection", "comparison", "unsigned", "range", "ordering", "outside",
}


def _tail(text: str, limit: int = 12000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _interesting_log_excerpt(output: str, limit: int = 12000) -> str:
    lines = []
    patterns = re.compile(
        r"(TB_FAIL|SIM FAIL|ASSERT|assert|fatal|%Error|%Warning|CANDIDATE_PATCH_APPLY_FAILED|"
        r"\bFAIL\b|mismatch|expected|observed|illegal|timeout|No HWE_BENCH_RESULTS)",
        re.IGNORECASE,
    )
    for line in (output or "").splitlines():
        if patterns.search(line):
            lines.append(line)
    if not lines:
        return _tail(output, limit)
    excerpt = "\n".join(lines[-120:])
    return _tail(excerpt, limit)


def _extract_file_line(text: str) -> tuple[str, Optional[int]]:
    for pattern in (
        r"(?P<file>(?:/[A-Za-z0-9_./$-]+|[A-Za-z0-9_./$-]+)\.(?:sv|v|svh|vh)):(?P<line>\d+)",
        r"%Error(?:-[A-Z0-9_]+)?:\s*(?P<file>[A-Za-z0-9_./$-]+\.(?:sv|v|svh|vh)):(?P<line>\d+)",
    ):
        match = re.search(pattern, text or "", re.IGNORECASE)
        if not match:
            continue
        file_path = _clean_repo_file(match.group("file"))
        try:
            line = int(match.group("line"))
        except (TypeError, ValueError):
            line = None
        return file_path, line
    return "", None


def _extract_file_lines(text: str, limit: int = 20) -> List[tuple[str, int]]:
    pattern = re.compile(
        r"(?P<file>(?:/[A-Za-z0-9_./$-]+|[A-Za-z0-9_./$-]+)\.(?:sv|v|svh|vh))"
        r":(?P<line>\d+)",
        re.IGNORECASE,
    )
    locations: List[tuple[str, int]] = []
    seen = set()
    for match in pattern.finditer(text or ""):
        location = (_clean_repo_file(match.group("file")), int(match.group("line")))
        if not location[0] or location in seen:
            continue
        seen.add(location)
        locations.append(location)
    return locations[-max(1, int(limit)):]


def _clean_repo_file(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    for marker in (
        "/home/ibex/",
        "/home/cva6/",
        "/home/caliptra-rtl/",
        "/home/opentitan/",
    ):
        if marker in text:
            return text.split(marker, 1)[1]
    for marker in ("rtl/", "core/", "src/", "hw/", "include/", "vendor/"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:]
    return text.lstrip("./")


def _extract_signal_tokens(text: str, limit: int = 16) -> List[str]:
    tokens: List[str] = []
    for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*=", text or ""):
        lowered = token.lower()
        if lowered in _IDENTIFIER_STOPWORDS or len(token) <= 2 or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= limit:
            return tokens
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", text or ""):
        lowered = token.lower()
        if lowered in _IDENTIFIER_STOPWORDS or len(token) <= 2:
            continue
        if token.upper() == token and "_" not in token:
            continue
        if re.fullmatch(r"[0-9a-fA-F_xzXZ]+", token):
            continue
        if "_" not in token and "$" not in token:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _extract_module_name(text: str) -> str:
    for pattern in (
        r"--top-module\s+([A-Za-z_][A-Za-z0-9_$]*)",
        r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)",
        r"\bV([A-Za-z_][A-Za-z0-9_$]*)\b",
    ):
        match = re.search(pattern, text or "")
        if match:
            return match.group(1)
    return ""


def hwe_result_to_validation_report(result: dict) -> ValidationReport:
    """Convert raw HWE Docker validator output into KGVeri ValidationReport."""
    result = result or {}
    output = result.get("output") or ""
    tests = result.get("tests") or {}
    repo = str(result.get("repo") or "")
    pr_number = result.get("pr_number")
    image = str(result.get("image") or "")
    metadata = {
        "validator": "hwe_bench_docker_final",
        "repo": repo,
        "pr_number": pr_number,
        "image": image,
        "tests": tests,
        "returncode": result.get("returncode"),
        "error": result.get("error") or "",
        "output_tail": _tail(output, 12000),
    }

    if result.get("passed"):
        return build_validation_report([], metadata=metadata)

    raw_excerpt = _interesting_log_excerpt(output)
    diagnostic_locations = _extract_file_lines(raw_excerpt or output)
    if diagnostic_locations:
        file_path, line_no = diagnostic_locations[-1]
    else:
        file_path, line_no = _extract_file_line(raw_excerpt or output)
    metadata["diagnostic_locations"] = [
        {"file": path, "line": line}
        for path, line in diagnostic_locations
    ]
    signal_source = "\n".join(
        line for line in (raw_excerpt or output).splitlines()
        if not line.strip().startswith("TEST:") and "HWE_BENCH_RESULTS" not in line
    )
    signals = _extract_signal_tokens(signal_source or raw_excerpt or output)
    module = _extract_module_name(raw_excerpt or output)
    evidence: List[ValidationEvidence] = []

    if "CANDIDATE_PATCH_APPLY_FAILED" in output:
        evidence.append(
            ValidationEvidence(
                type="HWE_PATCH_APPLY_FAILED",
                severity="blocker",
                message="Candidate diff did not apply inside the HWE-Bench Docker worktree.",
                file=file_path,
                line=line_no,
                signal=signals[0] if signals else "",
                module=module,
                raw_log=raw_excerpt,
                validator="hwe_bench_docker_final",
            )
        )
        return build_validation_report(evidence, metadata=metadata)

    failing_tests = {name: status for name, status in tests.items() if str(status).upper() != "PASS"}
    if failing_tests:
        for test_name, status in failing_tests.items():
            evidence.append(
                ValidationEvidence(
                    type="HWE_FAIL_TO_PASS_TEST_FAILED",
                    severity="blocker",
                    message=f"HWE-Bench Docker test {test_name} ended with status {status}.",
                    file=file_path,
                    line=line_no,
                    module=module,
                    signal=signals[0] if signals else "",
                    check_id=str(test_name),
                    raw_log=raw_excerpt,
                    validator="hwe_bench_docker_final",
                )
            )
    else:
        error = result.get("error") or "HWE-Bench Docker validation failed without PASS/FAIL test markers."
        evidence.append(
            ValidationEvidence(
                type="HWE_DOCKER_VALIDATION_FAILED",
                severity="blocker",
                message=str(error),
                file=file_path,
                line=line_no,
                module=module,
                signal=signals[0] if signals else "",
                raw_log=raw_excerpt,
                validator="hwe_bench_docker_final",
            )
        )

    primary_location = (file_path, line_no)
    for diagnostic_file, diagnostic_line in reversed(diagnostic_locations):
        if (diagnostic_file, diagnostic_line) == primary_location:
            continue
        evidence.append(
            ValidationEvidence(
                type="HWE_DIAGNOSTIC_LOCATION",
                severity="info",
                message=(
                    "Additional RTL location reported by the HWE-Bench compiler or simulation log."
                ),
                file=diagnostic_file,
                line=diagnostic_line,
                module=module,
                raw_log="",
                validator="hwe_bench_docker_final",
            )
        )

    for extra_signal in signals[1:6]:
        evidence.append(
            ValidationEvidence(
                type="HWE_FAILURE_SIGNAL_ANCHOR",
                severity="info",
                message=f"Signal-like token extracted from HWE failure output: {extra_signal}",
                file=file_path,
                line=line_no,
                module=module,
                signal=extra_signal,
                raw_log="",
                validator="hwe_bench_docker_final",
            )
        )

    return build_validation_report(evidence, metadata=metadata)
