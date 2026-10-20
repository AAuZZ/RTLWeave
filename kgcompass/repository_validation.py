from __future__ import annotations

"""Lightweight execution of repository-provided build or test targets.

The validator intentionally accepts an argument vector rather than shell text.
Default commands come from a small, trusted profile table for the three RTL
repositories used by KGVeri.  Callers may provide another explicit argv vector
for a controlled experiment, but benchmark prose is never parsed or executed.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from diagnostic_evidence import build_validation_report
    from repair_types import ValidationEvidence, ValidationReport
except Exception:  # pragma: no cover - package import path
    from .diagnostic_evidence import build_validation_report
    from .repair_types import ValidationEvidence, ValidationReport


SCHEMA_VERSION = "kgveri-repository-validation-v1"
VALIDATOR_NAME = "repository_regression"
SUPPORTED_REPOSITORIES = frozenset({"ibex", "cva6", "caliptra-rtl"})
REPOSITORY_ALIASES = {
    "ibex": "ibex",
    "lowrisc_ibex": "ibex",
    "cva6": "cva6",
    "openhwgroup_cva6": "cva6",
    "caliptra": "caliptra-rtl",
    "caliptra-rtl": "caliptra-rtl",
    "chipsalliance_caliptra-rtl": "caliptra-rtl",
}


@dataclass(frozen=True)
class RepositoryValidationProfile:
    repo: str
    name: str
    argv: tuple[str, ...]
    working_subdir: str = "."
    timeout_seconds: int = 600
    environment: Mapping[str, str] = field(default_factory=dict)
    required_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "name": self.name,
            "argv": list(self.argv),
            "working_subdir": self.working_subdir,
            "timeout_seconds": self.timeout_seconds,
            "environment": dict(self.environment),
            "required_files": list(self.required_files),
        }


# These profiles invoke bounded targets that are defined by each repository.
# They intentionally avoid complete nightly regressions.  A historical checkout
# that does not provide the selected target is reported as unsupported
# infrastructure rather than silently replaced with another static lint pass.
TRUSTED_PROFILES: dict[str, RepositoryValidationProfile] = {
    "ibex": RepositoryValidationProfile(
        repo="ibex",
        name="ibex_native_simple_system_lint",
        argv=(
            "fusesoc",
            "--cores-root",
            ".",
            "run",
            "--target=lint",
            "--tool=verilator",
            "lowrisc:ibex:ibex_simple_system",
        ),
        timeout_seconds=300,
        required_files=("examples/simple_system/ibex_simple_system.core",),
    ),
    "cva6": RepositoryValidationProfile(
        repo="cva6",
        name="cva6_native_verilate",
        argv=("make", "verilate", "NUM_JOBS=2"),
        timeout_seconds=900,
        environment={
            "CVA6_REPO_DIR": "{repo}",
            "TARGET_CFG": "cv64a6_imafdc_sv39",
        },
        required_files=("Makefile",),
    ),
    "caliptra-rtl": RepositoryValidationProfile(
        repo="caliptra-rtl",
        name="caliptra_native_smoke_test",
        argv=(
            "make",
            "-C",
            "tools/scripts",
            "TESTNAME=smoke_test_soc_ifc",
            "verilator",
            "VERILATOR_RUN_ARGS=+CLP_REGRESSION",
        ),
        timeout_seconds=1200,
        environment={"CALIPTRA_ROOT": "{repo}"},
        required_files=("tools/scripts/Makefile",),
    ),
}


@dataclass
class RepositoryRunResult:
    phase: str
    repo: str
    profile_name: str
    command_id: str
    argv: list[str]
    working_subdir: str
    cwd: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    infra_error: str = ""
    failure_signatures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.infra_error and not self.timed_out and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({"schema_version": SCHEMA_VERSION, "passed": self.passed})
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositoryRunResult":
        return cls(
            phase=str(value.get("phase") or "baseline"),
            repo=str(value.get("repo") or ""),
            profile_name=str(value.get("profile_name") or ""),
            command_id=str(value.get("command_id") or ""),
            argv=[str(item) for item in value.get("argv") or []],
            working_subdir=str(value.get("working_subdir") or "."),
            cwd=str(value.get("cwd") or ""),
            returncode=int(value.get("returncode", -1)),
            stdout=str(value.get("stdout") or ""),
            stderr=str(value.get("stderr") or ""),
            duration_seconds=float(value.get("duration_seconds") or 0.0),
            timed_out=bool(value.get("timed_out")),
            infra_error=str(value.get("infra_error") or ""),
            failure_signatures=[str(item) for item in value.get("failure_signatures") or []],
        )


CommandRunner = Callable[
    [list[str], Path, dict[str, str], int], subprocess.CompletedProcess[str]
]


class RepositoryRegressionValidator:
    """Run the same trusted repository command on base and candidate trees."""

    def __init__(
        self,
        repo: str,
        command_argv: Sequence[str] | None = None,
        working_subdir: str | None = None,
        timeout_seconds: int | None = None,
        runner: CommandRunner | None = None,
        docker_image: str | None = None,
    ) -> None:
        normalized_repo = normalize_repository_name(repo)
        profile = get_repository_profile(normalized_repo)
        explicit_command = command_argv is not None
        argv = _validate_argv(command_argv if explicit_command else profile.argv)
        subdir = working_subdir if working_subdir is not None else (
            "." if explicit_command else profile.working_subdir
        )
        self.working_subdir = _validate_working_subdir(subdir)
        self.timeout_seconds = int(
            timeout_seconds if timeout_seconds is not None else profile.timeout_seconds
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.repo = normalized_repo
        self.profile_name = f"{normalized_repo}_explicit_argv" if explicit_command else profile.name
        self.command_argv = tuple(argv)
        self.profile_environment = dict(profile.environment)
        self.required_files = () if explicit_command else tuple(profile.required_files)
        self.docker_image = str(docker_image or "").strip()
        self.command_id = _command_id(
            self.repo,
            self.profile_name,
            self.command_argv,
            self.working_subdir,
            {
                **self.profile_environment,
                "__execution_mode": "docker" if self.docker_image else "host",
                "__docker_image": self.docker_image,
                "__checkout_binding": "after_shell_setup_v2",
            },
        )
        self._runner = runner or (self._docker_runner if self.docker_image else _default_runner)

    def profile_base(
        self,
        base_repo: str | os.PathLike[str],
        artifact_dir: str | os.PathLike[str],
    ) -> RepositoryRunResult:
        artifact_path = Path(artifact_dir).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        self._persist_profile(artifact_path)
        result = self.run_once(base_repo, phase="baseline")
        self._persist_run(artifact_path, result)
        return result

    def validate_candidate(
        self,
        baseline: RepositoryRunResult | Mapping[str, Any],
        candidate_repo: str | os.PathLike[str],
        artifact_dir: str | os.PathLike[str],
    ) -> ValidationReport:
        if not isinstance(baseline, RepositoryRunResult):
            baseline = RepositoryRunResult.from_dict(baseline)
        self._validate_baseline_identity(baseline)

        artifact_path = Path(artifact_dir).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        self._persist_profile(artifact_path)
        _write_json(artifact_path / "baseline_reference.json", baseline.to_dict())
        candidate = self.run_once(candidate_repo, phase="candidate")
        self._persist_run(artifact_path, candidate)
        report = self.compare_runs(baseline, candidate, artifact_path=artifact_path)
        _write_json(artifact_path / "validation_report.json", report.to_dict())
        return report

    def validate_pair(
        self,
        base_repo: str | os.PathLike[str],
        candidate_repo: str | os.PathLike[str],
        artifact_dir: str | os.PathLike[str],
    ) -> ValidationReport:
        artifact_path = Path(artifact_dir).resolve()
        baseline = self.profile_base(base_repo, artifact_path)
        return self.validate_candidate(baseline, candidate_repo, artifact_path)

    def run_once(
        self,
        repo_path: str | os.PathLike[str],
        *,
        phase: str,
    ) -> RepositoryRunResult:
        if phase not in {"baseline", "candidate", "clean_replay"}:
            raise ValueError(f"unsupported repository validation phase: {phase}")
        root = Path(repo_path).resolve()
        cwd, cwd_error = _resolve_working_directory(root, self.working_subdir)
        if cwd_error:
            return RepositoryRunResult(
                phase=phase,
                repo=self.repo,
                profile_name=self.profile_name,
                command_id=self.command_id,
                argv=list(self.command_argv),
                working_subdir=self.working_subdir,
                cwd=str(cwd),
                returncode=-1,
                infra_error=cwd_error,
                failure_signatures=[f"infrastructure:{_normalize_text(cwd_error)}"],
            )
        missing_files = [
            rel for rel in self.required_files if not (root / rel).is_file()
        ]
        if missing_files:
            reason = (
                "repository checkout does not provide the configured lightweight "
                f"target: missing {missing_files}"
            )
            return RepositoryRunResult(
                phase=phase,
                repo=self.repo,
                profile_name=self.profile_name,
                command_id=self.command_id,
                argv=list(self.command_argv),
                working_subdir=self.working_subdir,
                cwd=str(cwd),
                returncode=-1,
                infra_error=reason,
                failure_signatures=[f"infrastructure:{_normalize_text(reason)}"],
            )

        environment = dict(os.environ)
        environment["KGVERI_REPO_ROOT"] = str(root)
        for key, raw_value in self.profile_environment.items():
            environment[str(key)] = str(raw_value).replace("{repo}", str(root))

        started = time.monotonic()
        stdout = ""
        stderr = ""
        returncode = -1
        timed_out = False
        infra_error = ""
        try:
            completed = self._runner(
                list(self.command_argv), cwd, environment, self.timeout_seconds
            )
            returncode = int(completed.returncode)
            stdout = _coerce_text(completed.stdout)
            stderr = _coerce_text(completed.stderr)
            infra_error = _detect_infrastructure_failure(returncode, stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_text(exc.stdout)
            stderr = _coerce_text(exc.stderr)
            timed_out = True
            infra_error = f"repository validation timed out after {self.timeout_seconds}s"
        except FileNotFoundError as exc:
            stderr = str(exc)
            infra_error = f"required validation tool is unavailable: {self.command_argv[0]}"
        except OSError as exc:
            stderr = str(exc)
            infra_error = f"repository validation could not start: {exc}"

        duration = max(0.0, time.monotonic() - started)
        combined = _combined_output(stdout, stderr)
        signatures = normalize_failure_signatures(
            combined,
            repo_root=root,
            returncode=returncode,
        )
        if infra_error:
            signatures = _dedupe(
                [f"infrastructure:{_normalize_text(infra_error)}", *signatures]
            )
        return RepositoryRunResult(
            phase=phase,
            repo=self.repo,
            profile_name=self.profile_name,
            command_id=self.command_id,
            argv=list(self.command_argv),
            working_subdir=self.working_subdir,
            cwd=str(cwd),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
            infra_error=infra_error,
            failure_signatures=signatures,
        )

    def compare_runs(
        self,
        baseline: RepositoryRunResult,
        candidate: RepositoryRunResult,
        *,
        artifact_path: Path | None = None,
    ) -> ValidationReport:
        self._validate_baseline_identity(baseline)
        if candidate.command_id != self.command_id or candidate.repo != self.repo:
            raise ValueError("candidate was produced by a different repository command")

        evidence: list[ValidationEvidence] = []
        if baseline.infra_error or baseline.timed_out:
            evidence.append(
                self._infra_evidence("baseline", baseline)
            )
        if candidate.infra_error or candidate.timed_out:
            evidence.append(
                self._infra_evidence("candidate", candidate)
            )

        baseline_signatures = set(baseline.failure_signatures)
        candidate_signatures = set(candidate.failure_signatures)
        new_failures = sorted(candidate_signatures - baseline_signatures)

        if not evidence:
            if baseline.passed and candidate.passed:
                evidence.append(
                    ValidationEvidence(
                        type="REPOSITORY_REGRESSION_PASS",
                        severity="info",
                        message="Repository regression passed on both base and candidate.",
                        raw_log=_tail(_combined_output(candidate.stdout, candidate.stderr)),
                        validator=VALIDATOR_NAME,
                    )
                )
            elif baseline.passed:
                evidence.append(
                    ValidationEvidence(
                        type="REPOSITORY_REGRESSION_CANDIDATE_FAILED",
                        severity="blocker",
                        message=(
                            "Repository regression passed on the base but failed on the "
                            "candidate."
                        ),
                        expected="base=PASS; candidate=PASS",
                        observed=f"candidate=FAIL; signatures={new_failures[:12]}",
                        raw_log=_tail(_combined_output(candidate.stdout, candidate.stderr)),
                        validator=VALIDATOR_NAME,
                    )
                )
            elif candidate.passed:
                evidence.append(
                    ValidationEvidence(
                        type="REPOSITORY_REGRESSION_IMPROVED",
                        severity="info",
                        message=(
                            "Repository regression failed on the base and passed on the "
                            "candidate; no regression was introduced."
                        ),
                        expected=f"existing={sorted(baseline_signatures)[:12]}",
                        observed="candidate=PASS",
                        raw_log=_tail(_combined_output(candidate.stdout, candidate.stderr)),
                        validator=VALIDATOR_NAME,
                    )
                )
            elif new_failures:
                evidence.append(
                    ValidationEvidence(
                        type="REPOSITORY_REGRESSION_NEW_FAILURE",
                        severity="blocker",
                        message=(
                            "Candidate introduced normalized repository-regression failure "
                            f"signatures: {new_failures[:12]}"
                        ),
                        expected=f"subset of baseline signatures ({len(baseline_signatures)})",
                        observed=f"new signatures ({len(new_failures)})",
                        raw_log=_tail(_combined_output(candidate.stdout, candidate.stderr)),
                        validator=VALIDATOR_NAME,
                    )
                )
            else:
                evidence.append(
                    ValidationEvidence(
                        type="REPOSITORY_REGRESSION_NO_NEW_FAILURE",
                        severity="info",
                        message=(
                            "Base and candidate both have repository-regression failures, "
                            "but the candidate introduced no normalized failure signature."
                        ),
                        expected=f"existing={sorted(baseline_signatures)[:12]}",
                        observed=f"candidate={sorted(candidate_signatures)[:12]}",
                        raw_log=_tail(_combined_output(candidate.stdout, candidate.stderr)),
                        validator=VALIDATOR_NAME,
                    )
                )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "validator": VALIDATOR_NAME,
            "repo": self.repo,
            "profile_name": self.profile_name,
            "command_id": self.command_id,
            "argv": list(self.command_argv),
            "working_subdir": self.working_subdir,
            "timeout_seconds": self.timeout_seconds,
            "execution_mode": "docker" if self.docker_image else "host",
            "docker_image": self.docker_image,
            "network": "none" if self.docker_image else "host_process",
            "baseline_returncode": baseline.returncode,
            "candidate_returncode": candidate.returncode,
            "baseline_passed": baseline.passed,
            "candidate_passed": candidate.passed,
            "baseline_failure_signatures": sorted(baseline_signatures),
            "candidate_failure_signatures": sorted(candidate_signatures),
            "new_failure_signatures": new_failures,
            "artifact_dir": str(artifact_path) if artifact_path else "",
        }
        return build_validation_report(evidence, metadata=metadata)

    def _infra_evidence(
        self,
        phase: str,
        result: RepositoryRunResult,
    ) -> ValidationEvidence:
        reason = result.infra_error or "repository validation infrastructure failure"
        return ValidationEvidence(
            type="REPOSITORY_REGRESSION_INFRA_ERROR",
            severity="blocker",
            message=f"{phase} repository regression could not run: {reason}",
            observed=f"returncode={result.returncode}; timed_out={result.timed_out}",
            raw_log=_tail(_combined_output(result.stdout, result.stderr)),
            validator=VALIDATOR_NAME,
        )

    def _validate_baseline_identity(self, baseline: RepositoryRunResult) -> None:
        if baseline.phase != "baseline":
            raise ValueError("repository baseline result must have phase='baseline'")
        if baseline.repo != self.repo or baseline.command_id != self.command_id:
            raise ValueError("baseline was produced by a different repository command")

    def _persist_profile(self, artifact_path: Path) -> None:
        _write_json(
            artifact_path / "profile.json",
            {
                "schema_version": SCHEMA_VERSION,
                "repo": self.repo,
                "profile_name": self.profile_name,
                "command_id": self.command_id,
                "argv": list(self.command_argv),
                "working_subdir": self.working_subdir,
                "timeout_seconds": self.timeout_seconds,
                "environment_keys": sorted(self.profile_environment),
                "required_files": list(self.required_files),
                "execution_mode": "docker" if self.docker_image else "host",
                "docker_image": self.docker_image,
                "network": "none" if self.docker_image else "host_process",
                "repository_mount": (
                    "disposable_copy" if self.docker_image else "local_checkout"
                ),
            },
        )

    def _docker_runner(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        root = Path(env["KGVERI_REPO_ROOT"]).resolve()
        relative_cwd = cwd.resolve().relative_to(root)
        container_cwd = "/repo" if str(relative_cwd) == "." else f"/repo/{relative_cwd.as_posix()}"
        tool = argv[0]
        translated_env = {
            key: str(env.get(key, "")).replace(str(root), "/repo")
            for key in self.profile_environment
        }

        # Repository-native targets normally create build products.  Execute
        # them in a disposable checkout copy so neither the immutable baseline
        # nor the candidate worktree is polluted by validation artifacts.
        with tempfile.TemporaryDirectory(prefix=f"kgveri_{self.repo}_regression_") as tmp:
            disposable_repo = Path(tmp) / "repo"
            shutil.copytree(
                root,
                disposable_repo,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "build", "out", "work", "work-*"
                ),
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=512m",
                "-v",
                f"{disposable_repo}:/repo:rw",
                "-w",
                container_cwd,
                "-e",
                "HOME=/tmp",
            ]
            for key, value in sorted(translated_env.items()):
                command.extend(["-e", f"{key}={value}"])
            # The benchmark images expose their toolchains through BASH_ENV
            # (for example, FuseSoC in a micromamba environment).  Positional
            # forwarding keeps the validated argv intact and does not evaluate
            # any command text derived from the issue or patch.
            command.extend([
                self.docker_image,
                "/bin/bash",
                "-c",
                (
                    'for d in /tools/verilator*/bin /tools/verilator/*/bin '
                    '/opt/verilator/bin; do '
                    'if [[ -x "$d/verilator" ]]; then export PATH="$d:$PATH"; break; fi; '
                    'done; exec "$@"'
                ),
                "kgveri-repository-validation",
                # Image BASH_ENV files may reset CALIPTRA_ROOT/CVA6_REPO_DIR
                # to the baked-in checkout after Docker applies -e. Reassert
                # the candidate paths after shell setup, and prevent child
                # shells from silently switching back to the image baseline.
                "env",
                "BASH_ENV=/dev/null",
                *[f"{key}={value}" for key, value in sorted(translated_env.items())],
                tool,
                *argv[1:],
            ])
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                check=False,
            )

    @staticmethod
    def _persist_run(artifact_path: Path, result: RepositoryRunResult) -> None:
        _write_json(artifact_path / f"{result.phase}.json", result.to_dict())
        command = json.dumps(result.argv, ensure_ascii=False)
        (artifact_path / f"{result.phase}.log").write_text(
            f"COMMAND_ARGV: {command}\n"
            f"CWD: {result.cwd}\n"
            f"RETURN_CODE: {result.returncode}\n"
            f"TIMED_OUT: {result.timed_out}\n"
            f"INFRA_ERROR: {result.infra_error}\n"
            f"DURATION_SECONDS: {result.duration_seconds:.6f}\n\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n",
            encoding="utf-8",
        )


def normalize_repository_name(repo: str) -> str:
    normalized = str(repo or "").strip().lower().replace("/", "_")
    try:
        result = REPOSITORY_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported repository {repo!r}; expected one of {sorted(SUPPORTED_REPOSITORIES)}"
        ) from exc
    return result


def get_repository_profile(repo: str) -> RepositoryValidationProfile:
    return TRUSTED_PROFILES[normalize_repository_name(repo)]


_FAILURE_LINE = re.compile(
    r"(?:%Error|\b(?:error|fatal|fail(?:ed|ure)?|assert(?:ion)?|timeout|"
    r"undefined reference|segmentation fault|command not found)\b)",
    re.IGNORECASE,
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_FILE_LOCATION = re.compile(
    r"(?P<file>(?:[A-Za-z]:)?[^\s:]+\.(?:svh?|vh?|cc|cpp|c|h|py))"
    r":(?P<line>\d+)(?::(?P<column>\d+))?",
    re.IGNORECASE,
)


def normalize_failure_signatures(
    output: str,
    repo_root: str | os.PathLike[str] | None = None,
    returncode: int | None = None,
) -> list[str]:
    """Extract stable failure identities for base/candidate delta comparison."""

    text = _ANSI_ESCAPE.sub("", _coerce_text(output)).replace("\\", "/")
    if repo_root is not None:
        root = str(Path(repo_root).resolve()).replace("\\", "/").rstrip("/")
        if root:
            text = text.replace(root, "<repo>")

    signatures: list[str] = []
    for match in re.finditer(
        r"TEST:\s*(?P<name>[^\s.]+)\s*\.{2,}\s*(?P<status>FAIL|ERROR|TIMEOUT)",
        text,
        re.IGNORECASE,
    ):
        signatures.append(
            f"test:{match.group('name').lower()}:{match.group('status').lower()}"
        )
    for match in re.finditer(
        r"(?:^|\n)FAILED\s+(?P<name>[^\s]+)", text, re.IGNORECASE
    ):
        signatures.append(f"test:{_normalize_text(match.group('name'))}:failed")

    # Successful repository commands often print benign fallback or assertion
    # setup messages containing words such as "failed" or "assert".  Treat
    # only explicit test-result markers as signatures on exit code zero; the
    # broader diagnostic scan is reserved for a failing command.
    if returncode not in (None, 0):
        failure_lines = [line for line in text.splitlines() if _FAILURE_LINE.search(line)]
        for line in failure_lines[-200:]:
            normalized = _normalize_failure_line(line)
            if normalized:
                signatures.append(normalized)

    if not signatures and returncode not in (None, 0):
        nonempty = [_normalize_failure_line(line) for line in text.splitlines() if line.strip()]
        signatures.extend(item for item in nonempty[-5:] if item)
        if not signatures:
            signatures.append(f"exit_code:{int(returncode)}")
    return sorted(_dedupe(signatures))


def _normalize_failure_line(line: str) -> str:
    value = _ANSI_ESCAPE.sub("", str(line or "")).replace("\\", "/")
    value = _FILE_LOCATION.sub(lambda match: f"{match.group('file')}:<line>", value)
    value = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", value)
    value = re.sub(
        r"\b(?:time|cycle|seed|pid)\s*[=:]?\s*\d+(?:\.\d+)?\b",
        lambda match: re.sub(r"\d+(?:\.\d+)?", "<n>", match.group(0)),
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|us|ns|seconds?|secs?|s)\b", "<time>", value, flags=re.IGNORECASE)
    return _normalize_text(value)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _validate_argv(argv: Sequence[str] | None) -> list[str]:
    if argv is None or isinstance(argv, (str, bytes)):
        raise TypeError("repository validation command must be an argv sequence, not shell text")
    result = [str(item) for item in argv]
    if not result or not result[0].strip():
        raise ValueError("repository validation argv must contain an executable")
    if any("\x00" in item or "\n" in item or "\r" in item for item in result):
        raise ValueError("repository validation argv contains an invalid control character")
    executable = Path(result[0]).name.lower()
    shell_executables = {
        "sh", "bash", "dash", "zsh", "ksh", "fish", "pwsh", "powershell", "cmd", "cmd.exe"
    }
    if executable in shell_executables:
        raise ValueError("shell interpreters are not accepted as repository validation commands")
    if executable == "env":
        first_command = next(
            (Path(item).name.lower() for item in result[1:] if "=" not in item and not item.startswith("-")),
            "",
        )
        if first_command in shell_executables:
            raise ValueError("shell interpreters are not accepted through env")
    return result


def _validate_working_subdir(value: str | os.PathLike[str]) -> str:
    path = Path(str(value or "."))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("working_subdir must stay inside the repository")
    return path.as_posix() or "."


def _resolve_working_directory(root: Path, subdir: str) -> tuple[Path, str]:
    if not root.is_dir():
        return root, f"repository checkout is unavailable: {root}"
    cwd = (root / subdir).resolve()
    try:
        cwd.relative_to(root)
    except ValueError:
        return cwd, f"repository validation working directory escapes checkout: {cwd}"
    if not cwd.is_dir():
        return cwd, f"repository validation working directory is unavailable: {cwd}"
    return cwd, ""


def _default_runner(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        check=False,
    )


def _detect_infrastructure_failure(returncode: int, stdout: str, stderr: str) -> str:
    if returncode == 127:
        return "required validation tool is unavailable (exit code 127)"
    if returncode == 126:
        return "repository validation tool is not executable (exit code 126)"
    output = _combined_output(stdout, stderr)
    patterns = (
        r"(?:^|\n)[^\n]*\bcommand not found\b",
        r"(?:^|\n)[^\n]*\b(?:fusesoc|verilator|riscv\S*)\b[^\n]*(?:\bnot found\b|no such file)",
        r"no rule to make target",
        r"no targets specified and no makefile found",
        r"\briscv(?:-v)?\b[^\n]*\bnot set\b",
        r"risc-v toolchain not found",
        r"modulenotfounderror",
        r"cannot connect to the docker daemon",
        r"permission denied while trying to connect to the docker daemon",
        r"docker:\s*(?:command )?not found",
        r"no such image",
    )
    if any(re.search(pattern, output, re.IGNORECASE) for pattern in patterns):
        return "required repository-validation infrastructure is unavailable"
    return ""


def _command_id(
    repo: str,
    profile_name: str,
    argv: Sequence[str],
    working_subdir: str,
    environment: Mapping[str, str],
) -> str:
    payload = json.dumps(
        {
            "repo": repo,
            "profile_name": profile_name,
            "argv": list(argv),
            "working_subdir": working_subdir,
            "environment": dict(environment),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combined_output(stdout: str, stderr: str) -> str:
    return (str(stdout or "") + "\n" + str(stderr or "")).strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tail(value: str, limit: int = 12000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[-limit:]


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RepositoryRegressionValidator",
    "RepositoryRunResult",
    "RepositoryValidationProfile",
    "SUPPORTED_REPOSITORIES",
    "TRUSTED_PROFILES",
    "get_repository_profile",
    "normalize_failure_signatures",
    "normalize_repository_name",
]
