"""Repository-grounded validation harness: native assets, fixed commands, typed feedback.

This is a small optional adapter, not an LLM agent or a replacement for an
independent correctness test. Discovery is advisory; execution uses a reviewed
manifest. It reuses the repository validator's isolated, network-disabled runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml

from .diagnostic_evidence import anchors_from_evidence, build_validation_report
from .repair_types import ValidationEvidence, ValidationReport
from .repository_validation import RepositoryRegressionValidator, RepositoryRunResult


@dataclass
class ValidationTask:
    task_id: str
    kind: str  # build, lint, simulation
    argv: list[str]
    assets: list[str]
    working_subdir: str = "."
    timeout_seconds: int = 180
    pass_pattern: str = ""
    fail_pattern: str = ""
    rationale: str = ""
    # Only an explicit behavioral/reproduction argument may justify this label.
    # Merely selecting a relevant module or seeing baseline FAIL is insufficient.
    issue_claim: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.task_id):
            raise ValueError("task_id must be a simple name")
        if self.kind not in {"build", "lint", "simulation"} or not self.assets:
            raise ValueError("task requires a supported kind and repository assets")
        if self.kind == "simulation" and not self.pass_pattern:
            raise ValueError("simulation requires an observed completion/pass marker")
        for pattern in (self.pass_pattern, self.fail_pattern):
            re.compile(pattern)


def discover_assets(repo: Path, query: str = "", related_files: list[str] = ()) -> list[dict]:
    """Shortlist tracked build/test descriptions; never scan benchmark artifacts.

    Related source paths can come from the localizer's support paths. Lexical
    overlap is only a cheap selection hint, not a behavioral-coverage estimate.
    """
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"], check=True,
        capture_output=True, text=True,
    ).stdout.split("\0")
    terms = set(re.findall(r"[a-z][a-z0-9_]{2,}", (query + " " + " ".join(related_files)).lower()))
    results = []
    for rel in sorted(tracked):
        path = repo / rel
        if not (path.name == "Makefile" or path.suffix == ".core") or not path.is_file():
            continue
        if not path.resolve().is_relative_to(repo.resolve()):
            continue
        body = path.read_text(errors="replace")[:80000]
        choices = []
        if path.suffix == ".core":
            try:
                core = yaml.safe_load(body) or {}
                for target in (core.get("targets") or {}):
                    if target in {"lint", "sim"}:
                        choices.append({"target": target, "argv": [
                            "fusesoc", "--cores-root", ".", "run", f"--target={target}",
                            "--tool=verilator", str(core["name"]),
                        ], "working_subdir": "."})
            except (yaml.YAMLError, KeyError, TypeError, AttributeError):
                continue
        else:
            targets = re.findall(r"^([A-Za-z][\w.-]*)\s*:(?!=)", body, re.M)
            for target in targets:
                if target in {"test", "check", "lint", "verilator", "verilate"} or target.startswith(("lint-", "run-", "build-")):
                    choices.append({"target": target, "argv": ["make", "-j2", target],
                                    "working_subdir": path.parent.relative_to(repo).as_posix()})
        if choices:
            words = set(re.findall(r"[a-z][a-z0-9_]{2,}", (rel + " " + body).lower()))
            hits = sorted(terms & words)
            results.append({"asset": rel, "matched_terms": hits, "score": len(hits),
                            "commands": choices})
    return sorted(results, key=lambda r: (-r["score"], r["asset"]))


def _hash_assets(root: Path, assets: list[str]) -> dict[str, str]:
    hashes = {}
    for rel in assets:
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"validation asset unavailable or outside repository: {rel}")
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def execution_status(task: ValidationTask, run: RepositoryRunResult) -> str:
    if run.timed_out:
        return "TIMEOUT"
    if run.infra_error:
        return "UNAVAILABLE"
    output = run.stdout + "\n" + run.stderr
    if run.returncode or (task.fail_pattern and re.search(task.fail_pattern, output, re.M)):
        return "FAIL"
    if task.pass_pattern and not re.search(task.pass_pattern, output, re.M):
        return "INCONCLUSIVE"
    return "PASS"


def _locations(output: str, root: Path) -> list[tuple[str, int]]:
    """Return only source coordinates that can be grounded in this checkout."""
    locations = []
    for filename, line in re.findall(r"([\w./-]+\.(?:svh?|vh?|cc|cpp)):(\d+)", output):
        rel = filename.removeprefix("/repo/")
        path = (root / rel).resolve()
        if not (path.is_relative_to(root.resolve()) and path.is_file()):
            matches = list(root.rglob(Path(rel).name))
            if len(matches) != 1:
                continue
            rel = matches[0].relative_to(root).as_posix()
        pair = (rel, int(line))
        if pair not in locations:
            locations.append(pair)
    return locations[:12]


class RepositoryValidationHarness:
    """Run reviewed tasks against the base and each candidate, returning the
    existing ValidationReport interface consumed by diagnosis and KG grounding.
    Native test checks/configuration are held fixed across the comparison.
    """

    def __init__(self, repo_kind: str, base_repo: Path, docker_image: str,
                 tasks: list[ValidationTask], output: Path):
        if not docker_image or not tasks or len({t.task_id for t in tasks}) != len(tasks):
            raise ValueError("need a local Docker image and uniquely named tasks")
        self.base_repo = base_repo.resolve()
        self.tasks, self.output = tasks, output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.validators = {t.task_id: RepositoryRegressionValidator(
            repo_kind, command_argv=t.argv, working_subdir=t.working_subdir,
            timeout_seconds=t.timeout_seconds, docker_image=docker_image,
        ) for t in tasks}
        self.hashes = {t.task_id: _hash_assets(self.base_repo, t.assets) for t in tasks}
        self.baseline: dict[str, RepositoryRunResult] = {}
        self._write("manifest.json", {"version": "repository-harness-v1", "repo": repo_kind,
                    "image": docker_image, "tasks": [asdict(t) for t in tasks],
                    "asset_hashes": self.hashes, "network": "none", "llm_calls": 0})

    def _write(self, rel: str, value: dict) -> None:
        path = self.output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    def profile_baseline(self) -> dict:
        outcomes = {}
        for task in self.tasks:
            run = self.validators[task.task_id].profile_base(
                self.base_repo, self.output / "baseline" / task.task_id)
            self.baseline[task.task_id] = run
            outcomes[task.task_id] = execution_status(task, run)
        self._write("baseline/status.json", outcomes)
        return outcomes

    def validate_candidate(self, candidate_repo: Path, label: str = "candidate") -> ValidationReport:
        if not re.fullmatch(r"[\w.-]+", label):
            raise ValueError("label must be a simple name")
        if not self.baseline:
            self.profile_baseline()
        candidate_repo = candidate_repo.resolve()
        evidence, outcomes = [], {}
        for task in self.tasks:
            baseline = self.baseline[task.task_id]
            base_status = execution_status(task, baseline)
            try:
                if _hash_assets(candidate_repo, task.assets) != self.hashes[task.task_id]:
                    raise ValueError("candidate changed a fixed validation asset")
            except ValueError as exc:
                outcomes[task.task_id] = {"baseline": base_status, "candidate": "ASSET_CHANGED"}
                evidence.append(ValidationEvidence("VALIDATION_ASSET_CHANGED", "blocker",
                    str(exc), check_id=task.task_id, validator="repository_harness"))
                continue
            validator = self.validators[task.task_id]
            pair = validator.validate_candidate(baseline, candidate_repo, self.output / label / task.task_id)
            run = RepositoryRunResult.from_dict(json.loads(
                (self.output / label / task.task_id / "candidate.json").read_text()))
            status = execution_status(task, run)
            def signatures(result: RepositoryRunResult) -> set[str]:
                values = set(result.failure_signatures)
                if task.fail_pattern and re.search(task.fail_pattern, result.stdout + "\n" + result.stderr):
                    values.add("native_failure_marker:" + task.fail_pattern)
                return values
            new_failures = sorted(signatures(run) - signatures(baseline))
            unchanged = status == base_status == "FAIL" and not new_failures
            transition = "UNCHANGED_FAILURE" if unchanged else f"{base_status}_TO_{status}"
            raw = run.stdout + "\n" + run.stderr
            build_failure = task.kind in {"build", "lint"} or bool(re.search(
                r"%Error|syntax error|compilation (?:failed|terminated)", raw, re.I))
            outcomes[task.task_id] = {"baseline": base_status, "candidate": status,
                "transition": transition, "kind": task.kind, "issue_claim": task.issue_claim,
                "new_failure_signatures": new_failures}
            kind = "HARNESS_PASS" if status == "PASS" else (
                "HARNESS_INFRA_ERROR" if status in {"UNAVAILABLE", "TIMEOUT", "INCONCLUSIVE"}
                else "HARNESS_BUILD_FAILURE" if build_failure
                else "HARNESS_BEHAVIOR_FAILURE")
            message = f"{task.task_id}: {transition}. {task.rationale}"
            evidence.append(ValidationEvidence(kind, "info" if status == "PASS" else "blocker",
                message, check_id=task.task_id, expected="selected check passes",
                observed=status, raw_log=raw[-12000:], validator="repository_harness"))
            if status == "FAIL":
                for file, line in _locations(raw, candidate_repo):
                    evidence.append(ValidationEvidence(kind, "warning", message,
                        file=file, line=line, check_id=task.task_id, validator="repository_harness"))
        report = build_validation_report(evidence, metadata={
            "validator": "repository_harness", "tasks": outcomes,
            "coverage": "selected_checks_only", "independent_evaluation_required": True,
            "target_behavior_proven": False,
        })
        self._write(f"{label}/report.json", report.to_dict())
        self._write(f"{label}/anchors.json", anchors_from_evidence(report.evidence))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--related-file", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidate-repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.manifest:
        print(json.dumps(discover_assets(args.base_repo.resolve(), args.query, args.related_file), indent=2))
        return
    if not args.output:
        parser.error("--manifest requires --output")
    manifest = json.loads(args.manifest.read_text())
    harness = RepositoryValidationHarness(manifest["repo"], args.base_repo, manifest["image"],
        [ValidationTask(**t) for t in manifest["tasks"]], args.output)
    print(json.dumps(harness.profile_baseline(), indent=2))
    if args.candidate_repo:
        print(json.dumps(harness.validate_candidate(args.candidate_repo).to_dict(), indent=2))


if __name__ == "__main__":
    main()
