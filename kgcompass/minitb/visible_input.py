from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .models import InputBoundaryError, VisibleCase


def load_visible_cases(path: Path) -> dict[str, VisibleCase]:
    if not path.is_file():
        raise InputBoundaryError(f"visible case file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise InputBoundaryError("visible_cases.json must contain a JSON list")
    cases: dict[str, VisibleCase] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise InputBoundaryError("each visible case must be a JSON object")
        case = VisibleCase.from_dict(item)
        if case.instance_id in cases:
            raise InputBoundaryError(f"duplicate instance_id: {case.instance_id}")
        cases[case.instance_id] = case
    return cases


def select_case(path: Path, instance_id: str) -> VisibleCase:
    cases = load_visible_cases(path)
    try:
        return cases[instance_id]
    except KeyError as exc:
        raise InputBoundaryError(f"instance not found in sanitized cases: {instance_id}") from exc


def ensure_base_checkout(case: VisibleCase, destination: Path) -> Path:
    """Create an isolated checkout without changing the source repository."""
    source = Path(os.path.expanduser(case.repo_path)).resolve()
    if not (source / ".git").exists() and not _is_git_repo(source):
        raise InputBoundaryError(f"repo_path is not a git repository: {source}")

    marker = destination / ".kgveri_base_commit"
    if destination.exists() and marker.is_file():
        if marker.read_text(encoding="utf-8").strip() == case.base_commit:
            result = _git(destination, "rev-parse", "HEAD", check=False)
            if result.returncode == 0 and result.stdout.strip() == case.base_commit:
                return destination

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(destination)],
        capture_output=True,
        text=True,
    )
    if clone.returncode != 0:
        raise InputBoundaryError(f"failed to clone base repository: {clone.stderr.strip()}")
    checkout = _git(destination, "checkout", "--quiet", "--detach", case.base_commit, check=False)
    if checkout.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        raise InputBoundaryError(
            f"base commit {case.base_commit} is unavailable in {source}: {checkout.stderr.strip()}"
        )
    marker.write_text(case.base_commit + "\n", encoding="utf-8")
    return destination


def derive_pr_number(instance_id: str) -> int:
    match = re.search(r"_pr(\d+)(?:-|$)", instance_id)
    if not match:
        raise InputBoundaryError(f"cannot derive PR number from {instance_id}")
    return int(match.group(1))


def _is_git_repo(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise InputBoundaryError(result.stderr.strip() or "git command failed")
    return result
