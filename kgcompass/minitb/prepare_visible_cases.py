#!/usr/bin/env python3
"""Trusted one-way sanitizer for the offline local benchmark snapshot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ALLOWED_REPOS = {"ibex", "cva6", "caliptra-rtl"}
IMAGE_PREFIX = {
    "ibex": "ghcr.io/pku-liang/lowrisc_m_ibex",
    "cva6": "ghcr.io/pku-liang/openhwgroup_m_cva6",
    "caliptra-rtl": "ghcr.io/pku-liang/chipsalliance_m_caliptra-rtl",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", help="Prepare only this case; other repository checkouts are not required")
    args = parser.parse_args()

    raw = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("benchmark must contain a JSON list")
    output = []
    for item in raw:
        if args.instance_id and item.get("instance_id") != args.instance_id:
            continue
        repo = item.get("repo")
        if repo not in ALLOWED_REPOS:
            continue
        instance_id = str(item.get("instance_id") or "")
        match = re.search(r"_pr(\d+)(?:-|$)", instance_id)
        if not match:
            raise SystemExit(f"cannot derive PR number: {instance_id}")
        pr_number = int(match.group(1))
        repo_identifier = str(item.get("repo_identifier") or instance_id.rsplit("-", 1)[0])
        repo_path = args.repo_root / repo_identifier
        if not repo_path.is_dir():
            shared = re.sub(r"_pr\d+$", "", repo_identifier)
            repo_path = args.repo_root / shared
        if not repo_path.is_dir():
            raise SystemExit(f"repository is unavailable for {instance_id}: {repo_path}")
        problem = str(item.get("problem_statement") or "").strip()
        title = str(item.get("title") or "").strip()
        if not title:
            title = re.split(r"(?<=[.!?])\s+", problem, maxsplit=1)[0][:300]
        output.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "pr_number": pr_number,
                "base_commit": str(item.get("base_commit") or (item.get("base") or {}).get("commit") or ""),
                "title": title,
                "problem_statement": problem,
                "repo_path": str(repo_path.resolve()),
                "docker_image": f"{IMAGE_PREFIX[repo]}:pr-{pr_number}",
            }
        )
    if args.instance_id and not output:
        raise SystemExit(f"unsupported or missing MiniTB case: {args.instance_id}")
    output.sort(key=lambda value: (value["repo"], value["pr_number"], value["instance_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {len(output)} sanitized Mini-TB cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
