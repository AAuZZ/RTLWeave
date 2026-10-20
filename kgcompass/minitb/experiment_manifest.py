"""Prevent mixing template/model-generation revisions in one experiment root."""
import hashlib
import json
from pathlib import Path


def generation_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    # No configuration, model keys, benchmark answers or generated artifacts.
    paths = sorted([*root.glob("*.py"), *root.glob("prompts/*.md"), *root.glob("templates/*")])
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_output_policy(output: Path) -> str:
    fingerprint = generation_fingerprint()
    manifest = output / "run_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("generation_policy") != "kg-self-correction-v1" or old.get("generator_sha256") != fingerprint:
            raise SystemExit("Output directory belongs to a different or unversioned generator. Set a NEW MINITB_AUTOGEN_ROOT; historical results will not be overwritten.")
    elif any(output.glob("*/attempts.jsonl")):
        raise SystemExit("Output has unversioned attempts. Set a NEW MINITB_AUTOGEN_ROOT.")
    return fingerprint
