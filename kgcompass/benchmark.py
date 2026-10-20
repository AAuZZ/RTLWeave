import json
import os
from datetime import datetime, timezone

try:
    from datasets import load_dataset
except Exception:  # local JSON benchmarks do not need HuggingFace datasets.
    load_dataset = None

from config import DATASET_NAME, LOCAL_BENCHMARK_PATH


LOCAL_BENCHMARK_NAMES = {"local", "verilog-local"}


def _as_utc_created_at(value):
    if isinstance(value, datetime):
        dt = value
    elif value:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(str(value), fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_local_path(dataset_name=None):
    path = LOCAL_BENCHMARK_PATH or dataset_name
    if not path:
        raise ValueError("LOCAL_BENCHMARK_PATH or DATASET_NAME must point to a local benchmark JSON/JSONL file.")
    return os.path.abspath(path)


def _load_local_items(dataset_name=None, split="test"):
    path = _resolve_local_path(dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local benchmark file not found: {path}")

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if split in data and isinstance(data[split], list):
            return data[split]
        if "instances" in data and isinstance(data["instances"], list):
            return data["instances"]
    raise ValueError(f"Unsupported local benchmark format: {path}")


def load_benchmark_items(dataset_name=None, benchmark_name="swe-bench", split="test"):
    dataset_name = dataset_name or DATASET_NAME
    if benchmark_name in LOCAL_BENCHMARK_NAMES or (dataset_name and os.path.exists(dataset_name)):
        return _load_local_items(dataset_name, split)

    if benchmark_name == "multi-swe-bench":
        split = "java_verified"
        dataset_name = "Daoguang/Multi-SWE-bench"

    if load_dataset is None:
        raise ImportError("datasets is required for non-local benchmarks; use verilog-local or install datasets.")
    ds = load_dataset(dataset_name, split=split)
    return [dict(item) for item in ds]


def get_target_sample(instance_id, repo_name=None, benchmark_name="swe-bench", dataset_name=None, split="test"):
    for item in load_benchmark_items(dataset_name, benchmark_name, split):
        if item.get("instance_id") != instance_id:
            continue
        if repo_name and item.get("repo") and item.get("repo") != repo_name:
            continue

        sample = dict(item)
        sample["created_at"] = _as_utc_created_at(sample.get("created_at"))
        sample.setdefault("problem_statement", sample.get("text", ""))
        sample.setdefault("test_patch", "")
        sample.setdefault("patch", "")
        sample.setdefault("pull_number", None)
        sample.setdefault("language", "python")
        return sample
    return None
