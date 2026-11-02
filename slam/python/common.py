"""Small provenance helpers. No model imports or network actions."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def new_run(parent, prefix):
    p = Path(parent) / (prefix + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    p.mkdir(parents=True, exist_ok=False)
    return p

def source_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and not any(x in p.parts for x in ("__pycache__", ".pytest_cache", ".venv", "results", "build", ".git"))}

def environment():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    packages = {}
    for name in ("numpy", "scipy", "opencv-python-headless", "matplotlib", "evo"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
            "cpu": cpu, "logical_cpus": os.cpu_count(), "packages": packages,
            "environment": "WSL2" if "microsoft" in platform.release().lower() else platform.system(),
            "gpu": "not used by CPU SLAM", "gpu_driver": "not queried; CPU-only run"}

def git_state(repo, executable):
    command = [executable, "-c", "safe.directory=" + str(Path(repo).resolve()), "-C", str(repo)]
    sha = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=all", "--", "slam", ":(exclude)slam/results"], text=True)
    if len(sha) != 40 or dirty:
        raise ValueError("SLAM source must be committed and clean; do not bypass with a fabricated SHA")
    whole = subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=all"], text=True)
    return {"git_sha": sha, "git_executable": executable, "selected_source_status": dirty,
            "selected_source_scope": "slam excluding derived slam/results", "whole_worktree_status": whole,
            "git_version": subprocess.check_output([executable, "--version"], text=True).strip()}

def percentile(samples, q):
    if not samples:
        return None
    x = sorted(samples)
    t = (len(x) - 1) * q
    lo, hi = math.floor(t), math.ceil(t)
    return x[lo] + (x[hi] - x[lo]) * (t - lo)

def summary(samples):
    return {"count": len(samples), "mean_ms": sum(samples) / len(samples) if samples else None,
            "p95_ms": percentile(samples, .95), "percentile_method": "linear (n-1)*q"}
