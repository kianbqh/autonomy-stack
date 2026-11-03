"""Immutable attempts, input identities, provenance and failure journals."""
from __future__ import annotations
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def verify_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"immutable file hash mismatch: {path}")
    return str(Path(path).resolve())

def source_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.suffix in (".py", ".json", ".yaml", ".sh", ".md")
            and "results" not in p.parts and "__pycache__" not in p.parts}

def environment_identity():
    system,release=platform.system(),platform.release()
    cgroup=Path('/proc/1/cgroup')
    try: cgroup_text=cgroup.read_text() if cgroup.is_file() else ''
    except OSError: cgroup_text=''
    evidence={'platform_system':system,'platform_release':release,
              'github_actions':os.environ.get('GITHUB_ACTIONS')=='true',
              'runner_environment':os.environ.get('RUNNER_ENVIRONMENT'),
              'wsl_interop_present':bool(os.environ.get('WSL_INTEROP')),
              'wsl_distro_name':os.environ.get('WSL_DISTRO_NAME'),
              'docker_marker':Path('/.dockerenv').is_file(),
              'containerenv_marker':Path('/run/.containerenv').is_file(),
              'cgroup_container_marker':any(name in cgroup_text.lower() for name in ('docker','containerd','kubepods'))}
    if evidence['github_actions']: detected='GitHub Actions'
    elif 'microsoft' in release.lower() or evidence['wsl_interop_present'] or evidence['wsl_distro_name']:
        detected='WSL2' if 'wsl2' in release.lower() else 'WSL'
    elif evidence['docker_marker'] or evidence['cgroup_container_marker']: detected='Docker/container'
    elif evidence['containerenv_marker']: detected='Container'
    elif system=='Windows': detected='native Windows'
    else: detected=system
    label=os.environ.get('M3_ENVIRONMENT')
    return {'label':label or detected,'label_source':'user-provided M3_ENVIRONMENT' if label else 'detected',
            'detected':detected,'evidence':evidence}

def environment():
    packages = {}
    for name in ("numpy", "torch", "gymnasium", "gymnasium-robotics", "mujoco", "stable-baselines3", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    cpu = platform.processor()
    info = Path("/proc/cpuinfo")
    if info.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False)
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else {"status": "unavailable", "stderr": gpu.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_info = {"status": "unavailable", "reason": str(exc)}
    identity=environment_identity()
    return {"os": platform.platform(), "python": sys.version, "executable": sys.executable,
            "environment": identity['label'], "environment_identity":identity,
            "cpu": cpu, "gpu_driver": gpu_info, "packages": packages,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}

def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def checked_config(path):
    config = read_json(path)
    if config.get("mode") not in ("formal", "smoke"):
        raise ValueError("config must explicitly label formal or smoke")
    return config

def artifact_root(path):
    path=Path(path).expanduser().resolve()
    if any(part.lower()=='.git' for part in path.parts): raise ValueError("invalid artifact root")
    return path

class Attempt:
    def __init__(self, output_root, name, config_path, inputs=None):
        self.started = time.monotonic()
        self.config = checked_config(config_path)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = artifact_root(output_root) / f"{stamp}-{name}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        self.identity = {"config_sha256": sha256(config_path), "config": self.config,
                         "source_sha256": source_manifest(root), "inputs": inputs or {}}
        self.identity["identity_sha256"] = digest(self.identity)
        try:
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, timeout=10)
            git_sha = git.stdout.strip() if git.returncode == 0 else "not a git checkout"
        except (OSError, subprocess.TimeoutExpired):
            git_sha = "unavailable"
        self.base = {**self.identity, "git_sha": git_sha, "environment": environment(),
                     "command_argv": sys.argv, "command_display": subprocess.list2cmdline(sys.argv),
                     "utc_start": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "monotonic_start": self.started, "status": "started"}
        dump_new(self.path / "provenance.json", self.base)
        self.event("started", identity_sha256=self.identity["identity_sha256"])

    def event(self, event, **fields):
        row = {"event": event, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "monotonic": time.monotonic(), **fields}
        with (self.path / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def write(self,name,value):
        dump_new(self.path/name,{"provenance":self.base,"wall_seconds":time.monotonic()-self.started,**value})

    def finish(self, result):
        result = {**self.base, **result, "status": "completed", "wall_seconds": time.monotonic()-self.started,
                  "utc_end": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump_new(self.path / "result.json", result)
        self.event("completed", wall_seconds=result["wall_seconds"])
        return result

    def failure(self, exc):
        dump_new(self.path / "failure.json", {**self.base, "status": "failed", "type": type(exc).__name__,
                 "reason": str(exc), "traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-self.started,
                 "utc_end": dt.datetime.now(dt.timezone.utc).isoformat()})
        self.event("failed", reason=str(exc))

@contextlib.contextmanager
def attempted(output_root, name, config_path, inputs=None):
    run = Attempt(output_root, name, config_path, inputs)
    try:
        yield run
    except BaseException as exc:
        run.failure(exc)
        raise
