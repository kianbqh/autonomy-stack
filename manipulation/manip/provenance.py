"""Immutable attempts and explicit source, configuration and environment receipts."""
from __future__ import annotations
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
from .protocol import digest

class Tee:
    def __init__(self,original,stream):
        self.original,self.stream=original,stream
    def write(self,value):
        self.stream.write(value)
        self.stream.flush()
        return self.original.write(value)
    def flush(self):
        self.stream.flush()
        self.original.flush()
    def __getattr__(self,name):
        return getattr(self.original,name)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

def command_output(args, cwd=None, optional=False):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if optional:
            return {"unavailable": str(exc)}
        raise

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

def artifact_root(path):
    path = Path(path).expanduser().resolve()
    if any(part.lower()=='.git' for part in path.parts):
        raise ValueError("invalid artifact root")
    return path

def inventory(repo, config):
    git = os.environ.get("M3_GIT", "git")
    packages = {}
    for name in ("numpy", "torch", "mujoco", "gymnasium", "gymnasium-robotics", "stable-baselines3"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    src = Path(__file__).parent.parent
    source_hashes = {str(p.relative_to(src)): sha256(p) for p in sorted(src.rglob("*.py"))}
    import torch  # Imported only while recording an actual attempt, never by pure checks.
    identity=environment_identity()
    return {"git_sha": command_output([git, "-C", str(repo), "rev-parse", "HEAD"]),
            "git_executable":git,"git_version":command_output([git,"--version"]),
            "git_status": command_output([git, "-C", str(repo), "status", "--porcelain"]),
            "config": config, "config_sha256": digest(config), "source_hashes": source_hashes,
            "packages": packages, "platform": platform.platform(), "environment": identity['label'],
            "environment_identity":identity,
            "cpu": command_output(["lscpu"], optional=True),
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], optional=True),
            "torch_cuda_build":torch.version.cuda,"torch_cuda_available":torch.cuda.is_available(),
            "python": sys.version, "executable": sys.executable, "command": sys.argv,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}

class Attempt:
    def __init__(self, root, label, config, repo):
        self.path = artifact_root(root) / f"{label}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        try:
            self.record = inventory(repo, config)
        except BaseException:
            dump_new(self.path/"failure-before-inventory.json",{"status":"failed","command":sys.argv,
                     "config":config,"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),
                     "wall_seconds":time.monotonic()-self.started,"exception":traceback.format_exc()})
            raise
        dump_new(self.path / "provenance.json", self.record)
        snapshot=self.path/"source"
        source_root=Path(__file__).parent.parent
        for name,expected in self.record["source_hashes"].items():
            destination=snapshot/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open("xb") as stream:
                stream.write((source_root/name).read_bytes())
            if sha256(destination)!=expected:
                raise ValueError("source changed while freezing runtime attempt")
        config_file=self.path/"config.json"
        dump_new(config_file,config)
        self.producer_base={"source_root":str(snapshot),"source_files":self.record["source_hashes"],
                            "source_sha256":digest(self.record["source_hashes"]),
                            "config":{"path":str(config_file),"sha256":sha256(config_file),"identity_sha256":digest(config)}}
        self.event("started", pid=os.getpid())

    def producer_receipt(self,policy,checkpoint=None):
        policy_id=digest(policy)
        path=self.path/f"policy-{policy_id}.json"
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(json.dumps(policy,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        if sha256(path)!=policy_id:
            raise ValueError("existing immutable policy specification bytes changed")
        result={**self.producer_base,"policy":{"path":str(path),"sha256":sha256(path)},"checkpoint":None}
        if checkpoint:
            result["checkpoint"]={"path":str(checkpoint),"sha256":sha256(checkpoint)}
        return result

    def event(self, kind, **fields):
        with (self.path / "journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "at": dt.datetime.now(dt.timezone.utc).isoformat(), **fields}, allow_nan=False)+"\n")

    def write(self, name, value):
        dump_new(self.path / name, {"provenance": self.record, "wall_seconds": time.monotonic()-self.started, **value})

    def __enter__(self):
        self.original_stdout,self.original_stderr=sys.stdout,sys.stderr
        self.stdout_log=(self.path/"stdout.txt").open("x",encoding="utf-8")
        self.stderr_log=(self.path/"stderr.txt").open("x",encoding="utf-8")
        sys.stdout=Tee(self.original_stdout,self.stdout_log)
        sys.stderr=Tee(self.original_stderr,self.stderr_log)
        return self

    def __exit__(self, kind, value, tb):
        try:
            if kind:
                self.write("failure.json", {"status": "failed", "exception": "".join(traceback.format_exception(kind, value, tb))})
            else:
                self.event("completed", wall_seconds=time.monotonic()-self.started)
        finally:
            sys.stdout,self.original_stdout=self.original_stdout,None
            sys.stderr,self.original_stderr=self.original_stderr,None
            self.stdout_log.close()
            self.stderr_log.close()
        return False

def resume_guard(old, current):
    keys = ("git_sha", "config_sha256", "source_hashes", "packages", "environment")
    changed = [key for key in keys if old[key] != current[key]]
    if changed:
        raise ValueError(f"resume provenance mismatch: {changed}")
