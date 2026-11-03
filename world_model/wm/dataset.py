"""Audited, episode/reset-group split data. NPZ arrays are never pickled or padded."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np
from .protocol import FETCH_LAYOUT, SOURCES, observation_to_state, split_for_reset, state_delta
from .provenance import digest, read_json, sha256, verify_file

VECTOR_SIZES = {"obs":25,"next_obs":25,"clean_obs":25,"next_clean_obs":25,
                "action":4,"applied_action":4,"desired_goal":3,"next_desired_goal":3,
                "achieved_goal":3,"next_achieved_goal":3,"clean_achieved_goal":3,"next_clean_achieved_goal":3}
SCALARS = ("reward", "terminated", "truncated", "is_success")

def safe_child(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root != path and root not in path.parents:
        raise ValueError("archive path escapes episode root")
    return path

def require_hash(value, name, allow_none=False):
    if allow_none and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"missing/invalid {name}")

def validate_metadata(m):
    if m.get("schema") != "fetch-episode-v1" or m.get("source_policy") not in SOURCES:
        raise ValueError("unsupported episode/source schema")
    for key in ("config_sha256", "source_sha256", "policy_sha256", "checkpoint_sha256"):
        require_hash(m.get(key), key, allow_none=key=="checkpoint_sha256" and m["source_policy"]=="expert")
    for key in ("reset_seed", "wrapper_seed", "randomization_seed"):
        if type(m.get(key)) is not int:
            raise ValueError(f"missing integer {key}")
    seed = m["reset_seed"]
    split = split_for_reset(seed)
    if m.get("dataset_eligible") is not True or m.get("split") != split or m.get("split_group") != f"reset:{seed}":
        raise ValueError("episode group/split mismatch or held-out evaluation record")
    env = m["env_contract"]
    if env["env_id"] != "FetchPush-v4" or env["field_map"] != FETCH_LAYOUT:
        raise ValueError("Fetch schema/field mapping changed")
    if env["max_episode_steps"] != 50 or env["reward_type"] != "sparse":
        raise ValueError("task horizon/reward changed")
    if not np.isfinite(float(m["control_dt"])) or not 0 < float(m["control_dt"]) == float(env["control_dt"]):
        raise ValueError("control dt mismatch")
    if not np.isfinite(env["physics_dt"]) or env["physics_dt"]<=0 or type(env["n_substeps"]) is not int or env["n_substeps"]<=0:
        raise ValueError("invalid native physics timebase")
    if not np.isclose(m["control_dt"],env["physics_dt"]*env["n_substeps"],rtol=0,atol=1e-12):
        raise ValueError("control timestep differs from native physics dt 脳substeps")
    if "env_contract_sha256" in m and m["env_contract_sha256"] != digest(env):
        raise ValueError("env contract hash mismatch")

def validate_arrays(a, m):
    length = int(m["transitions"])
    # Var-length slicing/batching is supported; native Push records must be complete.
    if length != 50:
        raise ValueError("formal native Push episode must have 50 transitions")
    if set(a) != set(VECTOR_SIZES) | set(SCALARS) | {"simulator_state"}:
        raise ValueError("NPZ array key schema mismatch")
    for key, size in VECTOR_SIZES.items():
        if a[key].shape != (length, size) or not np.isfinite(a[key]).all():
            raise ValueError(f"nonfinite/shape mismatch: {key}")
    for key in SCALARS:
        if a[key].shape != (length,) or not np.isfinite(a[key]).all():
            raise ValueError(f"nonfinite/shape mismatch: {key}")
    if a["simulator_state"].shape != (length+1, m["env_contract"]["integration_state_size"]) or not np.isfinite(a["simulator_state"]).all():
        raise ValueError("complete integration-state sequence required")
    if np.max(np.abs(a["action"])) > 1.000001 or np.max(np.abs(a["applied_action"])) > 1.000001:
        raise ValueError("action outside native bounds")
    if a["terminated"].any() or a["truncated"][:-1].any() or not a["truncated"][-1]:
        raise ValueError("native continuing-task TimeLimit semantics changed")
    if not np.isin(a["reward"], (-1, 0)).all():
        raise ValueError("native sparse reward changed")
    if any(not np.isin(a[k],(0,1)).all() for k in ("terminated","truncated","is_success")):
        raise ValueError("invalid native boolean scalar")
    if bool(a["is_success"][-1])!=m["terminal_success"] or bool(a["is_success"].any())!=m["ever_success"]:
        raise ValueError("sidecar final/ever success disagrees with episode arrays")
    for prefix in ("", "next_"):
        obs, clean = a[prefix+"obs"], a[prefix+"clean_obs"]
        for actual, expected in ((a[prefix+"achieved_goal"],obs[:,3:6]),
                                 (a[prefix+"clean_achieved_goal"],clean[:,3:6]),
                                 (obs[:,6:9],obs[:,3:6]-obs[:,:3]),
                                 (clean[:,6:9],clean[:,3:6]-clean[:,:3])):
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError("redundant physical/visible observation inconsistency")
    if not np.allclose(a["desired_goal"], a["desired_goal"][0], rtol=0, atol=0) or not np.array_equal(a["desired_goal"], a["next_desired_goal"]):
        raise ValueError("desired goal is not static")
    for previous, following in (("next_obs","obs"),("next_clean_obs","clean_obs"),
                                ("next_desired_goal","desired_goal"),("next_achieved_goal","achieved_goal")):
        if not np.array_equal(a[previous][:-1], a[following][1:]):
            raise ValueError("broken episode transition continuity")
    return length

class Episode:
    def __init__(self, arrays, metadata, record_identity):
        self.arrays, self.metadata, self.identity = arrays, metadata, record_identity
        self.length = len(arrays["action"])
        dt = metadata["control_dt"]
        self.visible = np.asarray([observation_to_state(x,dt) for x in arrays["obs"]], dtype=np.float32)
        self.clean = np.asarray([observation_to_state(x,dt) for x in arrays["clean_obs"]], dtype=np.float32)
        self.clean_next = np.asarray([observation_to_state(x,dt) for x in arrays["next_clean_obs"]], dtype=np.float32)
        self.delta = np.asarray([state_delta(x,y) for x,y in zip(self.visible,self.clean_next)], dtype=np.float32)

class Dataset:
    def __init__(self, episodes, audit):
        self.episodes, self.audit = episodes, audit
        self.by_split = {name:[ep for ep in episodes if ep.metadata["split"]==name] for name in ("train","val","test")}
        if any(not eps for eps in self.by_split.values()):
            raise ValueError("empty train/val/test episode split")
        self.flat = {}
        for split, episodes in self.by_split.items():
            self.flat[split] = {"state":np.concatenate([ep.visible for ep in episodes]),
                                "action":np.concatenate([ep.arrays["action"] for ep in episodes]).astype(np.float32),
                                "clean_state":np.concatenate([ep.clean for ep in episodes]),
                                "next_state":np.concatenate([ep.clean_next for ep in episodes]),
                                "episode_index":np.concatenate([np.full(ep.length,i,dtype=np.int64) for i,ep in enumerate(episodes)])}
        # There is no padded transition; valid starts are computed per episode.
        self.identity = digest(audit)

    def valid_starts(self, split, horizon):
        return [(ep, start) for ep in self.by_split[split] for start in range(max(0,ep.length-horizon+1))]


def load_dataset(manifest_specs, config):
    episodes, identities = [], []
    seen_ids, seen_npz, seen_content, seen_source_reset, reset_groups = set(), set(), set(), set(), {}
    verified_sources=set(); verified_files=set()
    def checked_file(path,expected):
        identity=(str(Path(path).resolve()),expected)
        if identity not in verified_files:
            verify_file(path,expected); verified_files.add(identity)
    counts = {s:{k:{"episodes":0,"transitions":0,"episode_ids":[]} for k in ("train","val","test")} for s in SOURCES}
    env_identity, control_dt, simulator_versions = None, None, None
    for spec in manifest_specs:
        path = Path(verify_file(spec["path"],spec["sha256"]))
        manifest = read_json(path)
        records = manifest.get("episodes", manifest.get("records"))
        if not isinstance(records,list):
            raise ValueError("manifest requires explicit episodes/records list")
        root = Path(spec["episode_root"]) if "episode_root" in spec else safe_child(path.parent,manifest.get("episode_directory","."))
        # Receipts bind every historical producer source/config/policy/checkpoint to bytes.
        defaults=spec.get("producer_receipts",manifest.get("producer_receipts",[]))
        receipts=defaults+[r["producer_receipt"] for r in records if "producer_receipt" in r]
        receipts=list({digest(r):r for r in receipts}.values())
        if not receipts: raise ValueError("byte-verifiable producer receipts required")
        for receipt in receipts:
            source_files=receipt["source_files"]
            source_identity=(receipt["source_root"],receipt["source_sha256"])
            if source_identity not in verified_sources:
                for relative,expected in source_files.items():
                    checked_file(safe_child(receipt["source_root"],relative),expected)
                verified_sources.add(source_identity)
            if digest(source_files)!=receipt["source_sha256"]:
                raise ValueError("producer source digest mismatch")
            for kind in ("config","policy","checkpoint"):
                item=receipt.get(kind)
                if item is not None:
                    checked_file(item["path"],item.get("sha256",item.get("file_sha256")))
                    if kind=="config":
                        parsed=read_json(item["path"])
                        if digest(parsed)!=item["identity_sha256"]:
                            raise ValueError("parsed config identity mismatch")
        for record in records:
            side = safe_child(root, record["sidecar"])
            verify_file(side, record["sidecar_sha256"])
            m = read_json(side)
            for key in record:
                if key in m and record[key] != m[key]:
                    raise ValueError("manifest/sidecar record mismatch")
            validate_metadata(m)
            for xml_path,xml_hash in m["env_contract"]["xml_inputs"].items():
                checked_file(xml_path,xml_hash)
            provenance=m["provenance"]
            packages=provenance.get("packages")
            if packages is None:
                env_provenance=provenance.get("environment",{})
                packages=env_provenance.get("packages",{}) if isinstance(env_provenance,dict) else {}
            versions={k:packages[k] for k in ("gymnasium","gymnasium-robotics","mujoco")}
            if any(not isinstance(v,str) or v in ("not installed","unknown","") for v in versions.values()):
                raise ValueError("actual simulator package versions required")
            if simulator_versions is None: simulator_versions=versions
            if simulator_versions!=versions: raise ValueError("mixed simulator package versions")
            matches=[r for r in receipts if r["source_sha256"]==m["source_sha256"]
                     and r["config"]["identity_sha256"]==m["config_sha256"] and r["policy"]["sha256"]==m["policy_sha256"]
                     and (None if r.get("checkpoint") is None else r["checkpoint"]["sha256"])==m["checkpoint_sha256"]]
            if len(matches)!=1:
                raise ValueError("no unique byte-verified producer source/config/policy/checkpoint receipt")
            npz = safe_child(root,m["npz"])
            verify_file(npz,m["npz_sha256"])
            ep_id, ep_hash, source = m["episode_id"], m["npz_sha256"], m["source_policy"]
            if ep_id in seen_ids or ep_hash in seen_npz:
                raise ValueError("duplicate episode ID or identical episode archive")
            source_reset = (source,m.get("model_seed"),m["checkpoint_sha256"],m["reset_seed"],m["randomization_seed"])
            if source_reset in seen_source_reset:
                raise ValueError("duplicate source/checkpoint/reset episode")
            seen_ids.add(ep_id); seen_npz.add(ep_hash); seen_source_reset.add(source_reset)
            # Same reset across sources is legal; collisions with different physical/noise identities are not.
            group = (m["randomization_seed"],m["wrapper_seed"],digest(m.get("perturbation",{})),digest(m["env_contract"]))
            if m["reset_seed"] in reset_groups and reset_groups[m["reset_seed"]] != group:
                raise ValueError("reset-group collision with different env/randomization identity")
            reset_groups[m["reset_seed"]] = group
            contract_hash = digest(m["env_contract"])
            if env_identity is None:
                env_identity, control_dt = contract_hash, m["control_dt"]
            if contract_hash != env_identity or control_dt != m["control_dt"]:
                raise ValueError("mixed environment contract or dt")
            with np.load(npz,allow_pickle=False) as archive:
                a = {key:archive[key] for key in archive.files}
            length = validate_arrays(a,m)
            content=hashlib.sha256()
            for key,array in sorted(a.items()):
                content.update(key.encode()); content.update(str(array.dtype).encode())
                content.update(str(array.shape).encode()); content.update(array.tobytes())
            content_hash=content.hexdigest()
            if content_hash in seen_content: raise ValueError("duplicate trajectory content despite different NPZ bytes")
            seen_content.add(content_hash)
            identity = {"episode_id":ep_id,"sidecar_sha256":record["sidecar_sha256"],"npz_sha256":ep_hash,
                        "source":source,"reset_seed":m["reset_seed"],"split":m["split"],"content_sha256":content_hash}
            identities.append(identity)
            episodes.append(Episode(a,m,identity))
            cell = counts[source][m["split"]]
            cell["episodes"] += 1; cell["transitions"] += length; cell["episode_ids"].append(ep_id)
    required = config.get("required_sources",list(SOURCES))
    if set(required) != set(SOURCES) or any(sum(c["episodes"] for c in counts[s].values()) == 0 for s in required):
        raise ValueError("all three reduced behavior sources must be present")
    for split in ("train","val","test"):
        if any(counts[s][split]["episodes"] == 0 for s in required):
            raise ValueError("each source must cover every split")
    if "expected_counts" in config and config["expected_counts"] != {s:{k:{q:c[q] for q in ("episodes","transitions")} for k,c in splits.items()} for s,splits in counts.items()}:
        raise ValueError("source/episode/transition counts differ from frozen intake counts")
    audit = {"schema":"wm-audit-v1","manifests":manifest_specs,"records":identities,"counts":counts,
             "env_contract_sha256":env_identity,"control_dt":control_dt,
             "simulator_versions":simulator_versions,
             "reset_groups":{str(k):split_for_reset(k) for k in sorted(reset_groups)},
             "split_rule":"SHA256(fetch-wm-v1:<reset_seed>) mod100; <80/<90/else", "action_input":"policy command action", "scope":"reduced v2; SAC replay omitted by direct user scope cut"}
    return Dataset(episodes,audit)
