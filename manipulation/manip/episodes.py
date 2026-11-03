"""Immutable episode interchange; external NPZ with audit sidecar and no pickle."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .protocol import FIELD_MAP, split_for_reset
from .provenance import dump_new,sha256

SIZES={"obs":25,"next_obs":25,"clean_obs":25,"next_clean_obs":25,"action":4,"applied_action":4,
       "desired_goal":3,"next_desired_goal":3,"achieved_goal":3,"next_achieved_goal":3,
       "clean_achieved_goal":3,"next_clean_achieved_goal":3}
SCALARS=("reward","terminated","truncated","is_success")

def validate(arrays,metadata):
    if metadata["schema"]!="fetch-episode-v1":
        raise ValueError("episode schema")
    length=len(arrays["reward"])
    if length!=50:
        raise ValueError("only complete native 50-step episodes")
    for key,size in SIZES.items():
        if np.asarray(arrays[key]).shape!=(length,size) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key}")
    for key in SCALARS:
        if np.asarray(arrays[key]).shape!=(length,):
            raise ValueError(key)
    if arrays["terminated"].any() or arrays["truncated"][:-1].any() or not arrays["truncated"][-1]:
        raise ValueError("native continuing/time-limit semantics changed")
    if not np.isin(arrays["reward"],[-1,0]).all():
        raise ValueError("native sparse reward changed")
    if arrays["simulator_state"].shape[0]!=length+1:
        raise ValueError("missing full simulator state sequence")
    if not np.isfinite(arrays["simulator_state"]).all():
        raise ValueError("nonfinite simulator state")
    if not np.allclose(arrays["clean_achieved_goal"],arrays["clean_obs"][:,3:6]):
        raise ValueError("clean position mapping")
    if not np.allclose(arrays["achieved_goal"],arrays["obs"][:,3:6]):
        raise ValueError("visible position mapping")
    return length

def write_episode(root,arrays,metadata):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    metadata={"schema":"fetch-episode-v1","physical_field_map":FIELD_MAP,**metadata}
    required=("episode_id","source_policy","model_seed","reset_seed","wrapper_seed","randomization_seed",
              "control_dt","env_contract","config_sha256","source_sha256","policy_sha256","checkpoint_sha256","provenance")
    missing=[key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"missing episode metadata {missing}")
    for key in ("config_sha256","source_sha256","policy_sha256"):
        if len(metadata[key])!=64:
            raise ValueError(f"invalid {key}")
    validate(arrays,metadata)
    seed=int(metadata["reset_seed"])
    if metadata.get("dataset_eligible",True):
        metadata["split"]=split_for_reset(seed)
        metadata["split_group"]=f"reset:{seed}"
    else:
        metadata["split"]="heldout_evaluation"
    path=root/f"{metadata['episode_id']}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    metadata.update({"npz":path.name,"npz_sha256":sha256(path),"transitions":len(arrays["reward"])})
    side=path.with_suffix(".json")
    dump_new(side,metadata)
    return {**metadata,"sidecar":side.name,"sidecar_sha256":sha256(side)}

def load_episode(root,record):
    root=Path(root)
    path=root/record["npz"]
    if sha256(path)!=record["npz_sha256"]:
        raise ValueError("NPZ hash mismatch")
    if "sidecar_sha256" in record and sha256(root/record["sidecar"])!=record["sidecar_sha256"]:
        raise ValueError("sidecar hash mismatch")
    with np.load(path,allow_pickle=False) as loaded:
        arrays={k:loaded[k] for k in loaded.files}
    validate(arrays,record)
    return arrays
