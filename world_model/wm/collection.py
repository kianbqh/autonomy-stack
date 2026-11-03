"""Fresh nominal rollouts from real Track A policies; no synthetic fallback."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
from .protocol import SOURCES, child_seed
from .provenance import artifact_root, digest, dump_new, sha256, verify_file

class NoisyPolicy:
    def __init__(self,policy,standard_deviation): self.policy=policy; self.std=standard_deviation
    def reset(self,seed=None):
        self.policy.reset(seed); self.rng=np.random.default_rng(child_seed("noisy_bc",seed))
    def act(self,obs,deterministic=True):
        return np.clip(self.policy.act(obs,deterministic=True)+self.rng.normal(0,self.std,4),-1,1).astype(np.float32)

def collect(config,run):
    from manip.envs import make_env
    from manip.evaluate import rollout
    from manip.episodes import write_episode
    from manip.policies import load_policy
    import manip
    source_specs=config["source_specs"]
    if set(s["source_policy"] for s in source_specs)!=set(SOURCES):
        raise ValueError("collection requires real expert, BC and noisy BC only")
    starts={"expert":800000,"bc":801000,"noisy_bc":802000}
    if len(source_specs)!=3 or config["episodes_per_source"]!=200 or config["noisy_bc_std"]!=.2:
        raise ValueError("root-accepted three-source200episode collection changed")
    for spec in source_specs:
        if spec["reset_start"]!=starts[spec["source_policy"]] or spec["model_seed"]!=(None if spec["source_policy"]=="expert" else 0):
            raise ValueError("predeclared reset allocation/controller seed changed")
    learned=[s["checkpoint"] for s in source_specs if s["source_policy"]!="expert"]
    if learned[0]!=learned[1] or learned[0] is None: raise ValueError("BC/noisyBC must use same actual nominalBC seed0 checkpoint")
    directory=artifact_root(config["artifact_root"])/run.path.name
    directory.mkdir(parents=True,exist_ok=False)
    source_root=directory/"producer-source"; source_files={}
    for name,root in (("world_model",Path(__file__).resolve().parents[1]),("manipulation",Path(manip.__file__).resolve().parents[1])):
        for source in sorted(root.rglob("*.py")):
            if "__pycache__" in source.parts: continue
            relative=Path(name)/source.relative_to(root)
            target=source_root/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target); source_files[str(relative).replace("\\","/")]=sha256(target)
    cfg=directory/"producer-config.json"; dump_new(cfg,config)
    common={"source_root":str(source_root),"source_files":source_files,"source_sha256":digest(source_files),
            "config":{"path":str(cfg),"sha256":sha256(cfg),"identity_sha256":digest(config)}}
    env=make_env(config["env_config"])
    records=[]
    try:
        for index,spec in enumerate(source_specs):
            source=spec["source_policy"]
            checkpoint=spec.get("checkpoint")
            if source!="expert" and checkpoint is None: raise ValueError("trained source requires actual checkpoint")
            if checkpoint: verify_file(checkpoint["path"],checkpoint["sha256"])
            kind="bc" if source=="noisy_bc" else source
            if source=="expert":
                from manip.expert import PushExpert
                policy=PushExpert(spec["controller"])
            else:
                policy=load_policy(kind,checkpoint["path"],config["policy_device"])
                if getattr(policy,"model_seed",None)!=spec["model_seed"]:
                    raise ValueError("source model seed differs from actual checkpoint seed")
            if source=="noisy_bc": policy=NoisyPolicy(policy,config["noisy_bc_std"])
            policy_path=directory/f"policy-{index:02d}.json"
            dump_new(policy_path,{**spec,"noisy_bc_std":config["noisy_bc_std"] if source=="noisy_bc" else None,
                                  "semantics":"fresh frozen-policy rollouts; command action stored before delay/gain/shield"})
            receipt={**common,"policy":{"path":str(policy_path),"sha256":sha256(policy_path)},"checkpoint":checkpoint}
            resets=range(spec["reset_start"],spec["reset_start"]+config["episodes_per_source"])
            for reset in resets:
                arrays,metadata=rollout(env,policy,reset,spec["model_seed"],source,dataset_eligible=True)
                metadata.update({"episode_id":f"wm-{source}-m{spec['model_seed']}-r{reset}-{run.identity['identity_sha256'][:12]}",
                    "config_sha256":digest(config),"source_sha256":common["source_sha256"],
                    "policy_sha256":sha256(policy_path),"checkpoint_sha256":checkpoint["sha256"] if checkpoint else None,
                    "env_contract_sha256":digest(env.contract),"provenance":run.base,
                    "collection":"fresh fixed-checkpoint policy rollout; not replay re-labelling"})
                record=write_episode(directory/"episodes",arrays,metadata)
                record["producer_receipt"]=receipt; records.append(record)
                run.event("episode_collected",episode_id=record["episode_id"],source=source,reset_seed=reset,
                          terminal_success=metadata["terminal_success"],transitions=len(arrays["action"]))
    finally:
        env.close()
    manifest=directory/"manifest.json"
    dump_new(manifest,{"schema":"fetch-wm-collection-v1","provenance":run.base,"episode_directory":"episodes","episodes":records,
                       "producer_receipts":[],"counts":{s:sum(r["source_policy"]==s for r in records) for s in SOURCES}})
    return {"manifest":str(manifest),"manifest_sha256":sha256(manifest),"episodes":len(records),
            "source_counts":{s:sum(r["source_policy"]==s for r in records) for s in SOURCES},"artifact_directory":str(directory)}
