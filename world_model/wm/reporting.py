"""Build comparisons only from byte-verified actual paired policy episodes."""
from __future__ import annotations
from pathlib import Path
from .protocol import EVAL_SEEDS, MODEL_SEEDS, population_summary
from .provenance import digest, dump_new, read_json, verify_file

def pointer(value,path):
    for key in path.split("/"):
        if not key: continue
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value

def compare(config,run):
    methods={}; identities=[]; task_identity=None
    for spec in config["policy_results"]:
        verify_file(spec["path"],spec["sha256"])
        payload=read_json(spec["path"])
        result=pointer(payload,spec.get("pointer",""))
        rows=result["episodes_detail"]
        if sorted(r["reset_seed"] for r in rows)!=list(EVAL_SEEDS) or len(rows)!=100:
            raise ValueError("comparison requires the same 100 held-out resets")
        if any(r["length"]!=50 for r in rows): raise ValueError("native step50 outcome required")
        seed=result["model_seed"]
        if seed not in MODEL_SEEDS: raise ValueError("comparison requires independent seeds0,1,2")
        source_config=payload.get("config",payload.get("provenance",{}).get("config",{}))
        if source_config.get("mode")!="formal":
            raise ValueError("smoke/ever-success is not a formal comparison outcome")
        rows=sorted(rows,key=lambda r:r["reset_seed"])
        if any(r["model_seed"]!=seed for r in rows): raise ValueError("episode/model seed mismatch")
        for row in rows:
            env=row["env_contract"]
            if env["env_id"]!="FetchPush-v4" or env["reward_type"]!="sparse" or env["max_episode_steps"]!=50:
                raise ValueError("different native task/reward/horizon")
            if row["control_dt"]!=env["control_dt"]: raise ValueError("dt mismatch")
        receipt={"mode":"formal","outcome":"native final-step50 is_success",
                 "env_contract_sha256":digest(rows[0]["env_contract"]),
                 "condition_sha256":digest([r["perturbation"] for r in rows]),
                 "reset_protocol_sha256":digest([[r[k] for k in ("reset_seed","wrapper_seed","randomization_seed")] for r in rows])}
        if any(digest(r["env_contract"])!=receipt["env_contract_sha256"] for r in rows):
            raise ValueError("env contract changed within policy evaluation")
        if "protocol_receipt" in spec and spec["protocol_receipt"]!=receipt:
            raise ValueError("provided protocol receipt disagrees with recorded episode metadata")
        identity=(receipt["env_contract_sha256"],receipt["condition_sha256"],receipt["reset_protocol_sha256"])
        if task_identity is None: task_identity=identity
        if task_identity!=identity: raise ValueError("different task/condition/noise/reset protocol")
        successes=sum(bool(r["terminal_success"]) for r in rows)
        if result["successes"]!=successes or result["success_rate"]!=successes/100:
            raise ValueError("summary does not match actual episode outcomes")
        methods.setdefault(spec["method"],[]).append({"model_seed":seed,"episodes":100,"successes":successes,
            "success_rate":successes/100,"mean_length_on_success":result["mean_length_on_success"],"source":spec,"protocol_receipt":receipt})
        identities.append({"path":spec["path"],"sha256":spec["sha256"]})
    required={"learned_mpc","bc","bc_dr"}
    if set(methods)!=required: raise ValueError("required small-model MPC/BC/DRBC comparison is incomplete")
    table=[]
    for method in sorted(methods):
        rows=methods[method]
        if sorted(r["model_seed"] for r in rows)!=list(MODEL_SEEDS):
            raise ValueError("each row requires exactly three independent seeds")
        table.append({"method":method,"success":population_summary([r["success_rate"] for r in rows]),"seeds":rows})
    learned=next(r for r in table if r["method"]=="learned_mpc")["success"]["mean"]
    negatives=[f"Learned MPC mean success ({learned}) was lower than {r['method']} ({r['success']['mean']})."
               for r in table if r["method"]!="learned_mpc" and r["success"]["mean"]>learned]
    lines=["# Learned-dynamics MPC comparison", "", "All outcomes are actual native FetchPush step50 success on paired held-out resets.", "",
           "| Method | Success mean | Population std | Episodes per seed |", "|---|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['method']} | {row['success']['mean']} | {row['success']['population_std']} | 100 |")
    lines += ["", "Negative results", ""] + (negatives or ["No lower learned-MPC mean was found in these exact paired inputs."])
    path=run.path/"comparison.md"; path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"table":table,"negative_results":negatives,"input_results":identities,"report":str(path),
            "std_definition":"population ddof=0 over independent trained small-model/policy seeds"}

def plot_prediction(spec,run):
    verify_file(spec["path"],spec["sha256"])
    payload=read_json(spec["path"])
    results=payload["models"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for model in results:
        seed=model["model_seed"]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,key in zip(axes,("object_mm","gripper_mm")):
            horizons=[r["horizon"] for r in model["summary"]]
            axis.errorbar(horizons,[r[key]["mean"] for r in model["summary"]],
                          yerr=[r[key]["population_std"] for r in model["summary"]],label="Learned dynamics")
            axis.errorbar(horizons,[r["cv_"+key]["mean"] for r in model["summary"]],
                          yerr=[r["cv_"+key]["population_std"] for r in model["summary"]],label="Constant velocity")
            axis.set(xlabel="Recorded-action open-loop horizon",ylabel=key.replace("_"," "),title=f"Small-model seed {seed}")
            axis.legend()
        fig.tight_layout(); fig.savefig(run.path/f"prediction-seed-{seed}.png",dpi=180); plt.close(fig)
    return {"source":spec,"plots":sorted(p.name for p in run.path.glob("*.png"))}
