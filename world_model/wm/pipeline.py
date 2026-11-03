"""Dependency-resolved explicit run plan; references point only to completed actual artifacts."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
from .provenance import digest,dump_new,read_json,sha256,verify_file

SCRIPTS={"collect":"collect_dataset.py","train":"train_model.py","prediction":"eval_prediction.py",
         "mpc":"eval_mpc.py","report":"report.py","plot":"plot_results.py"}

def run_plan(config,run,output_root):
    if [stage['stage'] for stage in config['stages']]!=list(SCRIPTS):
        raise ValueError('reduced run_all requires exactly collect/train/prediction/mpc/report/plot once, in order')
    stages=config["stages"]; context={}; outputs=[]
    names=[s["name"] for s in stages]
    if len(names)!=len(set(names)): raise ValueError("stage names must be unique")
    for stage in stages:
        if stage["stage"] not in SCRIPTS: raise ValueError("cut or unsupported stage in reduced run-all")
        verify_file(stage["config"]["path"],stage["config"]["sha256"])
    def reference(name):
        stage,key=name.split(".",1)
        if stage not in context or key not in context[stage]: raise ValueError(f"unfinished or unknown artifact reference: {name}")
        return context[stage][key]
    for index,stage in enumerate(stages):
        kind=stage["stage"]
        if kind not in SCRIPTS: raise ValueError("unsupported stage")
        verify_file(stage["config"]["path"],stage["config"]["sha256"])
        cfg=read_json(stage["config"]["path"])
        for key,ref in stage.get("config_bindings",{}).items(): cfg[key]=reference(ref)
        for key,refs in stage.get("append_bindings",{}).items():
            if not isinstance(cfg[key],list): raise ValueError("append binding target must be a list")
            cfg[key]=list(cfg[key])
            for ref in refs: cfg[key]+=reference(ref)
        cfg_path=run.path/f"stage-{index:02d}-{stage['name']}-config.json"
        dump_new(cfg_path,cfg)
        script=Path(__file__).resolve().parents[1]/"scripts"/SCRIPTS[kind]
        command=[sys.executable,str(script),"--config",str(cfg_path),"--output-root",str(output_root)]
        for flag,ref in stage.get("input_bindings",{}).items():
            if flag not in ("intake","checkpoints","resume"): raise ValueError("unsupported plan input flag")
            command += ["--"+flag,str(reference(ref))]
        for flag,path in stage.get("inputs",{}).items():
            if flag not in ("intake","checkpoints","resume"): raise ValueError("unsupported plan input flag")
            command += ["--"+flag,str(path)]
        run.event("stage_started",stage=stage["name"],command=command,effective_config_sha256=sha256(cfg_path))
        proc=subprocess.run(command,capture_output=True,text=True,check=False)
        stdout=run.path/f"stage-{index:02d}-stdout.log"; stderr=run.path/f"stage-{index:02d}-stderr.log"
        stdout.write_text(proc.stdout,encoding="utf-8"); stderr.write_text(proc.stderr,encoding="utf-8")
        if proc.returncode!=0: raise RuntimeError(f"stage {stage['name']} failed ({proc.returncode}); see {stderr}")
        lines=[line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines: raise RuntimeError("child returned no attempt path")
        attempt=Path(lines[-1]).resolve(); parent=Path(output_root).resolve()
        if parent not in attempt.parents: raise ValueError("child attempt escaped declared output root")
        result_path=attempt/"result.json"; result=read_json(result_path)
        if result["status"]!="completed": raise ValueError("child is not complete")
        record={"attempt":str(attempt),"result_path":str(result_path),
                "result_receipt":{"path":str(result_path),"sha256":sha256(result_path)}}
        if kind=="collect":
            intake={"schema":"wm-intake-v1","required_sources":["expert","bc","noisy_bc"],
                    "manifests":stage.get("additional_manifests",[])+[
                        {"path":result["manifest"],"sha256":result["manifest_sha256"]}]}
            intake_path=run.path/f"{stage['name']}-intake.json"; dump_new(intake_path,intake)
            record["intake"]=str(intake_path)
        if kind=="train":
            record["checkpoints"]=str(attempt/"checkpoints.json")
            policies=[]
            for checkpoint in result["checkpoints"]:
                policies.append({"kind":"external","method":"learned_mpc","adapter_module":"wm.adapter",
                    "adapter_sha256":sha256(Path(__file__).with_name("adapter.py")),"mode":"formal",
                    "checkpoint":checkpoint["path"],"checkpoint_sha256":checkpoint["sha256"],
                    "model_seed":checkpoint["model_seed"],"planner":config["planner"]})
            policy_path=run.path/f"{stage['name']}-policies.json"; dump_new(policy_path,{"policies":policies})
            record["policy_manifest_receipt"]={"path":str(policy_path),"sha256":sha256(policy_path)}
        if kind=="mpc":
            record["nominal_files"]=[{"path":str(attempt/f"nominal-seed-{s}.json"),
                                      "sha256":sha256(attempt/f"nominal-seed-{s}.json")} for s in result["model_seeds"]]
            record["policy_results"]=[{**item,"method":result["method"]} for item in record["nominal_files"]]
        context[stage["name"]]=record
        outputs.append({"name":stage["name"],"kind":kind,**record})
        run.event("stage_completed",stage=stage["name"],result=record["result_receipt"])
    return {"stages":outputs,"artifact_context":context}
