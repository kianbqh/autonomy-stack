"""Native generation and evaluator revisions have separate verified identities."""
import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from common import sha256,source_manifest,write_json
from eval_traj import validate_native_generation

@pytest.fixture
def generation(tmp_path):
    repo=tmp_path/"repo";slam=repo/"slam";run=tmp_path/"run";native=run/"native";native.mkdir(parents=True)
    for name in ["CMakeLists.txt","dependencies.lock.json","prerequisites.lock.json","src/cloud.cpp","src/geometry.cpp","src/pipeline.cpp","include/slam/core.hpp","apps/run_kitti_slam.cpp","python/eval_traj.py"]:
        path=slam/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    binary=tmp_path/"native-binary";binary.write_bytes(b"fixed native executable")
    config=run/"config.json";write_json(config,{"algorithm":{"max_iterations":40},"expected_frames":3})
    (run/"native.cfg").write_text("expected_frames=3\nmax_iterations=40\n")
    write_json(run/"input-manifest.json",{"verified":True,"frames":3})
    sources=source_manifest(slam);write_json(run/"source-sha256.json",sources)
    build={"verified":True,"git_sha":"native-generation-commit","binary":str(binary),"binary_sha256":sha256(binary),"source_files":sources};write_json(run/"build-receipt.json",build)
    write_json(run/"run.json",{"status":"complete","native_exit_code":0,"git_sha":build["git_sha"],"config_sha256":sha256(config),"binary_sha256":sha256(binary),"build_receipt_sha256":sha256(run/"build-receipt.json"),"input_manifest_sha256":sha256(run/"input-manifest.json")})
    write_json(native/"native_summary.json",{"status":"complete","frames":3})
    return native,config,repo,{"git_sha":"new-evaluator-commit","whole_worktree_status":""}

def test_evaluator_only_revision_keeps_native_generation_identity(generation):
    native,config,repo,state=generation
    (repo/"slam/python/eval_traj.py").write_text("new evaluator implementation")
    result=validate_native_generation(native,config,repo,state)
    assert result["native_generation_git_sha"]=="native-generation-commit"
    assert result["evaluation_git_sha"]=="new-evaluator-commit"

@pytest.mark.parametrize("target",["src/cloud.cpp","src/new_native.cpp","dependencies.lock.json"])
def test_modified_or_added_native_input_rejected(generation,target):
    native,config,repo,state=generation;(repo/"slam"/target).write_text("changed")
    with pytest.raises(ValueError,match="compiled native"):validate_native_generation(native,config,repo,state)

@pytest.mark.parametrize("target,match",[("config.json","runtime config"),("build-receipt.json","build receipt"),("source-sha256.json","source inventories"),("input-manifest.json","input manifest"),("native.cfg","flattened runtime")])
def test_altered_generation_receipts_rejected(generation,target,match):
    native,config,repo,state=generation;path=native.parent/target
    if target=="source-sha256.json":write_json(path,{})
    else:path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match=match):validate_native_generation(native,config,repo,state)

def test_changed_binary_rejected(generation):
    native,config,repo,state=generation;(repo.parent/"native-binary").write_bytes(b"changed")
    with pytest.raises(ValueError,match="binary changed"):validate_native_generation(native,config,repo,state)

def test_dirty_evaluator_checkpoint_rejected(generation):
    native,config,repo,state=generation;state["whole_worktree_status"]=" M slam/python/eval_traj.py"
    with pytest.raises(ValueError,match="clean source"):validate_native_generation(native,config,repo,state)
