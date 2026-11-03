import ast
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from manip.protocol import EVAL_SEEDS,VALIDATION_SEEDS,aggregate_policies,assert_formal,child_seed,split_for_reset,wilson,assert_panel,digest,summarize
from manip.provenance import Attempt,dump_new,resume_guard,sha256
from manip.shield import Shield,distance_segment
from manip.expert import PushExpert

ROOT=Path(__file__).resolve().parents[1]

class ProtocolTests(unittest.TestCase):
    def config(self,name):
        return json.loads((ROOT/"configs"/name).read_text())




    def test_split_group_reproducible_and_rejects_test(self):
        for seed in range(100):
            self.assertEqual(split_for_reset(seed),split_for_reset(seed))
        with self.assertRaises(ValueError):
            split_for_reset(EVAL_SEEDS[0])
        with self.assertRaises(ValueError):
            split_for_reset(VALIDATION_SEEDS[0])
        self.assertNotEqual(child_seed("wrapper",1),child_seed("wrapper",2))


    def test_frozen_producer_policy_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            attempt=Attempt.__new__(Attempt)
            attempt.path=Path(folder)
            attempt.producer_base={"source_sha256":"known"}
            checkpoint=Path(folder)/"actor.bytes"
            checkpoint.write_bytes(b"actual-checkpoint-fixture")
            policy={"name":"expert","gain":20.0}
            receipt=attempt.producer_receipt(policy,checkpoint)
            self.assertEqual(receipt["policy"]["sha256"],digest(policy))
            self.assertEqual(receipt["checkpoint"]["sha256"],sha256(checkpoint))
            Path(receipt["policy"]["path"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                attempt.producer_receipt(policy,checkpoint)


    def test_intervals_and_population_std(self):
        low,high=wilson(0,100)
        self.assertAlmostEqual(low,0)
        self.assertGreater(high,0)
        self.assertEqual(wilson(0,0),[None,None])
        data=aggregate_policies([{"success_rate":x} for x in (0.,.5,1.)])
        self.assertAlmostEqual(data["population_std"],math.sqrt(1/6))


    def test_immutable_outputs(self):
        # Temp directory is explicitly inside this G-backed candidate, never platform default C:.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path=Path(folder)/"a.json"
            dump_new(path,{"a":1})
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):
                dump_new(path,{"a":2})
            self.assertEqual(original,path.read_bytes())

    def test_resume_guard_each_dimension(self):
        original={k:"same" for k in ("git_sha","config_sha256","source_hashes","packages","environment")}
        resume_guard(original,original)
        for key in original:
            changed={**original,key:"changed"}
            with self.assertRaises(ValueError):
                resume_guard(original,changed)




class ReducedProtocolTests(unittest.TestCase):
    def config(self,name): return json.loads((ROOT/'configs'/name).read_text())
    def test_matched_bc_conditions_and_pool_budget(self):
        a=self.config('push_bc_mlp.yaml'); b=self.config('push_dr_train.yaml')
        assert_formal(a); assert_formal(b)
        self.assertEqual(a['bc'],b['bc']); self.assertEqual(a['demo_budget'],100)
        for key,value in [('demo_budget',25),('seeds',[0]),('horizon',100),('method','sac')]:
            with self.assertRaises(ValueError): assert_formal({**a,key:value})
        nominal=self.config('push_expert.yaml'); dr=self.config('push_dr_expert.yaml')
        self.assertEqual((nominal['reset_start'],nominal['max_attempts']),(dr['reset_start'],dr['max_attempts']))
        self.assertEqual(nominal['successful_demos'],dr['successful_demos'])
        self.assertIsNone(nominal['training_randomization']); self.assertIsNotNone(dr['training_randomization'])
    def test_frozen_panel_counts_axes_and_no_adaptive_cells(self):
        panel=self.config('dr_eval.yaml'); assert_panel(panel)
        self.assertEqual(len(panel['cells']),11)
        nominal=panel['cells'][0]
        for cell in panel['cells'][1:]: self.assertEqual(sum(cell[k]!=nominal[k] for k in ('mass','friction','noise_sigma','latency','gain')),1)
        changed=copy.deepcopy(panel); changed['cells'][1]['mass']=.75
        with self.assertRaises(ValueError): assert_panel(changed)
    def test_training_and_eval_seed_domains(self):
        self.assertTrue(set(EVAL_SEEDS).isdisjoint(VALIDATION_SEEDS))
        for seed in list(range(10))+[800000,801000,802000]: self.assertIn(split_for_reset(seed),('train','val','test'))
        for seed in [900000,1000000]:
            with self.assertRaises(ValueError): split_for_reset(seed)
    def test_features_have_only_public_observation_and_goal(self):
        tree=ast.parse((ROOT/'manip/bc.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
        fields={n.slice.value for n in ast.walk(fn) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='obs' and isinstance(n.slice,ast.Constant)}
        self.assertEqual(fields,{'observation','desired_goal'})
    def test_unmeasured_physical_audit_is_not_zero_violations(self):
        row={'terminal_success':False,'ever_success':False,'length':50,'physical_audit_active':False,'physical_violations':None}
        self.assertIsNone(summarize([row])['physical_violation_episodes'])
        active={**row,'physical_audit_active':True,'physical_violations':2}
        self.assertEqual(summarize([active])['physical_violation_episodes'],1)

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.config={"workspace_low":[-1,-1,-1],"workspace_high":[1,1,1],"end_effector_radius":.01,
                     "k_sigma":3.,"max_speed_m_s":.5,"native_cartesian_scale_m":.05,
                     "keepout_spheres":[{"center":[.04,0,0],"radius":.015}]}
        self.shield=Shield(self.config,.04)

    def test_swept_segment_crossing_detected(self):
        self.assertAlmostEqual(distance_segment([0,0,0],[-1,0,0],[1,0,0]),0)
        c={**self.config,"max_speed_m_s":100,"keepout_spheres":[{"center":[0,0,0],"radius":.1}]}
        shield=Shield(c,1)
        c["native_cartesian_scale_m"]=1.
        action,diagnostic=shield.project([1,0,0,0],[-.5,0,0])
        endpoint=[-.5+action[0],action[1],action[2]]
        self.assertEqual(shield.violations([-.5,0,0],endpoint),[])
        self.assertLess(endpoint[0],-.10999)

    def test_random_safe_projection_geometry(self):
        rng=random.Random(0)
        checked=0
        for _ in range(1000):
            p=[rng.uniform(-.8,.8) for _ in range(3)]
            a=[rng.uniform(-1,1) for _ in range(4)]
            projected,diag=self.shield.project(a,p,.005)
            if diag["already_unsafe"]:
                continue
            end=[x+.05*y for x,y in zip(p,projected[:3])]
            self.assertEqual(self.shield.violations(p,end,.005),[])
            self.assertLessEqual(math.sqrt(sum(x*x for x in projected[:3]))*.05,.02+1e-10)
            checked+=1
        self.assertGreater(checked,900)

    def test_unsafe_start_is_explicit(self):
        action,diagnostic=self.shield.project([1,1,1,0],[1.,1.,1.])
        self.assertEqual(action[:3],[0,0,0])
        self.assertTrue(diagnostic["already_unsafe"])
        self.assertTrue(diagnostic["infeasible"])

    def test_uncertainty_cannot_reduce_clearance(self):
        a,_=self.shield.project([1,0,0,0],[-.02,0,0],0.)
        b,_=self.shield.project([1,0,0,0],[-.02,0,0],.003)
        self.assertLessEqual(b[0],a[0]+1e-10)

class ExpertTests(unittest.TestCase):
    def test_deterministic_bounded_controller(self):
        expert=PushExpert()
        obs={"observation":[1.2,.7,.6,1.3,.7,.42]+[0.]*19,"desired_goal":[1.4,.8,.42]}
        first=expert.act(obs)
        expert.reset()
        self.assertEqual(first,expert.act(obs))
        self.assertEqual(len(first),4)
        self.assertTrue(all(-1<=x<=1 for x in first))
        self.assertEqual(first[3],0.)

if __name__=="__main__":
    unittest.main()
