"""Meaningful physical/grouped protocol/CEM checks; standard library only."""
import copy
import json
import math
from pathlib import Path
import unittest
from wm.cem_core import cem
from wm.protocol import (SOURCES,add_delta,constant_velocity,formal_planner,observation_to_state,population_summary,
                         split_for_reset,state_delta,state_to_observation,wrap_angle)
ROOT=Path(__file__).resolve().parents[1]
class PhysicsProtocolTests(unittest.TestCase):
    def test_scaled_relative_velocity_and_redundancy(self):
        o=[0.]*25; o[:6]=[1,2,3,4,6,8]; o[11:14]=[.1,.2,3.2]
        o[14:17]=[.02,-.04,.06]; o[20:23]=[.04,.02,-.02]; o[17:20]=[.01,.02,.03]
        s=observation_to_state(o,.02)
        for x,y in zip(s[9:12],[3,-1,2]): self.assertAlmostEqual(x,y)
        rebuilt=state_to_observation(s,o,.02); self.assertEqual(rebuilt['observation'][6:9],[3,4,5]); self.assertEqual(rebuilt['achieved_goal'],s[3:6])
        for x,y in zip(rebuilt['observation'][14:17],o[14:17]): self.assertAlmostEqual(x,y)
        self.assertEqual(rebuilt['observation'][9:11],o[9:11])
    def test_angular_delta_and_cv_units(self):
        a=[0.]*18; b=a.copy(); a[6]=math.pi-.01; b[6]=-math.pi+.01
        delta=state_delta(a,b); self.assertAlmostEqual(delta[6],.02); self.assertAlmostEqual(add_delta(a,delta)[6],b[6])
        a[9]=.2; a[15]=.1; predicted=constant_velocity(a,20,.02)
        self.assertAlmostEqual(predicted[3],.08); self.assertAlmostEqual(predicted[0],.04); self.assertEqual(wrap_angle(math.pi),-math.pi)
    def test_reset_grouped_split_disjoint_reserved_allocations(self):
        self.assertEqual(SOURCES,('expert','bc','noisy_bc'))
        groups=[set(range(s,s+200)) for s in (800000,801000,802000)]
        self.assertTrue(all(a.isdisjoint(b) for i,a in enumerate(groups) for b in groups[i+1:]))
        for group in groups: self.assertEqual({split_for_reset(s) for s in group},{'train','val','test'})
        for seed in range(100): self.assertEqual(len({split_for_reset(seed) for _ in SOURCES}),1)
        for seed in (-1,900000,1000000,True):
            with self.assertRaises(ValueError): split_for_reset(seed)
    def test_episode_population_aggregation(self):
        p=population_summary([2,10]); self.assertEqual(p['mean'],6); self.assertEqual(p['population_std'],4)
        self.assertEqual(population_summary([])['count'],0)
    def test_frozen_main_planner_and_single_model(self):
        config=json.loads((ROOT/'configs/mpc.json').read_text()); formal_planner(config['planner'])
        self.assertEqual(len(config['eval_reset_seeds']),100); self.assertEqual(config['model_seeds'],[0,1,2])
        for key,value in [('horizon',15),('candidates',500),('iterations',5),('elites',50)]:
            c={**config['planner'],key:value}
            with self.assertRaises(ValueError): formal_planner(c)
        model=json.loads((ROOT/'configs/small_model.json').read_text())
        self.assertEqual((model['hidden'],model['latent']),(64,8)); self.assertNotIn('members',model)
    def test_cem_bounded_reproducible_quadratic(self):
        target=[.35,-.45,.2]; loss=lambda x:sum((a-b)**2 for a,b in zip(x,target))
        result=cem(loss,3,seed=123); self.assertLess(result[1],.0005); self.assertTrue(all(-1<=x<=1 for x in result[0])); self.assertEqual(result,cem(loss,3,seed=123))
if __name__=='__main__': unittest.main()
