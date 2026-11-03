"""Opt-in actual Torch/physics checks. Fixtures are engineering checks only."""
import os
import unittest
@unittest.skipUnless(os.environ.get('M3_WORLD_MODEL_RUNTIME_TESTS')=='1','set M3_WORLD_MODEL_RUNTIME_TESTS=1 for bounded numerical fixtures')
class NumericalRuntimeTests(unittest.TestCase):
    def test_encoder_dynamics_decoder_gradients_reconstruction_and_rollout(self):
        import torch
        from wm.dynamics import SmallLatentDynamics
        torch.manual_seed(4)
        norms={k:{'mean':[0.]*d,'std':[1.]*d} for k,d in [('state',18),('action',4)]}
        model=SmallLatentDynamics(norms); x=torch.randn(32,18)*.1; a=torch.randn(32,4)*.1; y=x.clone(); y[:,:3]+=.02*a[:,:3]
        weights={'reconstruction':.25,'latent_consistency':.1}; opt=torch.optim.Adam(model.parameters(),lr=.003)
        first=float(model.loss(x,a,x,y,weights)[0].detach())
        for _ in range(40):
            loss,parts=model.loss(x,a,x,y,weights); opt.zero_grad(); loss.backward()
            for module in (model.encoder,model.dynamics,model.decoder):
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            opt.step()
        self.assertLess(float(model.loss(x,a,x,y,weights)[0].detach()),first)
        z=model.encode(x); self.assertEqual(tuple(z.shape),(32,8)); self.assertGreater(float(z.std(0).max()),1e-8)
        for _ in range(30): z=model.latent_step(z,a)
        decoded=model.decode(z); self.assertEqual(tuple(decoded.shape),(32,18)); self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue((decoded[:,6:9]>=-torch.pi).all()); self.assertTrue((decoded[:,6:9]<torch.pi).all())
    def test_torch_cem_quadratic(self):
        import torch
        from wm.mpc import Planner
        class ToyFixture:
            def encode(self,state): return state
            def latent_step(self,z,a):
                out=z.clone(); out[:,3:6]=a[:,:3]; return out
            def decode(self,z): return z
        c={'horizon':1,'candidates':128,'iterations':3,'elites':16,'scheme':'deterministic_latent','initial_std':.5,'minimum_std':.01,'action_penalty':0.}
        planner=Planner(c,'cpu',formal=False); action,stats=planner.plan([0]*18,[.35,-.45,.2],ToyFixture(),123)
        self.assertLess(sum((a-b)**2 for a,b in zip(action[:3],[.35,-.45,.2])),.03); self.assertEqual(stats['effective_horizon'],1)
        planner.reset(); self.assertIsNone(planner.warm)
@unittest.skipUnless(os.environ.get('M3_WORLD_MODEL_SIM_TESTS')=='1','set M3_WORLD_MODEL_SIM_TESTS=1 for actual MuJoCo checks')
class SimulatorTests(unittest.TestCase):
    def test_complete_clone_and_native_substep_equivalence(self):
        from manip.runtime_checks import stepping_regression
        config={'env_id':'FetchPush-v4','fault_stream_seed':710,'dr_stream_seed':720,'training_randomization':None}
        rows=stepping_regression(config); self.assertEqual(len(rows),6); self.assertEqual(sum(r['paired_control_steps'] for r in rows),300)
if __name__=='__main__': unittest.main()
