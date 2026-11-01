import os

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from fusion_core.env import GoalReach2DEnv, expert_action
from fusion_core.imitation import Policy, evaluate, make_policy, rollout, train
from fusion_core.numpy_policy import NumpyMLP


def test_env_passes_gymnasium_checker():
    check_env(GoalReach2DEnv("ekf"), skip_render_check=True)
    env = gym.make("GoalReach2D-v0", obs_mode="raw")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)


def test_expert_solves_task_with_truth():
    res = evaluate(lambda o, e: expert_action(e.truth_obs()), "truth", n=10)
    assert res["success_rate"] == 1.0


def test_episode_is_deterministic_given_seed():
    env = GoalReach2DEnv("ekf")
    a = rollout(env, lambda o, e: expert_action(o), seed=5)
    b = rollout(env, lambda o, e: expert_action(o), seed=5)
    assert a[0] == pytest.approx(b[0]) and a[2] == b[2]


def test_numpy_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    p = Policy()
    path = tmp_path / "w.npz"
    np.savez(path, **{k: v.detach().numpy() for k, v in p.state_dict().items()})
    m = NumpyMLP(path)
    x = np.random.default_rng(0).standard_normal((32, 4)).astype(np.float32)
    ref = p(torch.from_numpy(x)).detach().numpy()
    out = np.stack([m(xi) for xi in x])
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_training_repeats_from_the_same_seed():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    Y = np.tanh(X[:, :2]).astype(np.float32)
    first = make_policy(7)
    first_loss = train(first, X, Y, epochs=2, seed=7)
    torch.rand(100)
    second = make_policy(7)
    second_loss = train(second, X, Y, epochs=2, seed=7)
    assert first_loss == second_loss
    for name, weights in first.state_dict().items():
        torch.testing.assert_close(weights, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.skipif(not os.path.exists("models/bc_policy.pt"), reason="run fusion_core.imitation first")
def test_trained_policy_reaches_goals_with_ekf():
    p = Policy()
    p.load_state_dict(torch.load("models/bc_policy.pt", weights_only=True))
    res = evaluate(lambda o, e: p.act(o), "ekf", n=20, seed0=999)
    assert res["success_rate"] >= 0.9
