"""Behavior cloning (PyTorch MLP) for GoalReach2D, evaluated closed-loop with fused vs raw state estimates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env import GoalReach2DEnv, expert_action


class Policy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2), nn.Tanh())

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        with torch.no_grad():
            return self(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()


def make_policy(seed=0):
    torch.manual_seed(seed)
    return Policy()


def rollout(env, actor, seed, label_with_expert=False):
    """Run one episode. Returns (return, success, steps, [(obs, expert_label)])."""
    obs, _ = env.reset(seed=seed)
    ret, data = 0.0, []
    while True:
        if label_with_expert:
            data.append((obs, expert_action(env.truth_obs())))
        obs, r, term, trunc, info = env.step(actor(obs, env))
        ret += r
        if term or trunc:
            return ret, info["success"], env.steps, data


def train(policy, X, Y, epochs=60, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    X, Y = torch.as_tensor(np.array(X)), torch.as_tensor(np.array(Y))
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            loss = nn.functional.mse_loss(policy(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(nn.functional.mse_loss(policy(X), Y))


def evaluate(actor, obs_mode, n, seed0=100_000):
    env = GoalReach2DEnv(obs_mode)
    rs = [rollout(env, actor, seed0 + i) for i in range(n)]
    succ = [r[1] for r in rs]
    return {"success_rate": float(np.mean(succ)), "mean_return": float(np.mean([r[0] for r in rs])),
            "mean_steps_success": float(np.mean([r[2] for r in rs if r[1]])) if any(succ) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=200)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--out", default="results/policy_metrics.json")
    ap.add_argument("--model", default="models/bc_policy.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    expert = lambda o, env: expert_action(env.truth_obs())

    # 1) Expert demonstrations (privileged ground-truth observations).
    env = GoalReach2DEnv("truth")
    X, Y = [], []
    for i in range(args.demos):
        for o, a in rollout(env, expert, i, label_with_expert=True)[3]:
            X.append(o); Y.append(a)
    bc = make_policy(args.seed)
    bc_loss = train(bc, X, Y, seed=args.seed)
    print(f"BC: {len(X)} transitions, train MSE {bc_loss:.4f}")

    # 2) Closed-loop evaluation on held-out goals (unseen seeds).
    res = {
        "expert_truth_obs": evaluate(expert, "truth", args.eval),
        "expert_ekf_obs": evaluate(lambda o, e: expert_action(o), "ekf", args.eval),
        "expert_raw_obs": evaluate(lambda o, e: expert_action(o), "raw", args.eval),
        "bc_ekf_obs": evaluate(lambda o, e: bc.act(o), "ekf", args.eval),
        "bc_raw_obs": evaluate(lambda o, e: bc.act(o), "raw", args.eval),
    }
    res["meta"] = {"seed": args.seed, "demo_episodes": args.demos, "bc_transitions": len(X), "bc_train_mse": bc_loss,
                   "eval_episodes": args.eval, "success_radius_m": 0.75, "park_speed_mps": 0.3, "time_limit_s": 40.0,
                   "wall_time_s": time.time() - t0}
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, **{k: v.detach().numpy() for k, v in bc.state_dict().items()})
    torch.save(bc.state_dict(), model_path.with_suffix(".pt"))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
