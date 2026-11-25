"""Compare the trained SABLAS controller with the nominal slot-holding controller on
held-out episodes (collision = interpolated minimum separation < D_SAFE).

    python scripts/test_debris.py [--episodes 200] [--seed 99]
"""
import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import argparse
import numpy as np
import torch
from envs.env_debris import Debris
from modules.network import NNController
from envs import cw_env as E


def run(env, controller, episodes):
    """Run episodes; controller(state, obstacle, u_nominal, state_error) -> u, or None for the nominal law.
    Returns per-episode minimum separation [m]."""
    out = []
    for _ in range(episodes):
        state, obstacle, goal = env.reset()
        err, done = np.zeros(env.n_state, dtype=np.float32), False
        while not done:
            u_nom = env.nominal_controller(state, goal)
            if controller is None:
                u = u_nom
            else:
                with torch.no_grad():
                    u = controller(*(torch.from_numpy(np.asarray(x, dtype=np.float32)[None])
                                     for x in (state, obstacle, u_nom, err)))
                u = np.squeeze(u.cpu().numpy())
            state_next, state_nom_next, obstacle, goal, done = env.step(u)
            err = (state_next - state_nom_next) / env.dt
            state = state_next
        out.append(env.min_sep_m)
    return np.array(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=99, help='test seed (training uses 0)')
    ap.add_argument('--weights', default='./data/debris_controller_weights.pth')
    a = ap.parse_args()

    ctrl = NNController(n_state=6, k_obstacle=Debris.k_obstacle, m_control=3)
    ctrl.load_state_dict(torch.load(a.weights, map_location='cpu'))
    ctrl.eval()
    for name, c in (('nominal', None), ('SABLAS', ctrl)):
        ms = run(Debris(seed=a.seed), c, a.episodes)            # same seed -> same episodes
        print('{:8s} collisions {:5.1f}%   median min sep {:7.0f} m   5th pct {:7.0f} m'.format(
            name, 100 * np.mean(ms < E.D_SAFE), np.median(ms), np.percentile(ms, 5)))
