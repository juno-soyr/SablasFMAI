"""Train SABLAS (repo code, moving-obstacle patch) on the debris-avoidance environment.
Same loop as scripts/train_drone.py, with obstacle_next stored and used.

    python scripts/train_debris.py [--steps 1000000] [--seed 0]
"""
import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import argparse
import numpy as np
import torch
from envs.env_debris import Debris
from modules import config
from modules import utils
from modules.dataset import Dataset
from modules.trainer import Trainer
from modules.network import CBF, NNController
from envs import cw_env as E

np.set_printoptions(4)


def main(steps, seed):
    env = Debris(seed=seed)
    n_state, k, m, n_pos = env.n_state, env.k_obstacle, env.m_control, env.n_pos
    torch.manual_seed(seed); np.random.seed(seed)

    nn_controller = NNController(n_state=n_state, k_obstacle=k, m_control=m)
    cbf = CBF(n_state=n_state, k_obstacle=k, m_control=m)
    dataset = Dataset(n_state=n_state, k_obstacle=k, m_control=m, n_pos=n_pos,
                      safe_dist=env.safe_dist, dang_dist=env.dang_dist)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, n_pos=n_pos,
                      dt=env.dt, safe_dist=env.safe_dist, dang_dist=env.dang_dist, action_loss_weight=0.1)
    state, obstacle, goal = env.reset()
    state_error = np.zeros((n_state,), dtype=np.float32)

    safety_rate, collision_rate = 0.0, float('nan')
    os.makedirs('./data', exist_ok=True)

    for i in range(steps):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, n_state).astype(np.float32)),
            torch.from_numpy(obstacle.reshape(1, k, n_state).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, m).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, n_state).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        # obstacle_next: the debris moves, so the barrier at the next step needs its new position
        dataset.add_data(state, obstacle, u_nominal, state_next, state_error, obstacle_next)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=n_pos, dang_dist=env.dang_dist))
        safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

        # error between the true next state and the nominal model's prediction -> controller input
        state_error = (state_next - state_nominal_next) / env.dt

        state, obstacle, goal = state_next, obstacle_next, goal_next

        buffers = (dataset.buffer_safe, dataset.buffer_dang, dataset.buffer_mid)
        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0 and all(len(b) > 0 for b in buffers):
            loss_np, acc_np = trainer.train_cbf_and_controller()
            print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, '
                  'episode collision rate: {:.3f}, acc: {}'.format(i, loss_np, safety_rate, collision_rate, acc_np), flush=True)
            torch.save(cbf.state_dict(), './data/debris_cbf_weights.pth')
            torch.save(nn_controller.state_dict(), './data/debris_controller_weights.pth')

        if done:
            collided = float(env.min_sep_m < E.D_SAFE)          # interpolated minimum distance
            collision_rate = collided if np.isnan(collision_rate) else collision_rate * 0.98 + collided * 0.02
            state, obstacle, goal = env.reset()
            state_error = np.zeros((n_state,), dtype=np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=config.TRAIN_STEPS, help='environment steps (repo: 1e6)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    main(a.steps, a.seed)
