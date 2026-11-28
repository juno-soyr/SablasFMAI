"""Diagnose a trained debris CBF + controller (run from the repo root, after train_debris.py).

    python scripts/diagnose_debris.py

Prints (1) h against distance for debris straight ahead, (2) how many ReLU units are dead and
whether h is a constant, (3) how far the controller deviates from the nominal command.
"""
import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch
from envs.env_debris import Debris
from modules.network import CBF, NNController

env = Debris(seed=123)
k = env.k_obstacle
cbf = CBF(n_state=6, k_obstacle=k, m_control=3)
ctrl = NNController(n_state=6, k_obstacle=k, m_control=3)
cbf.load_state_dict(torch.load('./data/debris_cbf_weights.pth', map_location='cpu')); cbf.eval()
ctrl.load_state_dict(torch.load('./data/debris_controller_weights.pth', map_location='cpu')); ctrl.eval()

# 1. h vs distance: spacecraft at the origin, debris on the x axis, closing at 0 or 10 m/s
print('1) h vs distance (keep-out radius 2.5, safe distance 5.0; want h < 0 inside 2.5 and h > 0 beyond 5)')
print('   dist [km]   h (static)   h (closing 10 m/s)')
with torch.no_grad():
    for d in (0.5, 1.5, 2.5, 3.5, 5.0, 8.0, 12.0):
        row = []
        for vx in (0.0, -1.0):                                   # scaled velocity: -1 = -10 m/s
            obs = np.zeros((1, k, 6), dtype=np.float32); obs[:, :, 0] = d; obs[:, :, 3] = vx
            row.append(cbf(torch.zeros(1, 6), torch.from_numpy(obs))[0, 0].item())
        print('   {:8.1f}   {:10.3f}   {:10.3f}'.format(d, *row))

# 2. data from the nominal controller: spread of h and dead ReLU units
S, O = [], []
s, o, g = env.reset()
for _ in range(2000):
    s2, _, o2, g, done = env.step(env.nominal_controller(s, g))
    S.append(s); O.append(o); s, o = (env.reset()[:2] if done else (s2, o2))
S, O = (torch.from_numpy(np.array(a, dtype=np.float32)) for a in (S, O))
with torch.no_grad():
    h = cbf(S, O)
    x = (S.unsqueeze(2) - O.permute(0, 2, 1))
    dead = []
    for conv in (cbf.conv0, cbf.conv1, cbf.conv2, cbf.conv3):
        x = torch.relu(conv(x))
        dead.append(float((x.amax(dim=(0, 2)) == 0).float().mean()))
print('\n2) h over {} nominal-policy samples: mean {:.3f}, std {:.4f}, min {:.3f}, max {:.3f}'.format(
    h.numel(), h.mean(), h.std(), h.min(), h.max()))
print('   dead ReLU units per layer (never active on this data): ' + ', '.join('{:.0%}'.format(d) for d in dead))
print('   -> std ~ 0 or most units dead means the barrier has collapsed to a constant')

# 3. how far the controller moves away from the nominal command
dev = []
for _ in range(10):
    s, o, g = env.reset(); err = np.zeros(6, dtype=np.float32); done = False
    while not done:
        un = env.nominal_controller(s, g)
        with torch.no_grad():
            u = ctrl(*(torch.from_numpy(np.asarray(a, dtype=np.float32)[None]) for a in (s, o, un, err)))
        u = u.numpy()[0]; dev.append(np.abs(u - un).max())
        s2, sn, o, g, done = env.step(u); err = (s2 - sn) / env.dt; s = s2
print('\n3) controller deviation from nominal, max over axes (normalised, 1 = full thrust): '
      'mean {:.4f}, 95th pct {:.4f}'.format(np.mean(dev), np.percentile(dev, 95)))
print('   -> ~0 means the controller never learned to avoid anything')
