"""
Debris-avoidance environment with the interface of the SABLAS repo's envs (env_drone.py):

    state, obstacle, goal = env.reset()
    u_nominal = env.nominal_controller(state, goal)
    state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)
    env.nominal_dynamics_torch(state, u)      # used by modules/trainer.py

It wraps the numpy simulator in cw_env.py (real two-body + J2 ECI propagation, imperfect
thruster, moving non-cooperative debris, LVLH frame of a reference slot).

Units. The repo trainer is unit-less (dt = 0.1, alpha = 1, distances 0.6 / 1.0), so this
environment hands it SCALED quantities:
    positions / 1 km, velocities / 10 m/s, time unit = 100 s (so dt = 10 s = 0.1),
    control normalised by u_max (|u| <= 1, clipped by the environment).
In these units the Clohessy-Wiltshire model is exactly linear,  ds/dtau = A_s s + B_s u,
and the keep-out radius is dang_dist = 2.5, safe_dist = 5.0.
"""
import numpy as np
from envs import cw_env as E

TIME_UNIT = 100.0                                  # s per time unit
POS_SCALE, VEL_SCALE = 1000.0, 10.0                # m, m/s per state unit
SCALE = np.array([POS_SCALE] * 3 + [VEL_SCALE] * 3)

# CW in scaled coordinates: s_s = s / SCALE, tau = t / TIME_UNIT
A_S = TIME_UNIT * (np.diag(1.0 / SCALE) @ E.A_CW @ np.diag(SCALE))
B_S = np.zeros((6, 3))
B_S[3:] = np.eye(3) * (TIME_UNIT * E.U_MAX / VEL_SCALE)


class Debris(object):
    """One episode at a time (the repo's convention). Episodes are generated in pools of
    `pool` with cw_env.Episodes (which rejects unwinnable starts) and handed out one by one."""

    n_state, m_control, n_pos = 6, 3, 3
    k_obstacle = E.M_DEB
    dt = E.DT / TIME_UNIT                           # 0.1
    safe_dist = 5000.0 / POS_SCALE                  # 5.0
    dang_dist = E.D_SAFE / POS_SCALE                # 2.5

    def __init__(self, seed=0, pool=32):
        self.rng = np.random.default_rng(seed)
        self.pool_size, self._pool, self._i = pool, None, 0
        self.ep = None

    # ------------------------------------------------------------ nominal model
    @staticmethod
    def nominal_dynamics_np(state, u):
        """Known CW model in scaled units, numpy. state (..., 6), u (..., 3) normalised."""
        return state @ A_S.T + u @ B_S.T

    @staticmethod
    def nominal_dynamics_torch(state, u):
        """Same as nominal_dynamics_np, torch (state (bs, 6), u (bs, 3)); passed to the Trainer."""
        import torch
        A = torch.as_tensor(A_S, dtype=state.dtype, device=state.device)
        B = torch.as_tensor(B_S, dtype=state.dtype, device=state.device)
        return state @ A.T + u @ B.T

    def nominal_controller(self, state, goal=None):
        """Slot-holding controller of cw_env (cancels CW drift + PD), normalised by u_max."""
        own = (np.asarray(state, dtype=np.float64) * SCALE)[None]
        return (E.pi_nom(own)[0] / E.U_MAX).astype(np.float32)

    # ------------------------------------------------------------ episodes
    def _next_episode(self):
        """A single-episode cw_env.Episodes taken from the pool (refilled when empty)."""
        if self._pool is None or self._i >= self.pool_size:
            self._pool, self._i = E.Episodes(self.pool_size, self.rng), 0
        p, i = self._pool, self._i
        self._i += 1
        ep = E.Episodes.__new__(E.Episodes)            # a 1-episode view of the pool entry
        ep.n, ep.rng = 1, self.rng
        ep.ref, ep.sc, ep.deb = p.ref[i:i + 1].copy(), p.sc[i:i + 1].copy(), p.deb[i:i + 1].copy()
        return ep

    def _observe(self):
        """Scaled state (6,) and obstacles (k, 6): debris states in the slot's LVLH frame."""
        own, rel = self.ep.observe()
        return own[0] / SCALE, (rel[0] + own[0]) / SCALE

    def reset(self):
        """Start a new episode. Returns state (6,), obstacle (k, 6), goal (6,) (the slot = origin)."""
        self.ep, self.t = self._next_episode(), 0
        self.state, self.obstacle = self._observe()
        self.goal = np.zeros(6)
        self.min_sep_m = np.inf                         # interpolated minimum distance this episode
        return self.state.copy(), self.obstacle.copy(), self.goal.copy()

    def step(self, u):
        """Apply the normalised command u (3,) (clipped to +/-1) to the real system for one
        control period. Returns state_next, state_nominal_next (CW prediction from the
        saturated command), obstacle_next (moving debris!), goal_next, done."""
        u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
        state_nominal_next = self.state + self.dt * self.nominal_dynamics_np(self.state, u)
        prev_rel = (self.obstacle[:, :3] - self.state[:3]) * POS_SCALE
        self.ep.step(u[None] * E.U_MAX)
        self.t += 1
        self.state, self.obstacle = self._observe()
        rel = (self.obstacle[:, :3] - self.state[:3]) * POS_SCALE
        self.min_sep_m = min(self.min_sep_m, float(E.segment_min_dist(prev_rel, rel).min()))
        done = self.t >= E.STEPS
        return self.state.copy(), state_nominal_next, self.obstacle.copy(), self.goal.copy(), done
