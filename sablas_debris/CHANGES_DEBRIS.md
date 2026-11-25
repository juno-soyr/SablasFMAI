# Moving-obstacle patch for MIT-REALM/sablas (debris avoidance)

Base: upstream commit 2ba1c5c (2021-09-17). Everything not listed here is unchanged.

## Modified (moving obstacles)
- `modules/dataset.py`: stores `obstacle_next` with every sample; `add_data(..., obstacle_next=None)`
  (None = static obstacles, the old behaviour); `sample_data` returns a 6th array `o_next`.
- `modules/trainer.py`: unpacks `obstacle_next` and evaluates the next-step barrier as
  `cbf(state_next[_with_grad], obstacle_next)` in `train_cbf`, `train_controller`, `train_cbf_and_controller`.
  Losses, optimisers and constants are untouched.

## Added
- `envs/cw_env.py`: numpy simulator (two-body + J2 ECI propagation, imperfect thruster, moving debris).
- `envs/env_debris.py`: wraps it in the repo's env interface, in scaled units
  (positions / 1 km, velocities / 10 m/s, time unit 100 s so dt = 0.1, control normalised by u_max).
  Keep-out `dang_dist` = 2.5, `safe_dist` = 5.0 in these units.
- `scripts/train_debris.py`, `scripts/test_debris.py`.

## Run (from the repo root; needs torch)
    python scripts/train_debris.py --steps 1000000
    python scripts/test_debris.py --episodes 200

## Faithful-baseline notes
- No exploration noise, no error-signal gain, no multi-step loss, no run-time shield, action
  loss weight 0.1 and eps_action 0.2 as upstream.
- Upstream samples from all three class buffers, so the training script waits until each has data.
- `scripts/train_drone.py` still works (static obstacles via the default `obstacle_next=None`).
