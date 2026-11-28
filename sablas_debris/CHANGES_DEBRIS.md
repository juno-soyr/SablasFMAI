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
    python scripts/train_debris.py --steps 1000000            # GPU is used automatically if CUDA is available
    python scripts/train_debris.py --device cpu|cuda          # force a device
    python scripts/test_debris.py --episodes 200

## Faithful-baseline notes
- No exploration noise, no error-signal gain, no multi-step loss, no run-time shield, action
  loss weight 0.1 and eps_action 0.2 as upstream.
- Upstream samples from all three class buffers, so the training script waits until each has data.
- `scripts/train_drone.py` still works (static obstacles via the default `obstacle_next=None`).
- GPU: `train_debris.py --device` moves the networks to CUDA before the Trainer builds its optimisers
  (upstream's `gpu_id` only moves the batches). Data collection uses a CPU copy of the controller that is
  refreshed after every update; saved weights are always CPU tensors. Needs a CUDA build of PyTorch.
- Finding from a first full run: with upstream's eps_action = 0.2 the controller is pinned at +/-0.2 u_max
  (~0.01 m/s^2, too little to dodge) and the barrier settles at its floor h = eps_deriv = 0.03 inside the
  keep-out zone (loss 0.132, collision rate 1.0). `train_debris.py` therefore exposes `--eps-action` (default 1.0)
  and `--explore-noise` (default 0.1); use `scripts/diagnose_debris.py` to inspect a run.
