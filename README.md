# SABLAS for orbital debris avoidance

A PyTorch implementation of SABLAS for a satellite that has to hold its orbital slot while dodging non-cooperative debris.
The learner knows only a simplified model of the orbit (Clohessy-Wiltshire); the simulator it is trained and tested on is a black box with different dynamics.

This README describes how the code is organised and which part of the paper each piece implements.

**Paper:** Z. Qin, D. Sun, C. Fan, *SABLAS: Learning Safe Control for Black-box Dynamical Systems*, IEEE RA-L 2022, [arXiv:2201.01918v2](https://arxiv.org/abs/2201.01918).
Section, equation and algorithm numbers below refer to that version.

## Layout

| File | Role | Paper |
|---|---|---|
| [cw_env.py](cw_env.py) | The black-box simulator, the nominal model, the nominal controller, and episode generation (NumPy only) | Sec. III-A, IV-A, V-A |
| [sablas_torch.py](sablas_torch.py) | The barrier network, the controller network, the dataset, the loss and the training loop | Sec. III-B, III-C, IV-B, Alg. 1, V-D |
| [eval_torch.py](eval_torch.py) | Tests a trained agent against the nominal controller and checks the barrier conditions on held-out data | Sec. V-B, V-F, Eq. (2) |
| [run_sablas.py](run_sablas.py) | Runs everything in order: environment check, unit checks, smoke test, training, verification | Thm. 1 (unit check) |

The port follows the authors' reference code, [MIT-REALM/sablas](https://github.com/MIT-REALM/sablas), which is not included here.

The dependencies go one way: `cw_env` imports nothing from the project, `sablas_torch` imports `cw_env`, `eval_torch` imports both, and `run_sablas` calls the other two as subprocesses.

## How the pieces fit together

```
                 cw_env.py                                  sablas_torch.py
        ┌──────────────────────────┐              ┌──────────────────────────────┐
        │ Episodes.step()          │  s(t), s(t+Δt)│ Dataset                      │
        │   black box f            │ ─────────────▶│   safe / dangerous / mid     │
        │   (two-body + J2,        │              └──────────────┬───────────────┘
        │    imperfect thruster)   │                             │ batches
        │                          │              ┌──────────────▼───────────────┐
        │ pi_nom()   π_nom         │ ── u_nom ───▶│ Trainer.compute_loss()       │
        │ cw_step()  f_nom (CW)    │ ── f_nom ───▶│   L0 + Ld + Lp3 + λ·Lg       │
        └──────────▲───────────────┘              └──────────────┬───────────────┘
                   │ u = π(s)                                    │ gradients
                   │                              ┌──────────────▼───────────────┐
                   └──────────────────────────────│ NNController π    CBF h      │
                                                  └──────────────────────────────┘
```

The controller drives the black box, the transitions it produces fill the dataset, and the loss trains the controller and the barrier function together.
The gradient reaches the controller through the nominal model, never through the black box.

## `cw_env.py`: the environment

**Black-box system** — [`Episodes.step`](cw_env.py#L166), [`propagate`](cw_env.py#L70), [`_acc`](cw_env.py#L58).
Every body (reference slot, spacecraft, debris) is integrated in an inertial frame with two-body gravity plus J2, using RK4.
The thruster delivers 80 % of the commanded acceleration, rotated by 6° and with 3 % noise.
None of this is visible to the learner, which only sees states before and after a step.
*Paper: Definition 1 (Sec. III-A), a system `ṡ = f(s, u)` with `f` unknown. The thruster noise is the kind of disturbance discussed in Remark 2 (Sec. IV-B).*

**Nominal model** — [`cw_A`](cw_env.py#L38), [`cw_step`](cw_env.py#L50).
The Clohessy-Wiltshire linear model of relative motion, stepped with Euler.
It differs from the black box by J2, by the linearisation, and by the thruster errors.
*Paper: the differentiable nominal model `f_nom` of Sec. IV-A (text around Eq. 6). The paper fits `f_nom` from data (Sec. V-D); here it is the analytic CW equation instead, which the paper allows since it only says there are "many methods to obtain `f_nom`".*

**Nominal controller** — [`pi_nom`](cw_env.py#L180).
Cancels the CW drift and applies a PD law to stay at the slot. It ignores debris.
*Paper: `π_nom` in Sec. III-C (Eq. 4) and Sec. V-A, a controller "purely for goal-reaching" that "does not consider safety".*

**Episodes and observations** — [`Episodes.__init__`](cw_env.py#L124), [`Episodes.observe`](cw_env.py#L160), [`lvlh`](cw_env.py#L84).
Each episode samples a random orbit, a spacecraft near the slot, and `M_DEB` debris pieces placed so that they pass within 250 m of the slot at 3–20 m/s.
States are reported in the rotating frame of the slot: the spacecraft's own state, and each debris piece relative to the spacecraft.
*Paper: the initial set `S0` and dangerous set `Sd` of Definition 2 (Sec. III-A); the dangerous set is being closer than `D_SAFE` = 300 m to any debris. The moving obstacles correspond to the moving-NPC mode of Sec. V-A.*

**Helpers** — [`rollout`](cw_env.py#L187) runs a batch of episodes under any policy (the `RUN` function of Alg. 1), and [`min_separation`](cw_env.py#L200) gives the closest approach per episode.

## `sablas_torch.py`: the learner

**Barrier function** — [`CBF`](sablas_torch.py#L88).
A shared MLP applied to each `state − obstacle` difference, giving one value `h` per debris piece. The state is safe when every `h` is non-negative.
*Paper: the neural CBF `h(s; Ω)` of Sec. III-B and III-C. Sec. V-D gives the architecture only by reference to Sec. 4.2 of its ref. [17]; the layer sizes come from the authors' code (`modules/network.py`).*

**Controller** — [`NNController`](sablas_torch.py#L111).
Encodes each debris piece with a shared MLP, max-pools over the pieces so the order does not matter, then combines the result with the nominal command and outputs a bounded correction that is added to it.
*Paper: the neural controller `π(s; Θ)` of Sec. III-C, with observed obstacles as inputs as in Sec. V-D. The residual form and the `state_error` input (how far the last real step landed from the nominal prediction) are from the authors' code, not from the paper's text.*

**Nominal model in PyTorch** — [`nominal_dynamics`](sablas_torch.py#L142).
The same CW equation as `cw_env.cw_step`, written in PyTorch so that it can be differentiated.
*Paper: `f_nom` in Eq. (7), Sec. IV-B.*

**Dataset** — [`Dataset`](sablas_torch.py#L180).
Stores transitions `(s(t), obstacles, u_nom, s(t+Δt), next obstacles, state_error)` in three buffers, split by distance to the nearest debris: safe (≥ 3 km), dangerous (≤ 300 m) and in between. Batches take a third from each.
*Paper: the dataset `D` of Sec. III-C, which "is not fixed during training and will be periodically updated with new samples by running the current controller", and lines 3–7 of Alg. 1. The paper rebuilds `D` every iteration; the replay buffers and the middle class are from the authors' code (`modules/dataset.py`).*

**Loss** — [`Trainer.compute_loss`](sablas_torch.py#L237).

| Term in the code | What it enforces | Paper |
|---|---|---|
| `loss_h_safe` | `h ≥ 0` on safe states | `L0`, Eq. (3); first condition of Eq. (2) |
| `loss_h_dang` | `h < 0` on dangerous states | `Ld`, Eq. (3); second condition of Eq. (2) |
| `loss_d_safe`, `loss_d_dang`, `loss_d_mid` | `ḣ + α(h) ≥ 0` | `Lp3`, Eq. (7); third condition of Eq. (2) |
| `loss_action` | stay close to the nominal command | `Lg`, Eq. (4) |
| the sum | | total loss, Eq. (8), with λ = 0.08 |

The core of the method is [these four lines](sablas_torch.py#L241-L244):

```python
state_next_nominal   = state + nominal_dynamics(state, u) * E.DT
state_next_with_grad = state_next_nominal + (state_next - state_next_nominal).detach()
h_next = self.cbf(state_next_with_grad, obstacle_next)
deriv = (h_next - h) / dt_n() + h
```

`state_next_with_grad` has the value of the real next state from the black box, and the gradient of the nominal model.
*Paper: Sec. IV-B, the construction of `s̄(t+Δt)` with `g(s) = s.detach()`, Eq. (7), Theorem 1 and Fig. 2. This replaces `Lp1` (Eq. 5, real data but no gradient to the controller) and `Lp2` (Eq. 6, gradient but the wrong value), both from Sec. IV-A.*

Differences from the paper's equations, all inherited from the authors' code (`modules/trainer.py`):
- The losses use margins (`EPS`, `EPS_DERIV`) instead of a bare `max(0, ·)`.
- The derivative condition is enforced on all sampled states, not only on `Sp = {h ≥ 0}`.
- `α(h) = h`, a linear class-K function.
- `Lg` is a dead-zone L1 penalty instead of the squared norm in Eq. (4).

One difference is specific to this project: `h_next` is evaluated on the debris' next states, because here the obstacles move during training (the paper trains with static NPCs, Sec. V-D).

**Multi-step loss (optional)** — [`Trainer.unroll`](sablas_torch.py), enabled with `--horizon K`.
With the default `K = 1` the loss is the one-step loss above. With `K > 1` the dataset stores windows of `K` consecutive real transitions and the stop-gradient construction is chained, so each state keeps the value of the real state while its gradient reaches every earlier command in the window.
It was added because one thrust step changes the velocity by only 0.5 m/s, which gave the controller almost no gradient in a one-step look-ahead. Its effect on the collision rate has not been measured yet.
*Paper: an extension of Eq. (7), Sec. IV-B, which is one-step only. This part is not in the paper or in the authors' code.*

**Optimisation** — [`Trainer.train_cbf_and_controller`](sablas_torch.py#L265).
One Adam step on the summed loss updates both networks; 100 such steps per update with batch size 1024.
*Paper: the `UPDATE` function of Alg. 1 (line 8) and the settings of Sec. V-D (Adam, batch 1024, 100 iterations). The paper uses learning rate 1e-4; the code keeps 1e-4 for the CBF and uses 5e-4 for the controller, as the authors' code does.*

**Training loop** — [`train`](sablas_torch.py#L293).
Runs 32 episodes in parallel with the current controller, adds every transition to the dataset, and calls the trainer every 5000 environment steps until 10⁶ steps are collected.
*Paper: Alg. 1 as a whole, and the 10⁶ state samples of Sec. V-D.*

**Glue** — [`Agent`](sablas_torch.py#L148) bundles the two networks and turns the controller into a policy for `cw_env.rollout`; [`save`](sablas_torch.py#L280) and [`load`](sablas_torch.py#L285) handle checkpoints; [`preprocess_diff`](sablas_torch.py#L73) scales inputs and can add closest-approach features (`--ca-features`). None of these has a counterpart in the paper.

## `eval_torch.py`: verification

Runs the same held-out episodes (a different seed from training) under the nominal controller and under the learned one, then reports two kinds of result.

**Closed-loop performance** — [`metrics`](eval_torch.py#L32): collision rate, minimum separation, final distance from the slot and Δv used.
*Paper: Sec. V-B. Comparing the collision rate against the nominal controller plays the role of the relative safety rate, and the slot deviation plays the role of the tracking error. They are analogues, not the same formulas.*

**Barrier conditions** — [the `cbf` checks](eval_torch.py#L57-L74): the fraction of held-out transitions on which `h ≥ 0` (safe), `h < 0` (dangerous) and `ḣ + α(h) > 0` hold.
*Paper: the three CBF conditions of Eq. (2), Sec. III-B. They are checked empirically with thresholds below 100 % because of the limitation stated in Sec. V-F: the conditions are not guaranteed on the whole state space and "may still be occasionally violated during testing".*

The script writes a plot and a JSON file, and exits with 0 only if every criterion in [`criteria`](eval_torch.py#L80) passes.

## `run_sablas.py`: the pipeline

Five steps in order, each logged under `runs/<name>/`:

1. [`check_env`](run_sablas.py#L41): packages and device.
2. [`unit_checks`](run_sablas.py#L60): four groups of tests.
   - The PyTorch and NumPy nominal models agree.
   - `state_next_with_grad` equals the real next state while its Jacobian with respect to `u` is the nominal model's. *Paper: this is the statement of Theorem 1 (Sec. IV-B), that `s̄(t+Δt) = s(t+Δt)` and that the gradient with respect to the controller exists.*
   - The networks have the right shapes, one `h` per obstacle, and the controller ignores obstacle order.
   - The dataset labels transitions and one trainer step runs.
3. Smoke test: a tiny training run and a 20-episode verification.
4. Training: calls `sablas_torch.py`.
5. Verification: calls `eval_torch.py` and reports pass or fail.

The pipeline itself has no counterpart in the paper.

## What is not from the paper

| Component | Source |
|---|---|
| Clohessy-Wiltshire model, J2 propagation, LVLH frames, debris encounter geometry | Standard orbital mechanics, chosen for this project |
| Layer sizes, max-pooling, residual output, `state_error` input | Authors' code, `modules/network.py` |
| Replay buffers, the middle class, loss margins, λ = 0.08, controller learning rate | Authors' code, `modules/dataset.py` and `modules/trainer.py` |
| Time unit of 100 s, input scaling, moving obstacles in `h_next`, parallel episodes | Adaptations for this problem, listed in the docstring of [sablas_torch.py](sablas_torch.py) |
