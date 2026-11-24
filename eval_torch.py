"""Verify a trained SABLAS agent (repo-style port) against the nominal controller.

    python eval_torch.py --ckpt sablas_torch_agent.pt --episodes 500 --out results [--shield]

Barrier-function checks use the repo's distance labels per obstacle
(safe >= SAFE_DIST, dangerous <= DANG_DIST, mid in between) on held-out
real transitions. With --shield the agent is also run with the run-time CBF filter.
Writes <out>.png and <out>.json.

Exit codes: 0 all criteria pass, 2 some criteria fail, 1 the script crashed."""
import argparse, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cw_env as E
import sablas_torch as S

EXIT_PASS, EXIT_CRASH, EXIT_FAIL = 0, 1, 2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="sablas_torch_agent.pt")
ap.add_argument("--episodes", type=int, default=500)
ap.add_argument("--out", default="sablas_torch_results")
ap.add_argument("--seed", type=int, default=99, help="test seed (training uses 0)")
ap.add_argument("--max-collision", type=float, default=0.05)
ap.add_argument("--shield", action="store_true", help="also evaluate with the run-time CBF filter")
ap.add_argument("--shield-eps", type=float, default=0.0, help="margin in the filtered CBF condition")
ap.add_argument("--gain-margin", type=float, default=0.7, help="trust in the nominal thrust gain, < 1")
a = ap.parse_args()

ag, hist = S.load(a.ckpt)
N = a.episodes
On, Rn, Un = E.rollout(N, lambda o, r: E.pi_nom(o), np.random.default_rng(a.seed))
Ol, Rl, Ul = E.rollout(N, ag.make_policy(), np.random.default_rng(a.seed))
if a.shield:
    pol_s = ag.make_policy(shield=True, shield_eps=a.shield_eps, gain_margin=a.gain_margin)
    Os, Rs, Us = E.rollout(N, pol_s, np.random.default_rng(a.seed))


def metrics(O, R, U):
    """Closed-loop metrics for one set of rollouts (per-episode minimum separation is interpolated)."""
    ms = E.min_separation(R)
    slot = np.linalg.norm(O[..., :3], axis=-1)
    return dict(collision_rate=float(np.mean(ms < E.D_SAFE)),
                median_min_sep_m=float(np.median(ms)),
                p5_min_sep_m=float(np.percentile(ms, 5)),
                final_slot_dev_m=float(slot[-1].mean()),
                dv_mps=float((np.linalg.norm(U, axis=-1).sum(0) * E.DT).mean())), ms


m_nom, ms_n = metrics(On, Rn, Un)
m_lrn, ms_l = metrics(Ol, Rl, Ul)
if a.shield:
    m_shd, ms_s = metrics(Os, Rs, Us)


def transitions(O, R):
    """Flatten rollouts into (own, obstacle, own_next, obstacle_next) tensors."""
    own, own_n = S.T(O[:-1].reshape(-1, 6)), S.T(O[1:].reshape(-1, 6))
    obs = S.T((R[:-1] + O[:-1, :, None]).reshape(-1, E.M_DEB, 6))
    obs_n = S.T((R[1:] + O[1:, :, None]).reshape(-1, E.M_DEB, 6))
    return own, obs, own_n, obs_n


def cbf_chunked(own, obs, bs=20000):
    """ag.cbf over a large batch in chunks (avoids huge activations)."""
    return torch.cat([ag.cbf(own[i:i + bs], obs[i:i + bs]) for i in range(0, len(own), bs)])


def rate(mask, cond):
    """Fraction of cond over mask; NaN when the class has no samples."""
    return float(cond[mask].float().mean()) if bool(mask.any()) else float("nan")


with torch.no_grad():
    # sign of h: on closed-loop data plus nominal data (which has many dangerous states)
    own_a, obs_a = (torch.cat([x, y]) for x, y in zip(transitions(On, Rn)[:2], transitions(Ol, Rl)[:2]))
    h_a = cbf_chunked(own_a, obs_a)
    safe_a, dang_a, _ = S.Trainer.get_mask(own_a, obs_a)
    # derivative condition: closed-loop SABLAS data only (it certifies the controller)
    own, obs, own_n, obs_n = transitions(Ol, Rl)
    h, hn = cbf_chunked(own, obs), cbf_chunked(own_n, obs_n)
    deriv = (hn - h) / S.dt_n() + h
    safe, dang, mid = S.Trainer.get_mask(own, obs)

cbf = dict(h_pos_on_safe=rate(safe_a > 0, h_a >= 0),
           h_neg_on_dang=rate(dang_a > 0, h_a < 0),
           deriv_on_safe=rate(safe > 0, deriv > 0),
           deriv_on_mid=rate(mid > 0, deriv > 0),
           deriv_on_dang=rate(dang > 0, deriv > 0),
           n_dang_samples=int(dang_a.sum()))


def at_least(x, thr):
    """True if x >= thr; NaN (no samples of that class) counts as not applicable and is reported."""
    return bool(np.isnan(x) or x >= thr)


criteria = {
    f"collision rate <= {a.max_collision:.0%}": m_lrn["collision_rate"] <= a.max_collision,
    "fewer collisions than nominal": m_lrn["collision_rate"] < m_nom["collision_rate"],
    "h >= 0 on safe states (>= 99%)": at_least(cbf["h_pos_on_safe"], 0.99),
    "h < 0 on dangerous states (>= 99%)": at_least(cbf["h_neg_on_dang"], 0.99),
    "CBF condition on safe states (>= 95%)": at_least(cbf["deriv_on_safe"], 0.95),
    "CBF condition on mid states (>= 95%)": at_least(cbf["deriv_on_mid"], 0.95),
}
if a.shield:
    criteria[f"shielded collision rate <= {a.max_collision:.0%}"] = m_shd["collision_rate"] <= a.max_collision
    criteria["shield does not increase collisions"] = m_shd["collision_rate"] <= m_lrn["collision_rate"]

print(f"\ncheckpoint {a.ckpt} | debris {E.M_DEB} | u_max {E.U_MAX} m/s^2 | {N} test episodes (seed {a.seed})\n")
print(f"{'':10s}{'collisions':>12s}{'median sep':>12s}{'5th pct':>10s}{'slot dev':>10s}{'dv':>9s}")
rows = [("nominal", m_nom), ("SABLAS", m_lrn)] + ([("+shield", m_shd)] if a.shield else [])
for name, m in rows:
    print(f"{name:10s}{m['collision_rate']*100:11.1f}%{m['median_min_sep_m']:10.0f} m{m['p5_min_sep_m']:8.0f} m"
          f"{m['final_slot_dev_m']:8.0f} m{m['dv_mps']:6.2f} m/s")
if a.shield:
    print(f"shield: {pol_s.stats['active']} of {pol_s.stats['steps']} steps modified, "
          f"{pol_s.stats['infeasible']} infeasible (best effort)")
print("\nbarrier function (per obstacle):")
for k, v in cbf.items():
    print(f"  {k:20s} {v if k.startswith('n_') else f'{v*100:6.1f}%'}")
na = [k for k, v in cbf.items() if not k.startswith('n_') and np.isnan(v)]
if na:
    print(f"  note: no samples for {', '.join(na)}; those criteria count as not applicable")
print("\nverification:")
for k, ok in criteria.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

passed = bool(all(criteria.values()))
with open(a.out + ".json", "w") as f:
    json.dump(dict(checkpoint=a.ckpt, debris=E.M_DEB, u_max=E.U_MAX, episodes=N, seed=a.seed,
                   nominal=m_nom, sablas=m_lrn, shielded=(m_shd if a.shield else None),
                   shield_stats=(pol_s.stats if a.shield else None), cbf=cbf,
                   criteria={k: bool(v) for k, v in criteria.items()},
                   passed=passed), f, indent=2)

# ---------------------------------------------------------------- plots
t_min = np.arange(E.STEPS + 1) * E.DT / 60
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
dn = np.linalg.norm(Rn[..., :3], axis=-1).min(-1)
dl = np.linalg.norm(Rl[..., :3], axis=-1).min(-1)
for i in range(min(12, N)):
    ax[0].plot(t_min, dn[:, i], color="#c0504d", lw=0.8, alpha=0.6)
    ax[0].plot(t_min, dl[:, i], color="#1f6fb2", lw=0.8, alpha=0.8)
ax[0].axhline(E.D_SAFE, color="k", ls="--", lw=1)
ax[0].plot([], [], color="#c0504d", label="nominal"); ax[0].plot([], [], color="#1f6fb2", label="SABLAS")
ax[0].set(yscale="log", ylim=(5, 2e4), xlabel="time [min]", ylabel="distance to nearest debris [m]",
          title="Test episodes"); ax[0].legend()
bins = np.logspace(0.5, 4, 40)
ax[1].hist(ms_n, bins=bins, alpha=0.6, color="#c0504d", label="nominal")
ax[1].hist(ms_l, bins=bins, alpha=0.6, color="#1f6fb2", label="SABLAS")
if a.shield:
    ax[1].hist(ms_s, bins=bins, alpha=0.5, color="#2e8b57", label="SABLAS + shield")
ax[1].axvline(E.D_SAFE, color="k", ls="--")
ax[1].set(xscale="log", xlabel="minimum separation [m]", ylabel="episodes",
          title=f"Minimum separation, {N} episodes"); ax[1].legend()
if hist:
    st = [h_["step"] for h_ in hist]
    ax[2].plot(st, [h_["collision_rate"] for h_ in hist], label="collision rate (last 256 ep)")
    ax[2].plot(st, [h_["acc"][4] for h_ in hist], label="CBF condition acc, mid")
    ax[2].plot(st, [h_["acc"][1] for h_ in hist], label="h<0 acc, dangerous")
    ax[2].set(xlabel="environment steps", title="Training", ylim=(0, 1.05)); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(a.out + ".png", dpi=140)
print(f"\nsaved {a.out}.png and {a.out}.json")
sys.exit(EXIT_PASS if passed else EXIT_FAIL)