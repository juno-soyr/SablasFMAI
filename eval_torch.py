"""Verify a trained SABLAS agent (repo-style port) against the nominal controller.

    python eval_torch.py --ckpt sablas_torch_agent.pt --episodes 500 --out results

Barrier-function checks use the repo's distance labels per obstacle
(safe >= SAFE_DIST, dangerous <= DANG_DIST, mid in between) on held-out
real transitions. Writes <out>.png and <out>.json; exit code 0 if all
criteria pass, 1 otherwise."""
import argparse, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cw_env as E
import sablas_torch as S

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="sablas_torch_agent.pt")
ap.add_argument("--episodes", type=int, default=500)
ap.add_argument("--out", default="sablas_torch_results")
ap.add_argument("--seed", type=int, default=99, help="test seed (training uses 0)")
ap.add_argument("--max-collision", type=float, default=0.05)
a = ap.parse_args()

ag, hist = S.load(a.ckpt)
N = a.episodes
On, Rn, Un = E.rollout(N, lambda o, r: E.pi_nom(o), np.random.default_rng(a.seed))
Ol, Rl, Ul = E.rollout(N, ag.make_policy(), np.random.default_rng(a.seed))


def metrics(O, R, U):
    ms = E.min_separation(R)
    slot = np.linalg.norm(O[..., :3], axis=-1)
    return dict(collision_rate=float(np.mean(ms < E.D_SAFE)),
                median_min_sep_m=float(np.median(ms)),
                p5_min_sep_m=float(np.percentile(ms, 5)),
                final_slot_dev_m=float(slot[-1].mean()),
                dv_mps=float((np.linalg.norm(U, axis=-1).sum(0) * E.DT).mean())), ms


m_nom, ms_n = metrics(On, Rn, Un)
m_lrn, ms_l = metrics(Ol, Rl, Ul)


def transitions(O, R):
    own, own_n = S.T(O[:-1].reshape(-1, 6)), S.T(O[1:].reshape(-1, 6))
    obs = S.T((R[:-1] + O[:-1, :, None]).reshape(-1, E.M_DEB, 6))
    obs_n = S.T((R[1:] + O[1:, :, None]).reshape(-1, E.M_DEB, 6))
    return own, obs, own_n, obs_n


def rate(mask, cond):
    return float(cond[mask].float().mean()) if bool(mask.any()) else float("nan")


with torch.no_grad():
    # sign of h: on closed-loop data plus nominal data (which has many dangerous states)
    own_a, obs_a = (torch.cat([x, y]) for x, y in zip(transitions(On, Rn)[:2], transitions(Ol, Rl)[:2]))
    h_a = ag.cbf(own_a, obs_a)
    safe_a, dang_a, _ = S.Trainer.get_mask(own_a, obs_a)
    # derivative condition: closed-loop SABLAS data only (it certifies the controller)
    own, obs, own_n, obs_n = transitions(Ol, Rl)
    h, hn = ag.cbf(own, obs), ag.cbf(own_n, obs_n)
    deriv = (hn - h) / S.dt_n() + h
    safe, dang, mid = S.Trainer.get_mask(own, obs)

cbf = dict(h_pos_on_safe=rate(safe_a > 0, h_a >= 0),
           h_neg_on_dang=rate(dang_a > 0, h_a < 0),
           deriv_on_safe=rate(safe > 0, deriv > 0),
           deriv_on_mid=rate(mid > 0, deriv > 0),
           deriv_on_dang=rate(dang > 0, deriv > 0),
           n_dang_samples=int(dang_a.sum()))


def at_least(x, thr):
    return bool(np.isnan(x) or x >= thr)      # no samples of that class -> not applicable


criteria = {
    f"collision rate <= {a.max_collision:.0%}": m_lrn["collision_rate"] <= a.max_collision,
    "fewer collisions than nominal": m_lrn["collision_rate"] < m_nom["collision_rate"],
    "h >= 0 on safe states (>= 99%)": at_least(cbf["h_pos_on_safe"], 0.99),
    "h < 0 on dangerous states (>= 99%)": at_least(cbf["h_neg_on_dang"], 0.99),
    "CBF condition on safe states (>= 95%)": at_least(cbf["deriv_on_safe"], 0.95),
    "CBF condition on mid states (>= 95%)": at_least(cbf["deriv_on_mid"], 0.95),
}

print(f"\ncheckpoint {a.ckpt} | debris {E.M_DEB} | u_max {E.U_MAX} m/s^2 | {N} test episodes (seed {a.seed})\n")
print(f"{'':10s}{'collisions':>12s}{'median sep':>12s}{'5th pct':>10s}{'slot dev':>10s}{'dv':>9s}")
for name, m in [("nominal", m_nom), ("SABLAS", m_lrn)]:
    print(f"{name:10s}{m['collision_rate']*100:11.1f}%{m['median_min_sep_m']:10.0f} m{m['p5_min_sep_m']:8.0f} m"
          f"{m['final_slot_dev_m']:8.0f} m{m['dv_mps']:6.2f} m/s")
print("\nbarrier function (per obstacle):")
for k, v in cbf.items():
    print(f"  {k:20s} {v if k.startswith('n_') else f'{v*100:6.1f}%'}")
print("\nverification:")
for k, ok in criteria.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

with open(a.out + ".json", "w") as f:
    json.dump(dict(checkpoint=a.ckpt, debris=E.M_DEB, u_max=E.U_MAX, episodes=N, seed=a.seed,
                   nominal=m_nom, sablas=m_lrn, cbf=cbf,
                   criteria={k: bool(v) for k, v in criteria.items()},
                   passed=bool(all(criteria.values()))), f, indent=2)

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
sys.exit(0 if all(criteria.values()) else 1)
