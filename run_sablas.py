"""
One-command pipeline for SABLAS debris avoidance (PyTorch).

  1. environment check     torch / numpy / matplotlib, CPU or GPU
  2. unit checks           CW model matches the numpy simulator, the SABLAS
                           stop-gradient trick has the right value and Jacobian,
                           no gradient through a saturated thruster, per-obstacle h,
                           controller invariant to debris order, dataset labels,
                           one trainer step, the run-time shield filter
  3. smoke test            a few training iterations + a 20-episode verification
  4. training              full run, checkpoint after every network update
  5. verification          held-out episodes vs nominal controller, pass/fail

Usage
  python run_sablas.py                         # everything, 1e6 environment steps (as the repo)
  python run_sablas.py --quick                 # steps 1-3 only (~1 min)
  python run_sablas.py --train-steps 300000 --debris 1   # easier problem first
  python run_sablas.py --horizon 5 --cbf-prior --shield  # multi-step loss, evaluate with the shield
  python run_sablas.py --verify-only           # step 5 on an existing checkpoint

Logs go to runs/<name>/ (train.log, verify.log, results.png, results.json).
Defaults for --debris / --umax come from cw_env.py.
Exit codes: 0 all criteria pass, 2 some criteria fail, 1 anything crashed.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PASS, CRASH, FAIL = 0, 1, 2          # eval_torch.py exit codes


def banner(s):
    """Print a section header."""
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)


def run(cmd, log_path):
    """Run a command, stream its output to the console and to a log file; returns its exit code."""
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line); log.flush()
        return p.wait()


def remove(*paths):
    """Delete files if they exist (stale results must never be read)."""
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------- step 1
def check_env():
    """Verify packages and that all scripts sit in one folder; exits with a message otherwise."""
    banner("1/5  environment")
    missing = []
    for mod in ["numpy", "matplotlib", "torch"]:
        try:
            m = __import__(mod)
            print(f"  {mod:11s} {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"\nmissing packages: {' '.join(missing)}\ninstall with:  pip install {' '.join(missing)}")
    import torch
    print(f"  device      {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    for f in ["cw_env.py", "sablas_torch.py", "eval_torch.py"]:
        if not os.path.exists(os.path.join(HERE, f)):
            sys.exit(f"missing file {f}: keep all scripts in one folder")


# ---------------------------------------------------------------- step 2
def unit_checks(a, debris, umax):
    """Numerical sanity checks of the port, run with the configured network variant.
    Exits if any fails."""
    banner("2/5  unit checks")
    sys.path.insert(0, HERE)
    import numpy as np, torch
    import cw_env as E, sablas_torch as S
    E.M_DEB, E.U_MAX = debris, umax
    rng = np.random.default_rng(0)
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} {detail}")

    # (a) torch CW nominal dynamics == numpy CW step
    s = rng.normal(0, [500, 500, 500, 2, 2, 2], (64, 6)); un = rng.uniform(-1, 1, (64, 3))
    nxt = (S.T(s) + S.nominal_dynamics(S.T(s), S.T(un)) * E.DT).double().cpu().numpy()
    err = np.abs(nxt - E.cw_step(s, un * E.U_MAX)).max()
    report("CW nominal dynamics torch vs numpy", err < 1e-2, f"(max abs err {err:.1e})")

    # (b) SABLAS trick: value = real next state, d x_next / d u from CW
    O, R, _ = E.rollout(4, lambda o, r: E.pi_nom(o), rng, steps=3)
    st, st_n = S.T(O[1]), S.T(O[2])
    u = S.T(0.5 * E.pi_nom(O[1]) / E.U_MAX).requires_grad_(True)       # inside the box, away from |u| = 1
    x_nom = st + S.nominal_dynamics(st, u) * E.DT
    x_g = x_nom + (st_n - x_nom).detach()
    val_err = (x_g - st_n).abs().max().item()
    report("x_next_with_grad equals real next state", val_err < 1e-3, f"(max abs err {val_err:.1e})")
    g = torch.autograd.grad(x_g[0, 3], u)[0][0]
    jac_ok = abs(g[0].item() - E.DT * E.U_MAX) < 1e-6 and g[1:].abs().max().item() < 1e-9
    report("d v_next / d u = dt * u_max", jac_ok, f"(got {g[0].item():.4f}, expected {E.DT * E.U_MAX:.4f})")

    # (c) networks: shapes, per-obstacle h, order invariance of the controller
    ag = S.Agent(a.ca_features, a.cbf_act, a.cbf_prior).to(S.DEV)
    obs = S.T(R[1] + O[1][:, None])
    perm = torch.randperm(obs.shape[1])
    un0, er0 = S.T(E.pi_nom(O[1]) / E.U_MAX), torch.zeros_like(st)
    with torch.no_grad():
        h1, h2 = ag.cbf(st, obs), ag.cbf(st, obs[:, perm])
        u1 = ag.controller(st, obs, un0, er0); u2 = ag.controller(st, obs[:, perm], un0, er0)
    report("h has one value per obstacle", h1.shape == (len(st), E.M_DEB))
    report("h follows obstacle order (no mixing)", (h1[:, perm] - h2).abs().max().item() < 1e-5)
    report("controller invariant to obstacle order", (u1 - u2).abs().max().item() < 1e-5)
    scale = ag.controller.output_scale
    report("controller is a bounded residual on u_nominal", bool(((u1 - un0).abs() <= scale + 1e-6).all()),
           f"(|u - u_nom| <= {scale})")

    # (d) dataset labels + balanced sampling, one trainer step
    ds = S.Dataset()
    own, rel = O[:-1].reshape(-1, 6), R[:-1].reshape(-1, E.M_DEB, 6)
    own2, rel2 = O[1:].reshape(-1, 6), R[1:].reshape(-1, E.M_DEB, 6)
    ds.add_data(own, rel + own[:, None], E.pi_nom(own) / E.U_MAX, own2, rel2 + own2[:, None], np.zeros_like(own))
    report("dataset stores every transition", len(ds) == len(own), f"({len(ds)} of {len(own)})")
    tr = S.Trainer(ag.controller, ag.cbf, ds)
    loss, acc = tr.train_cbf_and_controller(rng, batch_size=32, opt_iter=2)
    report("trainer step runs, loss finite", np.isfinite(loss), f"(loss {loss:.3f})")

    # (e) multi-step loss: a window of 3 real transitions
    K = 3
    seq = lambda X: S.T(np.stack([X[i] for i in range(K + 1)], 1))
    st_w, ob_w = seq(O), seq(R + O[:, :, None])
    un_w = S.T(0.5 * np.stack([E.pi_nom(O[i]) / E.U_MAX for i in range(K)], 1))   # away from saturation
    er_w = torch.zeros(len(st), K, 6, device=S.DEV)
    xbar, _ = tr.unroll(st_w, ob_w, un_w, er_w)
    val_err = (xbar - st_w[:, 1:]).abs().max().item()
    report("multi-step: every chained state equals the real state", val_err < 1e-3, f"(max abs err {val_err:.1e})")

    def probe_grad(offset):
        """Gradient of position / velocity after the window w.r.t. a probe added to the first command."""
        probe = [torch.full((len(st), 3), offset, device=S.DEV, requires_grad=True) for _ in range(K)]
        it = iter(probe)
        net, tr.controller = tr.controller, lambda x, o, un_, e: un_ + next(it)
        xb, _ = tr.unroll(st_w, ob_w, un_w, er_w)
        tr.controller = net
        return (torch.autograd.grad(xb[0, 1, 0], probe[0], allow_unused=True, create_graph=True, retain_graph=True)[0],
                torch.autograd.grad(xb[0, 0, 3], probe[0], allow_unused=True)[0])

    gp, _ = probe_grad(0.0)
    report("multi-step: d p[t+2] / d u[t] = dt^2 * u_max", abs(gp[0, 0].item() - E.DT**2 * E.U_MAX) < 1e-3,
           f"(got {gp[0, 0].item():.3f}, expected {E.DT**2 * E.U_MAX:.3f})")
    gp_sat, gv_sat = probe_grad(5.0)                                   # command far beyond the thruster limit
    report("no gradient through a saturated thruster",
           (gp_sat is None or gp_sat.abs().max().item() < 1e-9) and (gv_sat is None or gv_sat.abs().max().item() < 1e-9))
    ds_w = S.Dataset(); ds_w.add_windows(*(x.cpu().numpy() for x in (st_w, ob_w, un_w, er_w)))
    loss, _ = S.Trainer(ag.controller, ag.cbf, ds_w).train_cbf_and_controller(rng, batch_size=16, opt_iter=2)
    report("multi-step: trainer step runs, loss finite", np.isfinite(loss), f"(loss {loss:.3f})")

    # (f) run-time shield: QP filter on random problems, then the shielded policy end to end
    n = 200
    u_star = rng.uniform(-0.9, 0.9, (n, 3)); A = rng.normal(size=(n, 3, 3))
    b = np.einsum('nkj,nj->nk', A, u_star) - rng.uniform(0, 0.3, (n, 3))      # u_star strictly feasible
    u0 = rng.uniform(-1, 1, (n, 3))
    uf, feas = S.qp_filter(u0, A, b)
    slack = (np.einsum('nkj,nj->nk', A, uf) - b).min(1)
    report("shield QP: result feasible and inside the box", bool(feas.all()) and bool((np.abs(uf) <= 1 + 1e-9).all()),
           f"(min slack {slack.min():.1e})")
    feas0 = (np.einsum('nkj,nj->nk', A, u0) - b).min(1) >= 0
    report("shield QP: leaves feasible commands unchanged", bool(np.abs(uf - u0)[feas0 & (np.abs(u0) <= 1).all(1)].max(initial=0) < 1e-9))
    pol = ag.make_policy(shield=True)
    O2, R2, U2 = E.rollout(4, pol, rng, steps=3)
    report("shielded policy runs, command within u_max", bool((np.abs(U2) <= E.U_MAX + 1e-12).all()) and np.isfinite(U2).all(),
           f"(stats {pol.stats})")
    if not ok:
        sys.exit("\nunit checks failed: fix before training")


# ---------------------------------------------------------------- main
def main():
    """Parse arguments and run the pipeline."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-steps", type=int, default=1_000_000, help="environment steps (repo: 1e6)")
    ap.add_argument("--n-env", type=int, default=32, help="episodes run in parallel")
    ap.add_argument("--ca-features", action="store_true", help="add closest-approach input features")
    ap.add_argument("--debris", type=int, default=None, help="debris pieces (default: cw_env.M_DEB)")
    ap.add_argument("--umax", type=float, default=None, help="per-axis thrust accel [m/s^2] (default: cw_env.U_MAX)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="transitions per training window; 1 = one-step loss of the paper (5+ gives the controller a useful gradient)")
    ap.add_argument("--cbf-act", default="silu", choices=["relu", "silu", "softplus", "tanh"],
                    help="CBF hidden activation (repo: relu)")
    ap.add_argument("--cbf-prior", action="store_true", help="distance prior + learned residual in the CBF")
    ap.add_argument("--eps-action", type=float, default=None, help="free residual size in units of u_max")
    ap.add_argument("--action-weight", type=float, default=None)
    ap.add_argument("--explore-noise", type=float, default=None, help="command noise while collecting data")
    ap.add_argument("--shield", action="store_true", help="also verify with the run-time CBF filter")
    ap.add_argument("--shield-eps", type=float, default=0.0)
    ap.add_argument("--gain-margin", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-episodes", type=int, default=500)
    ap.add_argument("--max-collision", type=float, default=0.05)
    ap.add_argument("--name", default=None, help="run folder name (default: from settings)")
    ap.add_argument("--quick", action="store_true", help="environment, unit checks, smoke test only")
    ap.add_argument("--verify-only", action="store_true", help="skip training, verify runs/<name>/agent.pt")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()

    check_env()
    sys.path.insert(0, HERE)
    import cw_env as E
    debris = a.debris if a.debris is not None else E.M_DEB
    umax = a.umax if a.umax is not None else E.U_MAX

    extras = "".join([f"_ne{a.n_env}" if a.n_env != 32 else "",
                      f"_ea{a.eps_action:g}" if a.eps_action is not None else "",
                      f"_aw{a.action_weight:g}" if a.action_weight is not None else "",
                      f"_ex{a.explore_noise:g}" if a.explore_noise is not None else ""])
    name = a.name or (f"deb{debris}_u{umax:g}_st{a.train_steps}_s{a.seed}{'_ca' if a.ca_features else ''}"
                      f"{f'_h{a.horizon}' if a.horizon > 1 else ''}{f'_{a.cbf_act}' if a.cbf_act != 'silu' else ''}"
                      f"{'_prior' if a.cbf_prior else ''}{extras}")
    out = os.path.join(HERE, "runs", name); os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "agent.pt")
    py = sys.executable
    cbf_args = ["--cbf-act", a.cbf_act] + (["--cbf-prior"] if a.cbf_prior else [])
    opt_args = [x for k, v in (("--eps-action", a.eps_action), ("--action-weight", a.action_weight),
                               ("--explore-noise", a.explore_noise)) if v is not None for x in (k, str(v))]
    shield_args = ["--shield", "--shield-eps", str(a.shield_eps), "--gain-margin", str(a.gain_margin)] if a.shield else []
    t0 = time.time()

    if not a.skip_checks and not a.verify_only:
        unit_checks(a, debris, umax)

        banner("3/5  smoke test")
        smoke = os.path.join(out, "smoke.pt")
        remove(os.path.join(out, "smoke.json"), os.path.join(out, "smoke.png"))
        rc = run([py, "sablas_torch.py", "--train-steps", "3000", "--n-env", "8", "--update-interval", "1000",
                  "--opt-iter", "5", "--batch", "128", "--debris", str(debris), "--umax", str(umax),
                  "--horizon", str(a.horizon), "--ckpt", smoke] + (["--ca-features"] if a.ca_features else [])
                 + cbf_args + opt_args, os.path.join(out, "smoke_train.log"))
        if rc != 0:
            sys.exit("smoke training failed, see smoke_train.log")
        rc = run([py, "eval_torch.py", "--ckpt", smoke, "--episodes", "20", "--out", os.path.join(out, "smoke")]
                 + shield_args, os.path.join(out, "smoke_verify.log"))
        # criteria are expected to fail after a few iterations (rc 2); a crash is rc 1 and no json
        if rc not in (PASS, FAIL) or not os.path.exists(os.path.join(out, "smoke.json")):
            sys.exit("smoke verification crashed, see smoke_verify.log")
        print("  smoke test ran end to end")
    if a.quick:
        print(f"\nquick check done in {time.time() - t0:.0f} s"); return

    if not a.verify_only:
        banner(f"4/5  training ({a.train_steps} environment steps) -> {ckpt}")
        rc = run([py, "sablas_torch.py", "--train-steps", str(a.train_steps), "--n-env", str(a.n_env),
                  "--debris", str(debris), "--umax", str(umax), "--seed", str(a.seed),
                  "--horizon", str(a.horizon), "--ckpt", ckpt]
                 + (["--ca-features"] if a.ca_features else []) + cbf_args + opt_args,
                 os.path.join(out, "train.log"))
        if rc != 0:
            sys.exit("training failed, see train.log")
    elif not os.path.exists(ckpt):
        sys.exit(f"no checkpoint at {ckpt}")

    banner("5/5  verification")
    results = os.path.join(out, "results")
    remove(results + ".json", results + ".png")                         # never read a stale result
    rc = run([py, "eval_torch.py", "--ckpt", ckpt, "--episodes", str(a.test_episodes),
              "--max-collision", str(a.max_collision), "--out", results] + shield_args,
             os.path.join(out, "verify.log"))
    if rc not in (PASS, FAIL) or not os.path.exists(results + ".json"):
        sys.exit("verification crashed, see verify.log")
    with open(results + ".json") as f:
        res = json.load(f)
    print(f"\n{'ALL CRITERIA PASSED' if res['passed'] else 'SOME CRITERIA FAILED'}  "
          f"({time.time() - t0:.0f} s total)  ->  {out}")
    sys.exit(PASS if res["passed"] else FAIL)


if __name__ == "__main__":
    main()