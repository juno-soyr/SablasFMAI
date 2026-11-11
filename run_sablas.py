"""
One-command pipeline for SABLAS debris avoidance (PyTorch).

  1. environment check     torch / numpy / matplotlib, CPU or GPU
  2. unit checks           CW model matches the numpy simulator, the SABLAS
                           stop-gradient trick has the right value and Jacobian,
                           per-obstacle h, controller invariant to debris order,
                           dataset labels and one trainer step
  3. smoke test            2 training iterations + a 20-episode verification
  4. training              full run, checkpoint after every network update
  5. verification          held-out episodes vs nominal controller, pass/fail

Usage
  python run_sablas.py                         # everything, 1e6 environment steps (as the repo)
  python run_sablas.py --quick                 # steps 1-3 only (~1 min)
  python run_sablas.py --train-steps 300000 --debris 1   # easier problem first
  python run_sablas.py --verify-only           # step 5 on an existing checkpoint

Logs go to runs/<name>/ (train.log, verify.log, results.png, results.json).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}", flush=True)


def run(cmd, log_path):
    """Run a command, stream its output to the console and to a log file."""
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line); log.flush()
        return p.wait()


# ---------------------------------------------------------------- step 1
def check_env():
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
def unit_checks():
    banner("2/5  unit checks")
    sys.path.insert(0, HERE)
    import numpy as np, torch
    import cw_env as E, sablas_torch as S
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
    u = S.T(E.pi_nom(O[1]) / E.U_MAX).requires_grad_(True)
    x_nom = st + S.nominal_dynamics(st, u) * E.DT
    x_g = x_nom + (st_n - x_nom).detach()
    val_err = (x_g - st_n).abs().max().item()
    report("x_next_with_grad equals real next state", val_err < 1e-3, f"(max abs err {val_err:.1e})")
    g = torch.autograd.grad(x_g[0, 3], u)[0][0]
    jac_ok = abs(g[0].item() - E.DT * E.U_MAX) < 1e-6 and g[1:].abs().max().item() < 1e-9
    report("d v_next / d u = dt * u_max", jac_ok, f"(got {g[0].item():.4f}, expected {E.DT * E.U_MAX:.4f})")

    # (c) networks: shapes, per-obstacle h, order invariance of the controller
    ag = S.Agent().to(S.DEV)
    obs = S.T(R[1] + O[1][:, None])
    perm = torch.randperm(obs.shape[1])
    un0, er0 = S.T(E.pi_nom(O[1]) / E.U_MAX), torch.zeros_like(st)
    with torch.no_grad():
        h1, h2 = ag.cbf(st, obs), ag.cbf(st, obs[:, perm])
        u1 = ag.controller(st, obs, un0, er0); u2 = ag.controller(st, obs[:, perm], un0, er0)
    report("h has one value per obstacle", h1.shape == (len(st), E.M_DEB))
    report("h follows obstacle order (no mixing)", (h1[:, perm] - h2).abs().max().item() < 1e-5)
    report("controller invariant to obstacle order", (u1 - u2).abs().max().item() < 1e-5)
    report("controller is residual on u_nominal", bool(((u1 - un0).abs() <= 1 + 1e-6).all()))

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
    un_w = S.T(np.stack([E.pi_nom(O[i]) / E.U_MAX for i in range(K)], 1))
    er_w = torch.zeros(len(st), K, 6, device=S.DEV)
    xbar, _ = tr.unroll(st_w, ob_w, un_w, er_w)
    val_err = (xbar - st_w[:, 1:]).abs().max().item()
    report("multi-step: every chained state equals the real state", val_err < 1e-3, f"(max abs err {val_err:.1e})")
    # gradient of the position two steps ahead w.r.t. the first command (probe in place of the network)
    probe = [torch.zeros(len(st), 3, device=S.DEV, requires_grad=True) for _ in range(K)]
    it = iter(probe)
    net, tr.controller = tr.controller, lambda x, o, un_, e: un_ + next(it)
    xbar, _ = tr.unroll(st_w, ob_w, un_w, er_w)
    tr.controller = net
    g = torch.autograd.grad(xbar[0, 1, 0], probe[0])[0][0, 0].item()
    report("multi-step: d p[t+2] / d u[t] = dt^2 * u_max", abs(g - E.DT**2 * E.U_MAX) < 1e-3,
           f"(got {g:.3f}, expected {E.DT**2 * E.U_MAX:.3f})")
    ds_w = S.Dataset(); ds_w.add_windows(*(x.cpu().numpy() for x in (st_w, ob_w, un_w, er_w)))
    loss, _ = S.Trainer(ag.controller, ag.cbf, ds_w).train_cbf_and_controller(rng, batch_size=16, opt_iter=2)
    report("multi-step: trainer step runs, loss finite", np.isfinite(loss), f"(loss {loss:.3f})")
    if not ok:
        sys.exit("\nunit checks failed: fix before training")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-steps", type=int, default=1_000_000, help="environment steps (repo: 1e6)")
    ap.add_argument("--n-env", type=int, default=32, help="episodes run in parallel")
    ap.add_argument("--ca-features", action="store_true", help="add closest-approach input features")
    ap.add_argument("--debris", type=int, default=3)
    ap.add_argument("--umax", type=float, default=0.05, help="per-axis thrust accel [m/s^2]")
    ap.add_argument("--horizon", type=int, default=1,
                    help="transitions per training window; 1 = one-step loss of the paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-episodes", type=int, default=500)
    ap.add_argument("--max-collision", type=float, default=0.05)
    ap.add_argument("--name", default=None, help="run folder name (default: from settings)")
    ap.add_argument("--quick", action="store_true", help="environment, unit checks, smoke test only")
    ap.add_argument("--verify-only", action="store_true", help="skip training, verify runs/<name>/agent.pt")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()

    name = a.name or f"deb{a.debris}_u{a.umax:g}_st{a.train_steps}_s{a.seed}{'_ca' if a.ca_features else ''}{f'_h{a.horizon}' if a.horizon > 1 else ''}"
    out = os.path.join(HERE, "runs", name); os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "agent.pt")
    py = sys.executable
    t0 = time.time()

    check_env()
    if not a.skip_checks and not a.verify_only:
        unit_checks()

        banner("3/5  smoke test")
        smoke = os.path.join(out, "smoke.pt")
        rc = run([py, "sablas_torch.py", "--train-steps", "3000", "--n-env", "8", "--update-interval", "1000",
                  "--opt-iter", "5", "--batch", "128", "--debris", str(a.debris), "--umax", str(a.umax),
                  "--horizon", str(a.horizon), "--ckpt", smoke] + (["--ca-features"] if a.ca_features else []),
                 os.path.join(out, "smoke_train.log"))
        if rc != 0:
            sys.exit("smoke training failed, see smoke_train.log")
        rc = run([py, "eval_torch.py", "--ckpt", smoke, "--episodes", "20",
                  "--out", os.path.join(out, "smoke")], os.path.join(out, "smoke_verify.log"))
        if rc not in (0, 1):            # 1 = criteria failed, expected after 2 iterations
            sys.exit("smoke verification crashed, see smoke_verify.log")
        print("  smoke test ran end to end")
    if a.quick:
        print(f"\nquick check done in {time.time() - t0:.0f} s"); return

    if not a.verify_only:
        banner(f"4/5  training ({a.train_steps} environment steps) -> {ckpt}")
        rc = run([py, "sablas_torch.py", "--train-steps", str(a.train_steps), "--n-env", str(a.n_env),
                  "--debris", str(a.debris), "--umax", str(a.umax), "--seed", str(a.seed),
                  "--horizon", str(a.horizon), "--ckpt", ckpt]
                 + (["--ca-features"] if a.ca_features else []),
                 os.path.join(out, "train.log"))
        if rc != 0:
            sys.exit("training failed, see train.log")
    elif not os.path.exists(ckpt):
        sys.exit(f"no checkpoint at {ckpt}")

    banner("5/5  verification")
    rc = run([py, "eval_torch.py", "--ckpt", ckpt, "--episodes", str(a.test_episodes),
              "--max-collision", str(a.max_collision), "--out", os.path.join(out, "results")],
             os.path.join(out, "verify.log"))
    if rc not in (0, 1):
        sys.exit("verification crashed, see verify.log")
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    print(f"\n{'ALL CRITERIA PASSED' if res['passed'] else 'SOME CRITERIA FAILED'}  "
          f"({time.time() - t0:.0f} s total)  ->  {out}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
