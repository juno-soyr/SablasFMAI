"""
SABLAS for non-cooperative crossing debris, ported from the authors' code
(github.com/MIT-REALM/sablas: modules/network.py, modules/trainer.py,
modules/dataset.py, scripts/train_drone.py).

What is taken from the repo as-is
  * CBF network: per-obstacle 1x1-conv MLP on (state - obstacle),
    n_in -> 64 -> 128 -> 128 -> 128 -> 1, ReLU. One h per obstacle, no pooling.
    Safe set = every h_k >= 0.
  * Controller: per-obstacle encoder n_in -> 64 -> 128 -> 128 (ReLU), max-pool over
    obstacles, concat [pooled, u_nominal, state_error] -> 128 -> 64 -> m, tanh * scale,
    residual: u = u_nominal + output.
  * state_error = (real next state - nominal next state) / dt from the previous step,
    fed to the controller.
  * Labels by distance per obstacle: safe (>= safe_dist), dangerous (<= dang_dist),
    mid (between). Replay buffers per class (10 000 each), batches sampled 1/3 each.
  * Losses (train_cbf_and_controller): relu(eps - h) on safe, relu(h + eps) on dang,
    relu(eps_deriv - deriv) on safe, dang and mid separately,
    deriv = (h(x_next_with_grad) - h(x)) / dt + h(x),
    x_next_with_grad = x_nom + (x_next - x_nom).detach(),
    action loss mean(relu(|u - u_nom| - eps_action)) * 0.08.
    eps 0.1, eps_deriv 0.03, eps_action 0.2.
  * Adam: controller lr 5e-4, CBF lr 1e-4, weight decay 1e-5; one joint step on the sum.
  * Online data collection; every 5000 environment steps, 100 updates with batch 1024.

What is adapted to this problem
  * Time is measured in units of 100 s, so the control step is dt = 0.1 exactly as in
    the repo, and alpha = 1 means 0.01 /s. The eps values then carry over unchanged.
  * Obstacles move, so h_next uses the obstacles' next states (the repo's obstacles
    are static and it reuses the current ones).
  * Inputs are scaled (positions / 1 km, velocities / 10 m/s) by the preprocess hook the
    repo provides; the control is normalised by u_max; the real thruster saturates.
  * The nominal model is the known Clohessy-Wiltshire equation instead of a learned
    control-affine network (the repo learns one by system identification first).
  * Several episodes run in parallel; data is added per environment step as in the repo.
  * Optional multi-step loss (--horizon K, default 1 = the repo's one-step loss). One
    thrust step only changes the velocity by u_max * dt, so a one-step look-ahead gives
    the controller almost no gradient. With K > 1 the dataset stores windows of K
    consecutive real transitions and the SABLAS construction is chained:
        x_nom[j+1]  = xbar[j] + f_nom(xbar[j], pi(xbar[j])) * dt
        xbar[j+1]   = x_nom[j+1] + (x_real[j+1] - x_nom[j+1]).detach()
    Every xbar[j] still has the value of the real state, but its gradient now reaches
    the commands of all earlier steps in the window. The CBF condition is applied at
    each step j with h(x_real[j]) as the reference and h(xbar[j+1]) as the next value.
"""
import argparse, time
import numpy as np
import torch
from torch import nn
import cw_env as E

TIME_UNIT = 100.0            # s per time unit -> dt = 0.1
SAFE_DIST = 3000.0           # m, repo: safe_dist = 1.0 (chosen so a dodge is still possible)
DANG_DIST = E.D_SAFE         # m, repo: dang_dist = 0.6
EPS, EPS_DERIV, EPS_ACTION = 0.1, 0.03, 0.2
ACTION_LOSS_WEIGHT = 0.08
LR_CTRL, LR_CBF, WEIGHT_DECAY = 5e-4, 1e-4, 1e-5
BUFFER_SIZE = 10000
POS_SCALE, VEL_SCALE, T_SCALE = 1000.0, 10.0, 300.0

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
A_CW = torch.tensor(E.A_CW, dtype=DTYPE, device=DEV)


def dt_n():
    """Control step in time units (0.1 with the defaults)."""
    return E.DT / TIME_UNIT


def T(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEV)


# ---------------------------------------------------------------- preprocessing
def scale_state(x):
    """x: (bs, 6, ...) -> positions / 1 km, velocities / 10 m/s."""
    s = torch.tensor([POS_SCALE] * 3 + [VEL_SCALE] * 3, dtype=x.dtype, device=x.device)
    return x / s.view([1, 6] + [1] * (x.dim() - 2))


def preprocess_diff(diff, ca_features):
    """diff = state - obstacle, (bs, 6, k). Optional straight-line closest-approach
    features (miss distance, time to closest approach) as two extra channels."""
    out = scale_state(diff)
    if not ca_features:
        return out
    p, v = diff[:, :3], diff[:, 3:]
    pv = (p * v).sum(1, keepdim=True)
    vv = (v * v).sum(1, keepdim=True) + 1e-6
    t = torch.relu(-pv / vv)
    dca = (p + v * t).norm(dim=1, keepdim=True)
    return torch.cat([out, dca / POS_SCALE, t / T_SCALE], dim=1)


# ---------------------------------------------------------------- networks (repo: modules/network.py)
class CBF(nn.Module):
    def __init__(self, ca_features=False):
        super().__init__()
        self.ca = ca_features
        n_in = 8 if ca_features else 6
        self.conv0 = nn.Conv1d(n_in, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 1, 1)
        self.activation = nn.ReLU()

    def forward(self, state, obstacle):
        """state (bs, 6), obstacle (bs, k, 6) -> h (bs, k)"""
        diff = state.unsqueeze(2) - obstacle.permute(0, 2, 1)
        x = preprocess_diff(diff, self.ca)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return self.conv4(x).squeeze(1)


class NNController(nn.Module):
    def __init__(self, m_control=3, ca_features=False, output_scale=2.0):
        super().__init__()
        self.ca = ca_features
        n_in = 8 if ca_features else 6
        self.conv0 = nn.Conv1d(n_in, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control + 6, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.output_scale = output_scale

    def forward(self, state, obstacle, u_nominal, state_error):
        """state (bs, 6), obstacle (bs, k, 6), u_nominal (bs, 3) normalised by u_max,
        state_error (bs, 6) -> u (bs, 3) normalised by u_max"""
        diff = state.unsqueeze(2) - obstacle.permute(0, 2, 1)
        x = preprocess_diff(diff, self.ca)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x, _ = torch.max(x, dim=2)
        x = torch.cat([x, u_nominal, scale_state(state_error)], dim=1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * self.output_scale
        return x + u_nominal


def nominal_dynamics(state, u_n):
    """Clohessy-Wiltshire, s_dot = A s + B u [per second]; u_n is normalised by u_max."""
    ds = state @ A_CW.T
    return torch.cat([ds[:, :3], ds[:, 3:] + u_n * E.U_MAX], dim=1)


class Agent(nn.Module):
    def __init__(self, ca_features=False):
        super().__init__()
        self.ca = ca_features
        self.cbf = CBF(ca_features)
        self.controller = NNController(ca_features=ca_features)

    def h_all(self, own, rel):
        return self.cbf(own, rel + own[:, None])

    def h(self, own, rel):
        return self.h_all(own, rel).min(dim=1).values

    def make_policy(self):
        """Policy for cw_env.rollout(own, rel) -> u [m/s^2]. Keeps the previous state
        and command to compute state_error, as in the repo; use a fresh one per rollout."""
        mem = {}

        @torch.no_grad()
        def policy(own, rel):
            if "own" in mem and len(mem["own"]) == len(own):
                err = (own - E.cw_step(mem["own"], mem["u"])) / dt_n()
            else:
                err = np.zeros_like(own)
            u_nom = E.pi_nom(own) / E.U_MAX
            u_n = self.controller(T(own), T(rel + own[:, None]), T(u_nom), T(err)).cpu().numpy().astype(np.float64)
            mem["own"], mem["u"] = own.copy(), u_n * E.U_MAX
            return np.clip(u_n * E.U_MAX, -E.U_MAX, E.U_MAX)
        return policy


# ---------------------------------------------------------------- dataset (repo: modules/dataset.py)
class Dataset:
    """Replay buffers of windows of K consecutive real transitions (K = 1: single
    transitions, as in the repo). One window holds
      state (K+1, 6), obstacle (K+1, k, 6), u_nominal (K, 3), state_error (K, 6)."""
    CLASSES = ("safe", "dang", "mid")

    def __init__(self, buffer_size=BUFFER_SIZE):
        self.size = buffer_size
        self.buf = {c: None for c in self.CLASSES}

    def __len__(self):
        return sum(0 if b is None else len(b[0]) for b in self.buf.values())

    def add_data(self, state, obstacle, u_nominal, state_next, obstacle_next, state_error):
        """Single transitions, arrays with leading dim n (a window of length 1)."""
        self.add_windows(np.stack([state, state_next], 1), np.stack([obstacle, obstacle_next], 1),
                         np.asarray(u_nominal)[:, None], np.asarray(state_error)[:, None])

    def add_windows(self, state, obstacle, u_nominal, state_error):
        """Batched windows, leading dim n. Labelled by the distance to the nearest
        obstacle at the first step of the window."""
        dist = np.linalg.norm(obstacle[:, 0, :, :3] - state[:, 0, None, :3], axis=2).min(1)
        cls = np.where(dist < DANG_DIST, 1, np.where(dist > SAFE_DIST, 0, 2))
        rows = [np.asarray(a, dtype=np.float32) for a in (state, obstacle, u_nominal, state_error)]
        for ci, c in enumerate(self.CLASSES):
            m = cls == ci
            if not m.any():
                continue
            new = [r[m] for r in rows]
            if self.buf[c] is None:
                self.buf[c] = new
            else:
                self.buf[c] = [np.concatenate([b, n_])[-self.size:] for b, n_ in zip(self.buf[c], new)]

    def sample_data(self, batch_size, rng):
        """1/3 from each class (split among the classes that have data yet)."""
        avail = [c for c in self.CLASSES if self.buf[c] is not None]
        sizes = [batch_size // len(avail)] * len(avail)
        sizes[-1] += batch_size - sum(sizes)
        parts = []
        for c, n_ in zip(avail, sizes):
            idx = rng.integers(0, len(self.buf[c][0]), n_)
            parts.append([b[idx] for b in self.buf[c]])
        return [np.concatenate([p[i] for p in parts]) for i in range(4)]


# ---------------------------------------------------------------- trainer (repo: modules/trainer.py)
class Trainer:
    def __init__(self, controller, cbf, dataset, action_loss_weight=ACTION_LOSS_WEIGHT, lr_decay_stepsize=-1):
        self.controller, self.cbf, self.dataset = controller, cbf, dataset
        self.controller_optimizer = torch.optim.Adam(controller.parameters(), lr=LR_CTRL, weight_decay=WEIGHT_DECAY)
        self.cbf_optimizer = torch.optim.Adam(cbf.parameters(), lr=LR_CBF, weight_decay=WEIGHT_DECAY)
        self.action_loss_weight = action_loss_weight
        self.lr_decay_stepsize = lr_decay_stepsize
        if lr_decay_stepsize >= 0:
            self.cbf_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.cbf_optimizer, lr_decay_stepsize, 0.5)
            self.controller_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.controller_optimizer, lr_decay_stepsize, 0.5)

    @staticmethod
    def get_mask(state, obstacle):
        dist = (state[:, None, :3] - obstacle[:, :, :3]).norm(dim=2)       # (bs, k)
        safe = (dist >= SAFE_DIST).float()
        dang = (dist <= DANG_DIST).float()
        return safe, dang, (1 - safe) * (1 - dang)

    def unroll(self, state, obstacle, u_nominal, state_error):
        """Chain the SABLAS construction over a window of K real transitions.
        state (bs, K+1, 6), obstacle (bs, K+1, k, 6), u_nominal (bs, K, 3), state_error (bs, K, 6).
        Returns xbar (bs, K, 6), the states at steps 1..K with the value of the real states
        and the gradient of the nominal model w.r.t. all earlier commands, and u (bs, K, 3)."""
        x, xbar, us = state[:, 0], [], []
        for j in range(u_nominal.shape[1]):
            u = self.controller(x, obstacle[:, j], u_nominal[:, j], state_error[:, j])
            x_nominal = x + nominal_dynamics(x, u) * E.DT
            x = x_nominal + (state[:, j + 1] - x_nominal).detach()
            xbar.append(x); us.append(u)
        return torch.stack(xbar, 1), torch.stack(us, 1)

    def compute_loss(self, state, obstacle, u_nominal, state_error):
        bs, K, k = u_nominal.shape[0], u_nominal.shape[1], obstacle.shape[2]
        state_next_with_grad, u = self.unroll(state, obstacle, u_nominal, state_error)
        s_now, o_now = state[:, :K].reshape(bs * K, 6), obstacle[:, :K].reshape(bs * K, k, 6)
        o_next = obstacle[:, 1:].reshape(bs * K, k, 6)
        safe, dang, mid = self.get_mask(s_now, o_now)
        h = self.cbf(s_now, o_now)                                  # real states, no controller gradient
        h_next = self.cbf(state_next_with_grad.reshape(bs * K, 6), o_next)
        deriv = (h_next - h) / dt_n() + h

        ns, nd, nm = safe.sum(), dang.sum(), mid.sum()
        relu = nn.ReLU()
        loss_h_safe = (relu(EPS - h) * safe).sum() / (1e-5 + ns)
        loss_h_dang = (relu(h + EPS) * dang).sum() / (1e-5 + nd)
        loss_d_safe = (relu(EPS_DERIV - deriv) * safe).sum() / (1e-5 + ns)
        loss_d_dang = (relu(EPS_DERIV - deriv) * dang).sum() / (1e-5 + nd)
        loss_d_mid = (relu(EPS_DERIV - deriv) * mid).sum() / (1e-5 + nm)
        loss_action = relu((u - u_nominal).abs() - EPS_ACTION).mean()
        loss = (loss_h_safe + loss_h_dang + loss_d_safe + loss_d_dang + loss_d_mid
                + loss_action * self.action_loss_weight)

        with torch.no_grad():
            acc = torch.stack([((h >= 0).float() * safe).sum() / (1e-5 + ns),
                               ((h < 0).float() * dang).sum() / (1e-5 + nd),
                               ((deriv > 0).float() * safe).sum() / (1e-5 + ns),
                               ((deriv > 0).float() * dang).sum() / (1e-5 + nd),
                               ((deriv > 0).float() * mid).sum() / (1e-5 + nm)])
        return loss, acc

    def train_cbf_and_controller(self, rng, batch_size=1024, opt_iter=100):
        loss_np, acc_np = 0.0, np.zeros(5)
        for _ in range(opt_iter):
            batch = [T(a) for a in self.dataset.sample_data(batch_size, rng)]
            loss, acc = self.compute_loss(*batch)
            self.controller_optimizer.zero_grad(); self.cbf_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step(); self.cbf_optimizer.step()
            loss_np += loss.item(); acc_np += acc.cpu().numpy()
        if self.lr_decay_stepsize >= 0:
            self.cbf_lr_scheduler.step(); self.controller_lr_scheduler.step()
        return loss_np / opt_iter, acc_np / opt_iter


# ---------------------------------------------------------------- training loop (repo: scripts/train_drone.py)
def save(path, agent, hist):
    torch.save({"cbf": agent.cbf.state_dict(), "controller": agent.controller.state_dict(),
                "ca_features": agent.ca, "hist": hist, "M_DEB": E.M_DEB, "U_MAX": E.U_MAX,
                "horizon": getattr(agent, "horizon", 1)}, path)


def load(path):
    ck = torch.load(path, map_location=DEV)
    E.M_DEB, E.U_MAX = ck["M_DEB"], ck["U_MAX"]
    ag = Agent(ck["ca_features"]).to(DEV)
    ag.cbf.load_state_dict(ck["cbf"]); ag.controller.load_state_dict(ck["controller"]); ag.eval()
    return ag, ck["hist"]


def train(train_steps=1_000_000, n_env=32, update_interval=5000, opt_iter=100, batch=1024,
          seed=0, ckpt="sablas_torch_agent.pt", ca_features=False, lr_decay_stepsize=-1, horizon=1):
    """train_steps and update_interval count environment steps (one spacecraft, one
    control period), as config.TRAIN_STEPS / POLICY_UPDATE_INTERVAL in the repo.
    horizon: number of consecutive transitions per training window (1 = one-step loss)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    agent = Agent(ca_features).to(DEV)
    agent.horizon = horizon
    dataset = Dataset()
    win = {k_: [] for k_ in ("state", "obstacle", "u_nominal", "state_error")}   # last `horizon` steps
    trainer = Trainer(agent.controller, agent.cbf, dataset, lr_decay_stepsize=lr_decay_stepsize)

    ep = E.Episodes(n_env, rng)
    err = np.zeros((n_env, 6))
    min_d = np.full(n_env, np.inf)
    t = env_steps = 0
    next_update = update_interval
    collisions, hist, t0 = [], [], time.time()

    while env_steps < train_steps:
        own, rel = ep.observe()
        obstacle = rel + own[:, None]
        u_nom = E.pi_nom(own) / E.U_MAX
        with torch.no_grad():
            u_n = agent.controller(T(own), T(obstacle), T(u_nom), T(err)).cpu().numpy().astype(np.float64)
        u_cmd = u_n * E.U_MAX
        ep.step(np.clip(u_cmd, -E.U_MAX, E.U_MAX))                 # black box, thruster saturates
        own2, rel2 = ep.observe()
        for k_, v in zip(win, (own, obstacle, u_nom, err)):
            win[k_] = (win[k_] + [v])[-horizon:]
        if len(win["state"]) == horizon:                           # windows never cross an episode reset
            dataset.add_windows(np.stack(win["state"] + [own2], 1),
                                np.stack(win["obstacle"] + [rel2 + own2[:, None]], 1),
                                np.stack(win["u_nominal"], 1), np.stack(win["state_error"], 1))
        err = (own2 - E.cw_step(own, u_cmd)) / dt_n()              # nominal-model error -> next input
        min_d = np.minimum(min_d, np.linalg.norm(rel2[..., :3], axis=-1).min(1))
        t += 1; env_steps += n_env

        if t == E.STEPS:                                           # episodes done -> reset
            collisions.extend((min_d < DANG_DIST).tolist())
            ep = E.Episodes(n_env, rng)
            err[:] = 0; min_d[:] = np.inf; t = 0
            win = {k_: [] for k_ in win}

        if env_steps >= next_update:
            next_update += update_interval
            loss, acc = trainer.train_cbf_and_controller(rng, batch, opt_iter)
            rate = float(np.mean(collisions[-256:])) if collisions else float("nan")
            hist.append(dict(step=env_steps, loss=loss, acc=acc.tolist(), collision_rate=rate))
            print(f"step {env_steps:8d}  loss {loss:.3f}  collision rate (last 256 ep) {rate:.3f}  "
                  f"acc h_safe {acc[0]:.2f} h_dang {acc[1]:.2f} deriv safe/dang/mid "
                  f"{acc[2]:.2f}/{acc[3]:.2f}/{acc[4]:.2f}  [{time.time() - t0:.0f} s]", flush=True)
            if ckpt:
                save(ckpt, agent, hist)
    if ckpt:
        save(ckpt, agent, hist)
    return agent, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=1_000_000, help="environment steps (repo: 1e6)")
    ap.add_argument("--n-env", type=int, default=32, help="episodes run in parallel")
    ap.add_argument("--update-interval", type=int, default=5000, help="env steps between updates (repo: 5000)")
    ap.add_argument("--opt-iter", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--debris", type=int, default=E.M_DEB)
    ap.add_argument("--umax", type=float, default=E.U_MAX)
    ap.add_argument("--ca-features", action="store_true", help="add miss distance / time to closest approach")
    ap.add_argument("--lr-decay", type=int, default=-1, help="halve lr every N updates (-1: off, as in repo)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="transitions per training window; 1 = one-step loss of the paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="sablas_torch_agent.pt")
    a = ap.parse_args()
    E.M_DEB, E.U_MAX = a.debris, a.umax
    print(f"device {DEV} | debris {E.M_DEB} | u_max {E.U_MAX} | horizon {a.horizon} | train steps {a.train_steps} | ckpt {a.ckpt}", flush=True)
    train(a.train_steps, a.n_env, a.update_interval, a.opt_iter, a.batch, a.seed, a.ckpt,
          a.ca_features, a.lr_decay, a.horizon)
