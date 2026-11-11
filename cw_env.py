"""
Debris-avoidance environment for SABLAS, Clohessy-Wiltshire nominal model.

Frame: LVLH of a reference "slot" on a ~500 km orbit
  x radial, y along-track, z cross-track.  Units: metres, seconds.

Nominal model (known to the learner): Clohessy-Wiltshire, control-affine
    d/dt [p; v] = A [p; v] + [0; I] u
    A = [[0, I], [diag(3n^2, 0, -n^2), [[0, 2n, 0], [-2n, 0, 0], [0, 0, 0]]]]
Debris relative to the spacecraft obeys the same linear model with -u.

Real system (black box): every body (slot, spacecraft, debris) is propagated in
ECI with two-body + J2; the thruster has efficiency, misalignment and noise.
Debris is non-cooperative: it never manoeuvres and ignores the spacecraft.
"""
import numpy as np

MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08263e-3
ALT = 500e3
A_REF = RE + ALT
N_MEAN = np.sqrt(MU / A_REF**3)          # ~1.107e-3 rad/s

DT = 10.0                                # control period [s]
N_SUB = 2
STEPS = 120                              # 20 min episodes
M_DEB = 3                                # debris pieces per episode
D_SAFE = 300.0                           # keep-out radius [m]
U_MAX = 0.05                             # per-axis thrust accel [m/s^2]

ETA, MIS_DEG, NOISE = 0.80, 6.0, 0.03
_c, _s = np.cos(np.radians(MIS_DEG)), np.sin(np.radians(MIS_DEG))
ROT_ERR = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1.0]])


# ---------------------------------------------------------------- CW nominal model
def cw_A(n=N_MEAN):
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 0] = 3 * n**2; A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n**2
    return A


A_CW = cw_A()


def cw_step(s, u):
    """Euler step of the CW model. s: (..., 6), u: (..., 3)."""
    ds = s @ A_CW.T
    ds[..., 3:] += u
    return s + DT * ds


# ---------------------------------------------------------------- real dynamics
def _acc(x):
    r = x[..., :3]
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    a = -MU * r / rn**3
    z2 = (r[..., 2:3] / rn) ** 2
    k = 1.5 * J2 * MU * RE**2 / rn**5
    a = a + k * np.concatenate([r[..., :1] * (5 * z2 - 1),
                                r[..., 1:2] * (5 * z2 - 1),
                                r[..., 2:3] * (5 * z2 - 3)], axis=-1)
    return np.concatenate([x[..., 3:], a], axis=-1)


def propagate(x, dt, a_ext=None):
    """RK4 for any batch shape (..., 6) with optional constant extra acceleration."""
    h = dt / N_SUB
    for _ in range(N_SUB):
        def F(y):
            d = _acc(y)
            if a_ext is not None:
                d[..., 3:] += a_ext
            return d
        k1 = F(x); k2 = F(x + h / 2 * k1); k3 = F(x + h / 2 * k2); k4 = F(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def lvlh(ref):
    """Rotation matrix C (columns x,y,z of LVLH in ECI) and angular velocity."""
    R, V = ref[..., :3], ref[..., 3:]
    H = np.cross(R, V)
    xh = R / np.linalg.norm(R, axis=-1, keepdims=True)
    zh = H / np.linalg.norm(H, axis=-1, keepdims=True)
    yh = np.cross(zh, xh)
    C = np.stack([xh, yh, zh], axis=-1)
    w = H / np.sum(R * R, axis=-1, keepdims=True)
    return C, w


def to_lvlh(ref, x):
    """Relative state of x (..., k, 6) w.r.t. ref (..., 6) in rotating LVLH."""
    C, w = lvlh(ref)
    dr = x[..., :3] - ref[..., None, :3]
    dv = x[..., 3:] - ref[..., None, 3:] - np.cross(w[..., None, :], dr)
    p = np.einsum('nji,nkj->nki', C, dr)
    v = np.einsum('nji,nkj->nki', C, dv)
    return np.concatenate([p, v], axis=-1)


def from_lvlh(ref, s):
    C, w = lvlh(ref)
    dr = np.einsum('nij,nkj->nki', C, s[..., :3])
    dv = np.einsum('nij,nkj->nki', C, s[..., 3:]) + np.cross(w[..., None, :], dr)
    return np.concatenate([ref[..., None, :3] + dr, ref[..., None, 3:] + dv], axis=-1)


# ---------------------------------------------------------------- episodes
def _random_frame(n, rng):
    q = rng.standard_normal((n, 3)); R = q / np.linalg.norm(q, axis=1, keepdims=True)
    w = rng.standard_normal((n, 3)); w -= np.sum(w * R, 1, keepdims=True) * R
    T = w / np.linalg.norm(w, axis=1, keepdims=True)
    return R, T


class Episodes:
    """A batch of n episodes. Holds ECI states: ref (n,6), sc (n,6), deb (n,M,6)."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        R, T = _random_frame(n, rng)
        vc = np.sqrt(MU / A_REF)
        ref0 = np.concatenate([R * A_REF, T * vc], axis=1)

        # reference trajectory on the control grid (needed to place debris)
        refs = [ref0]
        for _ in range(STEPS):
            refs.append(propagate(refs[-1], DT))
        refs = np.stack(refs)                                      # (T+1, n, 6)

        # spacecraft starts near the slot
        s0 = np.concatenate([rng.uniform(-50, 50, (n, 1, 3)),
                             rng.uniform(-0.05, 0.05, (n, 1, 3))], axis=-1)
        sc0 = from_lvlh(ref0, s0)[:, 0]

        # debris: closest approach to the slot at step k_ca, miss 0-250 m, 3-20 m/s
        k_ca = rng.integers(30, 111, (n, M_DEB))
        vdir = rng.standard_normal((n, M_DEB, 3)); vdir /= np.linalg.norm(vdir, axis=-1, keepdims=True)
        speed = rng.uniform(3, 20, (n, M_DEB, 1))
        pdir = rng.standard_normal((n, M_DEB, 3))
        pdir -= np.sum(pdir * vdir, -1, keepdims=True) * vdir
        pdir /= np.linalg.norm(pdir, axis=-1, keepdims=True)
        miss = rng.uniform(0, 250, (n, M_DEB, 1))
        s_ca = np.concatenate([pdir * miss, vdir * speed], axis=-1)
        ref_ca = refs[k_ca, np.arange(n)[:, None]]                   # (n, M, 6)
        deb = np.stack([from_lvlh(ref_ca[:, j], s_ca[:, j:j + 1])[:, 0]
                        for j in range(M_DEB)], axis=1)
        # propagate each piece backwards to t = 0
        for k in range(k_ca.max()):
            m = k < k_ca
            deb[m] = propagate(deb[m], -DT)

        self.ref, self.sc, self.deb = ref0, sc0, deb

    def observe(self):
        """own: slot-relative state (n,6); rel: debris minus spacecraft (n,M,6)."""
        own = to_lvlh(self.ref, self.sc[:, None])[:, 0]
        deb = to_lvlh(self.ref, self.deb)
        return own, deb - own[:, None]

    def step(self, u_lvlh):
        """Black-box step. u_lvlh: commanded accel in LVLH axes (n,3)."""
        C, _ = lvlh(self.ref)
        u_eff = ETA * (u_lvlh @ ROT_ERR.T) * (1 + NOISE * self.rng.standard_normal((self.n, 1)))
        a_eci = np.einsum('nij,nj->ni', C, u_eff)
        self.sc = propagate(self.sc, DT, a_eci)
        self.ref = propagate(self.ref, DT)
        self.deb = propagate(self.deb, DT)


# ---------------------------------------------------------------- nominal controller
KP, KD = 4e-4, 0.04          # ~50 s natural period, critically damped


def pi_nom(own):
    """Holds the slot: cancels the CW drift and applies a PD. Ignores debris."""
    p, v = own[:, :3], own[:, 3:]
    drift = (own @ A_CW.T)[:, 3:]
    return np.clip(-drift - KP * p - KD * v, -U_MAX, U_MAX)


def rollout(n, policy, rng, steps=STEPS):
    ep = Episodes(n, rng)
    OWN, REL, U = [], [], []
    for _ in range(steps):
        own, rel = ep.observe()
        u = policy(own, rel)
        OWN.append(own); REL.append(rel); U.append(u)
        ep.step(u)
    own, rel = ep.observe()
    OWN.append(own); REL.append(rel)
    return np.stack(OWN), np.stack(REL), np.stack(U)


def min_separation(REL):
    return np.linalg.norm(REL[..., :3], axis=-1).min(axis=(0, 2))   # per episode
