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

Notes on episode generation
  * Debris is aimed at the *slot* (not the spacecraft); the spacecraft starts
    within ~50 m of the slot, so the two are nearly equivalent at the start.
  * Episodes that start already inside D_SAFE + INIT_MARGIN of any debris piece
    are unwinnable, so they are rejected and resampled.
"""
import numpy as np

MU = 3.986004418e14                      # Earth gravitational parameter [m^3/s^2]
RE = 6378137.0                           # Earth equatorial radius [m]
J2 = 1.08263e-3                          # J2 oblateness coefficient
ALT = 500e3                              # reference orbit altitude [m]
A_REF = RE + ALT                         # reference orbit radius [m]
N_MEAN = np.sqrt(MU / A_REF**3)          # mean motion, ~1.107e-3 rad/s

DT = 10.0                                # control period [s]
N_SUB = 2                                # RK4 sub-steps per control period
STEPS = 120                              # 20 min episodes
M_DEB = 3                                # debris pieces per episode
D_SAFE = 2500.0                          # keep-out radius [m]
INIT_MARGIN = 500.0                      # extra clearance required at t = 0 [m]
U_MAX = 0.05                             # per-axis thrust accel [m/s^2]

ETA, MIS_DEG, NOISE = 0.80, 6.0, 0.03    # thruster efficiency, misalignment, noise
_c, _s = np.cos(np.radians(MIS_DEG)), np.sin(np.radians(MIS_DEG))
ROT_ERR = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1.0]])   # misalignment about z


# ---------------------------------------------------------------- CW nominal model
def cw_A(n=N_MEAN):
    """
    Build the Clohessy-Wiltshire (Hill) state matrix for relative motion about a
    circular reference orbit with mean motion n.

    State s = [x, y, z, vx, vy, vz]; x radial, y along-track, z cross-track.

        x'' =  3 n^2 x + 2 n y'
        y'' = -2 n x'
        z'' = -n^2 z

    ds/dt = A s, with

        A = [  0     0    0    1    0    0 ]
            [  0     0    0    0    1    0 ]
            [  0     0    0    0    0    1 ]
            [ 3n^2   0    0    0   2n    0 ]
            [  0     0    0  -2n    0    0 ]
            [  0     0  -n^2    0    0    0 ]
    """
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 0] = 3 * n**2; A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n**2
    return A


A_CW = cw_A()


def cw_step(s, u):
    """Euler step of the CW model over one control period DT. s: (..., 6), u: (..., 3)."""
    ds = s @ A_CW.T
    ds[..., 3:] += u
    return s + DT * ds


# ---------------------------------------------------------------- real dynamics
def _acc(x):
    """
    State derivative in ECI for two-body gravity plus J2.  x: (..., 6) -> (..., 6).

        a = -mu r / |r|^3 + (3/2) J2 mu Re^2 / |r|^5 *
            [ x (5 z^2/|r|^2 - 1),  y (5 z^2/|r|^2 - 1),  z (5 z^2/|r|^2 - 3) ]
    """
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
    """RK4 for any batch shape (..., 6) over time dt (N_SUB sub-steps), with an
    optional constant extra acceleration a_ext (..., 3) in ECI. dt may be negative."""
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
    """LVLH frame of the reference ref (..., 6): returns C (..., 3, 3), whose columns
    are the LVLH x (radial), y (along-track), z (orbit normal) axes in ECI, and the
    frame angular velocity w = (r x v) / |r|^2 (..., 3) in ECI."""
    R, V = ref[..., :3], ref[..., 3:]
    H = np.cross(R, V)
    xh = R / np.linalg.norm(R, axis=-1, keepdims=True)
    zh = H / np.linalg.norm(H, axis=-1, keepdims=True)
    yh = np.cross(zh, xh)
    C = np.stack([xh, yh, zh], axis=-1)
    w = H / np.sum(R * R, axis=-1, keepdims=True)
    return C, w


def to_lvlh(ref, x):
    """ECI -> LVLH. Relative state of x (..., k, 6) w.r.t. ref (..., 6), expressed in
    the rotating LVLH frame (velocity has the frame rotation w x dr removed)."""
    C, w = lvlh(ref)
    dr = x[..., :3] - ref[..., None, :3]
    dv = x[..., 3:] - ref[..., None, 3:] - np.cross(w[..., None, :], dr)
    p = np.einsum('nji,nkj->nki', C, dr)
    v = np.einsum('nji,nkj->nki', C, dv)
    return np.concatenate([p, v], axis=-1)


def from_lvlh(ref, s):
    """LVLH -> ECI. Inverse of to_lvlh: turns relative LVLH states s (..., k, 6) about
    ref (..., 6) back into absolute ECI states (..., k, 6)."""
    C, w = lvlh(ref)
    dr = np.einsum('nij,nkj->nki', C, s[..., :3])
    dv = np.einsum('nij,nkj->nki', C, s[..., 3:]) + np.cross(w[..., None, :], dr)
    return np.concatenate([ref[..., None, :3] + dr, ref[..., None, 3:] + dv], axis=-1)


# ---------------------------------------------------------------- episodes
def _random_frame(n, rng):
    """n random orthonormal pairs (R, T) of 3D unit vectors, R[i] perpendicular to T[i].
    R is uniform on the sphere; T is uniform on the circle perpendicular to R
    (Gram-Schmidt on a second Gaussian draw). Used as radial / velocity directions."""
    q = rng.standard_normal((n, 3)); R = q / np.linalg.norm(q, axis=1, keepdims=True)
    w = rng.standard_normal((n, 3)); w -= np.sum(w * R, 1, keepdims=True) * R
    T = w / np.linalg.norm(w, axis=1, keepdims=True)
    return R, T


def _build(n, rng):
    """Sample n raw episodes (no feasibility filter).
    Returns ECI states ref0 (n,6), sc0 (n,6), deb (n,M,6) at t = 0."""
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
    pdir -= np.sum(pdir * vdir, -1, keepdims=True) * vdir      # miss vector perpendicular to v
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
    return ref0, sc0, deb


def _init_clearance(ref0, sc0, deb):
    """Distance [m] from the spacecraft to the nearest debris piece at t = 0, per episode."""
    own = to_lvlh(ref0, sc0[:, None])[:, 0]
    d = to_lvlh(ref0, deb)
    return np.linalg.norm(d[..., :3] - own[:, None, :3], axis=-1).min(axis=1)


class Episodes:
    """A batch of n episodes. Holds ECI states: ref (n,6), sc (n,6), deb (n,M,6).
    Episodes starting within D_SAFE + INIT_MARGIN of any debris are resampled."""

    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        ref0, sc0, deb = _build(n, rng)
        bad = _init_clearance(ref0, sc0, deb) < D_SAFE + INIT_MARGIN
        for _ in range(200):                       # rejection sampling
            if not bad.any():
                break
            r2, s2, d2 = _build(int(bad.sum()), rng)
            ref0[bad], sc0[bad], deb[bad] = r2, s2, d2
            bad[bad] = _init_clearance(r2, s2, d2) < D_SAFE + INIT_MARGIN
        else:
            raise RuntimeError("could not sample feasible initial conditions")
        self.ref, self.sc, self.deb = ref0, sc0, deb

    def observe(self):
        """Return own: slot-relative spacecraft state (n,6) and
        rel: debris minus spacecraft (n,M,6), both in the slot's LVLH frame."""
        own = to_lvlh(self.ref, self.sc[:, None])[:, 0]
        deb = to_lvlh(self.ref, self.deb)
        return own, deb - own[:, None]

    def step(self, u_lvlh):
        """Black-box step over DT. u_lvlh: commanded accel in LVLH axes (n,3), clipped
        per-axis to +/-U_MAX, then degraded by efficiency, misalignment and noise."""
        u_lvlh = np.clip(u_lvlh, -U_MAX, U_MAX)
        C, _ = lvlh(self.ref)
        u_eff = ETA * (u_lvlh @ ROT_ERR.T) * (1 + NOISE * self.rng.standard_normal((self.n, 1)))
        a_eci = np.einsum('nij,nj->ni', C, u_eff)
        self.sc = propagate(self.sc, DT, a_eci)
        self.ref = propagate(self.ref, DT)
        self.deb = propagate(self.deb, DT)


# ---------------------------------------------------------------- nominal controller
# Gains: natural frequency sqrt(KP) = 0.02 rad/s (time constant 50 s, period ~314 s);
# KD = 2 sqrt(KP) makes the PD critically damped.
KP, KD = 4e-4, 0.04


def pi_nom(own):
    """Nominal controller, own: (n,6) slot-relative state -> u (n,3). Holds the slot by
    cancelling the CW drift and applying a PD law; clipped to +/-U_MAX. Ignores debris."""
    p, v = own[:, :3], own[:, 3:]
    drift = (own @ A_CW.T)[:, 3:]
    return np.clip(-drift - KP * p - KD * v, -U_MAX, U_MAX)


def rollout(n, policy, rng, steps=STEPS):
    """Run n episodes with policy(own, rel) -> u (n,3). Returns stacked OWN (T+1,n,6),
    REL (T+1,n,M,6) and U (T,n,3)."""
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


def segment_min_dist(r0, r1):
    """Minimum norm along the straight segment from r0 to r1 (both (..., 3)).
    Used so a closest approach between two control samples is not missed."""
    d = r1 - r0
    s = np.clip(-np.sum(r0 * d, -1) / np.maximum(np.sum(d * d, -1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(r0 + s[..., None] * d, axis=-1)


def min_separation(REL):
    """Per-episode minimum spacecraft-debris distance [m] over the rollout. Assumes
    straight-line relative motion between consecutive samples, so the minimum is not
    missed between control steps (debris can move ~200 m per DT)."""
    r = REL[..., :3]                                    # (T+1, n, M, 3)
    return segment_min_dist(r[:-1], r[1:]).min(axis=(0, 2))
