"""
geo_sim.py -- Minimal-cost dynamics of a satellite held in a geostationary slot.

A geostationary satellite sits at a fixed point of the Earth-fixed (rotating) frame, so we
linearise about that point and get a *linear time-invariant* system (Clohessy-Wiltshire / Hill
equations with mean motion n = Earth's rotation rate):

State  x = [dr, s, h, dr', s', h']   local frame at the slot, SI units
    dr : radial   (up, away from Earth)
    s  : along-track (east)  -> longitude offset  dlon = s / A_GEO
    h  : cross-track (north) -> latitude  offset  dlat = h / A_GEO

    dr'' =  3 n^2 dr + 2 n s' + u_r + d_r
    s''  =       -2 n dr'     + u_s + d_s
    h''  =   -n^2 h           + u_h + d_h

  u : control acceleration (thruster), m/s^2
  d : disturbance acceleration (both optional, cheap closed forms)
        d_s = J22 tesseral pull at the slot longitude (constant; drives E-W longitude drift)
        d_h = luni-solar inclination pump (one sinusoid at the orbit rate, ~0.85 deg/yr)

The continuous system is discretised exactly once (zero-order hold, matrix exponential), so a
time step is a single 6x6 mat-vec:   x[k+1] = Ad x[k] + Bd (u[k] + d[k]).

"Over a location": a geostationary satellite can only be overhead on the equator. For a site at
(lat, lon) the slot is placed at the site's longitude (best elevation); the site's latitude only
changes the look angles, see `look_angles`. Run `python geo_sim.py` for validation + demo plot.
"""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------- constants
MU = 3.986004418e14
R_E = 6.378137e6
W_E = 7.2921159e-5                         # Earth rotation rate = GEO mean motion [rad/s]
A_GEO = (MU / W_E ** 2) ** (1 / 3)         # geostationary radius, ~42 164 km
C22, S22 = 1.5744e-6, -9.0387e-7           # degree-2 order-2 (ellipticity of the equator)
J22 = np.hypot(C22, S22)
LON22 = 0.5 * np.arctan2(S22, C22)         # long axis of the equator, ~ -14.9 deg
DIDT_LUNISOLAR = np.radians(0.85) / (365.25 * 86400)   # mean inclination growth [rad/s]


# ------------------------------------------------------------------ the system
class GEOSlot:
    def __init__(self, lon_deg, dt=60.0, j22=True, lunisolar=True, phase=0.0):
        """lon_deg: slot (sub-satellite) longitude. dt: discrete step [s]."""
        self.lon0 = np.radians(lon_deg)
        self.dt = dt
        self.lunisolar = lunisolar
        self.phase = phase
        n = W_E
        self.A = np.zeros((6, 6))
        self.A[:3, 3:] = np.eye(3)
        self.A[3, 0], self.A[3, 4] = 3 * n ** 2, 2 * n
        self.A[4, 3] = -2 * n
        self.A[5, 2] = -n ** 2
        self.B = np.vstack([np.zeros((3, 3)), np.eye(3)])

        # exact ZOH discretisation: expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]
        M = np.zeros((9, 9))
        M[:6, :6], M[:6, 6:] = self.A, self.B
        E = expm(M * dt)
        self.Ad, self.Bd = E[:6, :6], E[:6, 6:]

        # J22 along-track acceleration at the slot (constant)
        self.d_s = (-6 * n ** 2 * A_GEO * (R_E / A_GEO) ** 2 * J22
                    * np.sin(2 * (self.lon0 - LON22))) if j22 else 0.0
        # luni-solar: cross-track a_h = D cos(n t + phase) gives mean di/dt = D / (2 n a)
        self.D_h = 2 * n * A_GEO * DIDT_LUNISOLAR

    # --- disturbance & stepping ----------------------------------------------
    def disturbance(self, t):
        """Disturbance acceleration [m/s^2]; t scalar -> (3,), t of shape (N,) -> (N, 3)."""
        t = np.asarray(t, float)
        d_h = self.D_h * np.cos(W_E * t + self.phase) if self.lunisolar else np.zeros_like(t)
        return np.stack([np.zeros_like(t), np.full_like(t, self.d_s), d_h], -1)

    def rhs(self, t, x, u=np.zeros(3)):
        return self.A @ x + self.B @ (u + self.disturbance(t))

    def step(self, x, u=np.zeros(3), t=0.0):
        """One discrete step from time t. Works on a single state (6,) or a batch (N, 6)."""
        w = u + self.disturbance(t + 0.5 * self.dt)       # mid-step sample of the disturbance
        return x @ self.Ad.T + w @ self.Bd.T

    def rollout(self, x0, steps, policy=None):
        """policy(t, x) -> u. Returns t (steps+1,), X (steps+1, 6), U (steps, 3)."""
        X = np.empty((steps + 1, 6)); X[0] = x0
        U = np.zeros((steps, 3))
        for k in range(steps):
            t = k * self.dt
            if policy is not None:
                U[k] = policy(t, X[k])
            X[k + 1] = self.step(X[k], U[k], t)
        return np.arange(steps + 1) * self.dt, X, U

    # --- geographic output -------------------------------------------------
    def latlon(self, X):
        """Sub-satellite lat, lon [deg] (small-angle map of h, s)."""
        X = np.atleast_2d(X)
        return np.degrees(X[:, 2] / A_GEO), np.degrees(self.lon0 + X[:, 1] / A_GEO)

    def ecef(self, X):
        X = np.atleast_2d(X)
        r = A_GEO + X[:, 0]
        lat, lon = X[:, 2] / A_GEO, self.lon0 + X[:, 1] / A_GEO
        return np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon),
                         r * np.sin(lat)], axis=1)


def look_angles(sat_ecef, site_lat_deg, site_lon_deg):
    """Azimuth / elevation [deg] of the satellite from a (spherical-Earth) ground site."""
    la, lo = np.radians(site_lat_deg), np.radians(site_lon_deg)
    site = R_E * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    d = np.atleast_2d(sat_ecef) - site
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    up = site / R_E
    e, nn, u = d @ east, d @ north, d @ up
    return np.degrees(np.arctan2(e, nn)) % 360, np.degrees(np.arctan2(u, np.hypot(e, nn)))


def slot_for_site(lat_deg, lon_deg, **kw):
    """GEO slot that best serves a site: same longitude. Warns if never visible."""
    slot = GEOSlot(lon_deg, **kw)
    _, el = look_angles(slot.ecef(np.zeros(6)), lat_deg, lon_deg)
    if el[0] < 0:
        print(f"warning: GEO is below the horizon at latitude {lat_deg} deg")
    return slot


# ------------------------------------------- nonlinear truth (validation only)
def nonlinear_rhs(t, y, j22=True):
    """Full two-body (+ C22/S22) motion in the Earth-fixed frame."""
    r, v = y[:3], y[3:]
    rn = np.linalg.norm(r)
    a = -MU * r / rn ** 3
    a += -2 * np.cross([0, 0, W_E], v) - np.cross([0, 0, W_E], np.cross([0, 0, W_E], r))
    if j22:  # gradient of U22 = 3 mu R^2 [C22 (x^2 - y^2) + 2 S22 x y] / r^5
        x, yy, _ = r
        f = C22 * (x * x - yy * yy) + 2 * S22 * x * yy
        g = np.array([2 * C22 * x + 2 * S22 * yy, -2 * C22 * yy + 2 * S22 * x, 0.0])
        a += 3 * MU * R_E ** 2 * (g / rn ** 5 - 5 * f * r / rn ** 7)
    return np.concatenate([v, a])


def local_to_ecef_state(slot, x):
    """Exact map of a local state to an Earth-fixed position/velocity (for the truth model)."""
    c, s = np.cos(slot.lon0), np.sin(slot.lon0)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])      # columns: up, east, north at slot
    return np.concatenate([R @ (np.array([A_GEO, 0, 0]) + x[:3]), R @ x[3:]])


def ecef_to_local(slot, y):
    c, s = np.cos(slot.lon0), np.sin(slot.lon0)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.concatenate([R.T @ y[:3] - [A_GEO, 0, 0], R.T @ y[3:]])


# ---------------------------------------------- black-box plant (the "real" satellite)
J2 = 1.08262668e-3
MU_SUN, MU_MOON = 1.32712440018e20, 4.9028e12
AU, R_MOON = 1.495978707e11, 3.844e8
N_SUN, N_MOON = 2 * np.pi / (365.25636 * 86400), 2 * np.pi / (27.321661 * 86400)
EPS = np.radians(23.439)                   # obliquity of the ecliptic
I_MOON = np.radians(5.145)                 # Moon's orbit vs ecliptic
P_SRP = 4.56e-6                            # solar radiation pressure at 1 AU [N/m^2]


def _rot_x(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([v[..., 0], c * v[..., 1] - s * v[..., 2], s * v[..., 1] + c * v[..., 2]], -1)


def _rot_z(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class BlackBoxGEO:
    """The "real" satellite in the slot: a simulator you may only query, x' = F(x, u, t).

    Same local state / control as GEOSlot, but inside it is nonlinear and carries effects the
    nominal model ignores or gets wrong:
        point mass + J2 + C22/S22 (full field, not frozen at the slot) + Sun & Moon third body
        + solar radiation pressure + thruster gain error & misalignment (+ optional noise),
    integrated with RK4 in the Earth-fixed frame. The parameters below are drawn per instance
    from `seed` and are meant to stay hidden from the controller / learner:
        Sun & Moon phases, Moon node, SRP coefficient Cr*A/m, thruster gain and misalignment.
    """

    def __init__(self, lon_deg, dt=60.0, seed=None, noise_std=0.0, max_substep=600.0):
        rng = np.random.default_rng(seed)
        self.dt, self.lon0 = dt, np.radians(lon_deg)
        c, s = np.cos(self.lon0), np.sin(self.lon0)
        self._R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])   # slot (up, east, north) -> ECEF
        self._nsub = int(np.ceil(dt / max_substep))
        self._h = dt / self._nsub
        self._noise, self._rng = noise_std, rng
        # --- hidden parameters
        self._sun0, self._moon0, self._node = rng.uniform(0, 2 * np.pi, 3)
        self._srp = P_SRP * rng.uniform(1.2, 1.5) * rng.uniform(0.01, 0.03)
        self._gain = rng.uniform(0.95, 1.05)
        k = rng.normal(size=3); k /= np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        ang = np.radians(rng.uniform(0, 1.0))
        self._Rthr = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        self.reset()

    # --- public interface: reset / step / query --------------------------------
    def reset(self, x0=None, t0=0.0):
        self.x = np.zeros(6) if x0 is None else np.array(x0, float)
        self.t = t0
        return self.x.copy()

    def step(self, u=np.zeros(3)):
        self.x = self.query(self.x, u, self.t)
        self.t += self.dt
        return self.x.copy()

    def query(self, X, U, t):
        """Stateless batch query: X (N,6) or (6,), U (N,3) or (3,), t scalar or (N,) -> next X."""
        single = np.ndim(X) == 1
        X = np.atleast_2d(X)
        N = len(X)
        U = np.broadcast_to(np.atleast_2d(U), (N, 3))
        t = np.broadcast_to(np.asarray(t, float), (N,))
        Y = np.hstack([(X[:, :3] + [A_GEO, 0, 0]) @ self._R.T, X[:, 3:] @ self._R.T])
        Ue = (self._gain * U @ self._Rthr.T) @ self._R.T
        h = self._h
        for k in range(self._nsub):
            tk = t + k * h
            k1 = self._rhs(Y, tk, Ue)
            k2 = self._rhs(Y + 0.5 * h * k1, tk + 0.5 * h, Ue)
            k3 = self._rhs(Y + 0.5 * h * k2, tk + 0.5 * h, Ue)
            k4 = self._rhs(Y + h * k3, tk + h, Ue)
            Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Xn = np.hstack([Y[:, :3] @ self._R - [A_GEO, 0, 0], Y[:, 3:] @ self._R])
        if self._noise:
            Xn[:, 3:] += self._noise * self.dt * self._rng.normal(size=(N, 3))
        return Xn[0] if single else Xn

    # --- hidden physics ---------------------------------------------------------
    def _bodies(self, t):
        """Sun and Moon positions in the Earth-fixed frame, circular analytic ephemerides."""
        L = self._sun0 + N_SUN * t
        sun = _rot_x(AU * np.stack([np.cos(L), np.sin(L), 0 * L], -1), EPS)
        u = self._moon0 + N_MOON * t
        moon = R_MOON * np.stack([np.cos(u), np.sin(u), 0 * u], -1)
        moon = _rot_x(_rot_z(_rot_x(moon, I_MOON), self._node), EPS)
        th = -W_E * t
        return _rot_z(sun, th), _rot_z(moon, th)

    def _rhs(self, Y, t, Ue):
        r, v = Y[:, :3], Y[:, 3:]
        x, y, z = r.T
        rn = np.linalg.norm(r, axis=1)
        a = -MU * r / rn[:, None] ** 3
        z2 = (z / rn) ** 2                                          # J2
        a += (-1.5 * J2 * MU * R_E ** 2 / rn ** 5)[:, None] * np.stack(
            [x * (1 - 5 * z2), y * (1 - 5 * z2), z * (3 - 5 * z2)], 1)
        F = C22 * (x * x - y * y) + 2 * S22 * x * y                 # C22 / S22
        g = np.stack([2 * C22 * x + 2 * S22 * y, -2 * C22 * y + 2 * S22 * x, 0 * x], 1)
        a += 3 * MU * R_E ** 2 * (g / rn[:, None] ** 5 - 5 * F[:, None] * r / rn[:, None] ** 7)
        a += np.stack([W_E ** 2 * x + 2 * W_E * v[:, 1],            # centrifugal + Coriolis
                       W_E ** 2 * y - 2 * W_E * v[:, 0], 0 * x], 1)
        r_sun, r_moon = self._bodies(t)
        for rb, mub in ((r_sun, MU_SUN), (r_moon, MU_MOON)):        # third bodies
            d = rb - r
            a += mub * (d / np.linalg.norm(d, axis=1, keepdims=True) ** 3
                        - rb / np.linalg.norm(rb, axis=1, keepdims=True) ** 3)
        d = r - r_sun                                               # SRP, pushes away from Sun
        a += self._srp * d / np.linalg.norm(d, axis=1, keepdims=True)
        return np.hstack([v, a + Ue])


def fit_linear_model(bb, Ad_known=None, n=4000, pos_box=20e3, vel_box=1.0, u_max=1e-4, seed=0):
    """System identification from black-box samples only: least squares for
        x[k+1] = Ad x[k] + Bd u[k] + c
    on random states in the box, random controls and random times over a year.
    The nominal structure is linear, so this is closed form; no neural net needed.

    Ad_known: grey-box fit (recommended). Keep the physics Ad (e.g. GEOSlot.Ad) and fit only
    Bd (thruster) and c (constant forces). Fitting Ad too lets the unmodelled Sun/Moon forcing
    leak into it, which nudges the marginally stable drift mode above |eig| = 1 and makes
    long rollouts blow up."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 6)) * np.r_[[pos_box] * 3, [vel_box] * 3]
    U = rng.uniform(-u_max, u_max, (n, 3))
    T = rng.uniform(0, 365.25 * 86400, n)
    Xn = bb.query(X, U, T)
    if Ad_known is None:
        Phi, target = np.hstack([X, U, np.ones((n, 1))]), Xn
    else:
        Phi, target = np.hstack([U, np.ones((n, 1))]), Xn - X @ Ad_known.T
    scale = np.abs(Phi).max(0)                       # column scaling for conditioning
    theta = np.linalg.lstsq(Phi / scale, target, rcond=None)[0] / scale[:, None]
    if Ad_known is None:
        return theta[:6].T, theta[6:9].T, theta[9]
    return Ad_known.copy(), theta[:3].T, theta[3]


def rollout_linear(Ad, Bd, c, x0, steps, policy=None, dt=1.0):
    """Roll out x[k+1] = Ad x + Bd u + c (e.g. a fitted model)."""
    X = np.empty((steps + 1, 6)); X[0] = x0
    for k in range(steps):
        u = np.zeros(3) if policy is None else policy(k * dt, X[k])
        X[k + 1] = Ad @ X[k] + Bd @ u + c
    return X


# ------------------------------------------------------------------ validation
def validate():
    print(f"GEO radius {A_GEO/1e3:.1f} km, step = one 6x6 mat-vec")

    print("== 1. Discrete map vs continuous integration (linear model, 2 days) ==")
    slot = GEOSlot(-30.0, dt=60.0)
    x0 = np.array([500.0, -2e3, 1e3, 0.01, 0.05, -0.02])
    steps = 2 * 1440
    t, X, _ = slot.rollout(x0, steps)
    s = solve_ivp(slot.rhs, (0, t[-1]), x0, t_eval=t, rtol=1e-11, atol=1e-9)
    print(f"  max position mismatch {np.abs(X[:, :3] - s.y[:3].T).max():.2e} m")

    print("== 2. Linear model vs nonlinear two-body + C22/S22 (1 day, 2 km / 0.1 m/s offsets) ==")
    slot = GEOSlot(-30.0, dt=60.0, lunisolar=False)
    x0 = np.array([2e3, -2e3, 2e3, 0.1, -0.1, 0.1])
    t, X, _ = slot.rollout(x0, 1440)
    sol = solve_ivp(nonlinear_rhs, (0, t[-1]), local_to_ecef_state(slot, x0), t_eval=t,
                    method="DOP853", rtol=1e-12, atol=1e-6)
    Xn = np.array([ecef_to_local(slot, sol.y[:, k]) for k in range(len(t))])
    # compare in curvilinear coordinates (s, h are arc lengths, the linear model's meaning)
    rn = np.linalg.norm(sol.y[:3], axis=0)
    lon_n = np.arctan2(sol.y[1], sol.y[0]) - slot.lon0
    lat_n = np.arcsin(sol.y[2] / rn)
    err = np.stack([X[:, 0] - (rn - A_GEO), X[:, 1] - A_GEO * lon_n, X[:, 2] - A_GEO * lat_n])
    print(f"  max error  radial {np.abs(err[0]).max():.1f} m  along {np.abs(err[1]).max():.1f} m"
          f"  cross {np.abs(err[2]).max():.1f} m   (Cartesian diff {np.abs(Xn[:, :3]-X[:, :3]).max():.0f} m)")

    print("== 3. J22 longitude drift vs analytic (30 days from rest) ==")
    for lon in (-30.0, 75.07, 165.0):
        slot = GEOSlot(lon, dt=600.0, lunisolar=False)
        t, X, _ = slot.rollout(np.zeros(6), 30 * 144)
        _, lo = slot.latlon(X)
        sim = 2 * (lo[-1] - lon) / (t[-1] / 86400) ** 2
        an = np.degrees(18 * W_E ** 2 * (R_E / A_GEO) ** 2 * J22
                        * np.sin(2 * (np.radians(lon) - LON22))) * 86400 ** 2
        print(f"  lon {lon:+7.2f}:  lon'' sim {sim:+.2e}  analytic {an:+.2e} deg/day^2")

    print("== 4. Luni-solar inclination growth (1 year) ==")
    slot = GEOSlot(0.0, dt=3600.0, j22=False)
    t, X, _ = slot.rollout(np.zeros(6), int(365.25 * 24))
    print(f"  peak latitude excursion after 1 yr {np.degrees(np.abs(X[-48:, 2]).max() / A_GEO):.3f} deg"
          f"  (target {np.degrees(DIDT_LUNISOLAR * 365.25 * 86400):.2f})")

    print("== 5. Black box: nominal model vs model fitted from black-box samples ==")
    bb = BlackBoxGEO(-30.0, dt=600.0, seed=1)
    nom = GEOSlot(-30.0, dt=600.0)
    Ad_free = fit_linear_model(bb)[0]
    print(f"  free fit (Ad learned too): max |eig(Ad)| = {np.abs(np.linalg.eigvals(Ad_free)).max():.5f}"
          f"  -> unstable rollouts; using the grey-box fit (Ad from physics)")
    Ad, Bd, c = fit_linear_model(bb, nom.Ad)
    rng = np.random.default_rng(99)                      # fresh test samples
    X = rng.uniform(-1, 1, (2000, 6)) * np.r_[[20e3] * 3, [1.0] * 3]
    U = rng.uniform(-1e-4, 1e-4, (2000, 3))
    T = rng.uniform(0, 365.25 * 86400, 2000)
    Xn = bb.query(X, U, T)
    pred_nom = nom.step(X, U, T)
    pred_fit = X @ Ad.T + U @ Bd.T + c
    for name, P in (("nominal", pred_nom), ("fitted ", pred_fit)):
        e = P - Xn
        print(f"  {name} one-step RMS error: pos {np.sqrt((e[:, :3]**2).mean()):.2e} m  "
              f"vel {np.sqrt((e[:, 3:]**2).mean())*1e3:.2e} mm/s")
    g_fit = np.linalg.norm(Bd[3:]) / np.linalg.norm(nom.Bd[3:])
    print(f"  thruster gain recovered from data {g_fit:.4f}  (hidden true value {bb._gain:.4f})")
    bb.reset()
    steps = 10 * 144
    Xb = np.array([bb.reset()] + [bb.step() for _ in range(steps)])
    _, Xnom, _ = nom.rollout(np.zeros(6), steps)
    Xfit = rollout_linear(Ad, Bd, c, np.zeros(6), steps)
    for name, P in (("nominal", Xnom), ("fitted ", Xfit)):
        print(f"  {name} 10-day free-drift error vs black box: "
              f"{np.linalg.norm(P[-1, :3] - Xb[-1, :3])/1e3:.1f} km  "
              f"(black box moved {np.linalg.norm(Xb[-1, :3])/1e3:.1f} km)")
    print("  (no time-invariant linear model can follow the time-varying Sun/Moon pull; that"
          " residual is the mismatch a robust / learned controller must absorb)")


# ----------------------------------------------------------------------- demo
def demo(site=(52.37, 4.90), path="geo_demo.png"):
    """Free drift vs a trivial station-keeping controller for a slot above `site` (lat, lon)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slot = slot_for_site(*site, dt=600.0)
    days, steps = 60, 60 * 144
    x0 = np.zeros(6)

    # PD on the local frame (plus feed-forward of the known J22 term) -- placeholder for SABLAS.
    # Kp must beat the 3 n^2 ~ 1.6e-8 radial instability of the CW equations to be stable.
    Kp, Kd = 1e-7, 5e-4

    def keep(t, x):
        return -Kp * x[:3] - Kd * x[3:] - np.array([0.0, slot.d_s, 0.0])

    t, Xf, _ = slot.rollout(x0, steps)
    _, Xc, Uc = slot.rollout(x0, steps, keep)
    d = t / 86400

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for X, lab in ((Xf, "free drift"), (Xc, "station keeping")):
        lat, lon = slot.latlon(X)
        ax[0, 0].plot(lon, lat, lw=.8, label=lab)
        ax[0, 1].plot(d, lon - np.degrees(slot.lon0), label=lab)
        ax[1, 0].plot(d, lat, lw=.6, label=lab)
        _, el = look_angles(slot.ecef(X), *site)
        ax[1, 1].plot(d, el, lw=.8, label=lab)
    ax[0, 0].set(title=f"Sub-satellite point, slot lon {np.degrees(slot.lon0):.2f} deg",
                 xlabel="longitude [deg]", ylabel="latitude [deg]")
    ax[0, 1].set(title="Longitude offset (J22 drift)", xlabel="days", ylabel="deg")
    ax[1, 0].set(title="Latitude (luni-solar inclination)", xlabel="days", ylabel="deg")
    ax[1, 1].set(title=f"Elevation seen from site {site}", xlabel="days", ylabel="deg")
    for a in ax.flat:
        a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    dv = np.linalg.norm(Uc, axis=1).sum() * slot.dt
    print(f"demo: site {site}, {days} d, station-keeping dv {dv:.2f} m/s -> {path}")


def demo_blackbox(lon=4.90, seed=1, path="geo_blackbox.png"):
    """30-day free drift of the black box vs the nominal and fitted models, and the nominal
    station-keeping controller applied to the black box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt, days = 600.0, 30
    steps = int(days * 86400 / dt)
    bb = BlackBoxGEO(lon, dt=dt, seed=seed)
    nom = GEOSlot(lon, dt=dt)
    Ad, Bd, c = fit_linear_model(bb, nom.Ad)
    Xb = np.array([bb.reset()] + [bb.step() for _ in range(steps)])
    _, Xn, _ = nom.rollout(np.zeros(6), steps)
    Xf = rollout_linear(Ad, Bd, c, np.zeros(6), steps)

    Kp, Kd = 1e-7, 5e-4                       # same controller as demo(), designed on the nominal
    keep = lambda x: -Kp * x[:3] - Kd * x[3:] - np.array([0.0, nom.d_s, 0.0])
    bb.reset()
    Xc, Uc = [bb.x.copy()], []
    for _ in range(steps):
        Uc.append(keep(Xc[-1]))
        Xc.append(bb.step(Uc[-1]))
    Xc = np.array(Xc)
    dv = np.linalg.norm(Uc, axis=1).sum() * dt
    _, _, Un = nom.rollout(np.zeros(6), steps, lambda t, x: keep(x))
    dv_nom = np.linalg.norm(Un, axis=1).sum() * dt
    d = np.arange(steps + 1) * dt / 86400

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for X, lab, sty in ((Xb, "black box", "k-"), (Xn, "nominal model", "C0--"),
                        (Xf, "fitted model", "C1:")):
        ax[0].plot(d, np.degrees(X[:, 1] / A_GEO), sty, label=lab)
        ax[1].plot(d, np.degrees(X[:, 2] / A_GEO), sty, lw=.7, label=lab)
    ax[2].plot(d, Xc[:, 0] / 1e3, label="radial [km]")
    ax[2].plot(d, Xc[:, 1] / 1e3, label="east [km]")
    ax[2].plot(d, Xc[:, 2] / 1e3, label="north [km]")
    ax[0].set(title="Free drift: longitude offset", xlabel="days", ylabel="deg")
    ax[1].set(title="Free drift: latitude", xlabel="days", ylabel="deg")
    ax[2].set(title="Nominal station keeping on the black box", xlabel="days")
    for a in ax:
        a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"demo_blackbox: lon {lon}, {days} d, controlled max offset "
          f"{np.abs(Xc[:, :3]).max()/1e3:.2f} km, dv {dv:.1f} m/s on the black box vs "
          f"{dv_nom:.1f} m/s on the nominal model -> {path}")


if __name__ == "__main__":
    validate()
    demo()
    demo_blackbox()
