"""
leo_sim.py -- A satellite in low Earth orbit as a 3D dynamical system.

State  y = [x, y, z, vx, vy, vz]   ECI (J2000-like equatorial, inertial), SI units

    r'' = -mu r/|r|^3  +  a_zonal(J2,J3,J4)  +  a_drag  [+ a_thrust]

  a_zonal : axisymmetric Earth gravity harmonics (z = spin axis)
  a_drag  : -1/2 rho (Cd A/m) |v_rel| v_rel,  v_rel = v - omega_E x r   (rotating atmosphere)
  rho     : piecewise-exponential atmosphere (static, no solar-cycle variation)

No Moon / Sun here. Python >= 3.9, numpy, scipy (matplotlib for plots).
Run `python leo_sim.py` for validation + demo plots.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------- constants
MU = 3.986004418e14
R_E = 6.378137e6
J2, J3, J4 = 1.08262668e-3, -2.5327e-6, -1.6196e-6
W_E = 7.2921159e-5                      # Earth rotation rate [rad/s]
OMEGA_E = np.array([0.0, 0.0, W_E])
RTOL, ATOL = 1e-12, 1e-9

# (base altitude km, base density kg/m^3, scale height km)  -- Vallado exponential table
_ATM = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682], [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714], [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382], [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298], [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00]])


_LEG = {n: np.polynomial.legendre.Legendre.basis(n) for n in (2, 3, 4)}
_LEGD = {n: p.deriv() for n, p in _LEG.items()}


def density(alt):
    """Atmospheric density [kg/m^3] at spherical altitude alt [m]."""
    h = alt / 1e3
    k = max(np.searchsorted(_ATM[:, 0], h, side="right") - 1, 0)
    h0, rho0, H = _ATM[k]
    return rho0 * np.exp(-(h - h0) / H)


# ------------------------------------------------------------------ rotations
def R1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ------------------------------------------------------- orbital elements <-> rv
def coe2rv(a, e, i, Om, w, nu, mu=MU):
    p = a * (1 - e ** 2)
    r_pqw = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    Q = R3(Om) @ R1(i) @ R3(w)
    return Q @ r_pqw, Q @ v_pqw


def rv2coe(r, v, mu=MU):
    """Osculating elements: a, e, i, Om, w, nu (+ u = argument of latitude), all rad/m."""
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0, 0, 1.0], h)
    nn = np.linalg.norm(n)
    evec = ((vn ** 2 - mu / rn) * r - (r @ v) * v) / mu
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - vn ** 2 / mu)
    i = np.arccos(np.clip(h[2] / hn, -1, 1))
    Om = np.arctan2(n[1], n[0]) % (2 * np.pi) if nn > 1e-12 else 0.0
    nhat = n / nn if nn > 1e-12 else np.array([1.0, 0, 0])
    u = np.arctan2((np.cross(nhat, r / rn)) @ (h / hn), nhat @ (r / rn)) % (2 * np.pi)
    if e > 1e-11:
        w = np.arctan2((np.cross(nhat, evec / e)) @ (h / hn), nhat @ (evec / e)) % (2 * np.pi)
        nu = np.arctan2((np.cross(evec / e, r / rn)) @ (h / hn), (evec / e) @ (r / rn)) % (2 * np.pi)
    else:
        w, nu = 0.0, u
    return dict(a=a, e=e, i=i, Om=Om, w=w, nu=nu, u=u)


# ------------------------------------------------------------------ the system
@dataclass
class Satellite:
    mass: float = 500.0          # kg
    area: float = 4.0            # m^2 (cross-section toward the flow)
    Cd: float = 2.2


@dataclass
class Forces:
    j2: bool = True
    j3: bool = False
    j4: bool = False
    drag: bool = True


class LEOSystem:
    def __init__(self, sat=None, forces=None, thrust=None):
        """thrust(t, r, v, m) -> acceleration vector [m/s^2] (optional, e.g. station keeping)."""
        self.sat = sat or Satellite()
        self.f = forces or Forces()
        self.thrust = thrust

    # --- accelerations ---------------------------------------------------
    def a_zonal(self, r):
        """Gradient of V_n = -mu J_n R^n P_n(sin lat) / r^(n+1), summed over enabled n."""
        rn = np.linalg.norm(r)
        s = r[2] / rn
        acc = np.zeros(3)
        for n, Jn, on in ((2, J2, self.f.j2), (3, J3, self.f.j3), (4, J4, self.f.j4)):
            if not on:
                continue
            P, dP = _LEG[n](s), _LEGD[n](s)
            k = MU * Jn * R_E ** n
            acc[:2] += k * r[:2] / rn ** (n + 3) * ((n + 1) * P + s * dP)
            acc[2] += k / rn ** (n + 2) * ((n + 1) * s * P - dP * (1 - s ** 2))
        return acc

    def a_drag(self, r, v):
        alt = np.linalg.norm(r) - R_E
        vrel = v - np.cross(OMEGA_E, r)
        B = self.sat.Cd * self.sat.area / self.sat.mass          # 1/ballistic coefficient
        return -0.5 * density(alt) * B * np.linalg.norm(vrel) * vrel

    # --- vector field ----------------------------------------------------
    def rhs(self, t, y):
        r, v = y[:3], y[3:6]
        a = -MU * r / np.linalg.norm(r) ** 3 + self.a_zonal(r)
        if self.f.drag:
            a = a + self.a_drag(r, v)
        if self.thrust is not None:
            a = a + self.thrust(t, r, v, self.sat.mass)
        return np.concatenate([v, a])

    # --- conserved quantities (J2 only, no drag/thrust) --------------------
    @staticmethod
    def energy_j2(y):
        r, v = y[:3], y[3:6]
        rn = np.linalg.norm(r)
        P2 = 0.5 * (3 * (r[2] / rn) ** 2 - 1)
        return 0.5 * v @ v - MU / rn * (1 - J2 * (R_E / rn) ** 2 * P2)

    @staticmethod
    def hz(y):
        return y[0] * y[4] - y[1] * y[3]

    # --- integration -------------------------------------------------------
    def propagate(self, y0, t_end, n=4000, reentry_alt=100e3):
        def reentry(t, y):
            return np.linalg.norm(y[:3]) - R_E - reentry_alt
        reentry.terminal, reentry.direction = True, -1
        s = solve_ivp(self.rhs, (0, t_end), y0, method="DOP853", rtol=RTOL, atol=ATOL,
                      events=reentry, dense_output=True)
        t = np.linspace(0, s.t[-1], n)
        return dict(t=t, y=s.sol(t), reentered=s.status == 1, t_end=s.t[-1])


# ----------------------------------------------------------------- analysis
def elements_history(t, y):
    E = [rv2coe(y[:3, k], y[3:6, k]) for k in range(len(t))]
    return {key: np.array([e[key] for e in E]) for key in E[0]}


def eci_to_latlon(t, y, gmst0=0.0):
    """Ground track: rotate ECI by Earth's rotation angle; returns lat, lon [deg]."""
    th = gmst0 + W_E * t
    x = np.cos(th) * y[0] + np.sin(th) * y[1]
    yy = -np.sin(th) * y[0] + np.cos(th) * y[1]
    z = y[2]
    r = np.sqrt(x * x + yy * yy + z * z)
    return np.degrees(np.arcsin(z / r)), np.degrees(np.arctan2(yy, x))


def j2_secular_rates(a, e, i):
    """Analytic secular drift of the node and perigee due to J2 [rad/s]."""
    n = np.sqrt(MU / a ** 3)
    p = a * (1 - e ** 2)
    f = 1.5 * J2 * n * (R_E / p) ** 2
    return -f * np.cos(i), 0.5 * f * (5 * np.cos(i) ** 2 - 1)


def mean_rate(t, ang):
    """Linear-fit slope of an unwrapped angle history [rad/s]."""
    return np.polyfit(t, np.unwrap(ang), 1)[0]


# ------------------------------------------------------------------ validation
def validate():
    print("== 1. Two-body: closed orbit, J2/drag off ==")
    r0, v0 = coe2rv(R_E + 500e3, 0.01, np.radians(51.6), np.radians(40), np.radians(30), 0.0)
    y0 = np.concatenate([r0, v0])
    sys_ = LEOSystem(forces=Forces(j2=False, drag=False))
    out = sys_.propagate(y0, 5 * 86400.0)
    el = elements_history(out["t"], out["y"])
    print(f"  a drift   {np.ptp(el['a']):.2e} m   e drift {np.ptp(el['e']):.2e}   "
          f"Om drift {np.ptp(np.unwrap(el['Om'])):.2e} rad")

    print("== 2. J2 only: conserved energy & h_z, secular rates vs analytic ==")
    sys_ = LEOSystem(forces=Forces(j2=True, drag=False))
    out = sys_.propagate(y0, 10 * 86400.0)
    y = out["y"]
    E = np.array([sys_.energy_j2(y[:, k]) for k in range(y.shape[1])])
    H = np.array([sys_.hz(y[:, k]) for k in range(y.shape[1])])
    print(f"  J2 energy drift (rel) {np.ptp(E)/abs(E[0]):.1e}   h_z drift (rel) {np.ptp(H)/H[0]:.1e}")
    el = elements_history(out["t"], y)
    a_m = el["a"].mean()
    dOm_an, dw_an = j2_secular_rates(a_m, el["e"].mean(), el["i"].mean())
    deg_day = np.degrees(86400.0)
    print(f"  node  rate: sim {mean_rate(out['t'], el['Om'])*deg_day:+.4f}  "
          f"analytic {dOm_an*deg_day:+.4f}  deg/day")
    print(f"  perig rate: sim {mean_rate(out['t'], el['w'])*deg_day:+.4f}  "
          f"analytic {dw_an*deg_day:+.4f}  deg/day")

    print("== 3. Sun-synchronous check (700 km, i = 98.19 deg) ==")
    r0s, v0s = coe2rv(R_E + 700e3, 0.0, np.radians(98.19), 0.0, 0.0, 0.0)
    out = LEOSystem(forces=Forces(j2=True, drag=False)).propagate(np.concatenate([r0s, v0s]), 10 * 86400.0)
    el = elements_history(out["t"], out["y"])
    print(f"  node rate {mean_rate(out['t'], el['Om'])*deg_day:+.4f} deg/day  (sun-sync = +0.9856)")

    print("== 4. Drag: decay of a 400 km circular orbit ==")
    r0d, v0d = coe2rv(R_E + 400e3, 0.0005, np.radians(51.6), 0.0, 0.0, 0.0)
    out = LEOSystem(forces=Forces(j2=True, drag=True)).propagate(np.concatenate([r0d, v0d]), 30 * 86400.0)
    el = elements_history(out["t"], out["y"])
    print(f"  a: {(el['a'][0]-R_E)/1e3:.1f} -> {(el['a'][-1]-R_E)/1e3:.1f} km altitude over "
          f"{out['t_end']/86400:.0f} d  (decay {(el['a'][0]-el['a'][-1])/1e3/(out['t_end']/86400):.2f} km/day)")


# ----------------------------------------------------------------------- demo
def demo(path="leo_demo.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sat = Satellite(mass=420e3, area=1600.0, Cd=2.2)           # ISS-like ballistic coefficient
    e0 = 0.01                                                   # eccentric enough that perigee is defined
    r0, v0 = coe2rv((R_E + 400e3) / (1 - e0), e0, np.radians(51.6), np.radians(30), np.radians(0), 0.0)
    sys_ = LEOSystem(sat, Forces(j2=True, j3=True, j4=True, drag=True))
    out = sys_.propagate(np.concatenate([r0, v0]), 10 * 86400.0, n=8000)
    t, y = out["t"], out["y"]
    el = elements_history(t, y)
    d = t / 86400

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    k = t < 3 * 5560
    ax.plot(*(y[:3, k] / 1e6), "C3", lw=2)
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    ax.plot_wireframe(R_E / 1e6 * np.cos(u) * np.sin(w), R_E / 1e6 * np.sin(u) * np.sin(w),
                      R_E / 1e6 * np.cos(w), color="steelblue", alpha=.5, lw=.6)
    ax.view_init(elev=22, azim=-55)
    #ax.quiver(0, 0, 0, 0, 0, 9, color="k", arrow_length_ratio=.05)
    ax.set(title="ECI orbit (first 3 revolutions), spin axis in black", xlabel="x [1000 km]",
           ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        f(-8, 8)

    lat, lon = eci_to_latlon(t, y)
    k = t < 4 * 5560
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(np.where(np.abs(np.diff(lon[k], prepend=lon[0])) > 180, np.nan, lon[k]), lat[k], lw=.9)
    ax2.set(title="Ground track (first 4 revolutions)", xlabel="longitude [deg]", ylabel="latitude [deg]",
            xlim=(-180, 180), ylim=(-90, 90))
    ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(d, np.degrees(np.unwrap(el["Om"] - el["Om"][0])), label="node  $\\Omega-\\Omega_0$")
    ax3.plot(d, np.degrees(np.unwrap(el["w"] - el["w"][0])), label="perigee  $\\omega-\\omega_0$")
    dOm, dw = j2_secular_rates(el["a"].mean(), el["e"].mean(), el["i"].mean())
    ax3.plot(d, np.degrees(dOm * t), "k--", lw=.8, label="analytic J2 rates")
    ax3.plot(d, np.degrees(dw * t), "k--", lw=.8)
    ax3.set(title="Secular J2 drift", xlabel="days", ylabel="deg"); ax3.legend(); ax3.grid(alpha=.3)

    ax4 = fig.add_subplot(2, 2, 4)
    win = max(int(5560 / (t[1] - t[0])), 1)                       # ~ one orbital period
    smooth = lambda q: np.convolve(q, np.ones(win) / win, mode="valid")
    ds = d[win // 2: win // 2 + len(smooth(el["a"]))]
    ax4.plot(d, (el["a"] * (1 - el["e"]) - R_E) / 1e3, color="C1", lw=.4, alpha=.5)
    ax4.plot(d, (el["a"] * (1 + el["e"]) - R_E) / 1e3, color="C0", lw=.4, alpha=.5)
    ax4.plot(ds, (smooth(el["a"] * (1 - el["e"])) - R_E) / 1e3, "C1", label="perigee altitude (1-rev mean)")
    ax4.plot(ds, (smooth(el["a"] * (1 + el["e"])) - R_E) / 1e3, "C0", label="apogee altitude (1-rev mean)")
    ax4.set(title="Drag decay", xlabel="days", ylabel="km"); ax4.legend(); ax4.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"demo: 10 d, mean a {smooth(el['a'])[0]/1e3:.1f} -> {smooth(el['a'])[-1]/1e3:.1f} km; "
          f"node {np.degrees(mean_rate(t, el['Om']))*86400:+.3f} deg/day, "
          f"perigee {np.degrees(mean_rate(t, el['w']))*86400:+.3f} deg/day -> {path}")


if __name__ == "__main__":
    validate()
    demo()
