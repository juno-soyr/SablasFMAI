#!/usr/bin/env python3
"""
Numerical simulation of an Earth -> Mars Hohmann transfer.

Part A  Heliocentric Kepler flow + two impulses, planets on circular orbits.
        Checks time of flight, delta-v, phasing and arrival miss distance.
Part B  Variable-mass, finite-thrust departure from 300 km LEO
        (state: x, y, vx, vy, m, theta).  Shows gravity losses vs thrust-to-weight
        and the impulsive limit as thrust-to-weight -> infinity.
Part C  Burn-error shear: small departure delta-v error -> arrival miss distance
        and growth of the separation from the nominal trajectory.

Units: km, s, kg (mass normalised to initial mass m0 = 1).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- constants
MU_S = 1.32712440018e11          # Sun GM, km^3/s^2
AU = 1.495978707e8               # km
R1 = 1.0 * AU                    # Earth orbit radius (circular model)
R2 = 1.523679 * AU               # Mars orbit radius (circular model)
MU_E = 398600.4418               # Earth GM, km^3/s^2
R_LEO = 6378.137 + 300.0         # km
G0 = 9.80665                     # m/s^2
ISP = 380.0                      # s (methalox-class, assumed)
VE = ISP * G0 / 1000.0           # exhaust speed, km/s
DAY = 86400.0

# ------------------------------------------------- analytic Hohmann reference
n_E = np.sqrt(MU_S / R1**3)
n_M = np.sqrt(MU_S / R2**3)
a_T = 0.5 * (R1 + R2)
T_H = np.pi * np.sqrt(a_T**3 / MU_S)             # half the transfer period
v_E = np.sqrt(MU_S / R1)
v_M = np.sqrt(MU_S / R2)
v_dep = np.sqrt(MU_S * (2 / R1 - 1 / a_T))
v_arr = np.sqrt(MU_S * (2 / R2 - 1 / a_T))
dv1 = v_dep - v_E                                # = v_inf at Earth
dv2 = v_M - v_arr                                # = v_inf at Mars
psi0 = np.pi - n_M * T_H                         # Mars lead angle at departure
n_T = np.sqrt(MU_S / a_T**3)


def earth_pos(t):
    return R1 * np.array([np.cos(n_E * t), np.sin(n_E * t)])


def mars_pos(t):
    ang = psi0 + n_M * t
    return R2 * np.array([np.cos(ang), np.sin(ang)])


# =========================================================== Part A: heliocentric
def kepler_rhs(t, s):
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return [vx, vy, -MU_S * x / r3, -MU_S * y / r3]


def propagate(dv, t_end):
    """Spacecraft leaves Earth (at angle 0) with Earth's velocity + dv (prograde)."""
    s0 = [R1, 0.0, 0.0, v_E + dv]
    return solve_ivp(kepler_rhs, [0.0, t_end], s0, method="DOP853",
                     rtol=1e-12, atol=1e-6, dense_output=True)


def miss_distances(sol):
    """Distance to Mars at nominal arrival time, and closest approach nearby."""
    arr = sol.sol(T_H)[:2]
    miss_arrival = np.linalg.norm(arr - mars_pos(T_H))
    ts = np.linspace(0.9 * T_H, 1.1 * T_H, 4001)
    d = [np.linalg.norm(sol.sol(t)[:2] - mars_pos(t)) for t in ts]
    return miss_arrival, min(d)


print("=" * 66)
print("PART A  heliocentric Hohmann (circular, coplanar planets)")
print("=" * 66)
print(f"semi-major axis a_T        : {a_T / AU:.4f} AU")
print(f"eccentricity e             : {(R2 - R1) / (R2 + R1):.4f}")
print(f"time of flight             : {T_H / DAY:.2f} days")
print(f"departure dv  (v_inf, E)   : {dv1:.4f} km/s")
print(f"arrival dv    (v_inf, M)   : {dv2:.4f} km/s")
print(f"total heliocentric dv      : {dv1 + dv2:.4f} km/s")
print(f"Mars lead angle at launch  : {np.degrees(psi0):.2f} deg")
print(f"synodic period             : {2 * np.pi / (n_E - n_M) / DAY:.1f} days")

sol0 = propagate(dv1, 1.05 * T_H)
miss0, min0 = miss_distances(sol0)
s_end = sol0.sol(T_H)
r_end = np.hypot(s_end[0], s_end[1])
E0 = 0.5 * v_dep**2 - MU_S / R1
E1 = 0.5 * (s_end[2]**2 + s_end[3]**2) - MU_S / r_end
print(f"radius at t=T_H            : {r_end / AU:.6f} AU (Mars orbit {R2 / AU:.6f})")
print(f"nominal arrival miss       : {miss0:.1f} km   (closest approach {min0:.1f} km)")
print(f"relative energy drift      : {abs(E1 / E0 - 1):.2e}")

# =========================================================== Part B: finite burn
VINF = dv1                       # required hyperbolic excess speed at Earth
v_c0 = np.sqrt(MU_E / R_LEO)
v_p = np.sqrt(VINF**2 + 2 * MU_E / R_LEO)
DV_IMP = v_p - v_c0              # impulsive injection from circular LEO
MF_IMP = np.exp(-DV_IMP / VE)


def finite_burn(tw):
    """Prograde (velocity-aligned) thrust from circular LEO until v_inf is reached.
    tw = initial thrust-to-weight (referenced to g0).  Returns dict of results."""
    a0 = tw * G0 / 1000.0        # initial thrust acceleration, km/s^2

    def rhs(t, s):
        x, y, vx, vy, m, th = s
        r = np.hypot(x, y)
        v = np.hypot(vx, vy)
        ax = -MU_E * x / r**3 + (a0 / m) * vx / v
        ay = -MU_E * y / r**3 + (a0 / m) * vy / v
        return [vx, vy, ax, ay, -a0 / VE, (x * vy - y * vx) / r**2]

    def reached(t, s):
        x, y, vx, vy = s[:4]
        return 0.5 * (vx * vx + vy * vy) - MU_E / np.hypot(x, y) - 0.5 * VINF**2
    reached.terminal = True
    reached.direction = 1

    t_guess = 1.5 * (DV_IMP + 6.0) / a0 + 1e4
    s0 = [R_LEO, 0.0, 0.0, v_c0, 1.0, 0.0]
    sol = solve_ivp(rhs, [0.0, t_guess], s0, method="DOP853", events=reached,
                    rtol=1e-10, atol=1e-9)
    sf = sol.y[:, -1]
    mf = sf[4]
    return dict(tw=tw, t=sol.t[-1], mf=mf, dv=VE * np.log(1.0 / mf),
                revs=sf[5] / (2 * np.pi), ok=sol.status == 1)


print()
print("=" * 66)
print("PART B  finite-thrust variable-mass departure from 300 km LEO")
print("=" * 66)
print(f"Isp = {ISP:.0f} s (v_e = {VE:.3f} km/s),  v_inf target = {VINF:.4f} km/s")
print(f"impulsive: dv = {DV_IMP:.4f} km/s, m_f/m_0 = {MF_IMP:.4f}, "
      f"propellant = {100 * (1 - MF_IMP):.1f}%")
print()
print(f"{'T/W0':>8} {'burn time':>12} {'revs':>7} {'dv [km/s]':>10} "
      f"{'loss':>8} {'m_f/m_0':>8}")
tws = np.logspace(np.log10(0.003), np.log10(5.0), 18)
res = []
for tw in tws:
    r = finite_burn(tw)
    res.append(r)
    tstr = f"{r['t'] / 3600:.2f} h" if r['t'] > 3600 else f"{r['t']:.0f} s"
    print(f"{tw:8.4f} {tstr:>12} {r['revs']:7.2f} {r['dv']:10.4f} "
          f"{100 * (r['dv'] / DV_IMP - 1):7.1f}% {r['mf']:8.4f}"
          + ("" if r['ok'] else "  (event not hit)"))

# =========================================================== Part C: burn errors
print()
print("=" * 66)
print("PART C  departure burn error -> arrival miss (shear)")
print("=" * 66)
errs = np.linspace(-20.0, 20.0, 41)              # m/s
miss_arr, miss_min = [], []
for e in errs:
    s = propagate(dv1 + e / 1000.0, 1.15 * T_H)
    a, b = miss_distances(s)
    miss_arr.append(a)
    miss_min.append(b)
miss_arr = np.array(miss_arr)
miss_min = np.array(miss_min)

for e in (1.0, 10.0):
    s = propagate(dv1 + e / 1000.0, 1.15 * T_H)
    a, b = miss_distances(s)
    da = 2 * a_T**2 * v_dep * (e / 1000.0) / MU_S
    print(f"dv error {e:5.1f} m/s : miss at T_H = {a:10.0f} km, "
          f"closest approach = {b:10.0f} km, delta a = {da:8.0f} km")

# separation growth for a +1 m/s error
e_demo = 1.0
sol_p = propagate(dv1 + e_demo / 1000.0, T_H)
ts = np.linspace(1.0, T_H, 800)
sep = np.array([np.linalg.norm(sol_p.sol(t)[:2] - sol0.sol(t)[:2]) for t in ts])
da_demo = 2 * a_T**2 * v_dep * (e_demo / 1000.0) / MU_S
shear = 1.5 * n_T * abs(da_demo) * ts            # secular along-track estimate

# ================================================================= figure
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (1) trajectory
th = np.linspace(0, 2 * np.pi, 400)
a1 = ax[0, 0]
a1.plot(R1 * np.cos(th) / AU, R1 * np.sin(th) / AU, color="tab:blue", lw=0.8, ls=":")
a1.plot(R2 * np.cos(th) / AU, R2 * np.sin(th) / AU, color="tab:red", lw=0.8, ls=":")
tt = np.linspace(0, T_H, 600)
tr = np.array([sol0.sol(t)[:2] for t in tt]) / AU
a1.plot(tr[:, 0], tr[:, 1], "k", lw=2, label="transfer ellipse")
for t, lab, ls in [(0, "launch", "o"), (T_H, "arrival", "s")]:
    e = earth_pos(t) / AU
    m = mars_pos(t) / AU
    a1.plot(*e, "o" if t == 0 else "x", color="tab:blue", ms=9)
    a1.plot(*m, "o" if t == 0 else "x", color="tab:red", ms=9)
mm = np.array([mars_pos(t) for t in np.linspace(0, T_H, 200)]) / AU
a1.plot(mm[:, 0], mm[:, 1], color="tab:red", lw=2, alpha=0.5, label="Mars during transfer")
a1.plot(0, 0, "o", color="gold", ms=12)
a1.set_aspect("equal")
a1.set_title(f"Heliocentric transfer: {T_H / DAY:.0f} d, Mars leads by {np.degrees(psi0):.1f}° at launch")
a1.set_xlabel("x [AU]"); a1.set_ylabel("y [AU]")
a1.legend(loc="lower left", fontsize=8)

# (2) gravity losses
a2 = ax[0, 1]
a2.semilogx([r["tw"] for r in res], [r["dv"] for r in res], "o-", label="finite thrust (prograde)")
a2.axhline(DV_IMP, color="k", ls="--", label=f"impulsive {DV_IMP:.2f} km/s")
a2.set_xlabel("initial thrust-to-weight  T/(m0 g0)")
a2.set_ylabel("Δv spent from LEO [km/s]")
a2.set_title("Gravity losses in the departure burn")
a2.legend(); a2.grid(alpha=0.3, which="both")

# (3) miss vs error
a3 = ax[1, 0]
a3.plot(errs, miss_arr / 1e6, "o-", label="distance to Mars at nominal arrival time")
a3.plot(errs, miss_min / 1e6, "s--", label="closest approach (±10% of T_H)")
a3.set_xlabel("departure Δv error [m/s]")
a3.set_ylabel("miss distance [10⁶ km]")
a3.set_title("Burn-error sensitivity")
a3.legend(fontsize=8); a3.grid(alpha=0.3)

# (4) separation growth
a4 = ax[1, 1]
a4.loglog(ts / DAY, sep, label=f"numerical separation ({e_demo:+.0f} m/s error)")
a4.loglog(ts / DAY, shear, "--", label="secular shear estimate 1.5·n·δa·t")
a4.loglog(ts / DAY, e_demo / 1000.0 * ts, ":", label="δv·t (early-time)")
a4.set_xlabel("time since departure [days]")
a4.set_ylabel("separation from nominal [km]")
a4.set_title("Shear: error grows with time")
a4.legend(fontsize=8); a4.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("hohmann_sim.png", dpi=140)
print("\nSaved hohmann_sim.png")
