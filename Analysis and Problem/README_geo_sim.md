# `geo_sim.py`: geostationary satellite dynamics

A satellite in geostationary orbit (GEO) stays above one point on Earth, and the model is
built around that fact. The goal is the cheapest model that still behaves like the real
system: **one 6×6 matrix–vector multiply per time step**. It is meant as the plant for
safe-control / station-keeping experiments (e.g. SABLAS).

## 1. Where the satellite sits

A circular orbit whose period equals one sidereal day (Earth's rotation rate
$\omega_E = 7.2921\times10^{-5}$ rad/s) has its radius fixed by $n^2 a = \mu / a^2$:

$$
a_{GEO} = \left(\frac{\mu}{\omega_E^2}\right)^{1/3} \approx 42\,164 \text{ km}
$$

If the orbit is also equatorial, the satellite never moves relative to the ground. Each
equatorial longitude is one possible **slot**.

**"Over a specified location":** a GEO satellite can only be directly overhead on the
equator. For a site at (lat, lon), `slot_for_site` puts the slot at the site's longitude.
That gives the highest elevation in the site's sky. It also warns if GEO is below the horizon,
which happens above roughly 81° latitude. The site's latitude changes only the look angles.

## 2. Simplification: linearise around the slot

Work in a frame that rotates with Earth, centred on the slot, with axes

| state | meaning | direction |
|---|---|---|
| $x$ | radial offset | up, away from Earth |
| $y$ | along-track offset | east |
| $z$ | cross-track offset | north |

In this frame the satellite feels gravity, the Coriolis force and the centrifugal force. Their
sum is exactly zero at the slot. For offsets that are small compared with $a_{GEO}$, a
first-order expansion gives the **Clohessy–Wiltshire (Hill) equations**, with $n = \omega_E$:

$$
\begin{aligned}
\ddot x &= 3n^2 x + 2n\dot y + u_x + d_x \\
\ddot y &= -2n\dot x + u_y + d_y \\
\ddot z &= -n^2 z + u_z + d_z
\end{aligned}
$$

Here $u$ is the thruster acceleration (your control) and $d$ is the disturbance acceleration
(section 4).

Where each term comes from:
- $3n^2 x$: going up, gravity weakens faster than the centrifugal force changes, so the
  radial balance tips outward.
- $\pm 2n\dot{(\cdot)}$: Coriolis force, which couples radial and east–west motion.
- $-n^2 z$: moving north of the equator, gravity pulls you back toward the equatorial plane.

The system is **linear and time-invariant**: $\dot{\mathbf x} = A\mathbf x + B(\mathbf u + \mathbf d)$,
with $\mathbf x = [x, y, z, \dot x, \dot y, \dot z]$. This is the step that makes the model so cheap.

### What the free motion looks like

The eigen-structure of $A$ explains everything the satellite does when left alone:
- **Radial/east–west plane:** a 2:1 ellipse traced once per day (a slightly eccentric orbit),
  plus a possible steady drift. Being $\delta a$ too high gives a westward drift
  $\dot y = -\tfrac32 n\,\delta a$, because a higher orbit is slower.
- **North–south:** a pure oscillation at $n$. A small inclination $i$ makes the sub-satellite
  point swing north and south by $\pm i$ once per day.

## 3. Discretisation: done once, exactly

With $\mathbf u + \mathbf d$ held constant over a step $\Delta t$ (zero-order hold), the exact
solution is $\mathbf x_{k+1} = A_d\mathbf x_k + B_d(\mathbf u_k + \mathbf d_k)$. One matrix
exponential, computed at construction, gives both matrices:

$$
\exp\!\left(\begin{bmatrix} A & B \\ 0 & 0\end{bmatrix}\Delta t\right)
= \begin{bmatrix} A_d & B_d \\ 0 & I\end{bmatrix}
$$

After that, stepping costs one mat-vec and never needs an ODE solver. Because the map is
exact, any $\Delta t$ (seconds to hours) is stable and accurate. `step()` also accepts a batch
of states of shape (N, 6).

The disturbance is sampled at the **middle** of each step. That is exact for the constant
term and second-order accurate for the sinusoidal one.

## 4. Disturbances (both optional, closed-form)

Real GEO satellites drift mainly because of two effects. Each is reduced to one formula.

### 4a. Elliptical equator → east–west drift (`j22=True`)

Earth's equator is slightly elliptical (degree-2, order-2 gravity:
$J_{22} = \sqrt{C_{22}^2 + S_{22}^2} \approx 1.815\times10^{-6}$, long axis at
$\lambda_{22} = \tfrac12\operatorname{atan2}(S_{22}, C_{22}) \approx -14.9°$). On the equator at
radius $a$ this adds an east–west acceleration:

$$
d_y = -6\,n^2 a\left(\frac{R_E}{a}\right)^2 J_{22}\,\sin 2(\lambda_0 - \lambda_{22})
$$

Over a slot's control box, $d_y$ barely changes, so it is treated as **constant at the slot
longitude** $\lambda_0$. An eastward push raises the orbit, which slows the satellite, so the
longitude accelerates the opposite way:

$$
\ddot\lambda = -\frac{3\,d_y}{a} = 18\,n^2\left(\frac{R_E}{a}\right)^2 J_{22}\,\sin 2(\lambda_0 - \lambda_{22})
$$

The maximum is about $1.7\times10^{-3}$ deg/day². Stable points are at about 75°E and 105°W;
unstable points are at about 15°W and 165°E.

### 4b. Moon and Sun → inclination growth (`lunisolar=True`)

The Moon and Sun tilt the orbit plane, so inclination grows about 0.75–0.95°/yr. Gauss's
equation for a circular orbit gives $\frac{di}{dt} = \frac{d_z\cos u}{n a}$, where $u$ is the
angle along the orbit. A north–south acceleration at the orbit rate

$$
d_z = D\cos(nt + \varphi)
$$

averages to $\frac{di}{dt} = \frac{D}{2na}$. So $D = 2na\cdot(0.85°/\text{yr})$ reproduces the
observed growth. In the linear model this shows up as resonant forcing of the $z$ oscillator,
so the daily north–south swing grows linearly in time. $\varphi$ (`phase`) sets the direction
of the growth.

## 5. Back to the ground

- **Sub-satellite point:** using small angles, $\text{lat} = z/a$ and $\text{lon} = \lambda_0 + y/a$ (`latlon`).
- **Earth-fixed position** (`ecef`): spherical coordinates with radius $a + x$.
- **Look angles from the site** (`look_angles`): take the satellite-minus-site vector and
  project it onto the site's local east, north and up axes:
  $\text{az} = \operatorname{atan2}(E, N)$, $\text{el} = \operatorname{atan2}(U, \sqrt{E^2+N^2})$.
  This assumes a spherical Earth.

## 6. Black-box plant (`BlackBoxGEO`) and system identification

SABLAS assumes the real system is a **black box**: you can query it, but you don't know its
equations. You only have an approximate **nominal** model. Here the nominal model is
`GEOSlot` (sections 2–4). The black box is `BlackBoxGEO`, a full nonlinear simulator with the
same state and control. It is used only through calls:

| call | use |
|---|---|
| `reset(x0, t0)` / `step(u)` | stateful rollout, like an environment |
| `query(X, U, t)` | stateless batch $x_{k+1} = F(x_k, u_k, t)$, for data collection |

### What's inside, and what the nominal model gets wrong

Inside, it integrates the full nonlinear motion in the Earth-fixed frame with RK4, including:
- **Gravity:** point mass, plus $J_2$ (Earth's oblateness, absent from the nominal model), plus
  $C_{22}/S_{22}$ as the full field rather than a constant frozen at the slot.
- **Sun and Moon third-body pull** from circular analytic ephemerides: Sun on the ecliptic
  (23.44° tilt); Moon at 27.3 days, inclined 5.145° to the ecliptic. The nominal model has only
  the one-sinusoid inclination pump.
- **Solar radiation pressure** (cannonball model), absent from the nominal model.
- **Thruster errors:** gain error and misalignment, so the actual acceleration is
  $g\,R_{thr}\,u$. The nominal model assumes the thruster is perfect.
- **Optional process noise** (`noise_std`).

**Hidden parameters**, drawn per instance from `seed`: Sun/Moon phases, Moon node,
$C_rA/m \in [0.012, 0.045]$ m²/kg, thruster gain $g \in [0.95, 1.05]$, and misalignment
$\le 1°$. They are stored as private `_attributes`; the controller and learner are not
supposed to read them.

RK4 is run with ≤ 600 s substeps. Results match 60 s substeps to 0.1 m, because GEO motion is
very smooth.

### System identification (`fit_linear_model`)

Because the nominal structure is linear, the SABLAS sysid step reduces to one least-squares
solve on black-box samples. Samples use random states in a ±20 km / ±1 m/s box, random
controls, and random times over a year, and the fit is

$$
x_{k+1} = A_d x_k + B_d u_k + c
$$

- **Grey box (recommended, `Ad_known=GEOSlot.Ad`):** keep $A_d$ from the physics (the CW terms
  are exact) and fit only $B_d$ (this recovers the thruster gain and misalignment) and $c$
  (the constant forces: $J_2$, $J_{22}$, the mean Sun/Moon pull).
- **Free fit (`Ad_known=None`):** also learns $A_d$. The unmodelled time-varying forcing leaks
  into $A_d$ and pushes the marginally stable east–west drift mode to $|\lambda| > 1$, so long
  rollouts diverge. Don't use it for multi-day predictions.

The leftover mismatch is mainly the **time-varying** Sun/Moon pull. No time-invariant linear
model can represent it; it's the part a robust or learned controller (e.g. SABLAS's
neural barrier certificate) has to absorb.

## 7. Validation (`python geo_sim.py`)

| check | result |
|---|---|
| Step model vs. integrating the same equations, 2 days | ≤ 2 mm |
| Linear model vs. full nonlinear motion in the rotating frame (point-mass + $C_{22}/S_{22}$), 1 day, starting 2 km / 0.1 m/s off | 25 m radial, 160 m east–west, 3 m north–south |
| Longitude drift $\ddot\lambda$ vs. formula in 4a, at −30°, 75°, 165° | matches; ≈ 0 at the stable point |
| Inclination after 1 year | 0.85° (target 0.85°) |
| One-step error vs black box ($\Delta t$ = 600 s), RMS velocity | nominal 3.3 mm/s, grey-box fit 2.2 mm/s |
| Thruster gain recovered by the fit | 0.9913 (hidden true value 0.9923) |
| 10-day free drift from rest, error vs black box (black box moved 85 km) | nominal 116 km, grey-box fit 48 km |

For scale, a typical station-keeping box is ±0.05°, about ±37 km. Linearisation errors of
around 100 m per day are negligible at that size.

**`geo_demo.png`** (site in Amsterdam) compares free drift with a simple
proportional-derivative controller plus feed-forward of $d_y$, run on the nominal model. It
holds the slot for about 58 m/s per year, a realistic GEO station-keeping budget dominated by
north–south control.

The proportional gain must exceed $3n^2 \approx 1.6\times10^{-8}$ s⁻²; otherwise the radial
instability of the CW equations wins. The demo uses $K_p = 10^{-7}$, $K_d = 5\times10^{-4}$,
with a maximum closed-loop $|\lambda| = 0.90$ at $\Delta t$ = 600 s.

**`geo_blackbox.png`** shows 30-day free drift of the black box against the nominal and fitted
models. It also runs the same controller on the black box: the slot is held within 0.14 km,
but it costs 24.6 m/s instead of 4.8 m/s. The extra fuel goes mostly to fighting the
unmodelled $J_2$ pull. A satellite parked about 0.5 km higher, at its true equilibrium radius,
wouldn't need it. That is a concrete example of what model mismatch costs.

## 8. Limits of the nominal model: what was left out to keep it cheap

- **Large offsets.** Valid for tens of km and fractions of a degree. It is not valid for
  relocating between slots or for large inclinations: the figure-8 ground track of an inclined
  geosynchronous orbit is a second-order effect the linear model doesn't have.
- **East–west push stays constant.** $d_y$ does not change as the satellite drifts. Over long
  free drifts of several degrees, it should be re-evaluated at the current longitude.
- **Moon/Sun simplified to one wave.** Only the secular inclination pump is modelled. The
  real perturbation varies over 14 days, half a year and 18.6 years.
- **Not modelled at all:** solar radiation pressure (eccentricity drift), Earth oblateness
  ($J_2$, which shifts the equilibrium radius by about +0.5 km), thruster mass loss and
  attitude. The black box (section 6) includes all of these except mass loss and attitude.

## 9. Usage

Run with the project venv (the system Python has no scipy):

```fish
source .venv/bin/activate.fish      # from the repo root
cd "Analysis and Problem"
python geo_sim.py                   # validation + geo_demo.png + geo_blackbox.png (~15 s)
```

```python
from geo_sim import slot_for_site, look_angles
import numpy as np

slot = slot_for_site(40.4, -3.7, dt=60.0)            # site lat, lon; 60 s steps
x0 = np.array([1e3, 0, 0, 0, 0, 0])                  # 1 km above the slot
t, X, U = slot.rollout(x0, 1440, policy=None)        # 1 day; policy(t, x) -> u [m/s^2]
lat, lon = slot.latlon(X)
az, el = look_angles(slot.ecef(X), 40.4, -3.7)

x1 = slot.step(x0, u=np.zeros(3), t=0.0)             # single step (or batch of shape (N, 6))
Ad, Bd = slot.Ad, slot.Bd                            # matrices for MPC / CBF / learning

# black box: the "real" satellite, query only
from geo_sim import BlackBoxGEO, fit_linear_model, rollout_linear
bb = BlackBoxGEO(-3.7, dt=60.0, seed=0)              # seed fixes the hidden parameters
x = bb.reset()
x = bb.step(np.array([0, 1e-5, 0]))                  # stateful
Xn = bb.query(X[:-1], np.zeros((1440, 3)), t[:-1])       # stateless batch (t: scalar or per-sample)

Ad_f, Bd_f, c_f = fit_linear_model(bb, slot.Ad)      # grey-box sysid from samples
Xf = rollout_linear(Ad_f, Bd_f, c_f, np.zeros(6), 1440)
```
