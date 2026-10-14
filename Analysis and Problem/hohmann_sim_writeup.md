# Hohmann transfer simulation: how it works

This script models the classic two-body, coplanar Earth-to-Mars transfer using a Hohmann ellipse. The goal is to move a spacecraft from Earth's circular orbit around the Sun to Mars's circular orbit with the minimum propellant for a simple impulsive transfer.

## 1. Physical model

The spacecraft is treated as a point mass moving under the Sun's gravity only:

$$
\ddot{\mathbf{r}} = -\frac{\mu_\odot}{\|\mathbf{r}\|^3}\mathbf{r}
$$

where:

- $\mu_\odot = GM_\odot$ is the Sun's gravitational parameter
- $\mathbf{r}$ is the spacecraft position vector in heliocentric coordinates
- the planets are modeled as circular orbits, not as full dynamical bodies with epicycles or perturbations

This is a Keplerian approximation, which is appropriate for a first-order mission design study.

## 2. Circular Earth and Mars orbits

The script assumes:

- Earth orbit radius: $R_1$
- Mars orbit radius: $R_2$
- both planets move on circular, coplanar paths around the Sun

The orbital angular rates are:

$$
n_E = \sqrt{\frac{\mu_\odot}{R_1^3}}, \qquad n_M = \sqrt{\frac{\mu_\odot}{R_2^3}}
$$

The transfer orbit is the ellipse tangent to both circular orbits at its apsides, so its semi-major axis is:

$$
a_T = \frac{R_1 + R_2}{2}
$$

This is the defining feature of a Hohmann transfer: the spacecraft departs Earth on an ellipse whose apoapsis is at Mars's orbit radius.

## 3. Time of flight and phasing

A Keplerian transfer from radius $R_1$ to $R_2$ takes half the period of the transfer ellipse:

$$
T_H = \pi \sqrt{\frac{a_T^3}{\mu_\odot}}
$$

This is the time required to travel from perihelion to aphelion on the transfer ellipse.

Because Mars is moving while the spacecraft is in flight, the launch must happen when the planet is ahead at the correct lead angle. The script computes the Mars lead angle as:

$$
\psi_0 = \pi - n_M T_H
$$

This tells you how much ahead of Earth the spacecraft must start relative to Mars so the spacecraft and Mars meet at the target orbit.

## 4. Launch and arrival delta-v

The spacecraft starts in Earth's circular orbit with speed:

$$
v_E = \sqrt{\frac{\mu_\odot}{R_1}}
$$

To leave on the transfer ellipse, it must increase speed to the transfer orbit's perihelion speed:

$$
v_{\text{dep}} = \sqrt{\mu_\odot\left(\frac{2}{R_1} - \frac{1}{a_T}\right)}
$$

The required departure impulse is:

$$
\Delta v_1 = v_{\text{dep}} - v_E
$$

At Mars arrival, the spacecraft is moving too fast for Mars's circular orbit. The target orbit speed there is:

$$
v_M = \sqrt{\frac{\mu_\odot}{R_2}}
$$

and the arrival speed on the transfer orbit is:

$$
v_{\text{arr}} = \sqrt{\mu_\odot\left(\frac{2}{R_2} - \frac{1}{a_T}\right)}
$$

The second burn is:

$$
\Delta v_2 = v_M - v_{\text{arr}}
$$

These are the classic impulsive Hohmann burns.

## 5. What the code does numerically

The script integrates the spacecraft trajectory using `solve_ivp` from SciPy with a high-order integrator (`DOP853`). That allows the equations of motion to be propagated directly from the initial condition rather than using only analytic formulas.

The spacecraft is initialized at Earth with the tangential velocity plus the departure burn:

```python
s0 = [R1, 0.0, 0.0, v_E + dv]
```

Then `kepler_rhs` integrates the two-body equations:

```python
def kepler_rhs(t, s):
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return [vx, vy, -MU_S * x / r3, -MU_S * y / r3]
```

This produces the transfer orbit, and the script evaluates the miss distance relative to Mars at the nominal arrival time and the closest approach nearby to show how close the impulsive solution is to a perfect intercept.

## 6. Part B: finite-thrust departure from LEO

The second part of the simulation is more realistic than the pure impulsive model. It starts from a low Earth orbit (LEO) and models a finite-thrust acceleration with variable mass.

The state includes:

- position $(x, y)$
- velocity $(v_x, v_y)$
- mass $m$
- heading angle $\theta$

The propulsion is assumed to be aligned with the velocity vector, and the mass decreases according to the rocket equation. This shows how the real burn is not instantaneous: gravity losses and finite burn duration cause the required delta-v to be larger than the ideal impulsive case.

## 7. Part C: burn error sensitivity

The script perturbs the departure burn by a small amount, such as $\pm 1$ m/s or $\pm 10$ m/s, and then propagates the trajectory again. Even a tiny burn mismatch can produce a large arrival miss because the transfer geometry amplifies the error over time. This is the “shear” effect: small, time-varying trajectory differences can grow substantially as the spacecraft progresses along the transfer arc.

This is why mission design includes navigation margins and correction burns.

## 8. What the plots show

The script generates a set of figures showing:

- the heliocentric transfer trajectory
- Earth and Mars orbits
- finite-thrust loss in LEO departure
- arrival miss distance versus departure burn error
- separation growth over time for a small burn mismatch

These visuals summarize the main ideas behind Hohmann transfers: the transfer ellipse, the required Delta-v, and the sensitivity of the final arrival to small errors.

## Summary

The simulation captures the central idea of orbital mechanics in a transfer design:

1. choose a transfer ellipse between the two circular orbits,
2. apply a tangential departure burn,
3. coast for half the transfer period,
4. apply a second tangential burn at arrival,
5. check the phasing and sensitivity to small errors.

That is the essence of the Hohmann transfer, and this script makes it visible numerically and graphically.
