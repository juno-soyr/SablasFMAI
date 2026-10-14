#!/usr/bin/env python3
"""
Interactive Hohmann transfer visualization with a time slider.

This script plots the Earth-to-Mars Hohmann transfer ellipse and allows the
user to scrub through the transfer using a matplotlib slider. A play button can
advance the trajectory in real time.

If no graphical display is available, the script saves a PNG instead.
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

MU_S = 1.32712440018e11
AU = 1.495978707e8
R1 = 1.0 * AU
R2 = 1.523679 * AU
DAY = 86400.0

n_E = np.sqrt(MU_S / R1**3)
n_M = np.sqrt(MU_S / R2**3)
a_T = 0.5 * (R1 + R2)
T_H = np.pi * np.sqrt(a_T**3 / MU_S)
psi0 = np.pi - n_M * T_H
v_E = np.sqrt(MU_S / R1)
v_M = np.sqrt(MU_S / R2)
v_dep = np.sqrt(MU_S * (2 / R1 - 1 / a_T))
dv1 = v_dep - v_E


def earth_pos(t):
    return R1 * np.array([np.cos(n_E * t), np.sin(n_E * t)])


def mars_pos(t):
    ang = psi0 + n_M * t
    return R2 * np.array([np.cos(ang), np.sin(ang)])


def kepler_rhs(t, s):
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return [vx, vy, -MU_S * x / r3, -MU_S * y / r3]


def propagate(dv, t_end):
    s0 = [R1, 0.0, 0.0, v_E + dv]
    return solve_ivp(
        kepler_rhs,
        [0.0, t_end],
        s0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-6,
        dense_output=True,
    )


sol0 = propagate(dv1, 1.05 * T_H)

# Sample the transfer trajectory for plotting and the slider state.
trajectory_t = np.linspace(0.0, T_H, 1000)
trajectory_xy = np.array([sol0.sol(t)[:2] for t in trajectory_t]) / AU

# Earth and Mars orbit rings for reference.
orbit_t = np.linspace(0, 2 * np.pi, 500)
earth_orbit = np.column_stack([
    R1 * np.cos(orbit_t) / AU,
    R1 * np.sin(orbit_t) / AU,
])
mars_orbit = np.column_stack([
    R2 * np.cos(orbit_t) / AU,
    R2 * np.sin(orbit_t) / AU,
])

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_aspect("equal")
ax.set_xlim(-1.5 * (R2 / AU), 1.5 * (R2 / AU))
ax.set_ylim(-1.5 * (R2 / AU), 1.5 * (R2 / AU))
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_title("Hohmann Transfer: Earth to Mars")
ax.grid(alpha=0.2)

# Static orbital paths.
ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], "--", color="tab:blue", lw=1.5, label="Earth orbit")
ax.plot(mars_orbit[:, 0], mars_orbit[:, 1], "--", color="tab:red", lw=1.5, label="Mars orbit")
ax.plot(0, 0, "o", color="gold", markersize=10, label="Sun")

# Trajectory of the transfer arc.
transfer_line, = ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color="black", lw=2, label="transfer ellipse")

# Spacecraft marker and trail.
spacecraft, = ax.plot([], [], "o", color="tab:green", markersize=8, label="spacecraft")
trail, = ax.plot([], [], color="tab:green", lw=2, alpha=0.7)

# Plot Earth and Mars current positions.
launch_marker, = ax.plot([], [], "o", color="tab:blue", markersize=7)
arrival_marker, = ax.plot([], [], "o", color="tab:red", markersize=7)

legend = ax.legend(loc="lower left", fontsize=9)

# Slider and play button.
ax_slider = plt.axes([0.2, 0.03, 0.55, 0.03])
slider = Slider(ax_slider, "time", 0.0, T_H, valinit=0.0)

play_ax = plt.axes([0.8, 0.03, 0.08, 0.04])
button = Button(play_ax, "Play")

animating = False


def update_state(t_val):
    pos = sol0.sol(t_val)[:2] / AU
    trail_xy = np.array([sol0.sol(tt)[:2] for tt in np.linspace(0.0, t_val, 200)]) / AU
    trail.set_data(trail_xy[:, 0], trail_xy[:, 1])
    spacecraft.set_data([pos[0]], [pos[1]])

    e_pos = earth_pos(t_val) / AU
    m_pos = mars_pos(t_val) / AU
    launch_marker.set_data([e_pos[0]], [e_pos[1]])
    arrival_marker.set_data([m_pos[0]], [m_pos[1]])

    fig.canvas.draw_idle()


def slider_update(val):
    update_state(float(val))


slider.on_changed(slider_update)


def play_toggle(_event):
    global animating
    animating = not animating
    button.label.set_text("Pause" if animating else "Play")


button.on_clicked(play_toggle)


def animate(_):
    if animating:
        t_val = slider.val + 0.5 * DAY
        if t_val >= T_H:
            t_val = 0.0
        slider.set_val(t_val)
    return []


# Use a long enough interval to make the motion feel smooth but not too fast.
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, animate, interval=20, blit=False, cache_frame_data=False)

# Initialize the state at t = 0.
update_state(0.0)

output_path = os.path.join(os.path.dirname(__file__), "hohmann_slider.png")
if "--save" in sys.argv:
    fig.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
else:
    plt.show()
