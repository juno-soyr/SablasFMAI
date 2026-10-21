
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from dataclasses import dataclass

# ============================================================
# 1. CONSTANTS
# ============================================================

MU_EARTH = 3.986004418e14   # m^3/s^2
MU_MOON = 4.9048695e12      # m^3/s^2

R_EARTH = 6_371_000.0       # m
R_MOON = 1_737_400.0        # m
D_EM = 384_400_000.0        # mean Earth-Moon distance, m

# Circular Earth-Moon angular velocity
N_MOON = np.sqrt((MU_EARTH + MU_MOON) / D_EM**3)


# ============================================================
# 2. EARTH AND MOON POSITIONS IN AN INERTIAL 3D FRAME
# ============================================================

def body_states(t):
    """Return (position, velocity) for Earth and Moon."""

    a_e = D_EM * MU_MOON / (MU_EARTH + MU_MOON)
    a_m = D_EM * MU_EARTH / (MU_EARTH + MU_MOON)

    theta = N_MOON * t

    r_e = np.array([
        -a_e * np.cos(theta),
        -a_e * np.sin(theta),
        0.0
    ])

    v_e = np.array([
        a_e * N_MOON * np.sin(theta),
        -a_e * N_MOON * np.cos(theta),
        0.0
    ])

    r_m = np.array([
        a_m * np.cos(theta),
        a_m * np.sin(theta),
        0.0
    ])

    v_m = np.array([
        -a_m * N_MOON * np.sin(theta),
        a_m * N_MOON * np.cos(theta),
        0.0
    ])

    return (r_e, v_e), (r_m, v_m)


# ============================================================
# 3. INTERACTIVE EARTH-MOON POSITIONS
# ============================================================

simulation_duration = 4 * 86400.0
plot_times = np.linspace(0.0, simulation_duration, 1000)
earth_positions = np.asarray([body_states(t)[0][0] for t in plot_times])
moon_positions = np.asarray([body_states(t)[1][0] for t in plot_times])

scale = 1e6
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(bottom=0.18)

ax.plot(
    earth_positions[:, 0] / scale,
    earth_positions[:, 1] / scale,
    earth_positions[:, 2] / scale,
    color="tab:blue",
    label="Earth trajectory"
)
ax.plot(
    moon_positions[:, 0] / scale,
    moon_positions[:, 1] / scale,
    moon_positions[:, 2] / scale,
    color="tab:gray",
    label="Moon trajectory"
)

earth_marker, = ax.plot(
    [earth_positions[0, 0] / scale],
    [earth_positions[0, 1] / scale],
    [earth_positions[0, 2] / scale],
    "o",
    color="tab:blue",
    markersize=10,
    label="Earth"
)
moon_marker, = ax.plot(
    [moon_positions[0, 0] / scale],
    [moon_positions[0, 1] / scale],
    [moon_positions[0, 2] / scale],
    "o",
    color="tab:orange",
    markersize=7,
    label="Moon"
)
line_of_centers, = ax.plot(
    [earth_positions[0, 0] / scale, moon_positions[0, 0] / scale],
    [earth_positions[0, 1] / scale, moon_positions[0, 1] / scale],
    [earth_positions[0, 2] / scale, moon_positions[0, 2] / scale],
    "--",
    color="0.55",
    linewidth=1
)

ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("X (million m)")
ax.set_ylabel("Y (million m)")
ax.set_zlabel("Z (million m)")
# Keep a visible negative-side buffer behind Earth (−200,000 km).
ax.set_xlim(-200.0, 450.0)
ax.set_ylim(-200.0, 450.0)
ax.set_zlim(-200.0, 200.0)
ax.set_title("Earth-Moon Motion in 3D: Day 0.00")
ax.grid(True, alpha=0.3)
ax.legend()

slider_axis = fig.add_axes([0.15, 0.07, 0.7, 0.035])
time_slider = Slider(
    slider_axis,
    "Time (days)",
    0.0,
    4.0,
    valinit=0.0,
    valstep=4.0 / (len(plot_times) - 1)
)


def update_plot(day):
    time_index = int(round(day / 4.0 * (len(plot_times) - 1)))
    earth_position = earth_positions[time_index] / scale
    moon_position = moon_positions[time_index] / scale

    earth_marker.set_data_3d(
        [earth_position[0]],
        [earth_position[1]],
        [earth_position[2]]
    )
    moon_marker.set_data_3d(
        [moon_position[0]],
        [moon_position[1]],
        [moon_position[2]]
    )
    line_of_centers.set_data_3d(
        [earth_position[0], moon_position[0]],
        [earth_position[1], moon_position[1]],
        [earth_position[2], moon_position[2]]
    )
    ax.set_title(f"Earth-Moon Motion in 3D: Day {day:.2f}")
    fig.canvas.draw_idle()


time_slider.on_changed(update_plot)
plt.show()
