#!/usr/bin/env python3
"""
Plot the Earth-to-Mars transfer geometry using JPL Horizons ephemerides for two
mission windows:

- 1996 Mars Pathfinder transfer window
- a 2015 Mars transfer window (replace with the exact mission you want if needed)

This script queries the heliocentric positions of Earth and Mars from JPL
Horizons and draws the transfer path in the ecliptic plane.

Install dependencies:
    python -m pip install astroquery matplotlib numpy

Run:
    python plot_mars_horizons_path.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astroquery.jplhorizons import Horizons


MISSION_WINDOWS = {
    "Mars Pathfinder (1996)": {
        "start": "1996-12-01",
        "stop": "1997-07-15",
        "step": "1d",
    },
    "Mars 2015 transfer window": {
        "start": "2015-03-01",
        "stop": "2015-12-31",
        "step": "1d",
    },
}


def get_planet_state(start, stop, step, target_id):
    """Return heliocentric Cartesian state vectors for a target body."""
    obj = Horizons(
        id=target_id,
        location="500@10",
        epochs={"start": start, "stop": stop, "step": step},
    )
    vec = obj.vectors()
    return vec


def to_xy(vec):
    x = np.asarray(vec["x"], dtype=float)
    y = np.asarray(vec["y"], dtype=float)
    return x, y


def plot_mission_window(label, window):
    earth = get_planet_state(window["start"], window["stop"], window["step"], "399")
    mars = get_planet_state(window["start"], window["stop"], window["step"], "499")

    x_e, y_e = to_xy(earth)
    x_m, y_m = to_xy(mars)

    ax = plt.gca()
    ax.plot(x_e, y_e, color="tab:blue", lw=1.5, alpha=0.8, label=f"Earth ({label})")
    ax.plot(x_m, y_m, color="tab:red", lw=1.5, alpha=0.8, label=f"Mars ({label})")

    # Draw a simple transfer arc between Earth and Mars at their respective mission dates.
    # This is a first-order visual approximation to the actual Earth-Mars transfer geometry.
    t0 = np.argmin(np.abs(np.asarray(earth["datetime_jd"]) - np.asarray(earth["datetime_jd"])[0]))
    t1 = np.argmin(np.abs(np.asarray(mars["datetime_jd"]) - np.asarray(mars["datetime_jd"])[-1]))
    x_transfer = np.array([x_e[t0], x_m[t1]])
    y_transfer = np.array([y_e[t0], y_m[t1]])
    ax.plot(x_transfer, y_transfer, color="k", ls="--", lw=1.5, alpha=0.8)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 8))

    for label, window in MISSION_WINDOWS.items():
        plot_mission_window(label, window)

    ax.set_aspect("equal")
    ax.set_xlabel("Heliocentric x [AU]")
    ax.set_ylabel("Heliocentric y [AU]")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.set_title("Earth-Mars transfer geometry from JPL Horizons")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig("mars_horizons_transfers.png", dpi=180)
    print("Saved mars_horizons_transfers.png")
    plt.show()
