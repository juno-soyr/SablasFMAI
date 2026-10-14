from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import get_body, GCRS
import astropy.units as u


# Save data file in `data` directory
FILE_PATH = "data/Artemis_II_OEM_2026_04_06_Pre"
OUTPUT_DIR = "figures"

# Create a Path object for the figures directory
figures_dir = Path("figures")

# Check if it exists, create it if not
if not figures_dir.exists():
    figures_dir.mkdir()
    print("Created 'figures' directory.")
else:
    pass


# Load Artemis II data
times_str = []
x_sc, y_sc, z_sc, vx_sc, vy_sc, vz_sc = [], [], [], [], [], []

with open(FILE_PATH) as f:
    for line in f:
        if line.startswith("202"):
            parts = line.split()
            times_str.append(parts[0])
            x_sc.append(float(parts[1]))
            y_sc.append(float(parts[2]))
            z_sc.append(float(parts[3]))
            vx_sc.append(float(parts[4]))
            vy_sc.append(float(parts[5]))
            vz_sc.append(float(parts[6]))

x_sc = np.array(x_sc)
y_sc = np.array(y_sc)
z_sc = np.array(z_sc)

vx_sc = np.array(vx_sc)
vy_sc = np.array(vy_sc)
vz_sc = np.array(vz_sc)

# velocity magnitude
v = np.sqrt(vx_sc**2 + vy_sc**2 + vz_sc**2)

# Convert times to Astropy Time
times = Time(times_str)

# Moon position in Earth-centered frame
moon_coords = get_body("moon", times)

# Transform to GCRS (Earth-centered inertial, close to EME2000)
moon_gcrs = moon_coords.transform_to(GCRS(obstime=times))

x_moon = moon_gcrs.cartesian.x.to(u.km).value
y_moon = moon_gcrs.cartesian.y.to(u.km).value
z_moon = moon_gcrs.cartesian.z.to(u.km).value

# Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(x_sc, y_sc, z_sc, c=v, cmap='viridis', s=2, label='Orion Spacecraft')

cbar = fig.colorbar(sc, ax=ax, pad=0.2, shrink=0.6)
cbar.set_label("Velocity (km/s)")

# Moon
ax.plot(x_moon, y_moon, z_moon, label="Moon")

ax.set_xlabel("X (km)", labelpad=15)
ax.set_ylabel("Y (km)", labelpad=15)
ax.set_zlabel("Z (km)", labelpad=15)
ax.legend()


plt.title("Trajectory with Moon Orbit")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectory_w_moon.png")
plt.close()



# Distance from Earth (km)
r = np.sqrt(x_sc**2 + y_sc**2 + z_sc**2)

# Indices
perigee_idx = np.argmin(r)
apogee_idx = np.argmax(r)

# Values
perigee_distance = r[perigee_idx]
apogee_distance = r[apogee_idx]

# Times
perigee_time = times_str[perigee_idx]
apogee_time = times_str[apogee_idx]

print("Perigee:")
print(f"  Time: {perigee_time}")
print(f"  Distance from Earth center: {perigee_distance:.2f} km")

print("\nApogee (furthest point):")
print(f"  Time: {apogee_time}")
print(f"  Distance from Earth center: {apogee_distance:.2f} km")

R_earth = 6371

print("\nPerigee altitude:", perigee_distance - R_earth, "km")
print("Apogee altitude:", apogee_distance - R_earth, "km")

R_moon = 1737.4  # km

# Spacecraft position at apogee
sc_apogee = np.array([
    x_sc[apogee_idx],
    y_sc[apogee_idx],
    z_sc[apogee_idx]
])

# Moon position at same time
moon_apogee = np.array([
    x_moon[apogee_idx],
    y_moon[apogee_idx],
    z_moon[apogee_idx]
])

# Distance to Moon center
distance_center = np.linalg.norm(sc_apogee - moon_apogee)

# Distance to Moon surface
distance_surface = distance_center - R_moon
