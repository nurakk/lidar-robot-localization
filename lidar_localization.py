#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Experiment 6: Localization of a Mobile Robot Using Simulated LiDAR

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# 1. CREATE THE 2D ENVIRONMENT
# --------------------------------------------------

WIDTH = 10
HEIGHT = 8
RESOLUTION = 0.1

nx = int(WIDTH / RESOLUTION)
ny = int(HEIGHT / RESOLUTION)

grid = np.zeros((ny, nx), dtype=int)

# Room boundaries
grid[0, :] = 1
grid[-1, :] = 1
grid[:, 0] = 1
grid[:, -1] = 1

# Function to add rectangular obstacles
def add_obstacle(x1, y1, x2, y2):
    ix1 = int(x1 / RESOLUTION)
    ix2 = int(x2 / RESOLUTION)
    iy1 = int(y1 / RESOLUTION)
    iy2 = int(y2 / RESOLUTION)

    grid[iy1:iy2, ix1:ix2] = 1

# Add obstacles
add_obstacle(4.5, 1.0, 5.5, 2.5)
add_obstacle(6.5, 4.5, 7.5, 5.5)
add_obstacle(2.5, 5.5, 3.5, 6.5)

# --------------------------------------------------
# 2. DEFINE THE ROBOT POSE
# --------------------------------------------------

actual_pose = np.array([2.0, 2.0, 0.3])

# --------------------------------------------------
# 3. SIMULATE THE LIDAR SENSOR
# --------------------------------------------------

MAX_RANGE = 6.0
RAY_STEP = 0.1

beam_angles = np.linspace(
    0, 2*np.pi, 8, endpoint=False
)

def is_occupied(x, y):

    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return True

    ix = int(x / RESOLUTION)
    iy = int(y / RESOLUTION)

    return grid[iy, ix] == 1


def lidar_scan(pose):

    x, y, theta = pose
    distances = []

    for angle in beam_angles:

        direction = theta + angle
        distance = 0.0

        while distance < MAX_RANGE:

            rx = x + distance * np.cos(direction)
            ry = y + distance * np.sin(direction)

            if is_occupied(rx, ry):
                break

            distance += RAY_STEP

        distances.append(min(distance, MAX_RANGE))

    return np.array(distances)


# Generate noisy LiDAR measurements
def noisy_lidar_scan(pose):

    scan = lidar_scan(pose)

    noise = np.random.normal(
        0, 0.04, len(beam_angles)
    )

    return np.clip(
        scan + noise, 0, MAX_RANGE
    )


# --------------------------------------------------
# 4. VISUALIZE THE INITIAL ENVIRONMENT AND LIDAR
# --------------------------------------------------

scan = lidar_scan(actual_pose)

plt.figure(figsize=(9, 6))

plt.imshow(
    grid,
    origin="lower",
    extent=[0, WIDTH, 0, HEIGHT],
    cmap="Greys"
)

x, y, theta = actual_pose

plt.scatter(
    x, y, color="red",
    s=100, label="Actual Robot"
)

for angle, distance in zip(beam_angles, scan):

    direction = theta + angle

    end_x = x + distance * np.cos(direction)
    end_y = y + distance * np.sin(direction)

    plt.plot([x, end_x], [y, end_y], "g-")

plt.title("Simulated LiDAR Scan")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 5. INITIALIZE PARTICLES
# --------------------------------------------------

NUM_PARTICLES = 200

particles = np.zeros((NUM_PARTICLES, 3))

particles[:, 0] = np.random.normal(
    actual_pose[0], 0.5, NUM_PARTICLES
)

particles[:, 1] = np.random.normal(
    actual_pose[1], 0.5, NUM_PARTICLES
)

particles[:, 2] = np.random.normal(
    actual_pose[2], 0.4, NUM_PARTICLES
)

particles[:, 2] = (
    particles[:, 2] + np.pi
) % (2*np.pi) - np.pi

# Remove particles inside obstacles
valid = np.array([
    not is_occupied(p[0], p[1])
    for p in particles
])

particles = particles[valid]

print("Valid particles:", len(particles))


# --------------------------------------------------
# 6. ROBOT MOTION MODEL
# --------------------------------------------------

def normalize_angle(angle):

    return (angle + np.pi) % (2*np.pi) - np.pi


def move_robot(pose, v, omega, dt):

    x, y, theta = pose

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt

    theta_new = normalize_angle(
        theta + omega * dt
    )

    return np.array([
        x_new, y_new, theta_new
    ])


# --------------------------------------------------
# 7. PARTICLE FILTER MEASUREMENT UPDATE
# --------------------------------------------------

def particle_update(particles, measured_scan):

    weights = []

    measurement_noise = 0.25

    for particle in particles:

        predicted_scan = lidar_scan(particle)

        error = measured_scan - predicted_scan

        log_weight = -0.5 * np.sum(
            (error / measurement_noise) ** 2
        )

        weights.append(log_weight)

    weights = np.array(weights)

    weights = np.exp(
        weights - np.max(weights)
    )

    weights /= np.sum(weights)

    return weights


# --------------------------------------------------
# 8. ESTIMATE ROBOT POSITION
# --------------------------------------------------

def estimate_pose(particles, weights):

    x_est = np.sum(
        particles[:, 0] * weights
    )

    y_est = np.sum(
        particles[:, 1] * weights
    )

    theta_est = np.arctan2(
        np.sum(weights * np.sin(particles[:, 2])),
        np.sum(weights * np.cos(particles[:, 2]))
    )

    return np.array([
        x_est, y_est, theta_est
    ])


# --------------------------------------------------
# 9. PARTICLE RESAMPLING
# --------------------------------------------------

def systematic_resampling(particles, weights):

    n = len(particles)

    positions = (
        np.random.random() + np.arange(n)
    ) / n

    cumulative_sum = np.cumsum(weights)

    indices = np.searchsorted(
        cumulative_sum, positions
    )

    return particles[indices].copy()


# --------------------------------------------------
# 10. RUN THE LOCALIZATION SIMULATION
# --------------------------------------------------

v = 0.15
dt = 1.0
steps = 30

actual_trajectory = []
estimated_trajectory = []

for step in range(steps):

    omega = 0.025 if step < 15 else -0.015

    # Move actual robot
    actual_pose = move_robot(
        actual_pose, v, omega, dt
    )

    actual_trajectory.append(
        actual_pose.copy()
    )

    # Move particles with motion noise
    noisy_v = v + np.random.normal(
        0, 0.025, len(particles)
    )

    noisy_omega = omega + np.random.normal(
        0, 0.02, len(particles)
    )

    for i in range(len(particles)):

        particles[i] = move_robot(
            particles[i],
            noisy_v[i],
            noisy_omega[i],
            dt
        )

    # Generate LiDAR measurement
    measured_scan = noisy_lidar_scan(
        actual_pose
    )

    # Measurement update
    weights = particle_update(
        particles, measured_scan
    )

    # Estimate robot pose
    estimated_pose = estimate_pose(
        particles, weights
    )

    estimated_trajectory.append(
        estimated_pose.copy()
    )

    # Resampling
    particles = systematic_resampling(
        particles, weights
    )

    # Add small noise to maintain diversity
    particles[:, 0] += np.random.normal(
        0, 0.015, len(particles)
    )

    particles[:, 1] += np.random.normal(
        0, 0.015, len(particles)
    )

    particles[:, 2] += np.random.normal(
        0, 0.01, len(particles)
    )

    particles[:, 2] = normalize_angle(
        particles[:, 2]
    )

    # Remove particles outside free space
    for i in range(len(particles)):

        if is_occupied(
            particles[i, 0],
            particles[i, 1]
        ):

            particles[i] = estimated_pose.copy()


# --------------------------------------------------
# 11. COMPARE ACTUAL AND ESTIMATED TRAJECTORIES
# --------------------------------------------------

actual_trajectory = np.array(
    actual_trajectory
)

estimated_trajectory = np.array(
    estimated_trajectory
)

plt.figure(figsize=(10, 7))

plt.imshow(
    grid,
    origin="lower",
    extent=[0, WIDTH, 0, HEIGHT],
    cmap="Greys"
)

plt.plot(
    actual_trajectory[:, 0],
    actual_trajectory[:, 1],
    "r-",
    linewidth=2,
    label="Actual Trajectory"
)

plt.plot(
    estimated_trajectory[:, 0],
    estimated_trajectory[:, 1],
    "b--",
    linewidth=2,
    label="Estimated Trajectory"
)

plt.scatter(
    actual_trajectory[0, 0],
    actual_trajectory[0, 1],
    color="green",
    s=100,
    label="Start"
)

plt.scatter(
    actual_trajectory[-1, 0],
    actual_trajectory[-1, 1],
    color="red",
    marker="*",
    s=180,
    label="End"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.title(
    "LiDAR-Based Mobile Robot Localization"
)

plt.legend()
plt.grid()
plt.show()


# --------------------------------------------------
# 12. CALCULATE LOCALIZATION ERROR
# --------------------------------------------------

position_errors = np.sqrt(
    (actual_trajectory[:, 0] -
     estimated_trajectory[:, 0])**2
    +
    (actual_trajectory[:, 1] -
     estimated_trajectory[:, 1])**2
)

mean_error = np.mean(position_errors)

maximum_error = np.max(position_errors)

orientation_error = abs(
    normalize_angle(
        actual_trajectory[-1, 2] -
        estimated_trajectory[-1, 2]
    )
)

print("\n----- LOCALIZATION RESULTS -----")

print(
    "Mean Position Error:",
    round(mean_error, 3), "m"
)

print(
    "Maximum Position Error:",
    round(maximum_error, 3), "m"
)

print(
    "Final Orientation Error:",
    round(np.degrees(orientation_error), 2),
    "degrees"
)

print("\nLocalization Simulation Completed")


