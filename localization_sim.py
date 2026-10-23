import pybullet as p
import numpy as np
import pybullet_data
import time
import matplotlib.pyplot as plt


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=18, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])

p.setGravity(0, 0, -9.81)  # gravity in z direction
planeId = p.loadURDF("plane.urdf")  # loads ground plane

# Starting position and orientation
SP = [-8.0, -8.0, 0.1]
SO = p.getQuaternionFromEuler([0, 0, 0])

# Loads robot
robotId = p.loadURDF("husky/husky.urdf", SP, SO)
print("Husky robot spawned ")

# GOAL and Enviroment

G_X, G_Y = -2.5, 5.0 #Goal
GOAL_THRESHOLD = 0.4

NOISE_PERCENT = 25  # odometry and lidar noise level, 0 to 100
noise = np.clip(NOISE_PERCENT, 0, 100) / 100.0
np.random.seed(42)  # repeatable runs

cube_positions = [
    [-6.0, -6.5, 0.25],
    [-3.0, -6.0, 0.25],
    [0.0, -8.0, 0.25],
    [6.0, -6.0, 0.25],
    [7.5, -1.5, 0.25],
    [-7.0, -1.0, 0.25],
    [-2.5, 0.0, 0.25],
    [2.0, -1.5, 0.25],
    [1.5, 4.0, 0.25],
    [-7.5, 7.0, 0.25],
    [7.0, 3.0, 0.25],
    [-2.0, 7.5, 0.25]
] #cube locations

for pos in cube_positions:
    p.loadURDF("cube.urdf", pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5, useFixedBase=True)

WALL_H = 0.4  # wall half height
wall_specs = [
    [10.0, 0.0, 0.1, 10.0],
    [-10.0, 0.0, 0.1, 10.0],
    [0.0, 10.0, 10.0, 0.1],
    [0.0, -10.0, 10.0, 0.1],  # arena boundary
    [-3.5, -4.0, 3.5, 0.1],
    [5.0, -1.0, 0.1, 3.0],
    [0.0, 2.0, 3.0, 0.1],
    [-5.0, 4.0, 0.1, 3.0],
    [4.0, 6.0, 3.0, 0.1],
    [2.0, -7.5, 0.1, 2.5]
] #walls as [x, y, half_x, half_y]

for wx, wy, hx, hy in wall_specs:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_H])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_H], rgbaColor=[0.6, 0.6, 0.6, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[wx, wy, WALL_H])  # mass 0 -> non moveable

for (mx, my), colour in [((SP[0], SP[1]), [0, 1, 0, 1]), ((G_X, G_Y), [1, 0, 0, 1])]:
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.6, length=0.01, rgbaColor=colour)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[mx, my, 0.005])  # start green, goal red (visual only)

print(f"Goal set at Target Coordinates: ({G_X}, {G_Y}) | Noise: {NOISE_PERCENT}%")

# Path tracking arrays
ground_truth_path = []
estimated_path = []
heading_errors = []

# Initialize dead-reckoning pose variables
est_x, est_y, est_theta = SP[0], SP[1], 0.0

# Systematic odometry errors, fixed for the whole run
v_bias = np.random.normal(0, 0.05 * noise)  # wheel scale error
w_bias = np.random.normal(0, 0.05 * noise)  # yaw rate drift rad/s

# Simulated 2D lidar
NUM_RAYS = 72
LIDAR_RANGE = 4.0  # m
LIDAR_Z = 0.25     # scan height
ray_angles = np.linspace(-np.pi, np.pi, NUM_RAYS, endpoint=False)
ray_start = np.minimum(0.55 / np.maximum(np.abs(np.cos(ray_angles)), 1e-6), 0.4 / np.maximum(np.abs(np.sin(ray_angles)), 1e-6))  # rays start just outside robot body

# Obstacle avoidance
SAFE_DIST = 0.9        # m, distance kept from obstacles
D0, K_REP = 1.2, 0.05  # repulsion range and strength
FOLLOW_TIMEOUT = 25.0  # s
mode = "GOAL"
d_best = np.inf
follow_side, follow_start = 1, 0

# Husky wheel joints: 2=front_left, 3=front_right, 4=rear_left, 5=rear_right
left_wheels = [2, 4]
right_wheels = [3, 5]

MAX_WHEEL_VEL = 10.0  # rad/s
MAX_FORCE = 50.0      # N*m per wheel

dt = 1.0 / 240.0  # Physics time step length
MAX_STEPS = 240 * 180  # timeout if robot gets stuck
step_count = 0
reached_goal = False

print("Starting Goal Attraction Steering Simulation...\n")

while p.isConnected() and not reached_goal and step_count < MAX_STEPS:
    p.stepSimulation()
    time.sleep(dt)
    step_count += 1

    # Ground truth - used only for evaluation
    pos, orient = p.getBasePositionAndOrientation(robotId)
    t_x, t_y = pos[0], pos[1]
    true_theta = p.getEulerFromQuaternion(orient)[2]

    # Simulated odometry reading (forward speed, yaw rate) with noise
    linear_vel, angular_vel = p.getBaseVelocity(robotId)
    v = linear_vel[0] * np.cos(true_theta) + linear_vel[1] * np.sin(true_theta)
    omega_z = angular_vel[2]

    v_meas = v * (1 + v_bias + np.random.normal(0, noise))
    w_meas = omega_z + w_bias + np.random.normal(0, noise * (abs(omega_z) + 0.1))

    # Dead reckoning
    est_x += v_meas * np.cos(est_theta) * dt
    est_y += v_meas * np.sin(est_theta) * dt
    est_theta += w_meas * dt

    ground_truth_path.append((t_x, t_y))
    estimated_path.append((est_x, est_y))
    heading_errors.append(np.arctan2(np.sin(true_theta - est_theta), np.cos(true_theta - est_theta)))

    # Lidar scan - rays move with the robot body, readings are ranges relative to the robot
    ray_world = true_theta + ray_angles
    ray_from = np.column_stack([t_x + ray_start * np.cos(ray_world), t_y + ray_start * np.sin(ray_world), np.full(NUM_RAYS, LIDAR_Z)])
    ray_to = np.column_stack([t_x + LIDAR_RANGE * np.cos(ray_world), t_y + LIDAR_RANGE * np.sin(ray_world), np.full(NUM_RAYS, LIDAR_Z)])
    hits = p.rayTestBatch(ray_from.tolist(), ray_to.tolist())
    ranges = ray_start + np.array([h[2] for h in hits]) * (LIDAR_RANGE - ray_start)
    ranges = np.clip(ranges + np.random.normal(0, 0.05 * noise, NUM_RAYS), 0.05, LIDAR_RANGE)

    # Robot navigates on its own estimate
    dx_attract = G_X - est_x
    dy_attract = G_Y - est_y
    distance_to_goal = np.sqrt(dx_attract**2 + dy_attract**2)

    if step_count % 40 == 0 or step_count == 1:
        print(f"Step {step_count:04d} | {mode:6s} | True: ({t_x:.2f}, {t_y:.2f}) | Est: ({est_x:.2f}, {est_y:.2f}) | Distance to Goal: {distance_to_goal:.2f} m")# Print live progress every 40 steps

    if distance_to_goal <= GOAL_THRESHOLD:
        print(f"\nSUCCESS: Robot believes it reached Goal ({G_X}, {G_Y}) in {step_count} steps (~{step_count*dt:.1f} seconds)!")
        print(f"True distance to goal: {np.hypot(G_X - t_x, G_Y - t_y):.2f} m")
        reached_goal = True
        for wheel in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=MAX_FORCE)
        break # Stop condition when inside goal attraction threshold

    # Goal direction in robot frame and free space around it
    goal_bearing = np.arctan2(dy_attract, dx_attract) - est_theta
    goal_bearing = np.arctan2(np.sin(goal_bearing), np.cos(goal_bearing))
    offs = np.arctan2(np.sin(ray_angles - goal_bearing), np.cos(ray_angles - goal_bearing))
    goal_clear = ranges[np.abs(offs) < np.radians(30)].min()
    front_clear = ranges[np.abs(ray_angles) < np.radians(30)].min()

    # Switch between goal seeking and wall following
    if mode == "GOAL":
        d_best = min(d_best, distance_to_goal)
    if mode == "GOAL" and goal_clear < min(SAFE_DIST, distance_to_goal):
        mode, follow_start = "FOLLOW", step_count
        open_rays = ranges > 0.95 * LIDAR_RANGE
        left_gap = np.min(offs[open_rays & (offs > 0)], initial=np.pi)
        right_gap = np.min(-offs[open_rays & (offs < 0)], initial=np.pi)
        follow_side = 1 if left_gap <= right_gap else -1  # go round the side with the nearest opening
    elif mode == "FOLLOW" and (step_count - follow_start) * dt > FOLLOW_TIMEOUT:
        mode, d_best = "GOAL", distance_to_goal  # circling too long, retry goal
    elif mode == "FOLLOW" and distance_to_goal - (goal_clear - SAFE_DIST) < d_best - 0.5 and goal_clear > min(distance_to_goal, 2.0):
        mode = "GOAL"  # clear path that gets closer than before

    # STEERING CONTROLLER
    if mode == "GOAL":
        # Attraction to goal + repulsion from nearby obstacles
        f_x, f_y = np.cos(goal_bearing), np.sin(goal_bearing)
        near = ranges < D0
        rep = K_REP * (1 / ranges[near] - 1 / D0) / ranges[near]**2
        f_x -= np.sum(rep * np.cos(ray_angles[near]))
        f_y -= np.sum(rep * np.sin(ray_angles[near]))
        heading_error = np.arctan2(f_y, f_x)
    else:
        # Keep closest obstacle on one side at SAFE_DIST
        side_rays = np.where(follow_side * ray_angles < np.radians(30), ranges, np.inf)
        i_min = np.argmin(side_rays)
        heading_error = ray_angles[i_min] + follow_side * (np.pi / 2 + np.clip(1.5 * (SAFE_DIST - ranges[i_min]), -0.8, 0.8))

    # Wrap angle error to [-pi, pi] to take shortest turn direction
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    forward_speed = min(8.0, 2.0 * distance_to_goal)  # Slows down smoothly near goal
    forward_speed *= max(0.0, np.cos(heading_error))  # Turn first if target direction is to the side/behind
    forward_speed *= np.clip((front_clear - 0.6) / 0.8, 0, 1)  # Slow down near obstacles ahead
    if mode == "FOLLOW":
        forward_speed = min(forward_speed, 5.0)
    turn_gain = 6.0

    # Differential drive control signals
    left_cmd = np.clip(forward_speed - (turn_gain * heading_error), -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    right_cmd = np.clip(forward_speed + (turn_gain * heading_error), -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

    for wheel in left_wheels:
        p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=left_cmd, force=MAX_FORCE)
    for wheel in right_wheels:
        p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=right_cmd, force=MAX_FORCE)

if not reached_goal:
    print("\nTIMEOUT: goal not reached")

if p.isConnected():
    p.disconnect()
print("\nSimulation is complete. Making plots of the robot's trajectory...")

#trag evolution 

gt = np.array(ground_truth_path)
est = np.array(estimated_path)
pos_err = np.linalg.norm(gt - est, axis=1)
head_err = np.degrees(np.array(heading_errors))
t = np.arange(len(pos_err)) * dt

print(f"Final position error: {pos_err[-1]:.3f} m | Mean: {pos_err.mean():.3f} m | Max: {pos_err.max():.3f} m | RMSE: {np.sqrt(np.mean(pos_err**2)):.3f} m")
print(f"Final heading error: {head_err[-1]:.2f} deg")

plt.figure(figsize=(9, 9))
for x, y, _ in cube_positions:
    plt.gca().add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, color="gray"))
for wx, wy, hx, hy in wall_specs:
    plt.gca().add_patch(plt.Rectangle((wx - hx, wy - hy), 2 * hx, 2 * hy, color="dimgray"))  # obstacles

plt.plot(gt[:, 0], gt[:, 1], label="Ground Truth Trajectory", color="blue", linewidth=2.5)
plt.plot(est[:, 0], est[:, 1], label="Estimated Trajectory (Dead Reckoning)", color="green", linestyle="--", linewidth=2)

plt.scatter([SP[0]], [SP[1]], color="green", s=150, marker="o", label=f"Start ({SP[0]}, {SP[1]})", zorder=5)
plt.scatter([G_X], [G_Y], color="red", s=300, marker="*", label=f"Goal ({G_X}, {G_Y})", zorder=5) # Mark Start and Goal

plt.title(f"Robot Localization: Ground Truth vs Dead Reckoning ({NOISE_PERCENT}% noise)")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.legend(loc="upper left")
plt.axis("equal")
plt.grid(True)

plt.savefig("localization_results_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("localization_results_plot.pdf", bbox_inches='tight')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(t, pos_err, color="red", label="Position Error")
ax1.set_ylabel("Position Error (meters)")
ax1.legend(loc="upper left")
ax1.grid(True)
ax2.plot(t, head_err, color="purple", label="Heading Error")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Heading Error (degrees)")
ax2.legend(loc="upper left")
ax2.grid(True)
ax1.set_title(f"Localization Error vs Ground Truth ({NOISE_PERCENT}% noise)")

plt.savefig("localization_error_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("localization_error_plot.pdf", bbox_inches='tight')
print("Plots saved as localization_results_plot.png/.pdf and localization_error_plot.png/.pdf!")

plt.show()
