import pybullet as p
import numpy as np
import pybullet_data
import time
import matplotlib.pyplot as plt


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)  # gravity in z direction
planeId = p.loadURDF("plane.urdf")  # loads ground plane

# Starting position and orientation
SP = [0, 0, 0.1]
SO = p.getQuaternionFromEuler([0, 0, 0])

# Loads robot
robotId = p.loadURDF("husky/husky.urdf", SP, SO)
print("Husky robot spawned ")

# GOAL and Enviroment

G_X, G_Y = 5.0, -4.0 #Goal
GOAL_THRESHOLD = 0.4


cube_positions = [
    [2.0, 4.0, 0.7],
    [3.0, 2.0, 0.4],
    [1.0, 3.0, 0.9],
    [-2.0, -4.0, 0.1],
    [-4.0, -3.0, 0.6],
    [-2.0, -3.0, 0.5]
] #cube locations and sizes

for pos in cube_positions:
    p.loadURDF("cube.urdf", pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5)

print(f"Goal set at Target Coordinates: ({G_X}, {G_Y})")

# Path tracking arrays
ground_truth_path = []
estimated_path = []

# Initialize dead-reckoning pose variables
est_x, est_y, est_theta = 0.0, 0.0, 0.0

# Husky wheel joints: 2=front_left, 3=front_right, 4=rear_left, 5=rear_right
left_wheels = [2, 4]
right_wheels = [3, 5]

MAX_WHEEL_VEL = 10.0  # rad/s
MAX_FORCE = 50.0      # N*m per wheel - real motor torque limit for pushing

dt = 1.0 / 240.0  # Physics time step length
step_count = 0
reached_goal = False

print("Starting Goal Attraction Steering Simulation...\n")

while p.isConnected() and not reached_goal:
    p.stepSimulation()
    time.sleep(dt)
    step_count += 1


    pos, orient = p.getBasePositionAndOrientation(robotId)
    t_x, t_y = pos[0], pos[1]

    
    euler = p.getEulerFromQuaternion(orient)
    true_theta = euler[2]

    ground_truth_path.append((t_x, t_y))

    #
    dx_attract = G_X - t_x
    dy_attract = G_Y - t_y
    distance_to_goal = np.sqrt(dx_attract**2 + dy_attract**2)

    
    if step_count % 40 == 0 or step_count == 1:
        print(f"Step {step_count:04d} | Pos: ({t_x:.2f}, {t_y:.2f}) | Target: ({G_X:.1f}, {G_Y:.1f}) | Distance to Goal: {distance_to_goal:.2f} m")# Print live progress every 40 steps

    
    if distance_to_goal <= GOAL_THRESHOLD:
        print(f"\nSUCCESS: Reached Attraction Goal ({G_X}, {G_Y}) in {step_count} steps (~{step_count*dt:.1f} seconds)!")
        reached_goal = True
        for wheel in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=MAX_FORCE)
        break # Stop condition when inside goal attraction threshold

    # STEERING CONTROLLER to go towards goal 
    # Calculate desired heading angle pointing directly toward goal
    target_heading = np.arctan2(dy_attract, dx_attract)
    heading_error = target_heading - true_theta

    # Wrap angle error to [-pi, pi] to take shortest turn direction
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Compute differential wheel velocities based on attraction vector
    forward_speed = min(8.0, 2.0 * distance_to_goal)  # Slows down smoothly near goal
    forward_speed *= max(0.0, np.cos(heading_error))  # Turn first if goal is to the side/behind
    turn_gain = 6.0

    # Differential drive control signals
    left_cmd = np.clip(forward_speed - (turn_gain * heading_error), -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    right_cmd = np.clip(forward_speed + (turn_gain * heading_error), -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

    for wheel in left_wheels:
        p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=left_cmd, force=MAX_FORCE)
    for wheel in right_wheels:
        p.setJointMotorControl2(robotId, wheel, p.VELOCITY_CONTROL, targetVelocity=right_cmd, force=MAX_FORCE)  # Apply wheel velocities (limited torque -> robot physically pushes against obstacles)
    
    linear_vel, angular_vel = p.getBaseVelocity(robotId)
    v_x, v_y = linear_vel[0], linear_vel[1]
    omega_z = angular_vel[2] # 4. REAL ODOMETRY INTEGRATION /Velocity Tracking

    est_theta += omega_z * dt
    est_x += v_x * dt
    est_y += v_y * dt
    estimated_path.append((est_x, est_y))

if p.isConnected():
    p.disconnect()
print("\nSimulation is complete. Making plots of the robot's trajectory...")

#trag evolution 

gt = np.array(ground_truth_path)
est = np.array(estimated_path)

plt.figure(figsize=(9, 7))
plt.plot(gt[:, 0], gt[:, 1], label="Ground Truth Trajectory", color="blue", linewidth=2.5)
plt.plot(est[:, 0], est[:, 1], label="Estimated Trajectory (Dead Reckoning)", color="green", linestyle="--", linewidth=2)

plt.scatter([0], [0], color="green", s=150, marker="o", label="Start (0,0)", zorder=5)
plt.scatter([G_X], [G_Y], color="gold", s=300, marker="*", label=f"Attraction Goal ({G_X}, {G_Y})", zorder=5) # Mark Start and Goal

plt.title("Robot Localization: Goal Attraction Vector Trajectory")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.legend(loc="upper left")
plt.grid(True)

plt.savefig("localization_results_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("localization_results_plot.pdf", bbox_inches='tight')
print("Plots saved successfully as localization_results_plot.png and localization_results_plot.pdf!")

plt.show()
