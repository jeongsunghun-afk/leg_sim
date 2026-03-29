import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import matplotlib as mpl

# Disable default matplotlib key bindings
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

def forward_kinematics(theta, link_lengths):
    x0, y0 = 0, 0
    x1 = x0 + link_lengths[0] * np.cos(theta[0])
    y1 = y0 + link_lengths[0] * np.sin(theta[0])
    x2 = x1 + link_lengths[1] * np.cos(theta[0] + theta[1])
    y2 = y1 + link_lengths[1] * np.sin(theta[0] + theta[1])
    x3 = x2 + link_lengths[2] * np.cos(theta[0] + theta[1] + theta[2])
    y3 = y2 + link_lengths[2] * np.sin(theta[0] + theta[1] + theta[2])
    x4 = x3 + link_lengths[3] * np.cos(theta[0] + theta[1] + theta[2] + theta[3])
    y4 = y3 + link_lengths[3] * np.sin(theta[0] + theta[1] + theta[2] + theta[3])
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def ik_objective(theta, target, link_lengths):
    toe = forward_kinematics(theta, link_lengths)[-2]
    return np.linalg.norm(np.array(toe) - np.array(target))

def compute_static_torques(theta, link_lengths, masses, g=9.81):
    torques = [0, 0, 0, 0]
    joints = forward_kinematics(theta, link_lengths)
    centers = [
        np.mean([joints[0], joints[1]], axis=0),
        np.mean([joints[1], joints[2]], axis=0),
        np.mean([joints[2], joints[3]], axis=0),
        np.mean([joints[3], joints[4]], axis=0)
    ]
    for i in range(4):
        for j in range(i, 4):
            r = np.array(centers[j]) - np.array(joints[i])
            f = np.array([0, -masses[j] * g])
            torques[i] += np.cross(r, f)
    return torques

# 설정값
link_lengths = [0.3, 0.3, 0.2, 0.1]
masses = [2.0, 1.5, 1.0, 0.5]
deg2rad = np.pi / 180
delta_angle = 5 * deg2rad
delta_pos = 0.005

initial_theta = [np.deg2rad(-135), np.deg2rad(85), np.deg2rad(-75), np.deg2rad(-55)]
theta = initial_theta.copy()
target_position = list(forward_kinematics(theta, link_lengths)[-2])  # toe 기준
history = []
playback_mode = False
frame_idx = 0

# 시각화
fig, ax = plt.subplots()
ax.set_xlim(-1.6, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=4, markersize=8)
text_info = ax.text(0.02, 0.72, '', transform=ax.transAxes, fontsize=9)
angle_info = ax.text(0.7, 0.72, '', transform=ax.transAxes, fontsize=9)
foot_info = ax.text(0.7, 0.60, '', transform=ax.transAxes, fontsize=9)
x_axis_line, = ax.plot([], [], 'r-', lw=2)
y_axis_line, = ax.plot([], [], 'g-', lw=2)

def update_visual():
    joints = forward_kinematics(theta, link_lengths)
    torques = compute_static_torques(theta, link_lengths, masses)
    x_data, y_data = zip(*joints)
    line.set_data(x_data, y_data)

    toe = joints[-2]
    angle = sum(theta[:3])
    dx = 0.05 * np.cos(angle)
    dy = 0.05 * np.sin(angle)
    x_axis_line.set_data([toe[0], toe[0] + dx], [toe[1], toe[1] + dy])
    x_ortho = -dy
    y_ortho = dx
    y_axis_line.set_data([toe[0], toe[0] + x_ortho], [toe[1], toe[1] + y_ortho])

    text_info.set_text(
        f"Torques:\n"
        f"Hip: {torques[0]:.2f} Nm\n"
        f"Knee: {torques[1]:.2f} Nm\n"
        f"Ankle: {torques[2]:.2f} Nm\n"
        f"Toe: {torques[3]:.2f} Nm"
    )
    angle_info.set_text(
        f"Angles (deg):\n"
        f"T1: {np.rad2deg(theta[0]):.1f}\n"
        f"T2: {np.rad2deg(theta[1]):.1f}\n"
        f"T3: {np.rad2deg(theta[2]):.1f}\n"
        f"T4: {np.rad2deg(theta[3]):.1f}"
    )
    foot_info.set_text(
        f"Toe Pos (m):\n"
        f"x: {toe[0]:.3f}\n"
        f"y: {toe[1]:.3f}"
    )

def on_key(event):
    global theta, target_position, playback_mode, frame_idx
    key = event.key

    if key == 'p':
        theta[:] = initial_theta
        update_visual()
        fig.canvas.draw()
        return
    elif key == 'l':
        playback_mode = not playback_mode
        frame_idx = 0
        return

    if playback_mode:
        return

    if key == '1':
        theta[0] += delta_angle
    elif key == 'q':
        theta[0] -= delta_angle
    elif key == '2':
        theta[1] += delta_angle
    elif key == 'w':
        theta[1] -= delta_angle
    elif key == '3':
        theta[2] += delta_angle
    elif key == 'e':
        theta[2] -= delta_angle
    elif key == '4':
        theta[3] += delta_angle
    elif key == 'r':
        theta[3] -= delta_angle
    elif key == 'a':
        target_position[0] += delta_pos
    elif key == 'z':
        target_position[0] -= delta_pos
    elif key == 's':
        target_position[1] += delta_pos
    elif key == 'x':
        target_position[1] -= delta_pos

    if key in ['a', 'z', 's', 'x']:
        res = minimize(ik_objective, theta, args=(target_position, link_lengths), bounds=[(-np.pi, np.pi)] * 4)
        if res.success:
            theta = list(res.x)

    history.append(list(theta))
    update_visual()
    fig.canvas.draw()

def init():
    update_visual()
    return line, text_info, angle_info, foot_info, x_axis_line, y_axis_line

def animate_playback(frame):
    global theta, frame_idx
    if playback_mode and frame_idx < len(history):
        theta = history[frame_idx]
        frame_idx += 1
        update_visual()
    return line, text_info, angle_info, foot_info, x_axis_line, y_axis_line

fig.canvas.mpl_connect('key_press_event', on_key)
ani = FuncAnimation(fig, animate_playback, init_func=init, frames=200, blit=True, interval=50)
plt.title("4DOF Leg Simulation (Toe IK)")
plt.show()
