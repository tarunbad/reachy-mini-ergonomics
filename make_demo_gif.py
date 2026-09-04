"""Render Reachy Mini's posture-nudge choreography to a GIF, fully offscreen.

No webcam, no screen recording — loads the MuJoCo model directly, drives the
same keyframes as reachy_nudge() in ergonomics_reminder.py through the SDK's
analytical IK, and renders frames with mujoco.Renderer.

Run:  python make_demo_gif.py   (outputs docs/nudge.gif)
"""

from importlib.resources import files

import mujoco
import numpy as np
from PIL import Image

import reachy_mini
from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics
from reachy_mini.utils import create_head_pose
from reachy_mini.daemon.backend.mujoco.utils import (
    get_actuator_names,
    get_joint_addr_from_name,
)

FPS = 15
SIZE = (360, 360)  # width, height
OUT = "docs/nudge.gif"


def min_jerk(t):
    """Smooth 0→1 profile (same family the SDK uses for gotos)."""
    t = np.clip(t, 0, 1)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def main():
    mjcf_root = str(files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/"))
    model = mujoco.MjModel.from_xml_path(f"{mjcf_root}/scenes/empty.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002

    joint_names = get_actuator_names(model)
    qpos_addr = [get_joint_addr_from_name(model, n) for n in joint_names]

    kin = AnalyticalKinematics(automatic_body_yaw=False)

    def head_joints(pose):
        return kin.ik(pose)

    # The nudge choreography — mirrors reachy_nudge() in ergonomics_reminder.py.
    # (head pose, antennas [right, left] rad, move duration s, hold after s)
    neutral = create_head_pose(degrees=True, mm=True)
    keyframes = [
        (neutral, [0.0, 0.0], 0.8, 0.6),  # settle at neutral
        (create_head_pose(pitch=-35, degrees=True, mm=True), [1.2, -1.2], 1.5, 1.2),
        (create_head_pose(pitch=-35, yaw=-20, degrees=True, mm=True), [1.2, -1.2], 0.6, 0.0),
        (create_head_pose(pitch=-35, yaw=20, degrees=True, mm=True), [1.2, -1.2], 0.7, 0.0),
        (create_head_pose(pitch=18, degrees=True, mm=True), [0.0, 0.0], 1.0, 0.3),
        (neutral, [0.0, 0.0], 1.2, 0.8),
    ]

    # Start the sim already sitting at neutral
    j0 = head_joints(neutral)
    data.qpos[qpos_addr[:7]] = np.array(j0).reshape(-1, 1)
    data.qpos[qpos_addr[-2:]] = 0.0
    data.ctrl[:7] = j0
    data.ctrl[-2:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Same camera the daemon's viewer uses
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.8
    cam.azimuth = 160
    cam.elevation = -20
    cam.lookat[:] = [0, 0, 0.15]

    renderer = mujoco.Renderer(model, height=SIZE[1], width=SIZE[0])

    frames = []
    frame_dt = 1.0 / FPS
    next_frame_t = 0.0
    sim_t = 0.0

    current_joints = np.array(j0, dtype=np.float64)
    current_ant = np.array([0.0, 0.0])

    for pose, antennas, duration, hold in keyframes:
        target_joints = np.array(head_joints(pose), dtype=np.float64)
        target_ant = np.array(antennas, dtype=np.float64)
        seg_start = sim_t
        total = duration + hold
        while sim_t - seg_start < total:
            alpha = min_jerk((sim_t - seg_start) / duration)
            data.ctrl[:7] = current_joints + (target_joints - current_joints) * alpha
            # backend convention: ctrl is negated antennas
            ant = current_ant + (target_ant - current_ant) * alpha
            data.ctrl[-2:] = -ant
            mujoco.mj_step(model, data)
            sim_t += model.opt.timestep
            if sim_t >= next_frame_t:
                renderer.update_scene(data, cam)
                frames.append(Image.fromarray(renderer.render()))
                next_frame_t += frame_dt
        current_joints = target_joints
        current_ant = target_ant

    import os
    os.makedirs("docs", exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT}: {len(frames)} frames, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
