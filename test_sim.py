"""
First test script for Reachy Mini in simulation mode.
Run with: mjpython test_sim.py
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time

with ReachyMini() as mini:
    print("Connected to simulated Reachy Mini!")

    # Look up and tilt head
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0
    )
    time.sleep(1.5)

    # Look left
    mini.goto_target(
        head=create_head_pose(yaw=30, degrees=True, mm=True),
        duration=1.0
    )
    time.sleep(1.5)

    # Look right
    mini.goto_target(
        head=create_head_pose(yaw=-30, degrees=True, mm=True),
        duration=1.0
    )
    time.sleep(1.5)

    # Back to neutral
    mini.goto_target(
        head=create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True, mm=True),
        duration=1.0
    )
    time.sleep(1.5)

    print("Done! If you saw the head move in the MuJoCo window, you're all set.")