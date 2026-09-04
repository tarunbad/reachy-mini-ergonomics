"""
Desk Ergonomics Reminder for Reachy Mini

Watches your posture via your Mac's webcam (MediaPipe Pose) and has
Reachy Mini give you a physical nudge when you've been slouched too long.

Prereqs (in your activated venv):
    pip install mediapipe opencv-python numpy

Run (daemon must already be running in another terminal):
    Terminal 1: mjpython -m reachy_mini.daemon.app.main --sim
    Terminal 2: python ergonomics_reminder.py

NOTE: use plain `python` for this script, NOT mjpython — mjpython keeps the
Mac main thread for MuJoCo, which breaks OpenCV's preview window (cv2.imshow).
Only the daemon needs mjpython.

Press 'q' in the video window to quit, or Ctrl+C in the terminal.

Optional: `python ergonomics_reminder.py --test` does one nudge on startup
so you can verify the robot connection without waiting for a real slouch.
"""

import os
import sys
import time
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ---- Tunable settings ----
SLOUCH_THRESHOLD_ANGLE = 54     # degrees; craniovertebral angle (ear-shoulder vs horizontal),
                                 # AVERAGED over both sides so a head tilt doesn't fake a slouch.
                                 # Below this = forward head posture / slouching.
                                 # Calibrated for Tarun: straight ~60.8 deg, slouched ~46.4 deg,
                                 # so 54 sits right in the middle of that gap.
HEAD_TILT_THRESHOLD = 14        # degrees; ear-to-ear line vs horizontal. More than this
                                 # sustained = head cocked toward one shoulder.
SHOULDER_TILT_THRESHOLD = 10    # degrees; shoulder-to-shoulder line vs horizontal. More than
                                 # this = leaning/slumping to one side.
SLOUCH_GRACE_PERIOD = 60        # seconds of sustained bad posture before Reachy reacts
NUDGE_COOLDOWN = 120             # min seconds between nudges, so it's not annoying
CHECK_INTERVAL = 0.5             # seconds between pose checks
GOOD_POSTURE_RESET = 5           # seconds of good posture needed to reset the slouch timer

TOO_CLOSE_RATIO = 1.25          # flagged when your shoulders appear >25% wider (closer)
                                 # than the baseline captured at startup
CALIBRATION_SECONDS = 5          # how long to sample your normal distance at startup
SIT_TIME_LIMIT = 45 * 60         # seconds of continuous sitting before a break reminder
ABSENCE_RESET = 120              # gone from frame this long = you took a break, timer resets

# BlazePose 33-point landmark indices (same topology mediapipe has always used)
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def ensure_model():
    """MediaPipe 1.0+ dropped the old mp.solutions API; PoseLandmarker needs
    a model file downloaded once (a few MB, cached locally after)."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (one-time, ~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def calculate_angle(a, b, c):
    """Angle at point b, formed by points a-b-c (2D pixel coords)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_cva(ear, shoulder, ref_direction=1):
    """Craniovertebral angle: angle between the ear-shoulder line and horizontal.
    Only needs ear + shoulder (both usually visible on a desk webcam), unlike a
    hip-based angle which needs landmarks that are often out of frame / guessed.

    ref_direction: +1 for the person's right side, -1 for the left side, so the
    horizontal reference always points toward the body midline and both sides
    give symmetric values."""
    horizontal_ref = [shoulder[0] + 100 * ref_direction, shoulder[1]]
    return calculate_angle(ear, shoulder, horizontal_ref)


def line_tilt_degrees(p1, p2):
    """How far the line p1-p2 is from horizontal, in degrees (0 = level)."""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return np.degrees(np.arctan2(dy, dx + 1e-6))


def check_posture(lm, w, h, baseline_shoulder_w=None):
    """Evaluate all posture rules from one set of pose landmarks.

    Returns (issues, metrics): issues is a list of human-readable problems
    (empty = posture OK), metrics is a dict of the raw numbers for display.
    baseline_shoulder_w: your shoulder width in pixels at normal distance
    (from startup calibration); None disables the too-close check.
    """
    r_ear = [lm[RIGHT_EAR].x * w, lm[RIGHT_EAR].y * h]
    l_ear = [lm[LEFT_EAR].x * w, lm[LEFT_EAR].y * h]
    r_shoulder = [lm[RIGHT_SHOULDER].x * w, lm[RIGHT_SHOULDER].y * h]
    l_shoulder = [lm[LEFT_SHOULDER].x * w, lm[LEFT_SHOULDER].y * h]

    # In a non-mirrored webcam frame the person's right side is on the image
    # left, so "toward midline" is +x for the right side and -x for the left.
    cva_right = calculate_cva(r_ear, r_shoulder, ref_direction=1)
    cva_left = calculate_cva(l_ear, l_shoulder, ref_direction=-1)
    neck_angle = (cva_right + cva_left) / 2  # averaging cancels out head tilt

    head_tilt = line_tilt_degrees(r_ear, l_ear)
    shoulder_tilt = line_tilt_degrees(r_shoulder, l_shoulder)
    shoulder_w = float(np.hypot(r_shoulder[0] - l_shoulder[0],
                                r_shoulder[1] - l_shoulder[1]))

    issues = []
    if neck_angle < SLOUCH_THRESHOLD_ANGLE:
        issues.append("slouching forward")
    if head_tilt > HEAD_TILT_THRESHOLD:
        issues.append("head tilted sideways")
    if shoulder_tilt > SHOULDER_TILT_THRESHOLD:
        issues.append("leaning to one side")
    if baseline_shoulder_w and shoulder_w > baseline_shoulder_w * TOO_CLOSE_RATIO:
        issues.append("too close to screen")

    metrics = {
        "neck": neck_angle,
        "head_tilt": head_tilt,
        "shoulder_tilt": shoulder_tilt,
        "shoulder_w": shoulder_w,
    }
    return issues, metrics


def reachy_nudge(mini):
    """Obvious 'hey, sit up' reaction: big droop, a little shake, then perk back up.

    goto_target already blocks until the move finishes, so sleeps below are
    only for holding a pose long enough to actually see it.
    """
    print("Posture nudge — look at the MuJoCo window!")
    # Droop + antennas down (concerned)
    mini.goto_target(
        head=create_head_pose(pitch=-35, degrees=True, mm=True),
        antennas=[1.2, -1.2],
        duration=1.5,
    )
    time.sleep(1.2)  # hold so it's unmissable
    # Shake "hey!" while still looking down
    mini.goto_target(
        head=create_head_pose(pitch=-35, yaw=-20, degrees=True, mm=True),
        antennas=[1.2, -1.2],
        duration=0.6,
    )
    mini.goto_target(
        head=create_head_pose(pitch=-35, yaw=20, degrees=True, mm=True),
        antennas=[1.2, -1.2],
        duration=0.7,
    )
    # Perk up, then back to rest
    mini.goto_target(
        head=create_head_pose(pitch=18, degrees=True, mm=True),
        antennas=[0.0, 0.0],
        duration=1.0,
    )
    mini.goto_target(
        head=create_head_pose(degrees=True, mm=True),
        antennas=[0.0, 0.0],
        duration=1.2,
    )


def reachy_break_reminder(mini):
    """Cheerful 'get up and move!' gesture: perks up and looks around the room,
    like it's suggesting you go take a walk. Distinct from the concerned nudge."""
    print("Break time — you've been sitting a while. Stand up and stretch!")
    mini.goto_target(
        head=create_head_pose(pitch=20, degrees=True, mm=True),
        antennas=[-0.6, 0.6],
        duration=1.0,
    )
    mini.goto_target(
        head=create_head_pose(pitch=10, yaw=45, degrees=True, mm=True),
        antennas=[-0.6, 0.6],
        duration=1.2,
    )
    time.sleep(0.5)
    mini.goto_target(
        head=create_head_pose(pitch=10, yaw=-45, degrees=True, mm=True),
        antennas=[-0.6, 0.6],
        duration=1.5,
    )
    time.sleep(0.5)
    mini.goto_target(
        head=create_head_pose(degrees=True, mm=True),
        antennas=[0.0, 0.0],
        duration=1.0,
    )


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    print("Connecting to Reachy Mini (no_media backend so we can use the webcam directly)...")
    with ReachyMini(media_backend="no_media") as mini:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        # Warm-up: on macOS the AVFoundation backend can take a few frames
        # before it actually starts returning real images.
        print("Warming up camera...")
        warm_ok = False
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                warm_ok = True
                break
            time.sleep(0.1)
        if not warm_ok:
            print("Camera opened but never returned a frame after 3s.")
            print("Check System Settings > Privacy & Security > Camera, and that no")
            print("other app (Zoom, Photo Booth, etc.) is currently holding the camera.")
            cap.release()
            return
        print("Camera is live.")

        if "--test" in sys.argv:
            print("Startup test nudge in 1s — watch the MuJoCo window, not the webcam.")
            time.sleep(1)
            reachy_nudge(mini)
            print("If Reachy just moved, the robot connection is good.")

        slouch_start = None
        good_posture_ticks = 0
        last_nudge_time = 0.0
        start_time = time.time()
        gui_ok = True  # falls back to terminal-only output if imshow can't work
        last_status_print = 0.0

        # Screen-distance calibration (samples your shoulder width for a few seconds)
        baseline_shoulder_w = None
        calib_samples = []
        calib_end = time.time() + CALIBRATION_SECONDS
        print(f"Calibrating your normal sitting distance for {CALIBRATION_SECONDS}s — "
              "sit straight at your usual distance from the screen.")

        # Continuous-sitting tracker
        sit_start = None
        last_seen = None

        print("Watching your posture. Press 'q' in the video window (or Ctrl+C) to stop.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame read failed, retrying...")
                    time.sleep(0.2)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]  # first detected person
                    h, w, _ = frame.shape

                    issues, metrics = check_posture(lm, w, h, baseline_shoulder_w)
                    bad_posture = len(issues) > 0

                    # One-time distance calibration during the first few seconds
                    if baseline_shoulder_w is None:
                        calib_samples.append(metrics["shoulder_w"])
                        if time.time() > calib_end:
                            baseline_shoulder_w = float(np.median(calib_samples))
                            print(f"Distance calibrated (shoulder width "
                                  f"{baseline_shoulder_w:.0f}px). Getting "
                                  f"{int((TOO_CLOSE_RATIO - 1) * 100)}% closer will be flagged.")

                    # Sitting-time tracking: you're in frame, so you're at your desk
                    now = time.time()
                    last_seen = now
                    if sit_start is None:
                        sit_start = now
                    elif now - sit_start > SIT_TIME_LIMIT:
                        reachy_break_reminder(mini)
                        sit_start = now  # remind again in another SIT_TIME_LIMIT

                    slouch_secs = 0.0  # how long posture has been bad this stretch
                    if bad_posture:
                        good_posture_ticks = 0
                        if slouch_start is None:
                            slouch_start = time.time()
                            print(f"Bad posture ({', '.join(issues)}) — timer started "
                                  f"(nudge in {SLOUCH_GRACE_PERIOD}s if you stay like this).")
                        slouch_secs = time.time() - slouch_start
                        if slouch_secs > SLOUCH_GRACE_PERIOD:
                            cooldown_left = NUDGE_COOLDOWN - (time.time() - last_nudge_time)
                            if cooldown_left <= 0:
                                print(f"Nudging for: {', '.join(issues)}")
                                reachy_nudge(mini)
                                last_nudge_time = time.time()
                            else:
                                print(f"Posture still bad, but nudge on cooldown "
                                      f"({cooldown_left:.0f}s left).")
                            slouch_start = time.time()  # restart grace period either way
                    else:
                        good_posture_ticks += 1
                        if good_posture_ticks * CHECK_INTERVAL > GOOD_POSTURE_RESET:
                            if slouch_start is not None:
                                print("Good posture held — timer reset.")
                            slouch_start = None

                    cv2.putText(frame,
                                f"Neck: {metrics['neck']:.0f}  "
                                f"HeadTilt: {metrics['head_tilt']:.0f}  "
                                f"ShoulderTilt: {metrics['shoulder_tilt']:.0f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    status = ", ".join(issues).upper() if bad_posture else "OK"
                    cv2.putText(frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255) if bad_posture else (0, 255, 0), 2)
                    if slouch_secs > 0:
                        cv2.putText(frame,
                                    f"Nudge in: {max(0, SLOUCH_GRACE_PERIOD - slouch_secs):.0f}s",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if sit_start is not None:
                        sit_min = (time.time() - sit_start) / 60
                        limit_min = SIT_TIME_LIMIT // 60
                        cv2.putText(frame, f"Sitting: {sit_min:.0f}m / {limit_min}m",
                                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if sit_min < limit_min * 0.8 else (0, 165, 255), 2)
                else:
                    # Nobody in frame — if gone long enough, count it as a real break
                    if (sit_start is not None and last_seen is not None
                            and time.time() - last_seen > ABSENCE_RESET):
                        print("Looks like you stepped away — sit timer reset. Nice.")
                        sit_start = None
                        slouch_start = None

                if gui_ok:
                    try:
                        cv2.imshow("Posture Check (press q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        gui_ok = False
                        print("Video window unavailable (are you running under mjpython?).")
                        print("Continuing headless — status prints here instead. Ctrl+C to stop.")

                if not gui_ok and result.pose_landmarks:
                    # Print a status line every few seconds so you can still see it working
                    if time.time() - last_status_print > 3:
                        status = ", ".join(issues) if issues else "OK"
                        print(f"Neck {metrics['neck']:.0f} deg, "
                              f"head tilt {metrics['head_tilt']:.0f} deg, "
                              f"shoulder tilt {metrics['shoulder_tilt']:.0f} deg — {status}")
                        last_status_print = time.time()

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()


if __name__ == "__main__":
    main()