# Reachy Mini Ergonomics Buddy

A desk robot that physically nags you into better posture.

Your webcam watches how you sit. When you slouch, tilt, lean, or creep toward
the screen for too long, [Reachy Mini](https://huggingface.co/docs/reachy-mini/index)
reacts — drooping its head in concern, shaking a disappointed "hey!", and
perking back up. Sit badly for 45 minutes straight and it cheerfully looks
around the room, suggesting you go take a walk.

No notification popup you can ignore. A robot judging you is much harder to tune out.

## What it detects

| Check | How | Trigger |
|---|---|---|
| Forward slouch ("tech neck") | Craniovertebral angle: ear–shoulder line vs horizontal, averaged over both sides | avg angle < 54° |
| Head tilted sideways | Ear-to-ear line vs horizontal | > 14° |
| Leaning to one side | Shoulder-to-shoulder line vs horizontal | > 10° |
| Too close to the screen | Shoulder width in pixels vs a baseline calibrated at startup | > 25% wider than baseline |
| Sitting too long | Continuous presence in frame (leaving for 2+ min counts as a break) | 45 min |

The slouch metric is a webcam approximation of the **craniovertebral angle
(CVA)** — the standard physiotherapy measure of forward head posture, where
research commonly uses cutoffs around 48–53°. Angles here are calibrated per
person and per camera setup, so the numbers are tuned to *your* straight vs
slouched posture rather than textbook values.

Any posture issue must persist for a full 60 seconds before Reachy reacts
(grabbing something off the floor won't set it off), and nudges have a
2-minute cooldown so it never gets annoying.

## How it works

```
webcam ──> MediaPipe Pose (BlazePose 33 landmarks)
              │
              ▼
      posture rules (angles + calibrated baselines)
              │
              ▼  sustained 60s?
      Reachy Mini SDK ──> goto_target() head/antenna choreography
              │
              ▼
      MuJoCo sim (or a real Reachy Mini)
```

- **Pose tracking:** MediaPipe `PoseLandmarker` (lite model, auto-downloaded on first run)
- **Robot:** Reachy Mini SDK — runs identically against the MuJoCo simulator or real hardware
- **Live overlay:** the webcam window shows all metrics, the current verdict, a countdown to the next nudge, and your sitting timer

## Setup

Requires Python 3.11+ and macOS/Linux with a webcam.

```bash
python3.11 -m venv reachy_mini_env
source reachy_mini_env/bin/activate
pip install reachy-mini mediapipe opencv-python numpy
```

## Run

Two terminals, both with the venv activated:

```bash
# Terminal 1 — Reachy Mini daemon + MuJoCo 3D viewer
mjpython -m reachy_mini.daemon.app.main --sim

# Terminal 2 — the posture watcher (plain python, NOT mjpython)
python ergonomics_reminder.py
```

> Why not mjpython for both? mjpython reserves the Mac main thread for
> MuJoCo's GUI, which is the same thread OpenCV's preview window needs —
> `cv2.imshow` crashes under it. Only the daemon needs mjpython. (If the
> preview window can't open, the script keeps running headless and prints
> status to the terminal instead.)

For the first 5 seconds, sit straight at your normal distance — the script
calibrates your baseline shoulder width for the too-close check. Then just work.
Press `q` in the video window (or Ctrl+C) to quit.

```bash
# Optional: verify the robot connection with one test nudge on startup
python ergonomics_reminder.py --test
```

## Tuning

Everything lives at the top of `ergonomics_reminder.py`:

| Setting | Default | Meaning |
|---|---|---|
| `SLOUCH_THRESHOLD_ANGLE` | 54° | Below this avg CVA = slouching. Calibrate to yourself: check the overlay while sitting straight vs slouched, pick the midpoint |
| `HEAD_TILT_THRESHOLD` | 14° | Ear line off horizontal |
| `SHOULDER_TILT_THRESHOLD` | 10° | Shoulder line off horizontal |
| `TOO_CLOSE_RATIO` | 1.25 | Flag when shoulders appear 25% wider than baseline |
| `SLOUCH_GRACE_PERIOD` | 60 s | Bad posture must persist this long |
| `NUDGE_COOLDOWN` | 120 s | Minimum gap between nudges |
| `SIT_TIME_LIMIT` | 45 min | Continuous sitting before a break reminder |
| `ABSENCE_RESET` | 120 s | Out of frame this long = you took a real break |

## Limitations

One front-facing webcam can't see everything: lower-back arch, pelvis
position, crossed legs, and monitor height are invisible from the front.
This covers what a front camera can legitimately judge — which turns out
to be most of the common desk sins.

## Roadmap

- Escalating nags: ignore Reachy and it gets more dramatic, more often
- Positive reinforcement: a happy wiggle when you correct your posture
- Session stats on quit: % time slouched, nudge count, longest good streak
- Real hardware: same code, minus `--sim`
