[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/KfEU5Azw)


# Assignment 7: Pointing

## Setup
1. Clone the repo and navigate to it via `cd assignment-07-pointing-daniel-felix`.
2. Set up a virtual environment by running `python -m venv .venv`.
3. Activate the virtual environment using `.venv\Scripts\activate` on Windows and `source .venv/bin/activate` on Linux/Mac.
4. Install the required dependencies via `pip install -r requirements.txt`.



## 1. Pose-Based Pointing Technique
[`pointing_input.py`](pointing_input.py) tracks hand movement using Google's [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) and controls the cursor with `pynput`. The cursor follows the **midpoint between the thumb and index fingertips**, so pinching the two together barely shifts it. **Pinching** the thumb and index finger triggers a left click. To keep the cursor steady, the position is smoothed with the [One Euro Filter](https://doi.org/10.1145/2207676.2208639), which cuts jitter when the hand is still while staying responsive during quick movements.

```bash
python pointing_input.py [camera_id]
```

| Key / Action | Result |
|--------------|--------|
| Move hand | Move the mouse pointer |
| Pinch thumb + index | Left click |
| `M` | Toggle mouse control (movement + clicking) on/off |
| `D` | Toggle the hand skeleton and live pinch distance |
| `Q` | Quit |

> Keys are read only while the camera preview window is focused (via `cv2.waitKey`), so they can't leak from another focused window and accidentally toggle or quit the tracker.


## 2 Fitts’ Law Application
[`fitts_law.py`](fitts_law.py) is a `pyglet` implementation of a **two-dimensional tapping task**. **Every click (hit or miss)** is logged automatically to `data/fitts_<technique>_<num_targets>_<W>_<D>_<pid>.csv`, one row per click, so both movement time and error rate / endpoint scatter can be analysed. Because the task reacts to the OS cursor, it works with the mid-air pointing technique from task 1, a mouse, or a touchpad.

Columns: study/config fields plus `target_id`, `step` (sequence position), `click_x`/`click_y` (click position), `target_x`/`target_y` (target center), `success` (1 = hit, 0 = miss), `mt_ms` (time since the previous successful click), `timestamp`.

```bash
python fitts_law.py [--config fitts_config.json] [--pid N] [--num-targets N] [--width W] [--distance D] [--iterations N] [--latency MS] [--technique pose|mouse|touchpad]
```

Parameters are read from [`fitts_config.json`](fitts_config.json), command-line argument overrides the config file.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Path to the JSON config file | `fitts_config.json` |
| `--pid` | Participant ID, used in the log file name and every logged row | `1` |
| `--num-targets` | Number of targets in the ring | `10` |
| `--width` | Target diameter `W` in pixels | `60` |
| `--distance` | Ring diameter `D` (movement amplitude) in pixels | `400` |
| `--iterations` | Number of full rings to repeat | `3` |
| `--latency` | Artificial pointer latency in ms (see task 4) | `0` |
| `--technique` | Input device label (`pose`/`mouse`/`touchpad`) | `mouse` |

| Key / Action | Result |
|--------------|--------|
| Click the red target | Record acquisition and advance to the next target |
| `R` | Restart the study |
| `Q` / `Esc` | Quit |


## 3 Steering Law Application

[`steering_law.py`](steering_law.py) is a `pyglet` implementation of a **steering (tunnel) task**. The cursor must enter the highlighted start goal, then steer through the horizontal tunnel to the goal at the opposite end without crossing the walls. Each completed one-way traversal (left→right or right→left) is logged automatically to `data/steering_<technique>_<W>_<A>_<pid>.csv`, one row per traversal. Timing starts when the cursor leaves the start goal and ends when it reaches the end goal; wall crossings are counted as violations. Like the Fitts' task it reacts to the OS cursor, so it works with mid-air pointing, a mouse, or a touchpad.

Columns: study/config fields plus `direction` (`LR`/`RL`), `start_ts`/`end_ts`, `duration_ms` (movement time), and `violations` (times the cursor left the corridor). `violations = 0` is a clean pass; use the count as the accuracy measure alongside `duration_ms`.

```bash
python steering_law.py [--config steering_config.json] [--pid N] [--width W] [--amplitude A] [--iterations N] [--latency MS] [--technique pose|mouse|touchpad]
```

Parameters are read from [`steering_config.json`](steering_config.json); command-line arguments override the config file.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Path to the JSON config file | `steering_config.json` |
| `--pid` | Participant ID, used in the log file name and every logged row | `1` |
| `--width` | Tunnel width `W` (wall-to-wall clearance) in pixels | `50` |
| `--amplitude` | Tunnel length `A` (goal-to-goal distance) in pixels | `500` |
| `--iterations` | Number of full go-and-return traversals | `5` |
| `--latency` | Artificial pointer latency in ms (see task 4) | `0` |
| `--technique` | Input device label (`pose`/`mouse`/`touchpad`) | `mouse` |

| Key / Action | Result |
|--------------|--------|
| Steer from start goal to the red goal | Record the traversal and turn around |
| `R` | Restart the study |
| `Q` / `Esc` | Quit |

## 4 Adding Latency

Both [`fitts_law.py`](fitts_law.py) and [`steering_law.py`](steering_law.py) accept a `--latency MS` option (also settable via the config files as `latency_ms`) that delays the pointer by a fixed number of milliseconds. Raw pointer positions are pushed into a time-stamped buffer and read back delayed; all hit detection (clicks, goal entry, wall crossings) uses the delayed position, and `latency_ms` is recorded in every logged row. The OS cursor is hidden so the participant follows only the delayed in-app cursor.

```bash
python fitts_law.py --latency 150
python steering_law.py --latency 150
```

## 5 Evaluating Input Techniques

[`run_study.py`](run_study.py) runs a full session for one participant, launching every Fitts'/Steering run as a subprocess (40 runs: 4 techniques — pose, mouse, mouse + 150 ms, touchpad x 5 Fitts' + 5 Steering conditions, each `--iterations 3`).

```bash
python run_study.py --pid 3            # run participant 3
python run_study.py --pid 3 --camera 1 # specify the camera for pose runs
```

The four techniques are ordered by a balanced Latin square (Williams design) keyed to `--pid`, so device-order effects cancel out across every four participants. The conditions within each technique are then shuffled reproducibly from `--pid`. Pose blocks start/stop [`pointing_input.py`](pointing_input.py) automatically. Per run: `Enter` start, `s` skip, `q` quit; after a run: `Enter` next, `r` redo, `q` quit.


### Problems encountered during the study runs

The first teammate ran their session on Linux (PID 1) without issues. The problems below came up when the second teammate later ran their session on Windows (PID 2), and the corresponding fixes were added in response.

- **Closing a task window also killed `pointing_input.py`.** Pressing **Q** to close a task window shut down the input script too, because it listened for keypresses globally and caught the **q** even when its window wasn't in focus. Fixed by reading the **q**/**m**/**d** keys directly from the camera preview window, so keypresses meant for other windows can't leak through.

- **No way to resume after an interruption.** If the script stopped partway through, rerunning it would start over and overwrite the CSV files already recorded. *Fixed:* `run_study` now checks what data has already been captured and only runs the missing conditions, which let us resume PID 2 after `pointing_input.py` was accidentally closed at 20 runs without repeating anything.

- **The study window flickered during pose blocks.** Under load, the screen refresh sometimes missed its timing and briefly showed a blank buffer. This appeared only on Windows (PID 2), not on Linux (PID 1). *Fixed:* turning off vsync in `fitts_law` and `steering_law`.
