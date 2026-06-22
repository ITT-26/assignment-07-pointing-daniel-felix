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


## 2 Fitts’ Law Application
[`fitts_law.py`](fitts_law.py) is a `pyglet` implementation of a **two-dimensional tapping task**. Successful clicks are logged automatically to `data/fitts_<num_targets>_<W>_<D>_<pid>.csv`. Because the task reacts to the OS cursor, it works with the mid-air pointing technique from task 1, a mouse, or a touchpad.

```bash
python fitts_law.py [--config fitts_config.json] [--pid N] [--num-targets N] [--width W] [--distance D] [--iterations N]
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

| Key / Action | Result |
|--------------|--------|
| Click the red target | Record acquisition and advance to the next target |
| `R` | Restart the study |
| `Q` / `Esc` | Quit |


## 3 Steering Law Application 

## 4 Adding Latency

## 5 Evaluating Input Techniques
