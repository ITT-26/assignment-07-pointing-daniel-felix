import os
import sys
import random
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FITTS_NUM_TARGETS = 9
FITTS_ITERATIONS = 3
FITTS_W = [40, 60, 90]
FITTS_D = [250, 400, 550]
FITTS_W_CENTER, FITTS_D_CENTER = 60, 400

STEER_ITERATIONS = 3
STEER_W = [60, 100, 150]
STEER_A = [350, 500, 650]
STEER_W_CENTER, STEER_A_CENTER = 100, 500

# (label, technique_arg, latency_ms, uses_pose)
TECHNIQUES = [
    ("pose",          "pose",     0,   True),
    ("mouse",         "mouse",    0,   False),
    ("mouse+150ms",   "mouse",    150, False),
    ("touchpad",      "touchpad", 0,   False),
]

LATIN_SQUARE = [
    [0, 1, 3, 2],
    [1, 2, 0, 3],
    [2, 3, 1, 0],
    [3, 0, 2, 1],
]


def fitts_conditions():
    combos = [(w, FITTS_D_CENTER) for w in FITTS_W]
    combos += [(FITTS_W_CENTER, d) for d in FITTS_D if d != FITTS_D_CENTER]
    return combos


def steering_conditions():
    combos = [(w, STEER_A_CENTER) for w in STEER_W]
    combos += [(STEER_W_CENTER, a) for a in STEER_A if a != STEER_A_CENTER]
    return combos


def build_runs(label, technique, latency, pid):
    runs = []
    for w, d in fitts_conditions():
        runs.append({
            "desc": f"Fitts  W={w} D={d}",
            "cmd": [PY, os.path.join(HERE, "fitts_law.py"),
                    "--pid", str(pid), "--technique", technique,
                    "--num-targets", str(FITTS_NUM_TARGETS),
                    "--width", str(w), "--distance", str(d),
                    "--iterations", str(FITTS_ITERATIONS),
                    "--latency", str(latency)],
        })
    for w, a in steering_conditions():
        runs.append({
            "desc": f"Steer  W={w} A={a}",
            "cmd": [PY, os.path.join(HERE, "steering_law.py"),
                    "--pid", str(pid), "--technique", technique,
                    "--width", str(w), "--amplitude", str(a),
                    "--iterations", str(STEER_ITERATIONS),
                    "--latency", str(latency)],
        })
    return runs


def build_plan(pid, rng):
    row = LATIN_SQUARE[(pid - 1) % len(LATIN_SQUARE)]
    plan = []
    for i in row:
        label, technique, latency, uses_pose = TECHNIQUES[i]
        runs = build_runs(label, technique, latency, pid)
        rng.shuffle(runs)
        plan.append((label, uses_pose, runs))
    return plan


def main():
    ap = argparse.ArgumentParser(description="Task 5 study driver.")
    ap.add_argument("--pid", type=int, required=True, help="Participant ID.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Shuffle seed (default: derived from pid).")
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera id passed to pointing_input.py for pose runs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without launching anything.")
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else args.pid)
    plan = build_plan(args.pid, rng)
    total = sum(len(runs) for _, _, runs in plan)

    print(f"\n=== Task 5 study  -  participant {args.pid}  -  {total} runs ===")
    for label, uses_pose, runs in plan:
        tag = "  (pose tracking)" if uses_pose else ""
        print(f"\n[{label}]{tag}")
        for r in runs:
            print(f"    {r['desc']}")
    if args.dry_run:
        print("\n(dry run -- nothing launched)\n")
        return

    done = 0
    for label, uses_pose, runs in plan:
        print(f"\n========== BLOCK: {label} ==========")
        pose_proc = None
        if uses_pose:
            input("  Set up the participant for POSE pointing, then press Enter "
                  "to start the camera...")
            pose_proc = subprocess.Popen(
                [PY, os.path.join(HERE, "pointing_input.py"), str(args.camera)])
            input("  Wait until hand tracking controls the cursor, then press "
                  "Enter to begin this block...")
        else:
            input(f"  Hand the participant the {label.upper()} input, then press "
                  "Enter to begin this block...")

        try:
            for r in runs:
                done += 1
                ans = input(
                    f"\n[{done}/{total}] {label}  -  {r['desc']}  "
                    f"-- Enter=start / s=skip / q=quit: ").strip().lower()
                if ans == "q":
                    print("Quitting.")
                    return
                if ans == "s":
                    print("  skipped.")
                    continue
                while True:
                    subprocess.run(r["cmd"])  # blocks until the window is closed
                    post = input("  done (CSV in data/). "
                                 "Enter=next / r=redo / q=quit: ").strip().lower()
                    if post == "r":
                        print("  redoing...")
                        continue
                    if post == "q":
                        print("Quitting.")
                        return
                    break
        finally:
            if pose_proc:
                pose_proc.terminate()
                try:
                    pose_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pose_proc.kill()

    print(f"\n=== Participant {args.pid} complete -- {done} runs. CSVs in data/ ===\n")


if __name__ == "__main__":
    main()
