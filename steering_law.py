import os
import json
import math
import time
import argparse

import pyglet
import pyglet.shapes


WINDOW_W, WINDOW_H = 1000, 800

BG_GL = (10 / 255, 10 / 255, 12 / 255, 1.0)

OFF_WHITE = (248, 248, 250)
LIGHT_GRAY = (229, 229, 234)
MID_GRAY = (174, 174, 178)
TEXT_LIGHT = (242, 242, 247)
TEXT_GRAY = (142, 142, 147)

# Apple HIG colors
APPLE_RED = (255, 56, 60)
APPLE_GREEN = (52, 199, 89)
APPLE_BLUE = (10, 132, 255)

IDLE_WALL_COLOR = MID_GRAY
ACTIVE_WALL_COLOR = APPLE_BLUE
GOAL_COLOR = APPLE_RED
DONE_GOAL_COLOR = APPLE_GREEN
CURSOR_COLOR = APPLE_GREEN
VIOLATION_COLOR = APPLE_RED

FONT = "Arial"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULTS = {
    "pid": 1,
    "tunnel_w": 50,      # tunnel width W in pixels (wall-to-wall clearance)
    "tunnel_a": 500,     # tunnel length A in pixels (center-to-center of goal zones)
    "iterations": 5,     # number of full go-and-return traversals per trial
    "num_trials": 3,     # how many (W, A) conditions to run back-to-back
}

# Each traversal = cursor moves from left goal to right goal (or vice versa).
# One "iteration" here = one full go-and-return (left→right, right→left).


def _rgba(rgb, a=255):
    return (*rgb, a)


def load_config():
    parser = argparse.ArgumentParser(description="Steering Law study application.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "steering_config.json"),
                        help="Path to JSON config file (default: steering_config.json).")
    parser.add_argument("--pid", type=int,
                        help=f"Participant ID (default: {DEFAULTS['pid']}).")
    parser.add_argument("--width", type=int, dest="tunnel_w",
                        help=f"Tunnel width W in pixels (default: {DEFAULTS['tunnel_w']}).")
    parser.add_argument("--amplitude", type=int, dest="tunnel_a",
                        help=f"Tunnel length A in pixels (default: {DEFAULTS['tunnel_a']}).")
    parser.add_argument("--iterations", type=int,
                        help=f"Go-and-return repetitions per trial (default: {DEFAULTS['iterations']}).")
    parser.add_argument("--num-trials", type=int, dest="num_trials",
                        help=f"Number of (W, A) conditions (default: {DEFAULTS['num_trials']}).")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))

    # command-line arguments override the config file
    for key in ("pid", "tunnel_w", "tunnel_a", "iterations", "num_trials"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    return cfg


# ---------- data logging ----------
class Logger:
    # One row per completed one-way traversal (half-crossing).
    HEADER = (
        "trial,iteration,direction,pid,tunnel_w,tunnel_a,"
        "start_ts,end_ts,duration_ms,violations\n"
    )

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(DATA_DIR, exist_ok=True)
        base = f"steering_{cfg['tunnel_w']}_{cfg['tunnel_a']}_{cfg['pid']}"
        path = os.path.join(DATA_DIR, base + ".csv")
        n = 1
        while os.path.exists(path):
            path = os.path.join(DATA_DIR, f"{base}_{n}.csv")
            n += 1
        self.path = path
        self.file = open(path, "w", newline="")
        self.file.write(self.HEADER)
        self.file.flush()

    def log(self, trial, iteration, direction, start_ts, end_ts, violations):
        c = self.cfg
        duration_ms = end_ts - start_ts
        self.file.write(
            f"{trial},{iteration},{direction},{c['pid']},"
            f"{c['tunnel_w']},{c['tunnel_a']},"
            f"{start_ts},{end_ts},{duration_ms},{violations}\n"
        )
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


# ---------- Tunnel geometry helpers ----------
def build_tunnel(cx, cy, a, w):
    """
    Returns the pixel coordinates of the tunnel's key geometry:
      left_x, right_x  – x centres of the two goal zones
      top_y, bot_y     – y edges of the tunnel walls
      half_w           – half the wall-to-wall clearance
    The tunnel is horizontal, centred at (cx, cy).
    """
    half_a = a / 2
    half_w = w / 2
    return (
        cx - half_a,   # left goal x
        cx + half_a,   # right goal x
        cy - half_w,   # top wall y
        cy + half_w,   # bottom wall y
        half_w,
    )


# ---------- App ----------
class SteeringLawApp:
    # Directions for alternating traversals
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1

    GOAL_RADIUS = 18   # visual radius of the circular goal targets at each end

    def __init__(self, window, cfg, logger):
        self.window = window
        self.cfg = cfg
        self.logger = logger
        self.batch = pyglet.graphics.Batch()

        g_bg = pyglet.graphics.Group(order=0)
        g_tunnel = pyglet.graphics.Group(order=1)
        g_goals = pyglet.graphics.Group(order=2)
        g_cursor = pyglet.graphics.Group(order=3)
        g_text = pyglet.graphics.Group(order=4)

        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        self.cx, self.cy = cx, cy

        lx, rx, ty, by, hw = build_tunnel(cx, cy, cfg["tunnel_a"], cfg["tunnel_w"])
        self.lx, self.rx = lx, rx
        self.ty, self.by = ty, by
        self.half_w = hw

        # Tunnel walls – two thin rectangles (top and bottom border)
        wall_thickness = 3
        self.wall_top = pyglet.shapes.Rectangle(
            lx, ty - wall_thickness, cfg["tunnel_a"], wall_thickness,
            color=IDLE_WALL_COLOR, batch=self.batch, group=g_tunnel)
        self.wall_bot = pyglet.shapes.Rectangle(
            lx, by, cfg["tunnel_a"], wall_thickness,
            color=IDLE_WALL_COLOR, batch=self.batch, group=g_tunnel)

        # Left and right goal circles
        self.goal_left = pyglet.shapes.Circle(
            lx, cy, self.GOAL_RADIUS, segments=48,
            color=GOAL_COLOR, batch=self.batch, group=g_goals)
        self.goal_right = pyglet.shapes.Circle(
            rx, cy, self.GOAL_RADIUS, segments=48,
            color=IDLE_WALL_COLOR, batch=self.batch, group=g_goals)

        # Cursor
        self.cursor = pyglet.shapes.Circle(
            cx, cy, 7, segments=48,
            color=CURSOR_COLOR, batch=self.batch, group=g_cursor)

        self.status_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=22,
            x=WINDOW_W // 2, y=WINDOW_H - 40,
            anchor_x="center", anchor_y="center",
            color=_rgba(TEXT_LIGHT), batch=self.batch, group=g_text)

        self.info_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=14,
            x=WINDOW_W // 2, y=28,
            anchor_x="center", anchor_y="center",
            color=_rgba(TEXT_GRAY), batch=self.batch, group=g_text)

        self.violation_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=13,
            x=WINDOW_W // 2, y=WINDOW_H - 75,
            anchor_x="center", anchor_y="center",
            color=_rgba(APPLE_RED), batch=self.batch, group=g_text)

        self.trial = 1
        self.restart()

    # ------------------------------------------------------------------
    def restart(self):
        self.iteration = 1
        self.direction = self.LEFT_TO_RIGHT   # always start left→right
        self.active = False      # waiting for cursor to enter start goal
        self.done = False
        self.violations = 0
        self.start_ts = None
        self._refresh()

    # ------------------------------------------------------------------
    @property
    def _start_goal(self):
        """The goal the cursor must leave from."""
        return self.goal_left if self.direction == self.LEFT_TO_RIGHT else self.goal_right

    @property
    def _end_goal(self):
        """The goal the cursor must arrive at."""
        return self.goal_right if self.direction == self.LEFT_TO_RIGHT else self.goal_left

    def _in_goal(self, goal, x, y):
        return (x - goal.x) ** 2 + (y - goal.y) ** 2 <= self.GOAL_RADIUS ** 2

    def _in_tunnel(self, x, y):
        """True when cursor is inside the tunnel corridor (between the two goals)."""
        in_x = self.lx <= x <= self.rx
        in_y = self.ty <= y <= self.by
        return in_x and in_y

    # ------------------------------------------------------------------
    def _refresh(self):
        if self.done:
            self.goal_left.color = DONE_GOAL_COLOR
            self.goal_right.color = DONE_GOAL_COLOR
            self.wall_top.color = IDLE_WALL_COLOR
            self.wall_bot.color = IDLE_WALL_COLOR
            self.status_lbl.text = "Done!  Press R to restart, Q to quit."
            self.status_lbl.color = _rgba(APPLE_GREEN)
            self.info_lbl.text = f"Logged to {os.path.basename(self.logger.path)}"
            self.violation_lbl.text = ""
            return

        # Colour active / idle goals
        self._start_goal.color = IDLE_WALL_COLOR
        self._end_goal.color = GOAL_COLOR

        if self.active:
            self.wall_top.color = ACTIVE_WALL_COLOR
            self.wall_bot.color = ACTIVE_WALL_COLOR
            self.status_lbl.text = "Steer to the red target"
        else:
            self.wall_top.color = IDLE_WALL_COLOR
            self.wall_bot.color = IDLE_WALL_COLOR
            # Highlight which end to start from
            self._start_goal.color = GOAL_COLOR
            self._end_goal.color = IDLE_WALL_COLOR
            self.status_lbl.text = "Move into the highlighted target to begin"

        self.status_lbl.color = _rgba(TEXT_LIGHT)

        c = self.cfg
        direction_str = "→" if self.direction == self.LEFT_TO_RIGHT else "←"
        self.info_lbl.text = (
            f"PID {c['pid']}  ·  trial {self.trial}/{c['num_trials']}  "
            f"·  iteration {self.iteration}/{c['iterations']}  {direction_str}  "
            f"·  W={c['tunnel_w']} A={c['tunnel_a']}"
        )
        self.violation_lbl.text = f"Wall hits: {self.violations}" if self.violations else ""

    # ------------------------------------------------------------------
    def on_mouse_move(self, x, y):
        self.cursor.x, self.cursor.y = x, y

        if self.done:
            return

        if not self.active:
            # Wait for cursor to enter the start goal before timing begins
            if self._in_goal(self._start_goal, x, y):
                self.active = True
                self.violations = 0
                self.start_ts = int(time.time() * 1000)
                self._refresh()
            return

        # --- Traversal in progress ---

        # Wall-crossing detection: cursor has left both the tunnel AND the goals
        if not self._in_tunnel(x, y) and not self._in_goal(self._start_goal, x, y) \
                and not self._in_goal(self._end_goal, x, y):
            # Only count one violation until cursor re-enters the tunnel
            if not getattr(self, "_outside", False):
                self.violations += 1
                self._outside = True
                self.violation_lbl.text = f"Wall hits: {self.violations}"
        else:
            self._outside = False

        # Check arrival at end goal
        if self._in_goal(self._end_goal, x, y):
            end_ts = int(time.time() * 1000)
            direction_str = "LR" if self.direction == self.LEFT_TO_RIGHT else "RL"
            self.logger.log(
                self.trial, self.iteration, direction_str,
                self.start_ts, end_ts, self.violations,
            )

            # Flip direction for return leg
            self.direction = (
                self.RIGHT_TO_LEFT if self.direction == self.LEFT_TO_RIGHT
                else self.LEFT_TO_RIGHT
            )
            self.active = False
            self._outside = False

            # Only increment iteration after both legs of a go-and-return
            if self.direction == self.LEFT_TO_RIGHT:
                self.iteration += 1
                if self.iteration > self.cfg["iterations"]:
                    self.done = True

            self._refresh()

    # ------------------------------------------------------------------
    def draw(self):
        self.batch.draw()


# ---------- main ----------
def main():
    cfg = load_config()
    logger = Logger(cfg)

    config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
    try:
        win = pyglet.window.Window(WINDOW_W, WINDOW_H, caption="Steering Law", resizable=False, config=config)
    except pyglet.window.NoSuchConfigException:
        win = pyglet.window.Window(WINDOW_W, WINDOW_H, caption="Steering Law", resizable=False)

    pyglet.gl.glClearColor(*BG_GL)
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    pyglet.gl.glEnable(pyglet.gl.GL_MULTISAMPLE)

    app = SteeringLawApp(win, cfg, logger)

    @win.event
    def on_draw():
        win.clear()
        app.draw()

    @win.event
    def on_mouse_motion(x, y, dx, dy):
        app.on_mouse_move(x, y)

    @win.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        app.on_mouse_move(x, y)

    @win.event
    def on_key_press(symbol, modifiers):
        if symbol in (pyglet.window.key.Q, pyglet.window.key.ESCAPE):
            pyglet.app.exit()
        elif symbol == pyglet.window.key.R:
            app.restart()

    try:
        pyglet.app.run()
    finally:
        logger.close()


if __name__ == "__main__":
    main()