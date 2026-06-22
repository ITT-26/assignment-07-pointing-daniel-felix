import os
import json
import math
import time
import argparse
from collections import deque

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

IDLE_COLOR = MID_GRAY
ACTIVE_COLOR = APPLE_RED
CURSOR_COLOR = APPLE_GREEN

FONT = "Arial"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULTS = {
    "pid": 1,
    "num_targets": 9,
    "target_w": 60,
    "target_d": 400,
    "iterations": 3,
    "latency_ms": 0,
}


def _rgba(rgb, a=255):
    return (*rgb, a)


def crossing_order(n):
    # e.g. for n=10 -> [0,5,1,6,2,7,3,8,4,9].
    half = (n + 1) // 2
    return [i // 2 + (half if i % 2 else 0) for i in range(n)]


def load_config():
    parser = argparse.ArgumentParser(description="Fitts' Law study application.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "fitts_config.json"),
                        help="Path to JSON config file (default: fitts_config.json).")
    parser.add_argument("--pid", type=int,
                        help=f"Participant ID (default: {DEFAULTS['pid']}).")
    parser.add_argument("--num-targets", type=int, dest="num_targets",
                        help=f"Number of targets in the ring (default: {DEFAULTS['num_targets']}).")
    parser.add_argument("--width", type=int, dest="target_w",
                        help=f"Target diameter W in pixels (default: {DEFAULTS['target_w']}).")
    parser.add_argument("--distance", type=int, dest="target_d",
                        help=f"Ring diameter D in pixels (default: {DEFAULTS['target_d']}).")
    parser.add_argument("--iterations", type=int,
                        help=f"Number of full rings to repeat (default: {DEFAULTS['iterations']}).")
    parser.add_argument("--latency", type=int, dest="latency_ms",
                        help=f"Artificial pointer latency in ms (default: {DEFAULTS['latency_ms']}).")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))

    # command-line arguments override the config file
    for key in ("pid", "num_targets", "target_w", "target_d", "iterations", "latency_ms"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    return cfg


# ---------- Latency buffer ----------
class LatencyBuffer:
    """Delays (x, y) positions by a fixed number of milliseconds.

    Call push(x, y) on every raw mouse event, then read .x and .y for the
    delayed position.  When latency_ms == 0 the buffer is a no-op.
    """
    def __init__(self, latency_ms):
        self.latency_ms = latency_ms
        self._buf = deque()   # (timestamp_ms, x, y)
        self.x = WINDOW_W // 2
        self.y = WINDOW_H // 2

    def push(self, x, y):
        now = time.perf_counter() * 1000
        self._buf.append((now, x, y))
        if self.latency_ms == 0:
            # Zero latency: always use the freshest position
            self.x, self.y = x, y
        else:
            cutoff = now - self.latency_ms
            while len(self._buf) > 1 and self._buf[1][0] <= cutoff:
                self._buf.popleft()
            _, self.x, self.y = self._buf[0]


# ---------- data logging ----------
class Logger:
    HEADER = "iteration,pid,num_targets,target_w,target_d,latency_ms,target_id,timestamp\n"

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(DATA_DIR, exist_ok=True)
        base = f"fitts_{cfg['num_targets']}_{cfg['target_w']}_{cfg['target_d']}_{cfg['pid']}"
        path = os.path.join(DATA_DIR, base + ".csv")
        # don't clobber existing data
        n = 1
        while os.path.exists(path):
            path = os.path.join(DATA_DIR, f"{base}_{n}.csv")
            n += 1
        self.path = path
        self.file = open(path, "w", newline="")
        self.file.write(self.HEADER)
        self.file.flush()

    def log(self, iteration, target_id):
        c = self.cfg
        ts = int(time.time() * 1000)
        self.file.write(f"{iteration},{c['pid']},{c['num_targets']},"
                        f"{c['target_w']},{c['target_d']},{c['latency_ms']},"
                        f"{target_id},{ts}\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


# ---------- App ----------
class FittsLawApp:
    def __init__(self, window, cfg, logger, latency_buf):
        self.window = window
        self.cfg = cfg
        self.logger = logger
        self.latency_buf = latency_buf
        self.batch = pyglet.graphics.Batch()

        g_target = pyglet.graphics.Group(order=1)
        g_cursor = pyglet.graphics.Group(order=2)
        g_text = pyglet.graphics.Group(order=3)

        self.n = cfg["num_targets"]
        self.radius = cfg["target_w"] / 2
        self.order = crossing_order(self.n)
        self.iterations = cfg["iterations"]

        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        ring_r = cfg["target_d"] / 2
        self.targets = []
        seg = max(64, int(self.radius * 3))
        for i in range(self.n):
            theta = math.pi / 2 + 2 * math.pi * i / self.n  # start at top, go counterclockwise
            tx = cx + ring_r * math.cos(theta)
            ty = cy + ring_r * math.sin(theta)
            self.targets.append(pyglet.shapes.Circle(
                tx, ty, self.radius, segments=seg,
                color=IDLE_COLOR, batch=self.batch, group=g_target))
        print(f"Targets at: {[f'({t.x:.1f}, {t.y:.1f})' for t in self.targets]}")

        self.cursor = pyglet.shapes.Circle(cx, cy, 8, segments=48, color=CURSOR_COLOR,
                                           batch=self.batch, group=g_cursor)

        self.info_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=14,
            x=WINDOW_W // 2, y=28, anchor_x="center", anchor_y="center",
            color=_rgba(TEXT_GRAY), batch=self.batch, group=g_text)

        self.status_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=22,
            x=WINDOW_W // 2, y=WINDOW_H - 40, anchor_x="center", anchor_y="center",
            color=_rgba(TEXT_LIGHT), batch=self.batch, group=g_text)

        self.latency_lbl = pyglet.text.Label(
            "", font_name=FONT, font_size=13,
            x=WINDOW_W - 12, y=28, anchor_x="right", anchor_y="center",
            color=_rgba(TEXT_GRAY), batch=self.batch, group=g_text)

        self.restart()

    def restart(self):
        self.iteration = 1
        self.step = 0
        self.done = False
        self._refresh()

    @property
    def active_target(self):
        return self.targets[self.order[self.step]]

    def _refresh(self):
        for t in self.targets:
            t.color = IDLE_COLOR
        lat = self.cfg["latency_ms"]
        self.latency_lbl.text = f"latency {lat} ms" if lat > 0 else ""
        if self.done:
            self.status_lbl.text = "Done!  Press R to restart, Q to quit."
            self.status_lbl.color = _rgba(APPLE_GREEN)
            self.info_lbl.text = f"Logged to {os.path.basename(self.logger.path)}"
            return
        self.active_target.color = ACTIVE_COLOR
        self.status_lbl.text = "Click the red target"
        self.status_lbl.color = _rgba(TEXT_LIGHT)
        c = self.cfg
        self.info_lbl.text = (f"PID {c['pid']}  ·  iteration {self.iteration}/{self.iterations}  "
                              f"·  target {self.step + 1}/{self.n}  ·  "
                              f"W={c['target_w']} D={c['target_d']}")

    def on_mouse_move(self, x, y):
        self.latency_buf.push(x, y)
        self.cursor.x = self.latency_buf.x
        self.cursor.y = self.latency_buf.y

    def on_click(self, x, y):
        if self.done:
            return
        # clicks use the delayed position too
        self.latency_buf.push(x, y)
        dx, dy = self.latency_buf.x, self.latency_buf.y
        t = self.active_target
        if (dx - t.x) ** 2 + (dy - t.y) ** 2 <= self.radius ** 2:
            self.logger.log(self.iteration, self.step)
            self.step += 1
            if self.step >= self.n:
                self.step = 0
                self.iteration += 1
                if self.iteration > self.iterations:
                    self.done = True
            self._refresh()

    def draw(self):
        self.batch.draw()


def main():
    cfg = load_config()
    logger = Logger(cfg)
    latency_buf = LatencyBuffer(cfg["latency_ms"])

    config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
    try:
        win = pyglet.window.Window(WINDOW_W, WINDOW_H, caption="Fitts' Law", resizable=False, config=config)
    except pyglet.window.NoSuchConfigException:
        win = pyglet.window.Window(WINDOW_W, WINDOW_H, caption="Fitts' Law", resizable=False)
    pyglet.gl.glClearColor(*BG_GL)
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    pyglet.gl.glEnable(pyglet.gl.GL_MULTISAMPLE)

    app = FittsLawApp(win, cfg, logger, latency_buf)

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
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            app.on_click(x, y)

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