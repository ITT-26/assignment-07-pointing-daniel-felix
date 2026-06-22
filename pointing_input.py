import os
import sys
import time
import threading
from collections import deque

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import math
import cv2
from pyglet.display import get_display
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput import keyboard
from pynput.mouse import Controller, Button
from OneEuroFilter import OneEuroFilter


VIDEO_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mediapipe_sample_code", "hand_landmarker.task")

# Landmarks for MediaPipe Hands
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

ACTIVE_LO, ACTIVE_HI = 0.20, 0.80  # only track hand when it within the central 60% of the camera frame
PINCH_ON, PINCH_OFF = 0.25, 0.35  # mouse down below ON, up above OFF to prevent jitter
LATCH_DELAY = 0.12  # lock click position to where the cursor was this long ago
CLICK_FLASH_FRAMES = 6
DISPLAY_W = 720

# One Euro Filter for smoothing: https://gery.casiez.net/1euro/
EURO_CONFIG = {"freq": 30, "mincutoff": 1.0, "beta": 0.01, "dcutoff": 1.0}


# ---------- Camera thread ----------
class CameraThread:
    def __init__(self, video_id):
        backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(video_id, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ---------- Helpers ----------
def remap(v, lo, hi):
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def dist(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def screen_to_frame(mx, my, w, h):
    fx = ACTIVE_LO + (mx / SCREEN_W) * (ACTIVE_HI - ACTIVE_LO)
    fy = ACTIVE_LO + (my / SCREEN_H) * (ACTIVE_HI - ACTIVE_LO)
    return int(fx * w), int(fy * h)


# --------- Setup ----------
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

mouse = Controller()
_screen = get_display().get_default_screen()
SCREEN_W, SCREEN_H = _screen.width, _screen.height
cap = CameraThread(VIDEO_ID)

# One Euro Filter per axis, created on demand
filter_x = None
filter_y = None
pinching = False
pos_history = deque(maxlen=60)
click_pos = None  # screen position of the last click
click_flash = 0
last_ts = 0

running = True
show_debug = True
control_active = True

# GUI runs in its own thread so a grabbed/dragged window can't freeze pointer control
display_frame = None
display_lock = threading.Lock()


def gui_loop():
    while running:
        with display_lock:
            f = display_frame
        if f is not None:
            cv2.imshow("pointing_input (m: control, d: skeleton, q: quit)", f)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def on_press(key):
    global running, show_debug, control_active, pinching
    if key == keyboard.KeyCode.from_char("q"):
        running = False
    elif key == keyboard.KeyCode.from_char("d"):  # toggle debug skeleton
        show_debug = not show_debug
    elif key == keyboard.KeyCode.from_char("m"):  # toggle mouse control
        control_active = not control_active
        pinching = False


keyboard.Listener(on_press=on_press).start()
threading.Thread(target=gui_loop, daemon=True).start()


# ---------- Main loop ----------
while running:
    frame = cap.read()
    if frame is None:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    last_ts = max(last_ts + 1, int(time.time() * 1000))  # must strictly increase for Medipipe's VIDEO mode
    result = detector.detect_for_video(mp_image, last_ts)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        # Average of thumb and index fingertips as the pointing position
        ax = (lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2
        ay = (lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2

        sx = remap(ax, ACTIVE_LO, ACTIVE_HI) * SCREEN_W
        sy = remap(ay, ACTIVE_LO, ACTIVE_HI) * SCREEN_H

        # One Euro Filter smoothing
        now = time.time()
        if filter_x is None:
            filter_x = OneEuroFilter(**EURO_CONFIG)
            filter_y = OneEuroFilter(**EURO_CONFIG)
        px = filter_x(sx, now)
        py = filter_y(sy, now)

        hand_size = dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6
        pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_size  # normalize by hand size

        mx = min(SCREEN_W - 1, max(0, int(px)))
        my = min(SCREEN_H - 1, max(0, int(py)))
        pos_history.append((now, mx, my))
        if click_flash > 0:
            click_flash -= 1

        if control_active:
            mouse.position = (mx, my)
            if not pinching and pinch < PINCH_ON:  # pinch down -> click
                lx, ly = mx, my
                for t, hx, hy in reversed(pos_history):
                    if t <= now - LATCH_DELAY:
                        lx, ly = hx, hy
                        break
                mouse.position = (lx, ly)
                mouse.click(Button.left)
                mouse.position = (mx, my)
                pinching = True
                click_pos = (lx, ly)
                click_flash = CLICK_FLASH_FRAMES
            elif pinching and pinch > PINCH_OFF:  # release -> re-arm
                pinching = False

        if click_flash > 0 and click_pos is not None:
            cx, cy = screen_to_frame(click_pos[0], click_pos[1], w, h)
        else:
            cx, cy = screen_to_frame(mx, my, w, h)
        if click_flash > 0:
            cv2.circle(frame, (cx, cy), 16, (0, 255, 0), -1, cv2.LINE_AA)  # click fired
        elif pinching:
            cv2.circle(frame, (cx, cy), 12, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 12, (90, 200, 255), 2, cv2.LINE_AA)  # ready
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

        if show_debug:
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"pinch {pinch:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        pinching = False
        click_flash = 0

    # Display border of active region
    cv2.rectangle(frame, (int(ACTIVE_LO * w), int(ACTIVE_LO * h)),
                  (int(ACTIVE_HI * w), int(ACTIVE_HI * h)), (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "control ON" if control_active else "control OFF (m)", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 0) if control_active else (0, 0, 255),
                2, cv2.LINE_AA)

    with display_lock:
        display_frame = cv2.resize(frame, (DISPLAY_W, int(DISPLAY_W * h / w)))

cap.release()