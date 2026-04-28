"""
Driver Awareness Monitor  (MediaPipe >= 0.10)
=============================================
Tracks 3 factors in real-time and computes an awareness score (0-100):

  Factor              Weight
  ─────────────────────────────────────────────
  PERCLOS (eye open%)   40%   -- drowsiness gold standard
  Head turn (yaw)       35%   -- looking away from road
  Head tilt (roll)      25%   -- nodding off / drooping

Dominant factor drives the alert message shown to the driver.
Message stays visible as long as the condition persists.
Cooldown only governs the flashing border (not the text).

Dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python face_analysis.py
"""

import os
import sys
import time
import urllib.request
import collections

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model download ───────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.", flush=True)

# ── Landmark indices ─────────────────────────────────────────────────────────
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57

MODEL_POINTS = np.array([
    [  0.0,    0.0,    0.0  ],
    [  0.0,  -63.6,  -12.5 ],
    [-43.3,   32.7,  -26.0 ],
    [ 43.3,   32.7,  -26.0 ],
    [-28.9,  -28.9,  -24.1 ],
    [ 28.9,  -28.9,  -24.1 ],
], dtype=np.float64)

EAR_CLOSED = 0.15
EAR_OPEN   = 0.32

# ── Scoring constants ────────────────────────────────────────────────────────
W_PERCLOS = 0.40
W_TURN    = 0.35
W_TILT    = 0.25

TURN_SAFE  = 35.0
TURN_MAX   = 50.0
TILT_SAFE  = 8.0
TILT_MAX   = 25.0

PERCLOS_WINDOW      = 50
EYE_CLOSED_THRESHOLD = 30.0

# Flash cooldown — controls how often the red border flashes (not the message)
FLASH_COOLDOWN = {3: 2.0, 2: 5.0, 1: 8.0}

# ── MediaPipe setup ──────────────────────────────────────────────────────────
base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
opts = mp_vision.FaceLandmarkerOptions(
    base_options=base_opts,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

# ── Colours ──────────────────────────────────────────────────────────────────
FONT     = cv2.FONT_HERSHEY_SIMPLEX
C_WHITE  = (230, 230, 230)
C_GRAY   = (110, 110, 110)
C_GREEN  = ( 80, 200, 120)
C_AMBER  = ( 40, 190, 240)
C_ORANGE = ( 30, 140, 255)
C_RED    = ( 60,  60, 220)
C_PANEL  = ( 22,  22,  22)
C_BAR_BG = ( 55,  55,  55)

# ── Math ─────────────────────────────────────────────────────────────────────
def eye_aspect_ratio(lm, indices, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)

def ear_to_pct(e):
    return float(np.clip((e - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED) * 100, 0, 100))

def head_pose(lm, w, h):
    img_pts = np.array([
        [lm[NOSE_TIP].x    * w, lm[NOSE_TIP].y    * h],
        [lm[CHIN].x        * w, lm[CHIN].y        * h],
        [lm[LEFT_EYE_L].x  * w, lm[LEFT_EYE_L].y  * h],
        [lm[RIGHT_EYE_R].x * w, lm[RIGHT_EYE_R].y * h],
        [lm[LEFT_MOUTH].x  * w, lm[LEFT_MOUTH].y  * h],
        [lm[RIGHT_MOUTH].x * w, lm[RIGHT_MOUTH].y * h],
    ], dtype=np.float64)
    focal   = float(w)
    cam_mat = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    dist    = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam_mat, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    yaw  = np.degrees(np.arctan2(-rmat[2,0], sy))
    roll = np.degrees(np.arctan2( rmat[1,0], rmat[0,0])) if sy > 1e-6 else 0.0
    return float(yaw), float(roll)

# ── Scoring ──────────────────────────────────────────────────────────────────
def perclos_score(perclos_pct):
    return float(np.clip(100 - perclos_pct, 0, 100))

def turn_score(yaw_deg):
    angle = abs(yaw_deg)
    if angle <= TURN_SAFE: return 100.0
    if angle >= TURN_MAX:  return 0.0
    return float(100 - (angle - TURN_SAFE) / (TURN_MAX - TURN_SAFE) * 100)

def tilt_score(roll_deg):
    angle = abs(roll_deg)
    if angle <= TILT_SAFE: return 100.0
    if angle >= TILT_MAX:  return 0.0
    return float(100 - (angle - TILT_SAFE) / (TILT_MAX - TILT_SAFE) * 100)

def compute_awareness(perclos_pct, yaw, roll):
    s_perclos = perclos_score(perclos_pct)
    s_turn    = turn_score(yaw)
    s_tilt    = tilt_score(roll)
    score     = s_perclos * W_PERCLOS + s_turn * W_TURN + s_tilt * W_TILT
    if s_perclos < 50:
        score = 0
    elif s_turn < 15:
        score = 0
    factors   = {"perclos": s_perclos, "turn": s_turn, "tilt": s_tilt}
    dominant  = min(factors, key=factors.get)
    return round(score, 1), dominant, s_perclos, s_turn, s_tilt

def alert_level(score):
    if score >= 70: return 0
    if score >= 50: return 1
    if score >= 30: return 2
    return 3

# Message table — keyed by (dominant, level)
# Message is ALWAYS shown while condition is active; no cooldown on text.
MESSAGES = {
    ("perclos", 1): ("Stay alert — eyes getting heavy",   C_AMBER),
    ("perclos", 2): ("You're drowsy — consider a break",  C_ORANGE),
    ("perclos", 3): ("WAKE UP!  Pull over now!",          C_RED),
    ("turn",    1): ("Eyes on the road",                  C_AMBER),
    ("turn",    2): ("Keep your focus on the road!",      C_ORANGE),
    ("turn",    3): ("LOOK AT THE ROAD!",                 C_RED),
    ("tilt",    1): ("Sit up straight",                   C_AMBER),
    ("tilt",    2): ("Head drooping — stay awake!",       C_ORANGE),
    ("tilt",    3): ("WAKE UP!  Pull over now!",          C_RED),
}

def score_color(score):
    if score >= 70: return C_GREEN
    if score >= 50: return C_AMBER
    if score >= 30: return C_ORANGE
    return C_RED

# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_eye_outline(frame, lm, indices, w, h):
    pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in indices], np.int32)
    cv2.polylines(frame, [pts], True, C_GREEN, 1, cv2.LINE_AA)

def factor_bar(panel, label, sub_score, x, y, bw=170):
    pct   = sub_score / 100.0
    color = (C_GREEN if sub_score >= 70 else
             C_AMBER if sub_score >= 50 else
             C_ORANGE if sub_score >= 30 else C_RED)
    cv2.rectangle(panel, (x, y), (x + bw, y + 7), C_BAR_BG, -1)
    if pct > 0:
        cv2.rectangle(panel, (x, y), (x + int(bw * pct), y + 7), color, -1)
    cv2.putText(panel, f"{label}  {sub_score:.0f}",
                (x, y - 7), FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)

def draw_panel(canvas, w, h,
               l_pct, r_pct, perclos_pct,
               yaw, roll,
               score, dominant,
               s_perclos, s_turn, s_tilt,
               alert_lvl, do_flash):

    PANEL_H = 210
    panel   = canvas[h: h + PANEL_H]
    panel[:] = C_PANEL

    # ── Left column: raw readings ────────────────────────────────────────────
    lx = 20
    cv2.putText(panel, f"Left eye   : {l_pct:.0f}%",
                (lx, 28),  FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Right eye  : {r_pct:.0f}%",
                (lx, 52),  FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"PERCLOS    : {perclos_pct:.0f}% closed",
                (lx, 76),  FONT, 0.45, C_AMBER, 1, cv2.LINE_AA)

    tilt_dir = "L" if roll < -2 else ("R" if roll >  2 else "-")
    turn_dir = "L" if yaw  >  2 else ("R" if yaw  < -2 else "-")
    cv2.putText(panel, f"Head tilt  : {abs(roll):.1f} deg {tilt_dir}",
                (lx, 100), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Head turn  : {abs(yaw):.1f} deg {turn_dir}",
                (lx, 124), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # ── Divider ──────────────────────────────────────────────────────────────
    mid = w // 2
    cv2.line(panel, (mid - 10, 10), (mid - 10, PANEL_H - 10), (50, 50, 50), 1)

    # ── Right column: factor sub-scores ──────────────────────────────────────
    rx = mid + 10
    cv2.putText(panel, "Factor scores", (rx, 20), FONT, 0.44, C_GRAY, 1, cv2.LINE_AA)
    factor_bar(panel, "PERCLOS", s_perclos, rx, 38,  bw=w - rx - 20)
    factor_bar(panel, "Turn   ", s_turn,    rx, 72,  bw=w - rx - 20)
    factor_bar(panel, "Tilt   ", s_tilt,    rx, 106, bw=w - rx - 20)

    # ── Awareness score ───────────────────────────────────────────────────────
    cv2.putText(panel, f"Awareness: {score:.0f} / 100",
                (lx, 158), FONT, 0.70, score_color(score), 2, cv2.LINE_AA)

    # ── Alert message — always visible while condition is active ─────────────
    if alert_lvl > 0:
        msg, msg_color = MESSAGES[(dominant, alert_lvl)]
        cv2.putText(panel, msg, (lx, 193), FONT, 0.62, msg_color, 2, cv2.LINE_AA)

        # Red border flash — gated by cooldown so it isn't constant noise
        if alert_lvl == 3 and do_flash:
            cv2.rectangle(canvas, (0, 0), (w, h + PANEL_H), C_RED, 5)
        elif alert_lvl == 2 and do_flash:
            cv2.rectangle(canvas, (0, 0), (w, h + PANEL_H), C_ORANGE, 3)

    cv2.putText(panel, "Q quit", (w - 75, PANEL_H - 8), FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Error: could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Driver Awareness Monitor running — press Q to quit.")

    PANEL_H       = 210
    ts_ms         = 0
    frame_count   = 0
    eye_history   = collections.deque(maxlen=PERCLOS_WINDOW)
    last_flash_t  = {1: 0.0, 2: 0.0, 3: 0.0}   # cooldown only for border flash

    # Cache last computed values so skipped frames still render
    last = dict(l_pct=0.0, r_pct=0.0, perclos_pct=0.0,
                yaw=0.0, roll=0.0, score=100.0,
                dominant="perclos", s_perclos=100.0,
                s_turn=100.0, s_tilt=100.0, alert_lvl=0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Only run inference every 3rd frame for performance ───────────────
        if frame_count % 5 == 0:
            continue
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms   += 100   # 3 frames × ~33ms
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        l_pct = r_pct = 0.0
        yaw   = roll  = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            l_pct = ear_to_pct(eye_aspect_ratio(lm, LEFT_EYE,  w, h))
            r_pct = ear_to_pct(eye_aspect_ratio(lm, RIGHT_EYE, w, h))
            yaw, roll = head_pose(lm, w, h)
            draw_eye_outline(frame, lm, LEFT_EYE,  w, h)
            draw_eye_outline(frame, lm, RIGHT_EYE, w, h)
            cv2.circle(frame,
                        (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)),
                        3, C_AMBER, -1)

        # PERCLOS rolling window
        avg_eye = (l_pct + r_pct) / 2.0 if result.face_landmarks else 0.0
        eye_history.append(avg_eye)
        closed_frames = sum(1 for e in eye_history if e < EYE_CLOSED_THRESHOLD)
        perclos_pct   = (closed_frames / len(eye_history)) * 100.0

        score, dominant, s_perclos, s_turn, s_tilt = compute_awareness(
            perclos_pct, yaw, roll)
        lvl = alert_level(score)

        last.update(l_pct=l_pct, r_pct=r_pct, perclos_pct=perclos_pct,
                    yaw=yaw, roll=roll, score=score, dominant=dominant,
                    s_perclos=s_perclos, s_turn=s_turn, s_tilt=s_tilt,
                    alert_lvl=lvl)
        # else:
        #     # Reuse landmarks on skipped frames
        #     if last["alert_lvl"] > 0:
        #         lm_cache = locals().get("lm")
        #         if lm_cache:
        #             draw_eye_outline(frame, lm_cache, LEFT_EYE,  w, h)
        #             draw_eye_outline(frame, lm_cache, RIGHT_EYE, w, h)

        # ── Flash cooldown — border only, never hides the message ────────────
        now      = time.time()
        lvl      = last["alert_lvl"]
        do_flash = False
        if lvl > 0 and (now - last_flash_t[lvl]) >= FLASH_COOLDOWN[lvl]:
            do_flash        = True
            last_flash_t[lvl] = now

        # ── Render ───────────────────────────────────────────────────────────
        canvas     = np.zeros((h + PANEL_H, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        draw_panel(canvas, w, h,
                   last["l_pct"], last["r_pct"], last["perclos_pct"],
                   last["yaw"],   last["roll"],
                   last["score"], last["dominant"],
                   last["s_perclos"], last["s_turn"], last["s_tilt"],
                   last["alert_lvl"], do_flash)

        cv2.imshow("Driver Awareness Monitor", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()