import cv2
import numpy as np
import time
import os, json
from collections import deque
from common.hands import HandTracker
from common.sound import Sounder
import mediapipe as mp

# ------------------- CONFIG -------------------
WIN_NAME    = 'Air Drums — pad mode'
MIRROR_VIEW = True
PAN_SWAP    = False

ZONE_NAMES  = ['SNARE', 'HIHAT', 'CLAP', 'KICK', 'BASS']
ZONE_SOUNDS = ['snare','hihat','clap','kick','bass']

SOUND_FILES = {
    'snare': 'snare.wav',
    'hihat': 'hihat.wav',
    'clap':  'clap.wav',
    'kick':  'kick.wav',
    'bass':  'bass.wav',
}

# speed -> volume
V_MIN_BASE_DEFAULT = 120
V_MAX  = 900
VOL_FLOOR = 0.65
VOL_EXP   = 0.75

# --- HITS ARE INDEX-FINGER SPEED ONLY ---
TIP_HIT_MIN_DEFAULT  = 200.0   # '[' lower, ']' raise
TIP_HIT_RESET_RATIO  = 0.55
MIN_INSIDE_FRAMES    = 2       # dwell before eligible

# cooldowns + flash
REFRACTORY_SLOT  = 0.08
PAD_COOLDOWN     = 0.10
HIT_FLASH_SEC    = 0.18

# optional head-kick
KICK_COOLDOWN = 0.3
YAW_THRESH    = 0.09
YAW_WINDOW    = 0.4

# ---- Layout tuned to your arrows ----
PAD_W_FRAC = 0.22        # bigger pads
PAD_H_FRAC = 0.26
PAD_INSET_ACTIVE = 0.12  # larger active area

# Top row: HIHAT (L), SNARE (center-high), CLAP (R)
# Bottom row: KICK (L), BASS (R) — both raised a little
PAD_LAYOUT = [
    (0.50, 0.19),  # 0 SNARE (higher)
    (0.17, 0.27),  # 1 HIHAT (slightly lower than before)
    (0.83, 0.27),  # 2 CLAP  (slightly lower than before)
    (0.30, 0.72),  # 3 KICK  (raised)
    (0.70, 0.72),  # 4 BASS  (raised)
]
LAYOUT_FILE = os.path.join(os.path.dirname(__file__), "pad_layout.json")

# colors
CLR_PAD_DEFAULT   = (110,  80, 220)  # pink-ish
CLR_PAD_OCCUPIED  = (180, 255, 180)  # light green when hand inside
CLR_PAD_HIT       = (  0, 180,   0)  # dark green flash on hit

CLR_BONES_DEFAULT = ( 30, 220, 220)  # teal
CLR_BONES_OCC     = (140, 255, 140)  # light green
CLR_BONES_HIT     = (  0, 200,   0)  # dark green

CLR_TEXT          = (255,255,255)
CLR_GRID          = (70,70,70)
# ------------------------------------------------

# -------------------- UTILS ---------------------
def ema(new, old, alpha): return alpha*new + (1-alpha)*old
def clamp(v, lo, hi): return max(lo, min(hi, v))

def median_tail(deq, n):
    arr = list(deq)
    if not arr:
        return 0.0
    return float(np.median(arr[-n:]))

def rect_from_center(cx, cy, w, h):
    x0 = int(cx - w/2); y0 = int(cy - h/2)
    x1 = int(cx + w/2); y1 = int(cy + h/2)
    return (x0, y0, x1, y1)

def inset_rect(rect, fx, fy):
    x0,y0,x1,y1 = rect
    w = (x1 - x0 + 1); h = (y1 - y0 + 1)
    dx = int(w * fx);  dy = int(h * fy)
    return (x0+dx, y0+dy, x1-dx, y1-dy)

def pt_in_rect(pt, rect):
    x,y = pt; x0,y0,x1,y1 = rect
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def draw_pad(img, rect, label, fill_color, text_color=CLR_TEXT):
    x0,y0,x1,y1 = rect
    cv2.rectangle(img, (x0,y0), (x1,y1), fill_color, -1)
    cv2.rectangle(img, (x0,y0), (x1,y1), (255,255,255), 2)
    cv2.putText(img, label, (x0+10, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

def put_text_with_bg(img, text, org, font_scale=0.75, color=CLR_TEXT, thick=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thick)
    x, y = org
    cv2.rectangle(img, (x-6, y-th-6), (x+tw+6, y+6), (0,0,0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thick, cv2.LINE_AA)

def draw_hud(frame, fps, yaw, last_hit, px_speed_max, vmin, tip_hit_min):
    put_text_with_bg(frame, f"FPS: {fps:.1f}", (10, 28))
    put_text_with_bg(frame, f"Yaw: {yaw:+.2f}", (10, 56))
    put_text_with_bg(frame, f"TipSpeed(max): {int(px_speed_max)} px/s  Vmin:{int(vmin)}", (10, 84))
    put_text_with_bg(frame, f"HitMin: {int(tip_hit_min)} px/s", (10, 112))
    lh = last_hit.get('left'); rh = last_hit.get('right')
    left_name  = ZONE_SOUNDS[lh['zone']] if lh else "-"
    right_name = ZONE_SOUNDS[rh['zone']] if rh else "-"
    put_text_with_bg(frame, f"Last left hit : {left_name}",  (10, 144))
    put_text_with_bg(frame, f"Last right hit: {right_name}", (10, 172))

# hand graph (21 points)
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
def draw_full_hand(frame, hand, color, label, thickness=2):
    for a,b in HAND_EDGES:
        pa = tuple(map(int, hand[a])); pb = tuple(map(int, hand[b]))
        cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(21):
        p = tuple(map(int, hand[i]))
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
    tip = tuple(map(int, hand[8]))
    cv2.putText(frame, label, (tip[0]+8, tip[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def draw_pads(frame, pad_rects, occupied, last_strike_time):
    overlay = frame.copy()
    now = time.time()
    for i, rect in enumerate(pad_rects):
        if now - last_strike_time[i] <= HIT_FLASH_SEC:
            col = CLR_PAD_HIT
        elif occupied[i]:
            col = CLR_PAD_OCCUPIED
        else:
            col = CLR_PAD_DEFAULT
        draw_pad(overlay, rect, ZONE_NAMES[i], col)
    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

def draw_grid_overlay(frame, step=0.1):
    h, w = frame.shape[:2]
    for f in np.arange(step, 1.0, step):
        x = int(w*f); y = int(h*f)
        cv2.line(frame, (x,0), (x,h), CLR_GRID, 1)
        cv2.line(frame, (0,y), (w,y), CLR_GRID, 1)

def save_layout(pad_layout, pad_w_frac, pad_h_frac):
    data = {"layout": pad_layout, "w_frac": pad_w_frac, "h_frac": pad_h_frac}
    with open(LAYOUT_FILE, "w") as f:
        json.dump(data, f)
    print("[layout] saved to", LAYOUT_FILE)

def load_layout():
    if not os.path.exists(LAYOUT_FILE):
        print("[layout] no file to load"); return None
    with open(LAYOUT_FILE, "r") as f:
        data = json.load(f)
    print("[layout] loaded from", LAYOUT_FILE)
    return data

def assign_slots(hands, width):
    if not hands: return []
    hs = sorted(hands, key=lambda h: h[8,0])
    if len(hs) == 1:
        slot = 'left' if hs[0][8,0] < (width/2) else 'right'
        return [(slot, hs[0])]
    return [('left', hs[0]), ('right', hs[-1])]
# ------------------------------------------------

# -------------------- MAIN ----------------------
def main():
    global PAD_LAYOUT, PAD_W_FRAC, PAD_H_FRAC

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    hand_tracker = HandTracker()
    sounder = Sounder(os.path.join(os.path.dirname(__file__), 'sounds'))
    for name, file in SOUND_FILES.items():
        sounder.load(name, file)

    # alias missing samples
    SOUND_ALIAS = {}
    loaded = set(sounder.sounds.keys())
    for s in ZONE_SOUNDS:
        if s not in loaded:
            if 'clap' in loaded: SOUND_ALIAS[s] = 'clap'
            elif loaded:         SOUND_ALIAS[s] = next(iter(loaded))
            else:                SOUND_ALIAS[s] = None

    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    tip_hit_min = TIP_HIT_MIN_DEFAULT

    yaw_baseline   = 0.0
    yaw_state      = deque(maxlen=10)
    last_kick_time = 0.0

    fps_hist = deque(maxlen=30)
    vmin_hist = deque(maxlen=30)
    vmin_base = V_MIN_BASE_DEFAULT

    prev_tip   = {'left': None, 'right': None}
    ema_speed  = {'left': 0.0,  'right': 0.0}

    last_strike_time = {i: 0.0 for i in range(5)}
    strike_ready     = {'left': True, 'right': True}
    last_fire_slot   = {'left': 0.0,  'right': 0.0}

    inside_pad    = {'left': None, 'right': None}
    inside_frames = {'left': 0,    'right': 0}

    last_hit = {'left': None, 'right': None}
    last_fired_pad = {'left': None, 'right': None}

    # calibrator
    selected_pad = None
    show_grid = False

    last_frame_time = time.time()
    running = True
    while running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
        if MIRROR_VIEW: frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # pad rects
        PAD_W = int(w * PAD_W_FRAC)
        PAD_H = int(h * PAD_H_FRAC)
        pad_rects, active_rects = [], []
        for (xf, yf) in PAD_LAYOUT:
            cx = int(w * xf); cy = int(h * yf)
            rect = rect_from_center(cx, cy, PAD_W, PAD_H)
            pad_rects.append(rect)
            active_rects.append(inset_rect(rect, PAD_INSET_ACTIVE, PAD_INSET_ACTIVE))

        # hands
        hands = hand_tracker.process(frame)
        slots = [(s, hnd) for s,hnd in assign_slots(hands, w)]

        # face/yaw
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = mp_face.process(rgb)
        yaw = 0.0
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark
            x33 = lm[33].x * w; x263 = lm[263].x * w; x1 = lm[1].x * w
            face_cx = (x33 + x263)/2; face_w = abs(x33 - x263)
            yaw = (x1 - face_cx) / (face_w + 1e-5) - yaw_baseline
            yaw_state.append((time.time(), yaw))

        # head-shake kick
        kick = False
        if len(yaw_state) >= 2:
            t_arr, y_arr = zip(*yaw_state)
            for i in range(len(y_arr)-2):
                if y_arr[i] < -YAW_THRESH and y_arr[-1] > YAW_THRESH and (t_arr[-1]-t_arr[i]) < YAW_WINDOW:
                    if time.time() - last_kick_time > KICK_COOLDOWN:
                        kick = True; last_kick_time = time.time(); break

        # per-hand logic
        dt = max(t0 - last_frame_time, 1/60)
        max_ema_speed = 0.0
        now = time.time()
        occupied = [False]*5
        if show_grid: draw_grid_overlay(frame)

        for slot, hand in slots:
            tip = hand[8]

            # tip speed
            if prev_tip[slot] is None:
                prev_tip[slot] = tip.copy()
            ds = np.linalg.norm(tip - prev_tip[slot]) / dt
            ema_speed[slot] = ema(ds, ema_speed[slot], 0.5)
            prev_tip[slot] = tip.copy()
            vmin_hist.append(ema_speed[slot])

            # which pad?
            pad_idx = None
            for i, ar in enumerate(active_rects):
                if pt_in_rect(tip, ar):
                    pad_idx = i; break
            if pad_idx is not None:
                # turn pad light-green immediately when inside
                occupied[pad_idx] = True

            # dwell to reduce jitter before eligible to hit
            if pad_idx == inside_pad[slot]:
                inside_frames[slot] += 1
            else:
                inside_pad[slot] = pad_idx
                inside_frames[slot] = 1 if pad_idx is not None else 0
            eligible_dwell = (pad_idx is not None) and (inside_frames[slot] >= MIN_INSIDE_FRAMES)

            # volume baseline (for loudness only)
            vmin = max(vmin_base, median_tail(vmin_hist, 15) * 1.15)

            # re-arm when finger slows
            if (not strike_ready[slot]) and (ema_speed[slot] < TIP_HIT_MIN_DEFAULT * TIP_HIT_RESET_RATIO):
                strike_ready[slot] = True

            can_pad  = (pad_idx is not None) and (now - last_strike_time[pad_idx] > PAD_COOLDOWN)
            can_slot = strike_ready[slot] and (now - last_fire_slot[slot] > REFRACTORY_SLOT)

            # hit on finger speed crossing threshold inside pad
            if can_pad and can_slot and eligible_dwell and (ema_speed[slot] >= tip_hit_min):
                raw = clamp((ema_speed[slot] - vmin) / (V_MAX - vmin + 1e-6), 0.0, 1.0)
                vol = VOL_FLOOR + (1.0 - VOL_FLOOR) * (raw ** VOL_EXP)
                pan = (-0.5 if slot == 'left' else +0.5)
                if PAN_SWAP: pan = -pan
                play_name = SOUND_ALIAS.get(ZONE_SOUNDS[pad_idx], ZONE_SOUNDS[pad_idx])
                if play_name:
                    sounder.play(play_name, vol, pan=pan)
                last_strike_time[pad_idx] = now
                last_fire_slot[slot] = now
                strike_ready[slot] = False
                last_hit[slot] = {'zone': pad_idx, 't': now}
                last_fired_pad[slot] = pad_idx

            max_ema_speed = max(max_ema_speed, ema_speed[slot])

            # draw full hand: default → light green inside → dark green just after hit
            if now - last_fire_slot[slot] <= HIT_FLASH_SEC:
                hand_col = CLR_BONES_HIT
            elif pad_idx is not None:
                hand_col = CLR_BONES_OCC
            else:
                hand_col = CLR_BONES_DEFAULT
            draw_full_hand(frame, hand, hand_col, 'L' if slot=='left' else 'R', thickness=2)

        # optional head kick
        if kick:
            if len(yaw_state) >= 2:
                yaws = np.array([y for t,y in yaw_state]); ts = np.array([t for t,y in yaw_state])
                speeds = np.abs(np.diff(yaws)) / np.maximum(np.diff(ts), 1e-3)
                v = np.max(speeds) if len(speeds) else 0.1
                raw = clamp(v/0.6, 0.0, 1.0)
                vol = VOL_FLOOR + (1.0 - VOL_FLOOR) * (raw ** VOL_EXP)
                sounder.play('kick', vol, pan=0.0)

        # draw pads & HUD
        frame = draw_pads(frame, pad_rects, occupied, last_strike_time)
        fps_hist.append(1.0/(time.time()-t0+1e-6))
        avg_fps = sum(fps_hist)/len(fps_hist)
        vmin_hud = max(vmin_base, median_tail(vmin_hist, 15) * 1.15)
        draw_hud(frame, avg_fps, yaw, last_hit, max_ema_speed, vmin_hud, tip_hit_min)

        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key in (ord('f'), ord('F')):
            fullscreen = not (cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key in (ord('r'), ord('R')): yaw_baseline = yaw
        elif key == ord('['): tip_hit_min  = max(60.0, tip_hit_min - 10.0);  print("HitMin:", int(tip_hit_min))
        elif key == ord(']'): tip_hit_min  = min(400.0, tip_hit_min + 10.0); print("HitMin:", int(tip_hit_min))
        # calibrator
        elif key in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            selected_pad = int(chr(key)) - 1; print(f"[cal] selected pad {selected_pad} ({ZONE_NAMES[selected_pad]})")
        elif key in (ord('g'), ord('G')): show_grid = not show_grid
        elif key in (ord('s'), ord('S')): save_layout(PAD_LAYOUT, PAD_W_FRAC, PAD_H_FRAC)
        elif key in (ord('l'), ord('L')):
            data = load_layout()
            if data:
                PAD_LAYOUT = [tuple(p) for p in data["layout"]]
                PAD_W_FRAC = float(data["w_frac"])
                PAD_H_FRAC = float(data["h_frac"])
        elif key in (81,82,83,84) and selected_pad is not None:  # ← ↑ → ↓
            step = 0.01; dx = dy = 0.0
            if key == 81: dx = -step
            if key == 83: dx = +step
            if key == 82: dy = -step
            if key == 84: dy = +step
            xf, yf = PAD_LAYOUT[selected_pad]
            PAD_LAYOUT[selected_pad] = (clamp(xf + dx, 0.05, 0.95), clamp(yf + dy, 0.05, 0.95))
        elif key == ord('z'): PAD_W_FRAC = max(0.08, PAD_W_FRAC - 0.01)
        elif key == ord('x'): PAD_W_FRAC = min(0.30, PAD_W_FRAC + 0.01)
        elif key == ord('c'): PAD_H_FRAC = max(0.08, PAD_H_FRAC - 0.01)
        elif key == ord('v'): PAD_H_FRAC = min(0.32, PAD_H_FRAC + 0.01)

        last_frame_time = t0

    cap.release()
    hand_tracker.close()
    sounder.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
