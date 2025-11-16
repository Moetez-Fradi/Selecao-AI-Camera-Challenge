import time
import collections
import math
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
import os
import urllib.request
from tqdm import tqdm

model = YOLO("yolov8x-pose.pt")
cap   = cv2.VideoCapture("./testcases/siu.mp4")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,
    refine_landmarks=True,  # Enables iris landmarks for better angled detection and potential gaze analysis
    min_detection_confidence=0.3,  # Lowered further
    min_tracking_confidence=0.3    # Lowered further
)

mp_drawing = mp.solutions.drawing_utils

TINY_FACE_URL  = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.pt"
TINY_FACE_PATH = "yolov8n-face-lindevs.pt"

if not os.path.exists(TINY_FACE_PATH):
    print("Downloading tiny face model …")
    def _download(url, dst):
        with tqdm(unit='B', unit_scale=True, desc=os.path.basename(dst)) as t:
            def _reporthook(b, bs, ts):
                if ts != -1: t.total = ts
                t.update(bs)
            urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
    _download(TINY_FACE_URL, TINY_FACE_PATH)
    print("Done!")

face_model = YOLO(TINY_FACE_PATH)

HISTORY_LEN               = 20  # Increased for more robustness
TORSO_ACTIVITY_THRESHOLD  = 0.0025   # the movement allowed
ARM_ACTIVITY_THRESHOLD    = 0.005
MIN_SHOULDER_WIDTH_FRAC   = 0.10     # smaller people are OK
STATIONARY_SECONDS        = 4.0      # Adjusted as per user

CLIENT_BOX_SCALE_W = 3  # Width scale
CLIENT_BOX_SCALE_H = 1  # Height scale
SCORE_INTERVAL   = 1.0  # Compute scores every 1 second
CENTROID_DISP_THRESH = 0.05  # Relaxed to 0.05 (5% of frame) for "close to same place"
LEAVING_THRESHOLD = 0.2  # Overlap below this considers client left
ENTERING_THRESHOLD = 0.90  # Overlap at or above this to count as new client

COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
]

# HELPERS
def _angle_between_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def _clamp01(x): return max(0.0, min(1.0, x))
def _map_to_01(v, a, b):
    if b == a: return 0.0
    return _clamp01((v - a) / (b - a))

# SATISFACTION (posture) - Updated for YOLO keypoints, added crossed arms penalty
def compute_satisfaction_score(history, fw, fh, prev_score=None, alpha=0.6):
    if len(history) < 3: return None, 0, {}
    pts_now  = history[-1]
    pts_prev = history[-2]
    pts_old  = history[-3]
    if pts_now.shape[0] < 17: return None, 0, {}

    disp1 = np.linalg.norm(pts_now - pts_prev, axis=1)
    disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)
    activity = float((disp1.mean() + disp2.mean()) / 2.0)

    L_SH, R_SH = 5, 6  # YOLO shoulders
    L_HIP, R_HIP = 11, 12
    NOSE = 0
    L_ELB, R_ELB = 7, 8
    L_WRIST, R_WRIST = 9, 10

    shoulders = (pts_now[L_SH] + pts_now[R_SH]) / 2.0
    hips      = (pts_now[L_HIP] + pts_now[R_HIP]) / 2.0
    nose      = pts_now[NOSE]

    torso_vec = hips - shoulders
    torso_len = np.linalg.norm(torso_vec) + 1e-6
    torso_dir = torso_vec / torso_len
    vertical  = np.array([0.0, 1.0])
    torso_angle = min(_angle_between_deg(torso_dir, vertical), 180 - _angle_between_deg(torso_dir, vertical))

    head_vec = nose - shoulders
    head_len = np.linalg.norm(head_vec) + 1e-6
    head_dir = head_vec / head_len
    head_torso_angle = min(_angle_between_deg(head_dir, torso_dir), 180 - _angle_between_deg(head_dir, torso_dir))

    sh_ys = abs(pts_now[L_SH][1] - pts_now[R_SH][1])
    shoulder_sym = sh_ys * fh / (torso_len * fh + 1e-6)

    shoulder_width = np.linalg.norm(pts_now[L_SH] - pts_now[R_SH]) + 1e-6
    wrist_dist     = np.linalg.norm(pts_now[L_WRIST] - pts_now[R_WRIST])
    arm_openness   = _clamp01((wrist_dist / shoulder_width) / 2.5)

    wrist_to_nose = min(np.linalg.norm(pts_now[L_WRIST] - nose),
                        np.linalg.norm(pts_now[R_WRIST] - nose))
    hands_face = _map_to_01(wrist_to_nose, 0.01, 0.20)

    # Detect crossed arms for penalty (robust check)
    crossed_arms_penalty = 0.0
    if all(pts_now[i].any() for i in [L_ELB, R_ELB, L_WRIST, R_WRIST, L_SH, R_SH, L_HIP, R_HIP]):
        # Wrists crossed over body
        if (pts_now[L_WRIST][0] > pts_now[R_SH][0] and pts_now[R_WRIST][0] < pts_now[L_SH][0]) or \
           (pts_now[L_WRIST][0] > pts_now[R_ELB][0] and pts_now[R_WRIST][0] < pts_now[L_ELB][0]):
            # Check if wrists are at chest level (between shoulders and hips y)
            chest_y_min = min(pts_now[L_SH][1], pts_now[R_SH][1])
            chest_y_max = max(pts_now[L_HIP][1], pts_now[R_HIP][1])
            if chest_y_min < pts_now[L_WRIST][1] < chest_y_max and chest_y_min < pts_now[R_WRIST][1] < chest_y_max:
                crossed_arms_penalty = 0.25  # Penalty value, adjust as needed

    activity_score = _map_to_01(activity, 0.0008, 0.018)
    upright_score = 1.0 if torso_angle <= 10 else 0.0 if torso_angle >= 40 else 1.0 - ((torso_angle-10)/(40-10))
    head_align_score = _clamp01(1.0 - (head_torso_angle/40.0))
    shoulder_sym_score = 1.0 - _clamp01(shoulder_sym*3.0)
    hands_open_score   = hands_face
    arm_open_score = arm_openness  # Weight more if needed

    combined = (0.30*upright_score + 0.25*activity_score + 0.15*hands_open_score +
                0.15*arm_open_score + 0.10*head_align_score + 0.05*shoulder_sym_score)
    combined -= crossed_arms_penalty  # Apply penalty for crossed arms
    combined = _clamp01(combined)

    if prev_score is not None:
        combined = alpha*combined + (1-alpha)*(prev_score/100.0)

    score = int(combined*100)
    label = "satisfied" if score >= 70 else "neutral" if score >= 45 else "dissatisfied"

    return label, score, {}

# FACE EXPRESSION
def compute_face_expression_score(face_landmarks, fw, fh, prev_score=None, alpha=0.6):
    if face_landmarks is None: return None, 0, {}
    pts = np.array([[p.x, p.y] for p in face_landmarks.landmark], dtype=np.float32)  # CHANGED: Access .landmark here since face_landmarks is now the full NormalizedLandmarkList (lm)
    try:
        lm_l_mouth = pts[61]; lm_r_mouth = pts[291]
        lm_top_lip = pts[13]; lm_bottom_lip = pts[14]
        lm_l_eye   = pts[33]; lm_r_eye   = pts[263]
    except Exception: return None, 0, {}

    mouth_w = np.linalg.norm(lm_r_mouth - lm_l_mouth)
    mouth_h = np.linalg.norm(lm_bottom_lip - lm_top_lip)
    eye_dist = np.linalg.norm(lm_r_eye - lm_l_eye) + 1e-6

    smile_ratio = mouth_w / eye_dist
    mouth_open_ratio = mouth_h / eye_dist

    smile_score = _map_to_01(smile_ratio, 0.35, 0.75)
    open_score  = _map_to_01(mouth_open_ratio, 0.02, 0.18)

    combined = 0.8*smile_score + 0.2*open_score
    combined = _clamp01(combined)

    if prev_score is not None:
        combined = alpha*combined + (1-alpha)*(prev_score/100.0)

    score = int(combined*100)
    label = "satisfied" if score >= 70 else "neutral" if score >= 45 else "dissatisfied"
    return label, score, {}

# TRANSACTION INDICATORS (torso still, arms moving) - Updated for YOLO
def compute_transaction_indicators(history, fw, fh):
    if len(history) < 3: return None, {}
    pts_now  = history[-1]; pts_prev = history[-2]; pts_old = history[-3]
    if pts_now.shape[0] < 17: return None, {}

    disp1 = np.linalg.norm(pts_now - pts_prev, axis=1)
    disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)

    torso_pts = [5,6,11,12]  # YOLO shoulders + hips
    arm_pts   = [7,8,9,10]   # elbows + wrists

    torso_disp = (disp1[torso_pts].mean() + disp2[torso_pts].mean())/2.0
    arm_disp   = (disp1[arm_pts].mean()   + disp2[arm_pts].mean())/2.0

    shoulders = (pts_now[5] + pts_now[6]) / 2.0
    nose      = pts_now[0]
    head_vec  = nose - shoulders
    head_len  = np.linalg.norm(head_vec) + 1e-6
    head_dir  = head_vec / head_len
    cam_dir   = np.array([0.0, -1.0])
    facing_angle = _angle_between_deg(head_dir, cam_dir)

    shoulder_width = np.linalg.norm(pts_now[5] - pts_now[6]) * fw
    shoulder_frac  = shoulder_width / fw

    diag = {"torso_activity":torso_disp, "arm_activity":arm_disp,
            "facing_angle":facing_angle, "shoulder_frac":shoulder_frac}

    is_transaction = (torso_disp < TORSO_ACTIVITY_THRESHOLD and
                      arm_disp   > ARM_ACTIVITY_THRESHOLD and
                      shoulder_frac > MIN_SHOULDER_WIDTH_FRAC)

    return "potential_transaction" if is_transaction else None, diag

# DATA STRUCTURES
landmark_histories      = {}
torso_activity_histories = {}
bbox_histories          = {}  # NEW: for client box estimation
stationary_since        = {}
prev_body_scores        = {}
prev_face_scores        = {}
stationary_start_time = {}  # NEW: tid -> time when it first became stationary

last_print = 0.0

transaction_locations = []

# NEW: Client tracking
current_client_tid = None
client_start_time  = None
client_scores      = []  # list of (posture_score, face_score, agg_score) per second
last_score_time    = 0.0
client_logs        = []  # Final per-client stats

# NEW: Guichet / Client Box
guichet_loc   = None
client_box    = None  # (x1,y1,x2,y2)

# NEW: Overlap calculation
def box_overlap(box1, box2):
    # box = (x1,y1,x2,y2)
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    ia  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return ia / a1 if a1 > 0 else 0.0

# NEW: Compute bbox area as proxy for closeness
def bbox_area(box):
    x1,y1,x2,y2 = box
    return (x2 - x1) * (y2 - y1)

# NEW: Compute centroid displacement
def compute_centroid_disp(bbox_list, fw, fh):
    if len(bbox_list) < 3: return 1.0
    disps = []
    for i in range(1, len(bbox_list)):
        x1,y1,x2,y2 = bbox_list[i]
        px1,py1,px2,py2 = bbox_list[i-1]
        c_i = ((x1 + x2)/2 / fw, (y1 + y2)/2 / fh)
        c_prev = ((px1 + px2)/2 / fw, (py1 + py2)/2 / fh)
        disps.append(np.linalg.norm(np.array(c_i) - np.array(c_prev)))
    return np.mean(disps)

# -----------------------------------------------------------------
# 0.5-second face detector – runs only when we need a new face
# -----------------------------------------------------------------
last_face_check = 0.0
cached_face_box = None          # (x1,y1,x2,y2)  – kept until next check
cached_face_landmarks = None   # MediaPipe landmarks (still needed for expression)

def get_face_for_person(person_bbox, now):
    """Return (face_box, face_landmarks) for the given person bbox.
       Detects only every 0.5 s, otherwise re-uses the cached result."""
    global last_face_check, cached_face_box, cached_face_landmarks

    # ---- 0.5 s throttle -------------------------------------------------
    if now - last_face_check < 0.5:
        # reuse previous detection (still valid for the same person)
        if cached_face_box and box_overlap(person_bbox, cached_face_box) > 0.4:
            return cached_face_box, cached_face_landmarks
        # otherwise fall-through to a fresh detection

    # ---- fresh YOLO-face detection --------------------------------------
    x1, y1, x2, y2 = person_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    results = face_model(crop, verbose=False, conf=0.25)   # tiny model, fast
    if results[0].boxes is None or len(results[0].boxes) == 0:
        # no face → clear cache
        cached_face_box = None
        cached_face_landmarks = None
        return None, None

    # take the highest-conf face
    best = results[0].boxes[0]
    fx1, fy1, fx2, fy2 = map(int, best.xyxy[0].tolist())
    # convert back to full-image coordinates
    fx1 += x1; fy1 += y1; fx2 += x1; fy2 += y1
    face_box = (fx1, fy1, fx2, fy2)

    # ---- optional: get landmarks from MediaPipe on the tiny crop -----
    crop_face_y1 = max(0, fy1 - y1)
    crop_face_y2 = min(crop.shape[0], fy2 - y1)
    crop_face_x1 = max(0, fx1 - x1)
    crop_face_x2 = min(crop.shape[1], fx2 - x1)
    face_crop = crop[crop_face_y1:crop_face_y2, crop_face_x1:crop_face_x2]
    if face_crop.size == 0:
        landmarks = None
    else:
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_res = face_mesh.process(rgb_crop)
        landmarks = None
        if mp_res.multi_face_landmarks:
            lm = mp_res.multi_face_landmarks[0]
            # shift to global coordinates (normalized [0,1])
            fw_face = crop_face_x2 - crop_face_x1
            fh_face = crop_face_y2 - crop_face_y1
            for p in lm.landmark:
                p.x = (p.x * fw_face + (fx1)) / fw
                p.y = (p.y * fh_face + (fy1)) / fh
            landmarks = lm  # CHANGED: Return the full lm (NormalizedLandmarkList) instead of lm.landmark

    # cache for the next 0.5 s
    cached_face_box       = face_box
    cached_face_landmarks = landmarks
    last_face_check       = now
    return face_box, landmarks

# Function to draw satisfaction graph on the frame
def draw_satisfaction_graph(frame, agg_history, fw, graph_height=200):
    if len(agg_history) == 0:
        return frame
    # Add border at bottom for graph
    graph_frame = np.zeros((graph_height, frame.shape[1], 3), dtype=np.uint8) + 255  # White background, use frame.shape[1] for width
    if len(agg_history) < 2:
        cv2.putText(graph_frame, "No data yet", (10, graph_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        # Draw axes
        cv2.line(graph_frame, (50, graph_height - 50), (frame.shape[1] - 50, graph_height - 50), (0, 0, 0), 2)  # X axis
        cv2.line(graph_frame, (50, graph_height - 50), (50, 50), (0, 0, 0), 2)  # Y axis
        cv2.putText(graph_frame, "0", (30, graph_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(graph_frame, "100", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(graph_frame, "Satisfaction over time", (frame.shape[1] // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Plot points
        max_x = frame.shape[1] - 100
        max_y = graph_height - 100
        points = []
        for i, score in enumerate(agg_history):
            x = 50 + int(i * max_x / max(1, (len(agg_history) - 1)))
            y = (graph_height - 50) - int(score / 100 * max_y)
            points.append((x, y))
            cv2.circle(graph_frame, (x, y), 3, (0, 0, 255), -1)

        # Draw lines
        for i in range(1, len(points)):
            cv2.line(graph_frame, points[i-1], points[i], (0, 0, 255), 2)

    # Concat to original frame
    frame_with_graph = np.vstack((frame, graph_frame))
    return frame_with_graph

# MAIN LOOP
monitoring_mode = False  # Phase 1: detect guichet; Phase 2: monitor clients

# Current scores for display (initialize to 0)
current_face_sc = 0
current_body_sc = 0
current_agg_sc = 0

# Agg history for graph
agg_history = []

# Set larger window
cv2.namedWindow("YOLO + Pose + Face (Satisfaction)", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("YOLO + Pose + Face (Satisfaction)", 1200, 900)

while True:
    ret, frame = cap.read()
    if not ret: break

    fh, fw = frame.shape[:2]

    results  = model.track(frame, persist=True, classes=[0], verbose=False)
    annotated = frame.copy()
    now = time.monotonic()

    # Phase 1: Detect guichet if not found (new logic: closest stationary person)
    if not monitoring_mode:
        current_candidates = {}  # tid -> (area, centroid, elapsed)
        for box in results[0].boxes:
            if box.id is None: continue
            tid = int(box.id.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

            if tid not in bbox_histories:
                bbox_histories[tid] = collections.deque(maxlen=30)  # ~1 sec buffer at 30 fps

            bbox_histories[tid].append((x1,y1,x2,y2))

            area = bbox_area((x1,y1,x2,y2))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Only consider if history is long enough for recent disp
            if len(bbox_histories[tid]) >= 15:  # ~0.5 sec
                avg_disp = compute_centroid_disp(list(bbox_histories[tid])[-15:], fw, fh)

                if avg_disp < CENTROID_DISP_THRESH:
                    if tid not in stationary_start_time:
                        stationary_start_time[tid] = now
                    elapsed = now - stationary_start_time[tid]

                    if elapsed >= STATIONARY_SECONDS:
                        current_candidates[tid] = (area, (cx, cy), elapsed)
                else:
                    # Reset if moving
                    if tid in stationary_start_time:
                        del stationary_start_time[tid]

            # Debug draw
            color = (0, 255, 0) if tid in current_candidates else (0, 0, 255)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            status = f"ID:{tid} {area:.0f}"
            if tid in current_candidates:
                status += f" [{current_candidates[tid][2]:.1f}s]"
            cv2.putText(annotated, status, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Pick the largest (closest) candidate that has been stationary long enough
        if current_candidates:
            best_tid = max(current_candidates, key=lambda tid: current_candidates[tid][0])
            _, best_centroid, _ = current_candidates[best_tid]
            best_bbox = bbox_histories[best_tid][-1]
            face_box, _ = get_face_for_person(best_bbox, now)
            if face_box is not None:
                transaction_locations.append(best_centroid)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Ignored candidate ID {best_tid}: no face detected.")

        # Check if enough data to find guichet
        if len(transaction_locations) > 10: 
            guichet_start_time = stationary_start_time[best_tid]
            from sklearn.cluster import DBSCAN
            locs = np.array(transaction_locations)
            cl = DBSCAN(eps=50, min_samples=3).fit(locs)
            lbls = cl.labels_
            if len(np.unique(lbls[lbls >= 0])) > 0:
                main = np.argmax(np.bincount(lbls[lbls >= 0]))
                guichet_loc = np.mean(locs[lbls == main], axis=0).astype(int)

                # Define client box based on average person size, scaled *3 width, *1.5 height
                avg_w = np.mean([bbox_area(b) for hist in bbox_histories.values() for b in hist]) ** 0.5  # approx width
                avg_h = avg_w * 1.5  # assume height > width
                cx, cy = guichet_loc
                client_box = (int(cx - avg_w/2 * CLIENT_BOX_SCALE_W),
                              int(cy - avg_h/2 * CLIENT_BOX_SCALE_H),
                              int(cx + avg_w/2 * CLIENT_BOX_SCALE_W),
                              int(cy + avg_h/2 * CLIENT_BOX_SCALE_H))

                monitoring_mode = True
                print(f"[{time.strftime('%H:%M:%S')}] Guichet found at {guichet_loc}. Switching to monitoring. Client box: {client_box}")

    else:
        # Phase 2: Monitor clients in client_box
        active_tracks_in_box = []
        for box in results[0].boxes:
            if box.id is None: continue
            tid = int(box.id.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

            overlap = box_overlap((x1,y1,x2,y2), client_box)
            if overlap >= LEAVING_THRESHOLD:
                active_tracks_in_box.append((tid, overlap, (x1,y1,x2,y2)))

        front_tid = None
        if active_tracks_in_box:
            high_overlap_candidates = [t for t in active_tracks_in_box if t[1] >= ENTERING_THRESHOLD]
            current_overlap = next((ov for tid,ov,_ in active_tracks_in_box if tid == current_client_tid), 0.0)
            if current_overlap > 0:
                front_tid = current_client_tid
            elif high_overlap_candidates:
                high_overlap_candidates.sort(key=lambda x: x[1], reverse=True)
                front_tid = high_overlap_candidates[0][0]

        if front_tid is not None:
            if current_client_tid is None or current_client_tid != front_tid:
                if current_client_tid is not None:
                    total_time = now - guichet_start_time
                    if client_scores:
                        posture_scores = [s[0] for s in client_scores]
                        face_scores    = [s[1] for s in client_scores]
                        agg_scores     = [s[2] for s in client_scores]
                        mean_posture   = np.mean(posture_scores)
                        mean_face      = np.mean(face_scores)
                        mean_total     = np.mean(agg_scores)
                    else:
                        mean_posture = mean_face = mean_total = 0

                    print(f"[{time.strftime('%H:%M:%S')}] Client ID {current_client_tid} left.")
                    print(f"Time spent: {total_time:.1f}s")
                    print(f"Scores list: {client_scores}")
                    print(f"Mean posture: {mean_posture:.1f}")
                    print(f"Mean face: {mean_face:.1f}")
                    print(f"Total mean score: {mean_total:.1f}")
                    client_logs.append((current_client_tid, total_time, mean_posture, mean_face, mean_total))

                current_client_tid = front_tid
                client_start_time  = now
                client_scores      = []
                last_score_time    = now
                agg_history = []
                print(f"[{time.strftime('%H:%M:%S')}] New client ID {front_tid} entered.")

            # Process pose/face/draw for current_client_tid
            tid = current_client_tid
            if tid not in landmark_histories:
                landmark_histories[tid]       = collections.deque(maxlen=HISTORY_LEN)
                torso_activity_histories[tid] = collections.deque(maxlen=5)
                prev_body_scores[tid]         = None
                prev_face_scores[tid]         = None

            person_bbox = next(b for t,o,b in active_tracks_in_box if t == tid)
            x1, y1, x2, y2 = person_bbox

            has_pose = False
            pts = None
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                for i, kp in enumerate(results[0].keypoints):
                    if results[0].boxes[i].id is not None and int(results[0].boxes[i].id.item()) == tid:
                        keypoints = kp.data.cpu().numpy()  # Shape: (1, 17, 3) -> [x, y, conf]
                        pts = keypoints[0, :, :2]  # Extract [x,y] for 17 keypoints
                        confidences = keypoints[0, :, 2]
                        # Filter low-conf keypoints
                        low_conf_mask = confidences < 0.5
                        pts[low_conf_mask] = [0, 0]
                        # Normalize to [0,1]
                        pts[:, 0] /= fw
                        pts[:, 1] /= fh
                        has_pose = True
                        # Draw pose mesh
                        for idx1, idx2 in COCO_CONNECTIONS:
                            if np.all(pts[idx1] != 0) and np.all(pts[idx2] != 0):
                                pt1 = (int(pts[idx1][0] * fw), int(pts[idx1][1] * fh))
                                pt2 = (int(pts[idx2][0] * fw), int(pts[idx2][1] * fh))
                                cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)
                        break

            # Face detection
            face_box, face_landmarks = get_face_for_person(person_bbox, now)

            # Draw face box
            if face_box:
                fx1, fy1, fx2, fy2 = face_box
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

            # Draw face mesh if landmarks
            if face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, 
                    face_landmarks,  # CHANGED: Pass face_landmarks directly (which is now lm, the full NormalizedLandmarkList)
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,100,0), thickness=0, circle_radius=0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1)
                )

            # Draw green box for current client
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated, f"ID:{tid}", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if has_pose and pts is not None and pts.shape[0] >= 17:
                landmark_histories[tid].append(pts)

            # Compute scores every second
            if now - last_score_time >= SCORE_INTERVAL:
                last_score_time = now

                # Posture
                body_sc = 0
                if tid in landmark_histories and len(landmark_histories[tid]) >= 3:
                    _, body_sc, _ = compute_satisfaction_score(
                        landmark_histories[tid], fw, fh,
                        prev_body_scores.get(tid))
                    prev_body_scores[tid] = body_sc

                # Face
                face_sc = 0
                if face_landmarks:
                    _, face_sc, _ = compute_face_expression_score(
                        face_landmarks, fw, fh,  # CHANGED: Pass face_landmarks (lm) directly; the function now accesses .landmark internally
                        prev_face_scores.get(tid))
                    prev_face_scores[tid] = face_sc

                # Agg - if no face, use posture only
                scores = [s for s in (body_sc, face_sc) if s > 0]
                agg_sc = sum(scores)/len(scores) if scores else 0

                client_scores.append((body_sc, face_sc, agg_sc))
                agg_history.append(agg_sc)

                # Update current scores for display
                current_face_sc = face_sc
                current_body_sc = body_sc
                current_agg_sc = agg_sc

        else:
            # No one in box → log if current was there
            if current_client_tid is not None:
                total_time = now - client_start_time
                if client_scores:
                    posture_scores = [s[0] for s in client_scores]
                    face_scores    = [s[1] for s in client_scores]
                    agg_scores     = [s[2] for s in client_scores]
                    mean_posture   = np.mean(posture_scores)
                    mean_face      = np.mean(face_scores)
                    mean_total     = np.mean(agg_scores)
                else:
                    mean_posture = mean_face = mean_total = 0

                print(f"[{time.strftime('%H:%M:%S')}] Client ID {current_client_tid} left.")
                print(f"Time spent: {total_time:.1f}s")
                print(f"Scores list: {client_scores}")
                print(f"Mean posture: {mean_posture:.1f}")
                print(f"Mean face: {mean_face:.1f}")
                print(f"Total mean score: {mean_total:.1f}")
                client_logs.append((current_client_tid, total_time, mean_posture, mean_face, mean_total))

                current_client_tid = None

        # Draw client box
        if client_box:
            cv2.rectangle(annotated, (client_box[0], client_box[1]), (client_box[2], client_box[3]), (255,0,0), 2)

    # Display satisfaction levels on top left of the frame
    cv2.putText(annotated, f"face : {current_face_sc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"posture : {current_body_sc}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"total : {current_agg_sc}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display chronometer in top right corner if a client is present
    if current_client_tid is not None:
        elapsed = now - client_start_time + guichet_start_time
        chron_text = f"Client Time: {elapsed:.1f}s"
        text_size = cv2.getTextSize(chron_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        chron_x = fw - text_size[0] - 10
        chron_y = 30
        cv2.putText(annotated, chron_text, (chron_x, chron_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw graph if in monitoring mode and have data
    if monitoring_mode and len(agg_history) > 0:
        annotated = draw_satisfaction_graph(annotated, agg_history, annotated.shape[1])

    cv2.imshow("YOLO + Pose + Face (Satisfaction)", annotated)
    if cv2.waitKey(1) == 27: break

# After video ends, log the last client if still present
if current_client_tid is not None:
    total_time = now - client_start_time
    if client_scores:
        posture_scores = [s[0] for s in client_scores]
        face_scores    = [s[1] for s in client_scores]
        agg_scores     = [s[2] for s in client_scores]
        mean_posture   = np.mean(posture_scores)
        mean_face      = np.mean(face_scores)
        mean_total     = np.mean(agg_scores)
    else:
        mean_posture = mean_face = mean_total = 0

    print(f"[{time.strftime('%H:%M:%S')}] Client ID {current_client_tid} left (video end).")
    print(f"Time spent: {total_time:.1f}s")
    print(f"Scores list: {client_scores}")
    print(f"Mean posture: {mean_posture:.1f}")
    print(f"Mean face: {mean_face:.1f}")
    print(f"Total mean score: {mean_total:.1f}")
    client_logs.append((current_client_tid, total_time, mean_posture, mean_face, mean_total))

# Final logs
print("All client logs:")
for log in client_logs:
    print(log)

cap.release()
cv2.destroyAllWindows()