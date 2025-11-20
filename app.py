# app.py
from flask import Flask, render_template, request, jsonify, Response
import threading, time, os
import cv2, numpy as np, collections, math
from ultralytics import YOLO
import urllib.request
from tqdm import tqdm
import torch, timm
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

lock = threading.Lock()
output_frame = None
metrics = {'face': 0, 'posture': 0, 'total': 0, 'agg_history': [], 'insight': ''}
worker = {'thread': None, 'stop': False, 'script': None, 'source': None, 'auto_detect': False}
client_box_norm = None  # normalized [x1,y1,x2,y2] or None

def encode_jpeg(frame):
    ret, buf = cv2.imencode('.jpg', frame)
    return buf.tobytes() if ret else None

def mjpeg_generator():
    global output_frame
    while True:
        if worker.get('stop'):
            time.sleep(0.05)
        with lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.05)
            continue
        jpg = encode_jpeg(frame)
        if jpg is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
               str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global client_box_norm
    script = request.form.get('script', 'script1')
    source = request.form.get('source', None)
    auto_detect = request.form.get('auto_detect', None) == 'on'
    if 'videoFile' in request.files and request.files['videoFile'].filename:
        f = request.files['videoFile']
        dst = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(dst)
        source = dst
    if source is None:
        return jsonify({'ok': False, 'error': 'no source specified'}), 400
    stop_worker()
    worker['stop'] = False
    worker['script'] = script
    worker['source'] = source
    worker['auto_detect'] = auto_detect
    client_box_norm = None
    t = threading.Thread(target=processing_worker, args=(script, source, auto_detect), daemon=True)
    worker['thread'] = t
    t.start()
    return jsonify({'ok': True})

@app.route('/stop', methods=['POST'])
def stop():
    stop_worker()
    return jsonify({'ok': True})

@app.route('/metrics')
def get_metrics():
    with lock:
        return jsonify({
            'face': metrics['face'],
            'posture': metrics['posture'],
            'total': metrics['total'],
            'agg_history': metrics['agg_history'][-240:],
            'insight': metrics.get('insight', '')
        })

@app.route('/set_box', methods=['POST'])
def set_box():
    global client_box_norm
    data = request.get_json()
    try:
        x1 = float(data['x1']); y1 = float(data['y1']); x2 = float(data['x2']); y2 = float(data['y2'])
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid payload'}), 400
    x1, y1, x2, y2 = max(0.0,min(1.0,x1)), max(0.0,min(1.0,y1)), max(0.0,min(1.0,x2)), max(0.0,min(1.0,y2))
    if x2 <= x1 or y2 <= y1:
        return jsonify({'ok': False, 'error': 'invalid rectangle'}), 400
    client_box_norm = (x1, y1, x2, y2)
    return jsonify({'ok': True})

@app.route('/clear_box', methods=['POST'])
def clear_box():
    global client_box_norm
    client_box_norm = None
    return jsonify({'ok': True})

def stop_worker():
    if worker.get('thread') and worker['thread'].is_alive():
        worker['stop'] = True
        worker['thread'].join(timeout=3.0)
    worker['thread'] = None
    worker['stop'] = False
    worker['script'] = None
    worker['source'] = None

# Utilities
COCO_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16),(11,12)
]
def _angle_between_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
def _clamp01(x): return max(0.0, min(1.0, x))
def _map_to_01(v, a, b):
    if b == a: return 0.0
    return _clamp01((v - a) / (b - a))

# Draw graph function (preserved)
def draw_satisfaction_graph(frame, agg_history, fw, graph_height=200):
    if len(agg_history) == 0:
        return frame
    graph_frame = np.zeros((graph_height, frame.shape[1], 3), dtype=np.uint8) + 255
    if len(agg_history) < 2:
        cv2.putText(graph_frame, "No data yet", (10, graph_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    else:
        cv2.line(graph_frame, (50, graph_height - 50), (frame.shape[1]-50, graph_height-50), (0,0,0), 2)
        cv2.line(graph_frame, (50, graph_height - 50), (50, 50), (0,0,0), 2)
        cv2.putText(graph_frame, "0", (30, graph_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(graph_frame, "100", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(graph_frame, "Satisfaction over time", (frame.shape[1]//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        max_x = frame.shape[1] - 100
        max_y = graph_height - 100
        points = []
        for i, score in enumerate(agg_history):
            x = 50 + int(i * max_x / max(1, (len(agg_history)-1)))
            y = (graph_height - 50) - int(score / 100 * max_y)
            points.append((x, y))
            cv2.circle(graph_frame, (x, y), 3, (0,0,255), -1)
        for i in range(1, len(points)):
            cv2.line(graph_frame, points[i-1], points[i], (0,0,255), 2)
    return np.vstack((frame, graph_frame))

# -----------------------
# Script1 implementation (keeps your original logic, with fixes + throttling)
# -----------------------
def run_script1(source, auto_detect_flag):
    global output_frame, metrics, client_box_norm
    model = YOLO("yolov8x-pose.pt")
    TINY_FACE_URL = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.pt"
    TINY_FACE_PATH = "yolov8n-face-lindevs.pt"
    if not os.path.exists(TINY_FACE_PATH):
        def _download(url, dst):
            with tqdm(unit='B', unit_scale=True, desc=os.path.basename(dst)) as t:
                def _reporthook(b, bs, ts):
                    if ts != -1: t.total = ts
                    t.update(bs)
                urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
        _download(TINY_FACE_URL, TINY_FACE_PATH)
    face_model = YOLO(TINY_FACE_PATH)

    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils

    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except Exception:
        cap = cv2.VideoCapture(source)

    # params from your script
    HISTORY_LEN = 20
    TORSO_ACTIVITY_THRESHOLD = 0.0025
    ARM_ACTIVITY_THRESHOLD = 0.005
    MIN_SHOULDER_WIDTH_FRAC = 0.10
    STATIONARY_SECONDS = 4.0
    CLIENT_BOX_SCALE_W = 2.5
    CLIENT_BOX_SCALE_H = 1
    SCORE_INTERVAL = 1.0
    CENTROID_DISP_THRESH = 0.05
    LEAVING_THRESHOLD = 0.2
    ENTERING_THRESHOLD = 0.90

    # runtime state
    landmark_histories = {}
    bbox_histories = {}
    stationary_start_time = {}
    prev_body_scores = {}
    prev_face_scores = {}
    transaction_locations = []
    monitoring_mode = False
    if client_box_norm is not None:
        monitoring_mode = True
    current_face_sc = 0; current_body_sc = 0; current_agg_sc = 0
    agg_history = []
    client_box_pixels = None
    current_client_tid = None
    client_start_time = None
    client_scores = []
    last_score_time = 0.0
    client_logs = []

    # face detection caching / throttle
    last_face_check = 0.0
    cached_face_box = None
    cached_face_landmarks = None
    FACE_CHECK_INTERVAL = 0.5

    def box_overlap(box1, box2):
        ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1]); ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
        ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        return ia / a1 if a1 > 0 else 0.0

    def bbox_area(box):
        x1,y1,x2,y2 = box
        return (x2-x1)*(y2-y1)

    def compute_centroid_disp(bbox_list, fw, fh):
        if len(bbox_list) < 3: return 1.0
        disps = []
        for i in range(1, len(bbox_list)):
            x1,y1,x2,y2 = bbox_list[i]
            px1,py1,px2,py2 = bbox_list[i-1]
            c_i = ((x1+x2)/2/fw, (y1+y2)/2/fh)
            c_prev = ((px1+px2)/2/fw, (py1+py2)/2/fh)
            disps.append(np.linalg.norm(np.array(c_i)-np.array(c_prev)))
        return np.mean(disps)

    def get_face_for_person(person_bbox, now, frame):
        nonlocal last_face_check, cached_face_box, cached_face_landmarks
        if person_bbox is None:
            return None, None
        # reuse cached if overlap high and recent
        if (now - last_face_check) < FACE_CHECK_INTERVAL:
            if cached_face_box and box_overlap(person_bbox, cached_face_box) > 0.4:
                return cached_face_box, cached_face_landmarks
        x1,y1,x2,y2 = person_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        results_f = face_model(crop, verbose=False, conf=0.25)
        if results_f[0].boxes is None or len(results_f[0].boxes) == 0:
            cached_face_box = None
            cached_face_landmarks = None
            last_face_check = now
            return None, None
        best = results_f[0].boxes[0]
        fx1,fy1,fx2,fy2 = map(int, best.xyxy[0].tolist())
        fx1 += x1; fy1 += y1; fx2 += x1; fy2 += y1
        face_box = (fx1,fy1,fx2,fy2)
        crop_face_y1 = max(0, fy1 - y1); crop_face_y2 = min(crop.shape[0], fy2 - y1)
        crop_face_x1 = max(0, fx1 - x1); crop_face_x2 = min(crop.shape[1], fx2 - x1)
        face_crop = crop[crop_face_y1:crop_face_y2, crop_face_x1:crop_face_x2]
        landmarks = None
        if face_crop.size > 0:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mp_res = face_mesh.process(rgb_crop)
            if mp_res.multi_face_landmarks:
                lm = mp_res.multi_face_landmarks[0]
                fw_face = crop_face_x2 - crop_face_x1
                fh_face = crop_face_y2 - crop_face_y1
                for p in lm.landmark:
                    p.x = (p.x * fw_face + (fx1)) / frame.shape[1]
                    p.y = (p.y * fh_face + (fy1)) / frame.shape[0]
                landmarks = lm
        cached_face_box = face_box
        cached_face_landmarks = landmarks
        last_face_check = now
        return face_box, landmarks

    def compute_satisfaction_score_local(history, fw, fh, prev_score=None, alpha=0.6):
        if len(history) < 3: return None, 0, {}
        pts_now = history[-1]; pts_prev = history[-2]; pts_old = history[-3]
        if pts_now.shape[0] < 17: return None, 0, {}
        disp1 = np.linalg.norm(pts_now - pts_prev, axis=1); disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)
        activity = float((disp1.mean() + disp2.mean()) / 2.0)
        L_SH,R_SH=5,6; L_HIP,R_HIP=11,12; NOSE=0; L_ELB,R_ELB=7,8; L_WRIST,R_WRIST=9,10
        shoulders = (pts_now[L_SH] + pts_now[R_SH]) / 2.0
        hips = (pts_now[L_HIP] + pts_now[R_HIP]) / 2.0
        nose = pts_now[NOSE]
        torso_vec = hips - shoulders
        torso_len = np.linalg.norm(torso_vec) + 1e-6
        torso_dir = torso_vec / torso_len
        vertical = np.array([0.0,1.0])
        torso_angle = min(_angle_between_deg(torso_dir,vertical), 180 - _angle_between_deg(torso_dir,vertical))
        head_vec = nose - shoulders
        head_len = np.linalg.norm(head_vec) + 1e-6
        head_dir = head_vec / head_len
        head_torso_angle = min(_angle_between_deg(head_dir, torso_dir), 180 - _angle_between_deg(head_dir, torso_dir))
        sh_ys = abs(pts_now[L_SH][1] - pts_now[R_SH][1])
        shoulder_sym = sh_ys * fh / (torso_len * fh + 1e-6)
        shoulder_width = np.linalg.norm(pts_now[L_SH] - pts_now[R_SH]) + 1e-6
        wrist_dist = np.linalg.norm(pts_now[L_WRIST] - pts_now[R_WRIST])
        arm_openness = _clamp01((wrist_dist / shoulder_width) / 2.5)
        wrist_to_nose = min(np.linalg.norm(pts_now[L_WRIST] - nose), np.linalg.norm(pts_now[R_WRIST] - nose))
        hands_face = _map_to_01(wrist_to_nose, 0.01, 0.20)
        crossed_arms_penalty = 0.0
        try:
            if all(pts_now[i].any() for i in [L_ELB,R_ELB,L_WRIST,R_WRIST,L_SH,R_SH,L_HIP,R_HIP]):
                if (pts_now[L_WRIST][0] > pts_now[R_SH][0] and pts_now[R_WRIST][0] < pts_now[L_SH][0]) or \
                   (pts_now[L_WRIST][0] > pts_now[R_ELB][0] and pts_now[R_WRIST][0] < pts_now[L_ELB][0]):
                    chest_y_min = min(pts_now[L_SH][1], pts_now[R_SH][1])
                    chest_y_max = max(pts_now[L_HIP][1], pts_now[R_HIP][1])
                    if chest_y_min < pts_now[L_WRIST][1] < chest_y_max and chest_y_min < pts_now[R_WRIST][1] < chest_y_max:
                        crossed_arms_penalty = 0.25
        except Exception:
            pass
        activity_score = _map_to_01(activity, 0.0008, 0.018)
        upright_score = 1.0 if torso_angle <= 10 else 0.0 if torso_angle >= 40 else 1.0 - ((torso_angle - 10) / (40 - 10))
        head_align_score = _clamp01(1.0 - (head_torso_angle / 40.0))
        shoulder_sym_score = 1.0 - _clamp01(shoulder_sym * 3.0)
        hands_open_score = hands_face
        arm_open_score = arm_openness
        combined = (0.30 * upright_score + 0.25 * activity_score + 0.15 * hands_open_score +
                    0.15 * arm_open_score + 0.10 * head_align_score + 0.05 * shoulder_sym_score)
        combined -= crossed_arms_penalty
        combined = _clamp01(combined)
        if prev_score is not None:
            combined = alpha * combined + (1 - alpha) * (prev_score / 100.0)
        score = int(combined * 100)
        label = "satisfied" if score >= 70 else "neutral" if score >= 45 else "dissatisfied"
        return label, score, {}

    try:
        while cap.isOpened() and not worker['stop']:
            ret, frame = cap.read()
            if not ret:
                break
            fh, fw = frame.shape[:2]

            # update early so UI continues receiving frames even if heavy ops follow
            with lock:
                output_frame = frame.copy()

            now = time.monotonic()
            # update client_box pixel coordinates if manual box set
            if client_box_norm is not None:
                x1n, y1n, x2n, y2n = client_box_norm
                client_box_pixels = (int(x1n * fw), int(y1n * fh), int(x2n * fw), int(y2n * fh))
                monitoring_mode = True

            results = model.track(frame, persist=True, classes=[0], verbose=False)
            annotated = frame.copy()

            if not monitoring_mode:
                current_candidates = {}
                if results and len(results) and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is None: continue
                        tid = int(box.id.item())
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        if tid not in bbox_histories:
                            bbox_histories[tid] = collections.deque(maxlen=30)
                        bbox_histories[tid].append((x1,y1,x2,y2))
                        area = bbox_area((x1,y1,x2,y2)); cx,cy = (x1+x2)//2, (y1+y2)//2
                        if len(bbox_histories[tid]) >= 15:
                            avg_disp = compute_centroid_disp(list(bbox_histories[tid])[-15:], fw, fh)
                            if avg_disp < CENTROID_DISP_THRESH:
                                if tid not in stationary_start_time: stationary_start_time[tid] = now
                                elapsed = now - stationary_start_time[tid]
                                if elapsed >= STATIONARY_SECONDS:
                                    current_candidates[tid] = (area, (cx,cy), elapsed)
                            else:
                                if tid in stationary_start_time: del stationary_start_time[tid]
                        color = (0,255,0) if tid in current_candidates else (0,0,255)
                        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
                        status = f"ID:{tid} {area:.0f}"
                        if tid in current_candidates: status += f" [{current_candidates[tid][2]:.1f}s]"
                        cv2.putText(annotated, status, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                if current_candidates:
                    best_tid = max(current_candidates, key=lambda t: current_candidates[t][0])
                    _, best_centroid, _ = current_candidates[best_tid]
                    best_bbox = bbox_histories[best_tid][-1]
                    face_box, _ = get_face_for_person(best_bbox, now, frame)
                    if face_box is not None:
                        transaction_locations.append(best_centroid)
                if auto_detect_flag and len(transaction_locations) > 10:
                    guichet_start_time = stationary_start_time.get(best_tid, now)
                    try:
                        from sklearn.cluster import DBSCAN
                        locs = np.array(transaction_locations)
                        cl = DBSCAN(eps=50, min_samples=3).fit(locs)
                        lbls = cl.labels_
                        if len(np.unique(lbls[lbls >= 0])) > 0:
                            main = np.argmax(np.bincount(lbls[lbls >= 0]))
                            guichet_loc = np.mean(locs[lbls == main], axis=0).astype(int)
                            avg_w = np.mean([bbox_area(b) for hist in bbox_histories.values() for b in hist])**0.5
                            avg_h = avg_w * 1.5
                            cx, cy = guichet_loc
                            client_box_pixels = (int(cx - avg_w/2 * CLIENT_BOX_SCALE_W),
                                                 int(cy - avg_h/2 * CLIENT_BOX_SCALE_H),
                                                 int(cx + avg_w/2 * CLIENT_BOX_SCALE_W),
                                                 int(cy + avg_h/2 * CLIENT_BOX_SCALE_H))
                            monitoring_mode = True
                    except Exception:
                        pass
            else:
                # monitoring mode
                active_tracks_in_box = []
                if results and len(results) and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is None: continue
                        tid = int(box.id.item())
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        overlap = 0.0
                        if client_box_pixels is not None:
                            ix1 = max(x1, client_box_pixels[0]); iy1 = max(y1, client_box_pixels[1])
                            ix2 = min(x2, client_box_pixels[2]); iy2 = min(y2, client_box_pixels[3])
                            ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            a1 = (x2-x1)*(y2-y1) if (x2-x1)>0 and (y2-y1)>0 else 1
                            overlap = ia / a1
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
                            total_time = now - client_start_time
                            if client_scores:
                                posture_scores = [s[0] for s in client_scores]; face_scores = [s[1] for s in client_scores]; agg_scores = [s[2] for s in client_scores]
                                mean_posture = np.mean(posture_scores); mean_face = np.mean(face_scores); mean_total = np.mean(agg_scores)
                            else:
                                mean_posture = mean_face = mean_total = 0
                            client_logs.append((current_client_tid, total_time, mean_posture, mean_face, mean_total))
                        current_client_tid = front_tid
                        client_start_time = now
                        client_scores = []
                        last_score_time = now
                        agg_history = []
                    tid = current_client_tid
                    # SAFELY extract bbox for this tid
                    entry = next((e for e in active_tracks_in_box if e[0] == tid), None)
                    if entry is None:
                        continue
                    person_bbox = entry[2]  # guaranteed tuple (x1,y1,x2,y2)
                    x1,y1,x2,y2 = person_bbox
                    has_pose = False; pts = None
                    if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                        for i,kp in enumerate(results[0].keypoints):
                            if results[0].boxes[i].id is not None and int(results[0].boxes[i].id.item()) == tid:
                                keypoints = kp.data.cpu().numpy(); pts = keypoints[0,:,:2]; confidences = keypoints[0,:,2]
                                low_conf_mask = confidences < 0.5; pts[low_conf_mask] = [0,0]
                                pts[:,0] /= fw; pts[:,1] /= fh; has_pose = True
                                for idx1,idx2 in COCO_CONNECTIONS:
                                    if np.all(pts[idx1] != 0) and np.all(pts[idx2] != 0):
                                        pt1=(int(pts[idx1][0]*fw), int(pts[idx1][1]*fh)); pt2=(int(pts[idx2][0]*fw), int(pts[idx2][1]*fh))
                                        cv2.line(annotated, pt1, pt2, (255,0,0), 2)
                                break
                    face_box, face_landmarks = get_face_for_person(person_bbox, now, frame)
                    if face_box:
                        fx1,fy1,fx2,fy2 = face_box; cv2.rectangle(annotated, (fx1,fy1),(fx2,fy2),(0,255,255),2)
                    if face_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                annotated,
                                face_landmarks,
                                mp_face.FACEMESH_TESSELATION,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,100,0), thickness=0, circle_radius=0),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1)
                            )
                        except Exception:
                            pass
                    cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(annotated, f"ID:{tid}", (x1,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    if has_pose and pts is not None and pts.shape[0] >= 17:
                        if tid not in landmark_histories:
                            landmark_histories[tid] = collections.deque(maxlen=HISTORY_LEN)
                        landmark_histories[tid].append(pts)
                    if now - last_score_time >= SCORE_INTERVAL:
                        last_score_time = now
                        body_sc = 0
                        if tid in landmark_histories and len(landmark_histories[tid]) >= 3:
                            _, body_sc, _ = compute_satisfaction_score_local(landmark_histories[tid], fw, fh, prev_body_scores.get(tid))
                            prev_body_scores[tid] = body_sc
                        face_sc = 0
                        if face_landmarks:
                            _, face_sc, _ = compute_satisfaction_score_local(landmark_histories[tid], fw, fh) if False else (None, 0, {})  # keep face pipeline separate
                            # use compute_face_expression if needed (we use mediapipe landmarks path in script1)
                            # compute face expression like original:
                            try:
                                # convert mp landmarks into same style as original compute_face_expression_score
                                pts_arr = np.array([[p.x, p.y] for p in face_landmarks.landmark], dtype=np.float32)
                                lm_l_mouth = pts_arr[61]; lm_r_mouth = pts_arr[291]; lm_top_lip = pts_arr[13]; lm_bottom_lip = pts_arr[14]
                                mouth_w = np.linalg.norm(lm_r_mouth - lm_l_mouth)
                                mouth_h = np.linalg.norm(lm_bottom_lip - lm_top_lip)
                                eye_dist = np.linalg.norm(pts_arr[263] - pts_arr[33]) + 1e-6
                                smile_ratio = mouth_w / eye_dist
                                mouth_open_ratio = mouth_h / eye_dist
                                smile_score = _map_to_01(smile_ratio, 0.35, 0.75)
                                open_score = _map_to_01(mouth_open_ratio, 0.02, 0.18)
                                combined_face = 0.8 * smile_score + 0.2 * open_score
                                combined_face = _clamp01(combined_face)
                                face_sc = int(combined_face * 100)
                                prev_face_scores[tid] = face_sc
                            except Exception:
                                face_sc = 0
                        scores = [s for s in (body_sc, face_sc) if s > 0]
                        agg_sc = sum(scores) / len(scores) if scores else 0
                        client_scores.append((body_sc, face_sc, agg_sc))
                        agg_history.append(agg_sc)
                        current_face_sc, current_body_sc, current_agg_sc = face_sc, body_sc, agg_sc
                else:
                    if current_client_tid is not None:
                        total_time = now - client_start_time
                        if client_scores:
                            posture_scores = [s[0] for s in client_scores]; face_scores = [s[1] for s in client_scores]; agg_scores = [s[2] for s in client_scores]
                            mean_posture = np.mean(posture_scores); mean_face = np.mean(face_scores); mean_total = np.mean(agg_scores)
                        else:
                            mean_posture = mean_face = mean_total = 0
                        client_logs.append((current_client_tid, total_time, mean_posture, mean_face, mean_total))
                        current_client_tid = None

            # overlays for debug/metrics (front-end shows metrics too)
            cv2.putText(annotated, f"face : {current_face_sc}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(annotated, f"posture : {current_body_sc}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(annotated, f"total : {current_agg_sc}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if monitoring_mode and len(agg_history) > 0:
                annotated = draw_satisfaction_graph(annotated, agg_history, annotated.shape[1])

            with lock:
                output_frame = annotated.copy()
                metrics['posture'] = int(current_body_sc)
                metrics['face'] = int(current_face_sc)
                metrics['total'] = int(current_agg_sc)
                metrics['agg_history'].append(int(current_agg_sc))

            if worker['stop']:
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass

# -----------------------
# Script2 implementation (keeps your original logic, with throttling similar to script1)
# -----------------------
def run_script2(source, auto_detect_flag):
    global output_frame, metrics, client_box_norm
    model = YOLO("yolov8x-pose.pt")
    TINY_FACE_URL = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.pt"
    TINY_FACE_PATH = "yolov8n-face-lindevs.pt"
    if not os.path.exists(TINY_FACE_PATH):
        def _download(url, dst):
            with tqdm(unit='B', unit_scale=True, desc=os.path.basename(dst)) as t:
                def _reporthook(b, bs, ts):
                    if ts != -1: t.total = ts
                    t.update(bs)
                urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
        _download(TINY_FACE_URL, TINY_FACE_PATH)
    face_model = YOLO(TINY_FACE_PATH)

    EMOTION_MODEL_URL = "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/enet_b0_8_best_afew.pt"
    EMOTION_MODEL_PATH = "enet_b0_8_best_afew.pt"
    if not os.path.exists(EMOTION_MODEL_PATH):
        def _download2(url,dst):
            with tqdm(unit='B', unit_scale=True, desc=os.path.basename(dst)) as t:
                def _reporthook(b, bs, ts):
                    if ts != -1: t.total = ts
                    t.update(bs)
                urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
        _download2(EMOTION_MODEL_URL, EMOTION_MODEL_PATH)

    emotion_model = timm.create_model('efficientnet_b0', num_classes=8, pretrained=False)
    emotion_model = torch.load(EMOTION_MODEL_PATH, map_location='cpu', weights_only=False)
    emotion_model.eval()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    def generate_insight(total_time, mean_posture, mean_face, mean_total, client_scores):
        prompt = f"""
Customer stayed for {total_time:.1f} seconds.
Mean posture score: {mean_posture:.1f}/100
Mean face score: {mean_face:.1f}/100
Total mean satisfaction: {mean_total:.1f}/100
Scores over time (posture, face, total): {client_scores}
Give a short, professional insight (1-2 sentences) about possible customer mood or service issue.
"""
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[{"role":"system","content":"You are a customer service analyst."},{"role":"user","content":prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def generate_insight_thread(total_time, mean_posture, mean_face, mean_total, client_scores):
        insight = generate_insight(total_time, mean_posture, mean_face, mean_total, client_scores)
        with lock:
            metrics['insight'] = insight

    try:
        idx = None
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx)
        except Exception:
            cap = cv2.VideoCapture(source)

        HISTORY_LEN = 20
        SCORE_INTERVAL = 1.0
        LEAVING_THRESHOLD = 0.2
        ENTERING_THRESHOLD = 0.90
        FACE_CHECK_INTERVAL = 0.5

        landmark_histories = {}
        prev_body_scores = {}
        prev_face_scores = {}
        monitoring_mode = False
        if client_box_norm is not None:
            monitoring_mode = True
        current_face_sc = 0; current_body_sc = 0; current_agg_sc = 0
        agg_history = []
        client_scores = []
        client_logs = []
        last_face_check = 0.0
        cached_face_crop = None
        cached_face_box = None

        while cap.isOpened() and not worker['stop']:
            ret, frame = cap.read()
            if not ret: break
            fh, fw = frame.shape[:2]
            with lock:
                output_frame = frame.copy()
            now = time.monotonic()
            if client_box_norm is not None:
                x1n, y1n, x2n, y2n = client_box_norm
                client_box_pixels = (int(x1n * fw), int(y1n * fh), int(x2n * fw), int(y2n * fh))
                monitoring_mode = True
            else:
                client_box_pixels = None
                if auto_detect_flag:
                    monitoring_mode = True

            results = model.track(frame, persist=True, classes=[0], verbose=False)
            annotated = frame.copy()

            # select candidate: prefer overlap with box, else the first detection
            candidate = None
            if results and len(results) and results[0].boxes is not None and len(results[0].boxes) > 0:
                if client_box_pixels is not None:
                    best = None; best_ov = 0.0
                    for box in results[0].boxes:
                        if box.id is None: continue
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        ix1 = max(x1, client_box_pixels[0]); iy1 = max(y1, client_box_pixels[1])
                        ix2 = min(x2, client_box_pixels[2]); iy2 = min(y2, client_box_pixels[3])
                        ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        a1 = (x2-x1)*(y2-y1) if (x2-x1)>0 and (y2-y1)>0 else 1
                        ov = ia / a1
                        if ov > best_ov:
                            best_ov = ov
                            best = (int(box.id.item()), (x1,y1,x2,y2), ov)
                    if best and best_ov >= LEAVING_THRESHOLD:
                        candidate = best
                else:
                    b = results[0].boxes[0]
                    candidate = (int(b.id.item()) if b.id is not None else 0, tuple(map(int, b.xyxy[0].tolist())), 1.0)

            if candidate is not None:
                tid, bbox, ov = candidate
                x1,y1,x2,y2 = bbox
                # try to compute keypoints and face crop (throttle face detection)
                pts = None
                if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    kp = results[0].keypoints[0].data.cpu().numpy()
                    pts = kp[0,:,:2].copy(); conf = kp[0,:,2]
                    low = conf < 0.5; pts[low] = 0
                    pts[:,0] /= fw; pts[:,1] /= fh
                face_crop = None
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and (now - last_face_check) >= FACE_CHECK_INTERVAL:
                    fres = face_model(crop, verbose=False, conf=0.25)
                    last_face_check = now
                    if fres and fres[0].boxes and len(fres[0].boxes):
                        b2 = fres[0].boxes[0]
                        fx1,fy1,fx2,fy2 = map(int, b2.xyxy[0].tolist())
                        face_crop = crop[max(0,fy1):min(crop.shape[0],fy2), max(0,fx1):min(crop.shape[1],fx2)]
                elif cached_face_crop is not None and cached_face_box is not None:
                    # reuse cached if available
                    face_crop = cached_face_crop

                body_sc = 0
                if pts is not None and pts.shape[0] >= 17:
                    if tid not in landmark_histories:
                        landmark_histories[tid] = collections.deque(maxlen=HISTORY_LEN)
                    landmark_histories[tid].append(pts)
                    if len(landmark_histories[tid]) >= 3:
                        pts_now = landmark_histories[tid][-1]; pts_prev = landmark_histories[tid][-2]; pts_old = landmark_histories[tid][-3]
                        disp1 = np.linalg.norm(pts_now - pts_prev, axis=1); disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)
                        activity = float((disp1.mean() + disp2.mean())/2.0)
                        L_SH,R_SH=5,6; L_HIP,R_HIP=11,12; NOSE=0; L_ELB,R_ELB=7,8; L_WRIST,R_WRIST=9,10
                        shoulders = (pts_now[L_SH] + pts_now[R_SH]) / 2.0
                        hips = (pts_now[L_HIP] + pts_now[R_HIP]) / 2.0
                        nose = pts_now[NOSE]
                        torso_vec = hips - shoulders
                        torso_len = np.linalg.norm(torso_vec) + 1e-6
                        torso_dir = torso_vec / torso_len
                        vertical = np.array([0.0,1.0])
                        torso_angle = min(_angle_between_deg(torso_dir, vertical), 180 - _angle_between_deg(torso_dir, vertical))
                        head_vec = nose - shoulders
                        head_len = np.linalg.norm(head_vec) + 1e-6
                        head_dir = head_vec / head_len
                        head_torso_angle = min(_angle_between_deg(head_dir, torso_dir), 180 - _angle_between_deg(head_dir, torso_dir))
                        sh_ys = abs(pts_now[L_SH][1] - pts_now[R_SH][1])
                        shoulder_sym = sh_ys * fh / (torso_len * fh + 1e-6)
                        shoulder_width = np.linalg.norm(pts_now[L_SH] - pts_now[R_SH]) + 1e-6
                        wrist_dist = np.linalg.norm(pts_now[L_WRIST] - pts_now[R_WRIST])
                        arm_openness = _clamp01((wrist_dist / shoulder_width) / 2.5)
                        wrist_to_nose = min(np.linalg.norm(pts_now[L_WRIST] - nose), np.linalg.norm(pts_now[R_WRIST] - nose))
                        hands_face = _map_to_01(wrist_to_nose, 0.01, 0.20)
                        crossed_arms_penalty = 0.0
                        try:
                            if all(pts_now[i].any() for i in [L_ELB,R_ELB,L_WRIST,R_WRIST,L_SH,R_SH,L_HIP,R_HIP]):
                                left_arm_vec = pts_now[L_WRIST] - pts_now[L_ELB]; right_arm_vec = pts_now[R_WRIST] - pts_now[R_ELB]
                                cross_prod = left_arm_vec[0]*right_arm_vec[1] - left_arm_vec[1]*right_arm_vec[0]
                                if abs(cross_prod) > 0.01: crossed_arms_penalty = 0.50
                        except Exception:
                            pass
                        activity_score = _map_to_01(activity, 0.0008, 0.018)
                        upright_score = 1.0 if torso_angle <= 10 else 0.0 if torso_angle >= 40 else 1.0 - ((torso_angle - 10)/(40 - 10))
                        head_align_score = _clamp01(1.0 - (head_torso_angle / 40.0))
                        shoulder_sym_score = 1.0 - _clamp01(shoulder_sym * 3.0)
                        hands_open_score = hands_face; arm_open_score = arm_openness
                        combined = (0.40 * upright_score + 0.20 * head_align_score + 0.10 * activity_score +
                                    0.10 * hands_open_score + 0.10 * arm_open_score + 0.10 * shoulder_sym_score)
                        combined -= crossed_arms_penalty
                        harsh_penalty = 0.0
                        if activity > 0.018: harsh_penalty = 0.30
                        combined -= harsh_penalty
                        close_arms_penalty = 0.0
                        if arm_openness < 0.3: close_arms_penalty = 0.20
                        combined -= close_arms_penalty
                        relaxed_boost = 0.0
                        if arm_openness > 0.7 and upright_score > 0.8 and activity < 0.005: relaxed_boost = 0.20
                        combined += relaxed_boost
                        combined = _clamp01(combined)
                        body_sc = int(combined * 100)

                face_sc = 0
                if face_crop is not None and face_crop.size > 0:
                    # emotion net inference (throttle done by FACE_CHECK_INTERVAL above)
                    resized = cv2.resize(face_crop, (224,224)); normalized = resized / 255.0
                    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
                    tensor = (tensor - mean) / std
                    with torch.no_grad():
                        output = emotion_model(tensor)
                        probs = torch.softmax(output, dim=1)
                    happy_prob = probs[0,4].item(); neutral_prob = probs[0,5].item(); surprise_prob = probs[0,7].item()
                    face_sc = int(min(happy_prob*100 + neutral_prob*50 + surprise_prob*30, 100.0))

                scores = [s for s in (body_sc, face_sc) if s > 0]
                agg_sc = sum(scores)/len(scores) if scores else 0
                client_scores.append((body_sc, face_sc, agg_sc))
                agg_history.append(agg_sc)
                current_face_sc, current_body_sc, current_agg_sc = face_sc, body_sc, agg_sc
                cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(annotated, f"posture:{current_body_sc} face:{current_face_sc} total:{current_agg_sc}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            with lock:
                output_frame = annotated.copy()
                metrics['posture'] = int(current_body_sc)
                metrics['face'] = int(current_face_sc)
                metrics['total'] = int(current_agg_sc)
                metrics['agg_history'].append(int(current_agg_sc))

            if worker['stop']:
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass

def processing_worker(script, source, auto_detect_flag):
    if script == 'script1':
        run_script1(source, auto_detect_flag)
    else:
        run_script2(source, auto_detect_flag)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
