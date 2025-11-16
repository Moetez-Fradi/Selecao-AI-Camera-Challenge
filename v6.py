import os
import time
import math
import json
import collections
import urllib.request
from tqdm import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from sklearn.cluster import DBSCAN

TINY_FACE_URL  = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.pt"
TINY_FACE_PATH = "yolov8n-face-lindevs.pt"

if not os.path.exists(TINY_FACE_PATH):
    def _download(url, dst):
        with tqdm(unit='B', unit_scale=True, desc=os.path.basename(dst)) as t:
            def _reporthook(b, bs, ts):
                if ts != -1:
                    t.total = ts
                t.update(bs)
            urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
    _download(TINY_FACE_URL, TINY_FACE_PATH)

model = YOLO("yolov8x-pose.pt")
face_model = YOLO(TINY_FACE_PATH)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                             min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

HISTORY_LEN = 20
SCORE_INTERVAL = 1.0
CENTROID_DISP_THRESH = 0.05
STATIONARY_SECONDS = 4.0
TORSO_ACTIVITY_THRESHOLD = 0.0025
ARM_ACTIVITY_THRESHOLD = 0.005
MIN_SHOULDER_WIDTH_FRAC = 0.10

def box_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _angle_between_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def _clamp01(x): return max(0.0, min(1.0, x))

def _map_to_01(v, a, b):
    if b == a: return 0.0
    return _clamp01((v - a) / (b - a))

# posture/face/transaction functions (adapted from your code)

def compute_satisfaction_score(history, fw, fh, prev_score=None, alpha=0.6):
    if len(history) < 3: return None, 0, {}
    pts_now  = history[-1]
    pts_prev = history[-2]
    pts_old  = history[-3]
    if pts_now.shape[0] < 17: return None, 0, {}

    disp1 = np.linalg.norm(pts_now - pts_prev, axis=1)
    disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)
    activity = float((disp1.mean() + disp2.mean()) / 2.0)

    L_SH, R_SH = 5, 6
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

    wrist_to_nose = min(np.linalg.norm(pts_now[L_WRIST] - nose), np.linalg.norm(pts_now[R_WRIST] - nose))
    hands_face = _map_to_01(wrist_to_nose, 0.01, 0.20)

    crossed_arms_penalty = 0.0
    try:
        if all(np.any(pts_now[i]) for i in [L_ELB, R_ELB, L_WRIST, R_WRIST, L_SH, R_SH, L_HIP, R_HIP]):
            if (pts_now[L_WRIST][0] > pts_now[R_SH][0] and pts_now[R_WRIST][0] < pts_now[L_SH][0]) or \
               (pts_now[L_WRIST][0] > pts_now[R_ELB][0] and pts_now[R_WRIST][0] < pts_now[L_ELB][0]):
                chest_y_min = min(pts_now[L_SH][1], pts_now[R_SH][1])
                chest_y_max = max(pts_now[L_HIP][1], pts_now[R_HIP][1])
                if chest_y_min < pts_now[L_WRIST][1] < chest_y_max and chest_y_min < pts_now[R_WRIST][1] < chest_y_max:
                    crossed_arms_penalty = 0.25
    except Exception:
        crossed_arms_penalty = 0.0

    activity_score = _map_to_01(activity, 0.0008, 0.018)
    upright_score = 1.0 if torso_angle <= 10 else 0.0 if torso_angle >= 40 else 1.0 - ((torso_angle-10)/(40-10))
    head_align_score = _clamp01(1.0 - (head_torso_angle/40.0))
    shoulder_sym_score = 1.0 - _clamp01(shoulder_sym*3.0)
    hands_open_score   = hands_face
    arm_open_score = arm_openness

    combined = (0.30*upright_score + 0.25*activity_score + 0.15*hands_open_score +
                0.15*arm_open_score + 0.10*head_align_score + 0.05*shoulder_sym_score)
    combined -= crossed_arms_penalty
    combined = _clamp01(combined)

    if prev_score is not None:
        combined = alpha*combined + (1-alpha)*(prev_score/100.0)

    score = int(combined*100)
    label = "satisfied" if score >= 70 else "neutral" if score >= 45 else "dissatisfied"

    return label, score, {}


def compute_face_expression_score(face_landmarks, fw, fh, prev_score=None, alpha=0.6):
    if face_landmarks is None: return None, 0, {}
    pts = np.array([[p.x, p.y] for p in face_landmarks.landmark], dtype=np.float32)
    try:
        lm_l_mouth = pts[61]; lm_r_mouth = pts[291]
        lm_top_lip = pts[13]; lm_bottom_lip = pts[14]
        lm_l_eye   = pts[33]; lm_r_eye   = pts[263]
    except Exception:
        return None, 0, {}

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


def compute_transaction_indicators(history, fw, fh):
    if len(history) < 3: return None, {}
    pts_now  = history[-1]; pts_prev = history[-2]; pts_old = history[-3]
    if pts_now.shape[0] < 17: return None, {}

    disp1 = np.linalg.norm(pts_now - pts_prev, axis=1)
    disp2 = np.linalg.norm(pts_prev - pts_old, axis=1)

    torso_pts = [5,6,11,12]
    arm_pts   = [7,8,9,10]

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

# face helper
last_face_check = 0.0
cached_face_box = None
cached_face_landmarks = None

def get_face_for_person(frame, person_bbox, now):
    global last_face_check, cached_face_box, cached_face_landmarks
    if now - last_face_check < 0.5:
        if cached_face_box and box_overlap(person_bbox, cached_face_box) > 0.4:
            return cached_face_box, cached_face_landmarks
    x1,y1,x2,y2 = person_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    results = face_model(crop, verbose=False, conf=0.25)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        cached_face_box = None
        cached_face_landmarks = None
        return None, None
    best = results[0].boxes[0]
    fx1, fy1, fx2, fy2 = map(int, best.xyxy[0].tolist())
    fx1 += x1; fy1 += y1; fx2 += x1; fy2 += y1
    face_box = (fx1, fy1, fx2, fy2)
    crop_face_y1 = max(0, fy1 - y1)
    crop_face_y2 = min(crop.shape[0], fy2 - y1)
    crop_face_x1 = max(0, fx1 - x1)
    crop_face_x2 = min(crop.shape[1], fx2 - x1)
    face_crop = crop[crop_face_y1:crop_face_y2, crop_face_x1:crop_face_x2]
    landmarks = None
    if face_crop.size != 0:
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

# IoU tracker fallback

def iou(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    inter = iw*ih
    aarea = max(1,(ax2-ax1)*(ay2-ay1))
    barea = max(1,(bx2-bx1)*(by2-by1))
    return inter / float(aarea + barea - inter)

class SimpleIOUTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_age = max_age
    def update(self, detections, frame_id):
        assigned = set()
        dets = [tuple(map(int,d[:4])) for d in detections]
        new_tracks = {}
        for tid, t in list(self.tracks.items()):
            best_iou = 0; best_j = -1
            for j,det in enumerate(dets):
                if j in assigned: continue
                val = iou(t['bbox'], det)
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou >= self.iou_thresh:
                new_tracks[tid] = {'bbox':dets[best_j], 'last_seen':frame_id}
                assigned.add(best_j)
            else:
                if frame_id - t['last_seen'] <= self.max_age:
                    new_tracks[tid] = t
        for j,det in enumerate(dets):
            if j in assigned: continue
            new_tracks[self.next_id] = {'bbox':det, 'last_seen':frame_id}
            self.next_id += 1
        self.tracks = new_tracks
        out = []
        for tid,t in self.tracks.items():
            x1,y1,x2,y2 = t['bbox']
            out.append(type('T', (), {'track_id': tid, 'tlbr': (x1,y1,x2,y2)}))
        return out

# cluster pixelrefer output to zones

def cluster_pixelrefer(pr_json, eps=50, min_samples=10, margin=80):
    with open(pr_json,'r') as f:
        data = json.load(f)
    centroids = []
    for entry in data:
        for b in entry.get('boxes',[]):
            x1,y1,x2,y2,score = b
            centroids.append([(x1+x2)/2.0,(y1+y2)/2.0])
    if len(centroids) == 0:
        return []
    centroids = np.array(centroids)
    cl = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = cl.labels_
    zones = []
    for lbl in np.unique(labels):
        if lbl < 0: continue
        pts = centroids[labels==lbl]
        cx,cy = pts.mean(axis=0)
        x_min,y_min = pts.min(axis=0) - margin
        x_max,y_max = pts.max(axis=0) + margin
        zones.append({'label':int(lbl),'bbox':[int(x_min),int(y_min),int(x_max),int(y_max)],'centroid':[int(cx),int(cy)]})
    with open('service_zones.json','w') as f:
        json.dump(zones,f,indent=2)
    return zones

# main run function

def run(video_path, pixelrefer_json=None, zone_index=0, out_csv='client_logs.csv'):
    zones = []
    if pixelrefer_json and os.path.exists(pixelrefer_json):
        zones = cluster_pixelrefer(pixelrefer_json)
    if not zones:
        print('No PixelRefer zones found; please run PixelRefer or provide pixelrefer_json')
        return
    zone = zones[zone_index]['bbox']

    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = SimpleIOUTracker(iou_thresh=0.3, max_age=int(fps*2))

    landmark_histories = {}
    prev_body_scores = {}
    prev_face_scores = {}
    active_clients = {}
    client_logs = []
    frame_id = 0
    last_score_time = time.time()

    while True:
        ret, frame = cap.read();
        if not ret: break
        frame_id += 1
        now = time.time()

        res = model.predict(frame, imgsz=1024, conf=0.25, classes=[0], verbose=False)
        dets = []
        if len(res) > 0:
            r = res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item()) if hasattr(box, 'conf') else 1.0
                    dets.append([x1,y1,x2,y2,conf])
        tracks = tracker.update(dets, frame_id)

        annotated = frame.copy()
        for tr in tracks:
            tid = int(tr.track_id)
            x1,y1,x2,y2 = map(int,tr.tlbr)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(annotated,f"ID:{tid}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            ov = box_overlap((x1,y1,x2,y2), zone)
            if ov > 0.6:
                if tid not in active_clients:
                    active_clients[tid] = {'start': now, 'frames': [], 'last_seen': now, 'pose': collections.deque(maxlen=HISTORY_LEN), 'scores': []}
                active_clients[tid]['last_seen'] = now
                active_clients[tid]['frames'].append((frame_id,(x1,y1,x2,y2)))

                # try to extract pose keypoints if available
                if res and hasattr(r,'keypoints') and r.keypoints is not None and len(r.keypoints)>0:
                    for i,kp in enumerate(r.keypoints):
                        if r.boxes is not None and i < len(r.boxes):
                            b = r.boxes[i]
                            bid = getattr(b,'id',None)
                        # assume matching by IoU: skip robust matching for brevity
                    # attempt: if keypoints present, pick nearest box
                    if r.keypoints is not None and len(r.keypoints)>0:
                        kps = r.keypoints[0].data.cpu().numpy()
                        pts = kps[0,:,:2]
                        pts[:,0] /= fw; pts[:,1] /= fh
                        low_conf = kps[0,:,2] < 0.5
                        pts[low_conf] = 0
                        if pts.shape[0] >= 17:
                            active_clients[tid]['pose'].append(pts)

                # face
                face_box, face_landmarks = get_face_for_person(frame, (x1,y1,x2,y2), now)
                if face_landmarks:
                    _, face_sc, _ = compute_face_expression_score(face_landmarks, fw, fh, prev_face_scores.get(tid))
                    prev_face_scores[tid] = face_sc
                else:
                    face_sc = 0

                if len(active_clients[tid]['pose'])>=3 and (now - last_score_time) >= SCORE_INTERVAL:
                    last_score_time = now
                    _, body_sc, _ = compute_satisfaction_score(active_clients[tid]['pose'], fw, fh, prev_body_scores.get(tid))
                    prev_body_scores[tid] = body_sc
                    # transaction check
                    trans, diag = compute_transaction_indicators(list(active_clients[tid]['pose']), fw, fh)
                    active_clients[tid]['scores'].append((body_sc, face_sc, (body_sc+face_sc)/2 if (body_sc+face_sc)>0 else 0))
                    if trans == 'potential_transaction':
                        active_clients[tid]['transaction_at'] = now

            else:
                if tid in active_clients:
                    elapsed = now - active_clients[tid]['start']
                    scores = active_clients[tid]['scores']
                    if scores:
                        posture = float(np.mean([s[0] for s in scores]))
                        face = float(np.mean([s[1] for s in scores]))
                        agg = float(np.mean([s[2] for s in scores]))
                    else:
                        posture=face=agg=0.0
                    client_logs.append({'zone':zones[zone_index]['label'],'track_id':tid,'enter_time':active_clients[tid]['start'],'leave_time':now,'duration':elapsed,'mean_posture':posture,'mean_face':face,'mean_agg':agg,'frames':len(active_clients[tid]['frames']), 'transaction_at': active_clients[tid].get('transaction_at',None)})
                    del active_clients[tid]

        x1,y1,x2,y2 = zone
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.imshow('service_monitor', annotated)
        if cv2.waitKey(1) == 27: break

    for tid, info in active_clients.items():
        elapsed = time.time() - info['start']
        scores = info['scores']
        if scores:
            posture = float(np.mean([s[0] for s in scores]))
            face = float(np.mean([s[1] for s in scores]))
            agg = float(np.mean([s[2] for s in scores]))
        else:
            posture=face=agg=0.0
        client_logs.append({'zone':zones[zone_index]['label'],'track_id':tid,'enter_time':info['start'],'leave_time':time.time(),'duration':elapsed,'mean_posture':posture,'mean_face':face,'mean_agg':agg,'frames':len(info['frames']),'transaction_at': info.get('transaction_at',None)})

    with open(out_csv,'w') as f:
        f.write('zone,track_id,enter_time,leave_time,duration,mean_posture,mean_face,mean_agg,frames,transaction_at\n')
        for c in client_logs:
            f.write(f"{c['zone']},{c['track_id']},{c['enter_time']},{c['leave_time']},{c['duration']},{c['mean_posture']},{c['mean_face']},{c['mean_agg']},{c['frames']},{c['transaction_at']}\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # usage: edit paths accordingly and run in your NVIDIA Jupyter
    video_path = './testcases/siu.mp4'
    pixelrefer_json = './pixelrefer_output.json'  # produced by PixelRefer demo
    run(video_path, pixelrefer_json=pixelrefer_json, zone_index=0)
