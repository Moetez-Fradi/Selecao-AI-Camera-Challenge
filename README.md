# Time Processing & Client Satisfaction (AI Vision)

This repo was created for the AI Camera Challenge.

It contains an AI-vision toolkit to monitor and analyze customer interactions at service counters ("guichets").

We provided two similar approches:

- They both use the service area, tracks people approaching the counter, and computes per-client metrics such as posture-based satisfaction and facial-expression score. They both also produces a real-time overlay, graphs and logs.

-  The first approach detects the service area automatically without user interaction algorithmically, we tried to implemet `PixelRefer` for a more intelligent system, but the time was so tight that we couldn't.

- The second approach prompts you to select the service zone manually, and has an additional feature: an LLM generated text that analysis the data overtime and reports the client experience.

**Contents**

- `autonomous.py` — Main autonomous pipeline. Automatically finds the counter (guichet), defines a client box, tracks clients, computes posture & face scores, draws overlays and a satisfaction graph, and logs per-client statistics.
- `manual-latest.py` — Manual client-zone selection with an emotion classifier and insight generation via an LLM. Useful for owners that don't want to analyze the scene by themselves (time and energy consuming)
- `versions/` — Historical/alternate variants (`v3.py`, `v4.py`, `v6.py`, `danger_zone-*.py`, `newStrategy.py`). These contain earlier or experimental logic for posture/face scoring and guichet detection.
- `scripts/` — Utility scripts for camera checks and streaming: `testCameras.py`, `stream.py`, `orchestration.py`, `cuda.py`, `chechCameras.py`.
- `testcases/` — Sample video files used for local testing (not included in repo by default).
- `requirements.txt` — Python package pins required to run the project.

**High-level features**

- Automatic guichet discovery via clustering of stationary, close-by people.
- Client detection and tracking using `ultralytics` YOLO tracking and keypoints (pose).
- Posture / satisfaction scoring based on pose keypoints (uprightness, arm openness, activity, symmetry, crossed-arms penalty).
- Face-expression scoring via MediaPipe face mesh (smile/open-mouth heuristics); optional emotion classifier (in `manual-latest.py`).
- Transaction detection (torso still, arms moving) to identify potential payment/exchange events (not implemented).
- Real-time overlay with bounding boxes, pose mesh, face mesh, scores, chronometer and a satisfaction graph.
- Per-client logging for later analysis; the app prints summaries when clients leave and stores them in `client_logs`.

## Getting started
---------------

1. Create and activate a Python environment (windows):

```bash
python -m venv .venv
source .venv/Scripts/activate 
pip install -U pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- The pinned `torch`/CUDA builds in `requirements.txt` are examples; ensure the installed `torch` matches your CUDA drivers (or use CPU-only builds if needed).
- `ultralytics` is required for YOLO models and tracking.

Run the main demos
-----------------

- Autonomous mode (automatic guichet detection and monitoring):

```bash
python autonomous.py
```

- Manual demo (select client zone manually + emotion model + LLM insights):

```bash
python manual-latest.py
```

- Quick camera test scripts:

```bash
python scripts/testCameras.py    # test local camera indexes, you can use a real time feed for both approaches
python scripts/cuda.py           # quick CUDA availability check
python scripts/stream.py         # run a YOLO stream example
```

## Configuration and important constants
------------------------------------

Key constants used to tune detection and scoring are defined near the top of `autonomous.py` and `manual-latest.py`. One frequently adjusted block is:

```python
HISTORY_LEN               = 20  # Increased for more robustness
TORSO_ACTIVITY_THRESHOLD  = 0.0025   # the movement allowed
ARM_ACTIVITY_THRESHOLD    = 0.005
MIN_SHOULDER_WIDTH_FRAC   = 0.10     # smaller people are OK
STATIONARY_SECONDS        = 4.0      # Adjusted as per user

CLIENT_BOX_SCALE_W = 2.5  # Width scale
CLIENT_BOX_SCALE_H = 1  # Height scale
SCORE_INTERVAL   = 1.0  # Compute scores every 1 second
CENTROID_DISP_THRESH = 0.05  # Relaxed to 0.05 (5% of frame) for "close to same place"
LEAVING_THRESHOLD = 0.2  # Overlap below this considers client left
ENTERING_THRESHOLD = 0.90  # Overlap at or above this to count as new client
```

You can tune these values to adapt to camera FOV, placement, and the expected behavior at the counter.

## Where models come from
----------------------

- `yolov8x-pose.pt`, `yolov8x.pt`, `yolov8n.pt` etc. are used from YOLO (Ultralytics). Pose keypoints are used for posture scoring.
- A small YOLO face model is downloaded automatically by `autonomous.py` and `manual-latest.py` (`yolov8n-face-lindevs.pt`) for quick face cropping.
- `manual-latest.py` optionally downloads an emotion model for a more fine-grained face/emotion classifier.

## Outputs
-------

- Real-time annotated window showing:
	- Bounding boxes for detected people
	- Pose skeleton (COCO connections)
	- Face boxes and MediaPipe face mesh
	- Scores (face / posture / aggregate) and client chronometer
	- A satisfaction graph appended to the window while monitoring
- Console logs with per-client timing and averaged scores.
- `client_logs` in memory — a list of tuples (client_id, time_spent, mean_posture, mean_face, mean_total). You can modify the code to persist these to CSV/DB.


## Contribution and Next steps
----------------------

We put so much effort into this solution, even if it didn't win, we see in its code a potential for reusing and enhancing it. If you see any contribution, we would be happy to include it!