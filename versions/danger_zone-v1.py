import time 
import collections 
import math 
import numpy as np 
import cv2 
from ultralytics import YOLO 
import mediapipe as mp 

model =YOLO ("yolov8n.pt")


cap =cv2 .VideoCapture ("./testcases/swey.mp4")

mp_pose =mp .solutions .pose 
pose =mp_pose .Pose (model_complexity =1 ,enable_segmentation =False )

mp_face =mp .solutions .face_mesh 
face_mesh =mp_face .FaceMesh (static_image_mode =False ,max_num_faces =1 ,refine_landmarks =False )

HISTORY_LEN =6 
landmark_history =collections .deque (maxlen =HISTORY_LEN )
bbox_centroid_history =collections .deque (maxlen =HISTORY_LEN )

last_print =0.0 
PRINT_INTERVAL =1.0 

prev_body_score =None 
prev_face_score =None 

stationary_since =None 
STATIONARY_SECONDS =4.0 
ACTIVITY_THRESHOLD =0.002 
FACING_ANGLE_DEG =60.0 

def _angle_between_deg (v1 ,v2 ):
    n1 =np .linalg .norm (v1 );n2 =np .linalg .norm (v2 )
    if n1 ==0 or n2 ==0 :return 0.0 
    c =np .clip (np .dot (v1 ,v2 )/(n1 *n2 ),-1.0 ,1.0 )
    return math .degrees (math .acos (c ))

def _clamp01 (x ):
    return max (0.0 ,min (1.0 ,x ))

def _map_to_01 (v ,a ,b ):
    if b ==a :return 0.0 
    return _clamp01 ((v -a )/(b -a ))

def compute_satisfaction_score (history ,frame_w ,frame_h ,prev_score =None ,alpha_smooth =0.6 ):
    if len (history )<3 :
        return None ,0 ,{}
    pts_now =history [-1 ];pts_prev =history [-2 ];pts_old =history [-3 ]
    if pts_now .shape [0 ]<25 :
        return None ,0 ,{}

    disp1 =np .linalg .norm (pts_now -pts_prev ,axis =1 )
    disp2 =np .linalg .norm (pts_prev -pts_old ,axis =1 )
    activity =float ((disp1 .mean ()+disp2 .mean ())/2.0 )

    L_SH ,R_SH =11 ,12 
    L_HIP ,R_HIP =23 ,24 
    NOSE =0 
    L_WRIST ,R_WRIST =15 ,16 

    shoulders =(pts_now [L_SH ]+pts_now [R_SH ])/2.0 
    hips =(pts_now [L_HIP ]+pts_now [R_HIP ])/2.0 
    nose =pts_now [NOSE ]

    torso_vec =hips -shoulders 
    torso_len =np .linalg .norm (torso_vec )+1e-6 
    torso_dir =torso_vec /torso_len 

    vertical =np .array ([0.0 ,1.0 ])
    torso_angle =_angle_between_deg (torso_dir ,vertical )
    torso_angle =min (torso_angle ,180 -torso_angle )

    head_vec =nose -shoulders 
    head_len =np .linalg .norm (head_vec )+1e-6 
    head_dir =head_vec /head_len 
    head_torso_angle =_angle_between_deg (head_dir ,torso_dir )
    head_torso_angle =min (head_torso_angle ,180 -head_torso_angle )

    sh_ys =abs (pts_now [L_SH ][1 ]-pts_now [R_SH ][1 ])
    shoulder_sym =sh_ys *frame_h /(torso_len *frame_h +1e-6 )

    shoulder_width =np .linalg .norm (pts_now [L_SH ]-pts_now [R_SH ])+1e-6 
    wrist_dist =np .linalg .norm (pts_now [L_WRIST ]-pts_now [R_WRIST ])
    arm_openness =_clamp01 ((wrist_dist /shoulder_width )/2.5 )

    wrist_to_nose =min (np .linalg .norm (pts_now [L_WRIST ]-nose ),np .linalg .norm (pts_now [R_WRIST ]-nose ))
    hands_face =_map_to_01 (wrist_to_nose ,0.01 ,0.20 )

    activity_score =_map_to_01 (activity ,0.0008 ,0.018 )
    if torso_angle <=10 :
        upright_score =1.0 
    elif torso_angle >=40 :
        upright_score =0.0 
    else :
        upright_score =1.0 -((torso_angle -10 )/(40 -10 ))
    head_align_score =_clamp01 (1.0 -(head_torso_angle /40.0 ))
    shoulder_sym_score =1.0 -_clamp01 (shoulder_sym *3.0 )
    hands_open_score =hands_face 

    combined =(
    0.35 *upright_score +
    0.30 *activity_score +
    0.15 *hands_open_score +
    0.12 *head_align_score +
    0.08 *shoulder_sym_score 
    )
    combined =_clamp01 (combined )

    if prev_score is not None :
        combined =alpha_smooth *combined +(1.0 -alpha_smooth )*(prev_score /100.0 )

    score =int (combined *100 )
    if score >=70 :
        label ="satisfied"
    elif score >=45 :
        label ="neutral"
    else :
        label ="dissatisfied"

    diag ={
    "activity":activity ,
    "torso_angle":torso_angle ,
    "head_torso_angle":head_torso_angle ,
    "arm_openness":arm_openness ,
    "hands_face_dist":wrist_to_nose ,
    "activity_score":activity_score ,
    "upright_score":upright_score ,
    "head_align_score":head_align_score ,
    "shoulder_sym_score":shoulder_sym_score ,
    "combined_raw":combined 
    }
    return label ,score ,diag 

def compute_face_expression_score (face_landmarks ,frame_w ,frame_h ,prev_score =None ,alpha_smooth =0.6 ):
    if face_landmarks is None :
        return None ,0 ,{}

    pts =np .array ([[p .x ,p .y ]for p in face_landmarks ],dtype =np .float32 )
    if pts .shape [0 ]<300 :
        pass 

    try :
        lm_l_mouth =pts [61 ]
        lm_r_mouth =pts [291 ]
        lm_top_lip =pts [13 ]
        lm_bottom_lip =pts [14 ]
        lm_l_eye =pts [33 ]
        lm_r_eye =pts [263 ]
    except Exception :
        return None ,0 ,{}

    mouth_w =np .linalg .norm ((lm_r_mouth -lm_l_mouth ))
    mouth_h =np .linalg .norm ((lm_bottom_lip -lm_top_lip ))
    eye_dist =np .linalg .norm (lm_r_eye -lm_l_eye )+1e-6 

    smile_ratio =mouth_w /eye_dist 
    mouth_open_ratio =mouth_h /eye_dist 

    smile_score =_map_to_01 (smile_ratio ,0.35 ,0.75 )
    open_score =_map_to_01 (mouth_open_ratio ,0.02 ,0.18 )

    combined =0.8 *smile_score +0.2 *open_score 
    combined =_clamp01 (combined )

    if prev_score is not None :
        combined =alpha_smooth *combined +(1.0 -alpha_smooth )*(prev_score /100.0 )

    score =int (combined *100 )
    if score >=70 :
        label ="satisfied"
    elif score >=45 :
        label ="neutral"
    else :
        label ="dissatisfied"

    diag ={
    "smile_ratio":float (smile_ratio ),
    "mouth_open_ratio":float (mouth_open_ratio ),
    "smile_score":float (smile_score ),
    "open_score":float (open_score ),
    "combined_raw":combined 
    }
    return label ,score ,diag 

while True :
    ret ,frame =cap .read ()
    if not ret :
        break 

    frame_h ,frame_w =frame .shape [:2 ]
    frame_rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )

    pose_res =pose .process (frame_rgb )
    face_res =face_mesh .process (frame_rgb )
    results =model .predict (frame ,verbose =False )
    annotated =frame .copy ()

    person_boxes =[]
    for box in results [0 ].boxes :
        cls =int (box .cls [0 ])
        if cls !=0 :
            continue 
        x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ].tolist ())
        area =(x2 -x1 )*(y2 -y1 )
        person_boxes .append ((box ,(x1 ,y1 ,x2 ,y2 ),area ))

    best_box =None 
    best_area =-1.0 
    best_facing =False 
    camera_dir =np .array ([0.0 ,-1.0 ])

    for box ,(x1 ,y1 ,x2 ,y2 ),area in person_boxes :
        has_pose_data =False 
        facing_ok =False 
        if pose_res .pose_landmarks :
            lm =pose_res .pose_landmarks .landmark 
            pts =np .array ([[p .x ,p .y ]for p in lm ],dtype =np .float32 )
            px_n =int (lm [0 ].x *frame_w );py_n =int (lm [0 ].y *frame_h )
            px_lsh =int (lm [11 ].x *frame_w );py_lsh =int (lm [11 ].y *frame_h )
            px_rsh =int (lm [12 ].x *frame_w );py_rsh =int (lm [12 ].y *frame_h )
            if x1 <=px_n <=x2 and y1 <=py_n <=y2 and x1 <=px_lsh <=x2 and x1 <=px_rsh <=x2 :
                has_pose_data =True 
                shoulders =(pts [11 ]+pts [12 ])/2.0 
                nose =pts [0 ]
                head_vec =nose -shoulders 
                head_dir =head_vec /(np .linalg .norm (head_vec )+1e-6 )
                ang =_angle_between_deg (head_dir ,camera_dir )
                if ang <=FACING_ANGLE_DEG :
                    facing_ok =True 

        if facing_ok and area >best_area :
            best_area =area 
            best_box =box 
            best_facing =True 

    if best_box is None and person_boxes :

        best_box ,(x1 ,y1 ,x2 ,y2 ),best_area =max (person_boxes ,key =lambda x :x [2 ])
        best_facing =False 
    elif best_box is not None :
        x1 ,y1 ,x2 ,y2 =map (int ,best_box .xyxy [0 ].tolist ())
        cv2 .rectangle (annotated ,(x1 ,y1 ),(x2 ,y2 ),(0 ,220 ,0 ),2 )

        if pose_res .pose_landmarks :
            lm =pose_res .pose_landmarks .landmark 
            pts =np .array ([[p .x ,p .y ]for p in lm ],dtype =np .float32 )
            landmark_history .append (pts )
            for p in lm :
                px =int (p .x *frame_w );py =int (p .y *frame_h )
                if x1 <=px <=x2 and y1 <=py <=y2 :
                    cv2 .circle (annotated ,(px ,py ),3 ,(255 ,100 ,0 ),-1 )

        centroid =((x1 +x2 )/2.0 ,(y1 +y2 )/2.0 )
        bbox_centroid_history .append (centroid )

        now =time .monotonic ()
        if now -last_print >=PRINT_INTERVAL :
            last_print =now 

            body_label ,body_score ,body_diag =compute_satisfaction_score (
            landmark_history ,frame_w ,frame_h ,prev_score =prev_body_score ,alpha_smooth =0.6 
            )
            if body_label is None :
                body_label ,body_score ="no_data",0 
            prev_body_score =body_score 


            activity =None 
            if isinstance (body_diag ,dict )and "activity"in body_diag :
                activity =body_diag ["activity"]
            else :
                if len (bbox_centroid_history )>=3 :
                    d0 =np .array (bbox_centroid_history [-1 ])
                    d1 =np .array (bbox_centroid_history [-2 ])
                    d2 =np .array (bbox_centroid_history [-3 ])
                    disp1 =np .linalg .norm (d0 -d1 )
                    disp2 =np .linalg .norm (d1 -d2 )
                    activity =float ((disp1 +disp2 )/2.0 )/max (frame_w ,frame_h )

            moving =True 
            if activity is not None and activity <ACTIVITY_THRESHOLD :
                if stationary_since is None :
                    stationary_since =now 
                elif now -stationary_since >=STATIONARY_SECONDS :
                    moving =False 
            else :
                stationary_since =None 
                moving =True 

            if face_res .multi_face_landmarks and len (face_res .multi_face_landmarks )>0 :
                f_lm =face_res .multi_face_landmarks [0 ].landmark 
                face_label ,face_score ,face_diag =compute_face_expression_score (
                f_lm ,frame_w ,frame_h ,prev_score =prev_face_score ,alpha_smooth =0.6 
                )
                if face_label is None :
                    face_label ,face_score ="no_face",0 
                prev_face_score =face_score 
            else :
                face_label ,face_score ="no_face",0 

            scores =[]
            if isinstance (body_score ,(int ,float )):scores .append (float (body_score ))
            if isinstance (face_score ,(int ,float )):scores .append (float (face_score ))
            avg_score =int (sum (scores )/len (scores ))if scores else 0 
            avg_label ="no_data"
            if avg_score >=70 :
                avg_label ="satisfied"
            elif avg_score >=45 :
                avg_label ="neutral"
            else :
                avg_label ="dissatisfied"

            state ="moving"if moving else "stationary"
            facing_text ="FACING"if best_facing else "unknown_facing"
            print (f"[{time .strftime ('%H:%M:%S')}] CHOSEN PERSON: {facing_text } {state }  BODY: {body_label } ({body_score })  FACE: {face_label } ({face_score })  AVG: {avg_label } ({avg_score })")

            cv2 .putText (annotated ,f"STATE:{state }",(12 ,26 ),
            cv2 .FONT_HERSHEY_SIMPLEX ,0.7 ,(200 ,200 ,200 ),2 )
            cv2 .putText (annotated ,f"FACE:{face_label [:6 ].upper ()} {face_score }",(12 ,56 ),
            cv2 .FONT_HERSHEY_SIMPLEX ,0.7 ,(200 ,200 ,200 ),2 )
            cv2 .putText (annotated ,f"AVG:{avg_label [:6 ].upper ()} {avg_score }",(12 ,86 ),
            cv2 .FONT_HERSHEY_SIMPLEX ,0.8 ,(0 ,255 ,200 ),2 )

    else :
        cv2 .putText (annotated ,"No person detected",(12 ,26 ),
        cv2 .FONT_HERSHEY_SIMPLEX ,0.9 ,(0 ,0 ,255 ),2 )
        stationary_since =None 
        landmark_history .clear ()
        bbox_centroid_history .clear ()

    if face_res .multi_face_landmarks and len (face_res .multi_face_landmarks )>0 :
        flm =face_res .multi_face_landmarks [0 ].landmark 
        xs =np .array ([p .x for p in flm ])*frame_w 
        ys =np .array ([p .y for p in flm ])*frame_h 
        fx1 ,fy1 =int (xs .min ()),int (ys .min ())
        fx2 ,fy2 =int (xs .max ()),int (ys .max ())
        cv2 .rectangle (annotated ,(fx1 ,fy1 ),(fx2 ,fy2 ),(160 ,160 ,255 ),1 )

    cv2 .imshow ("YOLO + Pose + Face (CX rating)",annotated )
    if cv2 .waitKey (1 )==27 :
        break 

cap .release ()
cv2 .destroyAllWindows ()
