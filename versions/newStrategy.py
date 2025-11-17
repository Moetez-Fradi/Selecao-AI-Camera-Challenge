import time 
import collections 
import math 
import numpy as np 
import cv2 
from ultralytics import YOLO 
import mediapipe as mp 


model =YOLO ("yolov8m.pt")
cap =cv2 .VideoCapture ("./testcases/preview.mp4")

mp_pose =mp .solutions .pose 
pose =mp_pose .Pose (model_complexity =1 ,enable_segmentation =False )

mp_face =mp .solutions .face_mesh 
face_mesh =mp_face .FaceMesh (
static_image_mode =False ,
max_num_faces =4 ,
refine_landmarks =True ,
min_detection_confidence =0.4 ,
min_tracking_confidence =0.4 
)

mp_drawing =mp .solutions .drawing_utils 

HISTORY_LEN =10 
TORSO_ACTIVITY_THRESHOLD =0.0025 
ARM_ACTIVITY_THRESHOLD =0.005 
MIN_SHOULDER_WIDTH_FRAC =0.10 
STATIONARY_SECONDS =4.0 


CLIENT_BOX_SCALE_W =3 
CLIENT_BOX_SCALE_H =1.5 
OVERLAP_THRESH =0.5 
SCORE_INTERVAL =1.0 
CENTROID_DISP_THRESH =0.05 


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

def compute_satisfaction_score (history ,fw ,fh ,prev_score =None ,alpha =0.6 ):
    if len (history )<3 :return None ,0 ,{}
    pts_now =history [-1 ]
    pts_prev =history [-2 ]
    pts_old =history [-3 ]
    if pts_now .shape [0 ]<25 :return None ,0 ,{}

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
    torso_angle =min (_angle_between_deg (torso_dir ,vertical ),180 -_angle_between_deg (torso_dir ,vertical ))

    head_vec =nose -shoulders 
    head_len =np .linalg .norm (head_vec )+1e-6 
    head_dir =head_vec /head_len 
    head_torso_angle =min (_angle_between_deg (head_dir ,torso_dir ),180 -_angle_between_deg (head_dir ,torso_dir ))

    sh_ys =abs (pts_now [L_SH ][1 ]-pts_now [R_SH ][1 ])
    shoulder_sym =sh_ys *fh /(torso_len *fh +1e-6 )

    shoulder_width =np .linalg .norm (pts_now [L_SH ]-pts_now [R_SH ])+1e-6 
    wrist_dist =np .linalg .norm (pts_now [L_WRIST ]-pts_now [R_WRIST ])
    arm_openness =_clamp01 ((wrist_dist /shoulder_width )/2.5 )

    wrist_to_nose =min (np .linalg .norm (pts_now [L_WRIST ]-nose ),
    np .linalg .norm (pts_now [R_WRIST ]-nose ))
    hands_face =_map_to_01 (wrist_to_nose ,0.01 ,0.20 )

    activity_score =_map_to_01 (activity ,0.0008 ,0.018 )
    upright_score =1.0 if torso_angle <=10 else 0.0 if torso_angle >=40 else 1.0 -((torso_angle -10 )/(40 -10 ))
    head_align_score =_clamp01 (1.0 -(head_torso_angle /40.0 ))
    shoulder_sym_score =1.0 -_clamp01 (shoulder_sym *3.0 )
    hands_open_score =hands_face 

    combined =(0.35 *upright_score +0.30 *activity_score +0.15 *hands_open_score +
    0.12 *head_align_score +0.08 *shoulder_sym_score )
    combined =_clamp01 (combined )

    if prev_score is not None :
        combined =alpha *combined +(1 -alpha )*(prev_score /100.0 )

    score =int (combined *100 )
    label ="satisfied"if score >=70 else "neutral"if score >=45 else "dissatisfied"

    return label ,score ,{}

def compute_face_expression_score (face_landmarks ,fw ,fh ,prev_score =None ,alpha =0.6 ):
    if face_landmarks is None :return None ,0 ,{}
    pts =np .array ([[p .x ,p .y ]for p in face_landmarks ],dtype =np .float32 )
    try :
        lm_l_mouth =pts [61 ];lm_r_mouth =pts [291 ]
        lm_top_lip =pts [13 ];lm_bottom_lip =pts [14 ]
        lm_l_eye =pts [33 ];lm_r_eye =pts [263 ]
    except Exception :return None ,0 ,{}

    mouth_w =np .linalg .norm (lm_r_mouth -lm_l_mouth )
    mouth_h =np .linalg .norm (lm_bottom_lip -lm_top_lip )
    eye_dist =np .linalg .norm (lm_r_eye -lm_l_eye )+1e-6 

    smile_ratio =mouth_w /eye_dist 
    mouth_open_ratio =mouth_h /eye_dist 

    smile_score =_map_to_01 (smile_ratio ,0.35 ,0.75 )
    open_score =_map_to_01 (mouth_open_ratio ,0.02 ,0.18 )

    combined =0.8 *smile_score +0.2 *open_score 
    combined =_clamp01 (combined )

    if prev_score is not None :
        combined =alpha *combined +(1 -alpha )*(prev_score /100.0 )

    score =int (combined *100 )
    label ="satisfied"if score >=70 else "neutral"if score >=45 else "dissatisfied"
    return label ,score ,{}

def compute_transaction_indicators (history ,fw ,fh ):
    if len (history )<3 :return None ,{}
    pts_now =history [-1 ];pts_prev =history [-2 ];pts_old =history [-3 ]
    if pts_now .shape [0 ]<25 :return None ,{}

    disp1 =np .linalg .norm (pts_now -pts_prev ,axis =1 )
    disp2 =np .linalg .norm (pts_prev -pts_old ,axis =1 )

    torso_pts =[11 ,12 ,23 ,24 ]
    arm_pts =[13 ,14 ,15 ,16 ]

    torso_disp =(disp1 [torso_pts ].mean ()+disp2 [torso_pts ].mean ())/2.0 
    arm_disp =(disp1 [arm_pts ].mean ()+disp2 [arm_pts ].mean ())/2.0 

    shoulders =(pts_now [11 ]+pts_now [12 ])/2.0 
    nose =pts_now [0 ]
    head_vec =nose -shoulders 
    head_len =np .linalg .norm (head_vec )+1e-6 
    head_dir =head_vec /head_len 
    cam_dir =np .array ([0.0 ,-1.0 ])
    facing_angle =_angle_between_deg (head_dir ,cam_dir )

    shoulder_width =np .linalg .norm (pts_now [11 ]-pts_now [12 ])*fw 
    shoulder_frac =shoulder_width /fw 

    diag ={"torso_activity":torso_disp ,"arm_activity":arm_disp ,
    "facing_angle":facing_angle ,"shoulder_frac":shoulder_frac }

    is_transaction =(torso_disp <TORSO_ACTIVITY_THRESHOLD and 
    arm_disp >ARM_ACTIVITY_THRESHOLD and 
    shoulder_frac >MIN_SHOULDER_WIDTH_FRAC )

    return "potential_transaction"if is_transaction else None ,diag 

landmark_histories ={}
torso_activity_histories ={}
bbox_histories ={}
stationary_since ={}
prev_body_scores ={}
prev_face_scores ={}
stationary_start_time ={}

last_print =0.0 

transaction_locations =[]


current_client_tid =None 
client_start_time =None 
client_scores =[]
last_score_time =0.0 
client_logs =[]


guichet_loc =None 
client_box =None 


def box_overlap (box1 ,box2 ):

    ix1 =max (box1 [0 ],box2 [0 ])
    iy1 =max (box1 [1 ],box2 [1 ])
    ix2 =min (box1 [2 ],box2 [2 ])
    iy2 =min (box1 [3 ],box2 [3 ])
    ia =max (0 ,ix2 -ix1 )*max (0 ,iy2 -iy1 )
    a1 =(box1 [2 ]-box1 [0 ])*(box1 [3 ]-box1 [1 ])
    return ia /a1 if a1 >0 else 0.0 


def bbox_area (box ):
    x1 ,y1 ,x2 ,y2 =box 
    return (x2 -x1 )*(y2 -y1 )


def compute_centroid_disp (bbox_list ,fw ,fh ):
    if len (bbox_list )<3 :return 1.0 
    disps =[]
    for i in range (1 ,len (bbox_list )):
        x1 ,y1 ,x2 ,y2 =bbox_list [i ]
        px1 ,py1 ,px2 ,py2 =bbox_list [i -1 ]
        c_i =((x1 +x2 )/2 /fw ,(y1 +y2 )/2 /fh )
        c_prev =((px1 +px2 )/2 /fw ,(py1 +py2 )/2 /fh )
        disps .append (np .linalg .norm (np .array (c_i )-np .array (c_prev )))
    return np .mean (disps )


monitoring_mode =False 

while True :
    ret ,frame =cap .read ()
    if not ret :break 

    fh ,fw =frame .shape [:2 ]
    rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )

    pose_res =pose .process (rgb )
    face_res =face_mesh .process (rgb )

    results =model .track (frame ,persist =True ,classes =[0 ],verbose =False )
    annotated =frame .copy ()
    now =time .monotonic ()


    if not monitoring_mode :
        current_candidates ={}
        for box in results [0 ].boxes :
            if box .id is None :continue 
            tid =int (box .id .item ())
            x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ].tolist ())

            if tid not in bbox_histories :
                bbox_histories [tid ]=collections .deque (maxlen =30 )

            bbox_histories [tid ].append ((x1 ,y1 ,x2 ,y2 ))

            area =bbox_area ((x1 ,y1 ,x2 ,y2 ))
            cx ,cy =(x1 +x2 )//2 ,(y1 +y2 )//2 


            if len (bbox_histories [tid ])>=15 :
                avg_disp =compute_centroid_disp (list (bbox_histories [tid ])[-15 :],fw ,fh )

                if avg_disp <CENTROID_DISP_THRESH :
                    if tid not in stationary_start_time :
                        stationary_start_time [tid ]=now 
                    elapsed =now -stationary_start_time [tid ]

                    if elapsed >=STATIONARY_SECONDS :
                        current_candidates [tid ]=(area ,(cx ,cy ),elapsed )
                else :

                    if tid in stationary_start_time :
                        del stationary_start_time [tid ]


            color =(0 ,255 ,0 )if tid in current_candidates else (0 ,0 ,255 )
            cv2 .rectangle (annotated ,(x1 ,y1 ),(x2 ,y2 ),color ,2 )
            status =f"ID:{tid } {area :.0f}"
            if tid in current_candidates :
                status +=f" [{current_candidates [tid ][2 ]:.1f}s]"
            cv2 .putText (annotated ,status ,(x1 ,y1 +20 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.6 ,(255 ,255 ,255 ),2 )


        if current_candidates :
            best_tid =max (current_candidates ,key =lambda tid :current_candidates [tid ][0 ])
            _ ,best_centroid ,_ =current_candidates [best_tid ]
            transaction_locations .append (best_centroid )


        if len (transaction_locations )>10 :
            from sklearn .cluster import DBSCAN 
            locs =np .array (transaction_locations )
            cl =DBSCAN (eps =50 ,min_samples =3 ).fit (locs )
            lbls =cl .labels_ 
            if len (np .unique (lbls [lbls >=0 ]))>0 :
                main =np .argmax (np .bincount (lbls [lbls >=0 ]))
                guichet_loc =np .mean (locs [lbls ==main ],axis =0 ).astype (int )


                avg_w =np .mean ([bbox_area (b )for hist in bbox_histories .values ()for b in hist ])**0.5 
                avg_h =avg_w *1.5 
                cx ,cy =guichet_loc 
                client_box =(int (cx -avg_w /2 *CLIENT_BOX_SCALE_W ),
                int (cy -avg_h /2 *CLIENT_BOX_SCALE_H ),
                int (cx +avg_w /2 *CLIENT_BOX_SCALE_W ),
                int (cy +avg_h /2 *CLIENT_BOX_SCALE_H ))

                monitoring_mode =True 
                print (f"[{time .strftime ('%H:%M:%S')}] Guichet found at {guichet_loc }. Switching to monitoring. Client box: {client_box }")

    else :

        active_tracks_in_box =[]
        for box in results [0 ].boxes :
            if box .id is None :continue 
            tid =int (box .id .item ())
            x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ].tolist ())

            overlap =box_overlap ((x1 ,y1 ,x2 ,y2 ),client_box )
            if overlap >OVERLAP_THRESH :
                active_tracks_in_box .append ((tid ,overlap ,(x1 ,y1 ,x2 ,y2 )))


                if tid ==current_client_tid :
                    if tid not in landmark_histories :
                        landmark_histories [tid ]=collections .deque (maxlen =HISTORY_LEN )
                        torso_activity_histories [tid ]=collections .deque (maxlen =5 )
                        prev_body_scores [tid ]=None 
                        prev_face_scores [tid ]=None 

                    has_pose =False 
                    face_landmarks =None 

                    if pose_res .pose_landmarks :
                        lm =pose_res .pose_landmarks .landmark 
                        pts =np .array ([[p .x ,p .y ]for p in lm ],dtype =np .float32 )
                        nx1 ,ny1 ,nx2 ,ny2 =x1 /fw ,y1 /fh ,x2 /fw ,y2 /fh 
                        if nx1 <=pts [0 ][0 ]<=nx2 and ny1 <=pts [0 ][1 ]<=ny2 :
                            has_pose =True 
                            landmark_histories [tid ].append (pts )


                            mp_drawing .draw_landmarks (
                            annotated ,pose_res .pose_landmarks ,mp_pose .POSE_CONNECTIONS ,
                            landmark_drawing_spec =mp_drawing .DrawingSpec (color =(255 ,100 ,0 ),thickness =0 ,circle_radius =0 ),
                            connection_drawing_spec =mp_drawing .DrawingSpec (color =(255 ,0 ,0 ),thickness =2 ))

                    if face_res .multi_face_landmarks :
                        for f_lm in face_res .multi_face_landmarks :
                            xs =np .array ([p .x for p in f_lm .landmark ])*fw 
                            ys =np .array ([p .y for p in f_lm .landmark ])*fh 
                            fx1 ,fy1 ,fx2 ,fy2 =int (xs .min ()),int (ys .min ()),int (xs .max ()),int (ys .max ())
                            if max (x1 ,fx1 )<min (x2 ,fx2 )and max (y1 ,fy1 )<min (y2 ,fy2 ):
                                face_landmarks =f_lm .landmark 

                                mp_drawing .draw_landmarks (
                                annotated ,f_lm ,mp_face .FACEMESH_TESSELATION ,
                                landmark_drawing_spec =mp_drawing .DrawingSpec (color =(255 ,100 ,0 ),thickness =0 ,circle_radius =0 ),
                                connection_drawing_spec =mp_drawing .DrawingSpec (color =(255 ,0 ,0 ),thickness =1 ))
                                break 


                    cv2 .rectangle (annotated ,(x1 ,y1 ),(x2 ,y2 ),(0 ,255 ,0 ),2 )
                    cv2 .putText (annotated ,f"ID:{tid }",(x1 ,y1 +20 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.6 ,(255 ,255 ,255 ),2 )


        if active_tracks_in_box :
            active_tracks_in_box .sort (key =lambda x :x [1 ],reverse =True )
            front_tid =active_tracks_in_box [0 ][0 ]

            if current_client_tid is None :
                current_client_tid =front_tid 
                client_start_time =now 
                client_scores =[]
                last_score_time =now 
                print (f"[{time .strftime ('%H:%M:%S')}] New client ID {front_tid } entered.")


            current_overlap =next ((ov for tid ,ov ,_ in active_tracks_in_box if tid ==current_client_tid ),0 )
            if current_overlap >0 :
                front_tid =current_client_tid 

            if current_client_tid ==front_tid :

                if now -last_score_time >=SCORE_INTERVAL :
                    last_score_time =now 


                    body_sc =0 
                    if current_client_tid in landmark_histories and len (landmark_histories [current_client_tid ])>=3 :
                        _ ,body_sc ,_ =compute_satisfaction_score (
                        landmark_histories [current_client_tid ],fw ,fh ,
                        prev_body_scores .get (current_client_tid ))
                        prev_body_scores [current_client_tid ]=body_sc 


                    face_sc =0 
                    if face_landmarks :
                        _ ,face_sc ,_ =compute_face_expression_score (
                        face_landmarks ,fw ,fh ,
                        prev_face_scores .get (current_client_tid ))
                        prev_face_scores [current_client_tid ]=face_sc 


                    scores =[s for s in (body_sc ,face_sc )if s >0 ]
                    agg_sc =sum (scores )/len (scores )if scores else 0 

                    client_scores .append ((body_sc ,face_sc ,agg_sc ))

            else :

                if current_client_tid is not None :
                    total_time =now -client_start_time 
                    if client_scores :
                        posture_scores =[s [0 ]for s in client_scores ]
                        face_scores =[s [1 ]for s in client_scores ]
                        agg_scores =[s [2 ]for s in client_scores ]
                        mean_posture =np .mean (posture_scores )
                        mean_face =np .mean (face_scores )
                        mean_total =np .mean (agg_scores )
                    else :
                        mean_posture =mean_face =mean_total =0 

                    print (f"[{time .strftime ('%H:%M:%S')}] Client ID {current_client_tid } left.")
                    print (f"Time spent: {total_time :.1f}s")
                    print (f"Scores list: {client_scores }")
                    print (f"Mean posture: {mean_posture :.1f}")
                    print (f"Mean face: {mean_face :.1f}")
                    print (f"Total mean score: {mean_total :.1f}")
                    client_logs .append ((current_client_tid ,total_time ,mean_posture ,mean_face ,mean_total ))

                current_client_tid =front_tid 
                client_start_time =now 
                client_scores =[]
                last_score_time =now 
                print (f"[{time .strftime ('%H:%M:%S')}] New client ID {front_tid } entered.")

        else :

            if current_client_tid is not None :
                total_time =now -client_start_time 
                if client_scores :
                    posture_scores =[s [0 ]for s in client_scores ]
                    face_scores =[s [1 ]for s in client_scores ]
                    agg_scores =[s [2 ]for s in client_scores ]
                    mean_posture =np .mean (posture_scores )
                    mean_face =np .mean (face_scores )
                    mean_total =np .mean (agg_scores )
                else :
                    mean_posture =mean_face =mean_total =0 

                print (f"[{time .strftime ('%H:%M:%S')}] Client ID {current_client_tid } left.")
                print (f"Time spent: {total_time :.1f}s")
                print (f"Scores list: {client_scores }")
                print (f"Mean posture: {mean_posture :.1f}")
                print (f"Mean face: {mean_face :.1f}")
                print (f"Total mean score: {mean_total :.1f}")
                client_logs .append ((current_client_tid ,total_time ,mean_posture ,mean_face ,mean_total ))

                current_client_tid =None 


        if client_box :
            cv2 .rectangle (annotated ,(client_box [0 ],client_box [1 ]),(client_box [2 ],client_box [3 ]),(255 ,0 ,0 ),2 )

    cv2 .imshow ("YOLO + Pose + Face (Satisfaction)",annotated )
    if cv2 .waitKey (1 )==27 :break 


print ("All client logs:")
for log in client_logs :
    print (log )

cap .release ()
cv2 .destroyAllWindows ()