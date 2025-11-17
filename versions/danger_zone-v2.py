import time ,math ,collections ,itertools 
import numpy as np 
import cv2 
from ultralytics import YOLO 
import mediapipe as mp 

model =YOLO ("yolov8x.pt")
cap =cv2 .VideoCapture ("./testcases/swey.mp4")

mp_pose =mp .solutions .pose 
pose =mp_pose .Pose (model_complexity =1 ,enable_segmentation =False )

mp_face_det =mp .solutions .face_detection 
face_det =mp_face_det .FaceDetection (model_selection =1 ,min_detection_confidence =0.5 )

mp_face_mesh =mp .solutions .face_mesh 
face_mesh =mp_face_mesh .FaceMesh (static_image_mode =False ,max_num_faces =1 ,refine_landmarks =False )

HISTORY_LEN =6 

last_print =0.0 
PRINT_INTERVAL =1.0 

CONFIRM_SECONDS =4.0 
IOU_ASSIGN_THRESH =0.3 
MAX_MISSING_SECONDS =1.0 
FACING_ANGLE_DEG =60.0 
ACTIVITY_THRESHOLD =0.002 

id_counter =itertools .count (1 )
tracks ={}

def iou (a ,b ):
    x1 =max (a [0 ],b [0 ]);y1 =max (a [1 ],b [1 ])
    x2 =min (a [2 ],b [2 ]);y2 =min (a [3 ],b [3 ])
    w =max (0 ,x2 -x1 );h =max (0 ,y2 -y1 )
    inter =w *h 
    area_a =(a [2 ]-a [0 ])*(a [3 ]-a [1 ])+1e-6 
    area_b =(b [2 ]-b [0 ])*(b [3 ]-b [1 ])+1e-6 
    return inter /(area_a +area_b -inter +1e-6 )

def _angle_between_deg (v1 ,v2 ):
    n1 =np .linalg .norm (v1 );n2 =np .linalg .norm (v2 )
    if n1 ==0 or n2 ==0 :return 180.0 
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

def is_facing_camera (pose_lm ,face_det_bbox ,person_bbox ,fw ,fh ):
    camera_dir =np .array ([0.0 ,-1.0 ])
    if pose_lm is not None :
        try :
            lm =pose_lm .landmark 
            shoulders =np .array ([(lm [11 ].x +lm [12 ].x )/2.0 ,(lm [11 ].y +lm [12 ].y )/2.0 ])
            nose =np .array ([lm [0 ].x ,lm [0 ].y ])
            head_vec =nose -shoulders 
            ang =_angle_between_deg (head_vec ,camera_dir )
            return ang <=FACING_ANGLE_DEG 
        except Exception :
            pass 
    if face_det_bbox is not None :
        fx1 ,fy1 ,fx2 ,fy2 =face_det_bbox 
        px =(fx1 +fx2 )/2.0 ;py =(fy1 +fy2 )/2.0 
        bx1 ,by1 ,bx2 ,by2 =person_bbox 
        bx_c =(bx1 +bx2 )/2.0 ;by_c =(by1 +by2 )/2.0 
        dist =math .hypot (px -bx_c ,py -by_c )
        diag =max (bx2 -bx1 ,by2 -by1 )
        return dist <=diag *0.35 
    return False 

while True :
    ret ,frame =cap .read ()
    if not ret :
        break 
    tnow =time .monotonic ()
    fh ,fw =frame .shape [:2 ]
    rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )

    pose_res =pose .process (rgb )
    face_det_res =face_det .process (rgb )
    face_mesh_res =face_mesh .process (rgb )

    results =model .predict (frame ,verbose =False )
    person_dets =[]
    for box in results [0 ].boxes :
        cls =int (box .cls [0 ])
        if cls !=0 :continue 
        x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ].tolist ())
        conf =float (box .conf [0 ])
        person_dets .append (((x1 ,y1 ,x2 ,y2 ),conf ))

    assigned =set ()
    new_tracks ={}
    for det_bbox ,conf in person_dets :
        best_id =None ;best_iou =0.0 
        for tid ,tr in tracks .items ():
            i =iou (det_bbox ,tr ['bbox'])
            if i >best_iou and i >=IOU_ASSIGN_THRESH :
                best_iou =i ;best_id =tid 
        if best_id is not None :
            tr =tracks [best_id ]
            tr ['bbox']=det_bbox 
            tr ['last_seen']=tnow 
            tr ['seen_frames']+=1 
            cx =(det_bbox [0 ]+det_bbox [2 ])/2.0 ;cy =(det_bbox [1 ]+det_bbox [3 ])/2.0 
            tr ['centroid_history'].append ((cx ,cy ))
            new_tracks [best_id ]=tr 
            assigned .add (best_id )
        else :
            nid =next (id_counter )
            tr ={
            'bbox':det_bbox ,
            'first_seen':tnow ,
            'last_seen':tnow ,
            'seen_frames':1 ,
            'face_hits':0 ,
            'centroid_history':collections .deque (maxlen =HISTORY_LEN ),
            'pose_history':collections .deque (maxlen =HISTORY_LEN ),
            'confirmed':False ,
            'stationary_since':None ,
            'face_det_bbox':None 
            }
            cx =(det_bbox [0 ]+det_bbox [2 ])/2.0 ;cy =(det_bbox [1 ]+det_bbox [3 ])/2.0 
            tr ['centroid_history'].append ((cx ,cy ))
            new_tracks [nid ]=tr 

    for tid ,tr in list (tracks .items ()):
        if tid not in assigned and (tnow -tr ['last_seen'])<=MAX_MISSING_SECONDS :
            new_tracks [tid ]=tr 

    tracks =new_tracks 

    face_bboxes =[]
    if face_det_res and face_det_res .detections :
        for det in face_det_res .detections :
            rb =det .location_data .relative_bounding_box 
            x1 =int (rb .xmin *fw );y1 =int (rb .ymin *fh )
            w =int (rb .width *fw );h =int (rb .height *fh )
            face_bboxes .append (((x1 ,y1 ,x1 +w ,y1 +h ),det ))

    for tid ,tr in tracks .items ():
        bx =tr ['bbox']
        cx =(bx [0 ]+bx [2 ])/2 ;cy =(bx [1 ]+bx [3 ])/2 
        face_found =None 
        for fb ,fdet in face_bboxes :
            if fb [0 ]<=cx <=fb [2 ]and fb [1 ]<=cy <=fb [3 ]:
                tr ['face_hits']+=1 
                tr ['face_det_bbox']=fb 
                face_found =fdet 
                break 

        pose_in_box =None 
        if pose_res .pose_landmarks :
            for p in pose_res .pose_landmarks .landmark :
                px =int (p .x *fw );py =int (p .y *fh )
                if bx [0 ]<=px <=bx [2 ]and bx [1 ]<=py <=bx [3 ]:
                    pose_in_box =pose_res 
                    break 
        tr ['pose_history'].append (pose_in_box )

        present_time =tnow -tr ['first_seen']
        continuous =(tnow -tr ['last_seen'])<=MAX_MISSING_SECONDS and present_time >=CONFIRM_SECONDS 


        moving =True 
        if len (tr ['centroid_history'])>=3 :
            d0 =np .array (tr ['centroid_history'][-1 ])
            d1 =np .array (tr ['centroid_history'][-2 ])
            d2 =np .array (tr ['centroid_history'][-3 ])
            disp1 =np .linalg .norm (d0 -d1 )
            disp2 =np .linalg .norm (d1 -d2 )
            activity =float ((disp1 +disp2 )/2.0 )/max (fw ,fh )
            if activity <ACTIVITY_THRESHOLD :
                if tr ['stationary_since']is None :
                    tr ['stationary_since']=tnow 
                elif tnow -tr ['stationary_since']>=CONFIRM_SECONDS :
                    stationary_ok =True 
                else :
                    stationary_ok =False 
            else :
                tr ['stationary_since']=None 
                stationary_ok =False 
        else :
            stationary_ok =False 

        facing =is_facing_camera (pose_in_box ,tr .get ('face_det_bbox'),bx ,fw ,fh )
        tr ['confirmed']=continuous and facing and stationary_ok 

    confirmed_tracks =[(tid ,tr )for tid ,tr in tracks .items ()if tr .get ('confirmed')]
    chosen =None 
    if confirmed_tracks :
        chosen =max (confirmed_tracks ,key =lambda x :(x [1 ]['bbox'][2 ]-x [1 ]['bbox'][0 ])*(x [1 ]['bbox'][3 ]-x [1 ]['bbox'][1 ]))
        tid_chosen ,tr_chosen =chosen 
        x1 ,y1 ,x2 ,y2 =tr_chosen ['bbox']
        cv2 .rectangle (frame ,(x1 ,y1 ),(x2 ,y2 ),(0 ,255 ,0 ),2 )

        if pose_res .pose_landmarks :
            lm =pose_res .pose_landmarks .landmark 
            pts =np .array ([[p .x ,p .y ]for p in lm ],dtype =np .float32 )



            local_hist =collections .deque (maxlen =HISTORY_LEN )
            local_hist .append (pts )

            body_label ,body_score ,body_diag =compute_satisfaction_score (local_hist ,fw ,fh ,prev_score =None ,alpha_smooth =0.6 )
            if body_label is None :
                body_label ,body_score ="no_data",0 
        else :
            body_label ,body_score ="no_data",0 

        if face_mesh_res .multi_face_landmarks and len (face_mesh_res .multi_face_landmarks )>0 :
            f_lm =face_mesh_res .multi_face_landmarks [0 ].landmark 
            face_label ,face_score ,face_diag =compute_face_expression_score (f_lm ,fw ,fh ,prev_score =None ,alpha_smooth =0.6 )
        else :
            face_label ,face_score ="no_face",0 

        scores =[]
        if isinstance (body_score ,(int ,float )):scores .append (float (body_score ))
        if isinstance (face_score ,(int ,float )):scores .append (float (face_score ))
        avg_score =int (sum (scores )/len (scores ))if scores else 0 
        avg_label ="no_data"
        if avg_score >=70 :avg_label ="satisfied"
        elif avg_score >=45 :avg_label ="neutral"
        else :avg_label ="dissatisfied"

        if tnow -last_print >=PRINT_INTERVAL :
            print (f"[{time .strftime ('%H:%M:%S')}] CONFIRMED id={tid_chosen } BODY:{body_label } ({body_score }) FACE:{face_label } ({face_score }) AVG:{avg_label } ({avg_score })")
            last_print =tnow 

        cv2 .putText (frame ,f"CONFIRMED id={tid_chosen }",(12 ,26 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.7 ,(200 ,200 ,200 ),2 )
        cv2 .putText (frame ,f"AVG:{avg_label } {avg_score }",(12 ,56 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.7 ,(0 ,255 ,200 ),2 )
    else :
        if tnow -last_print >=PRINT_INTERVAL :
            print (f"[{time .strftime ('%H:%M:%S')}] No confirmed client yet")
            last_print =tnow 

    for tid ,tr in tracks .items ():
        bx =tr ['bbox']
        cv2 .rectangle (frame ,(bx [0 ],bx [1 ]),(bx [2 ],bx [3 ]),(120 ,120 ,120 ),1 )

    cv2 .imshow ("Confirmed-client detection (Mediapipe face det)",frame )
    if cv2 .waitKey (1 )==27 :
        break 

cap .release ()
cv2 .destroyAllWindows ()
