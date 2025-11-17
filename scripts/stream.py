import cv2 
from ultralytics import YOLO 

model =YOLO ('yolov8n.pt')
cap =cv2 .VideoCapture (1 ,cv2 .CAP_DSHOW )

if not cap .isOpened ():
    print ("Cannot open camera")
    exit ()

window_name ="iPhone"
cv2 .namedWindow (window_name ,cv2 .WINDOW_NORMAL )
cv2 .resizeWindow (window_name ,1280 ,720 )

while True :
    ret ,frame =cap .read ()
    if not ret :
        print ("Failed to grab frame")
        break 

    results =model .predict (frame ,verbose =False )
    annotated =results [0 ].plot ()

    cv2 .imshow (window_name ,annotated )

    if cv2 .waitKey (1 )&0xFF ==27 :
        break 

cap .release ()
cv2 .destroyAllWindows ()