import cv2 
import time 

for i in [0 ,1 ]:
    print (f"Testing camera index {i }")
    cap =cv2 .VideoCapture (i )
    if not cap .isOpened ():
        print ("Cannot open",i )
        continue 

    ret ,frame =cap .read ()
    if ret :
        cv2 .imshow (f"Cam {i }",frame )
        cv2 .waitKey (1500 )
    cap .release ()
    cv2 .destroyAllWindows ()
