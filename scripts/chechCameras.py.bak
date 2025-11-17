import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.read()[0]:
        print("Camera index working:", i)
    cap.release()