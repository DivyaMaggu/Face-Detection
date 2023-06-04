import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("1.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if success==True:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        
        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                print(detection.location_data.relative_bounding_box)
                bboxc = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxc.xmin *w), int(bboxc.ymin *h),\
                       int(bboxc.width *w),int(bboxc.height *h)
                cv2.rectangle(img, bbox, (255,0,0), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0) , 2)

        cTime = time.time()
        fps = 1/ (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0) , 2)
        
        
        
        cv2.imshow("FaceDetection",img)
        cv2.waitKey(1)
