import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxc = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxc.xmin *w), int(bboxc.ymin *h),int(bboxc.width *w),int(bboxc.height *h)
                bboxs.append([id, bbox, detection.score])
                img = self.fancyDraw(img, bbox)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0) , 2)
        return img, bboxs
        
    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255,0,0), rt)
        # Top left x,y
        cv2.line(img, (x,y), (x+l, y), (255,0,0), t)
        cv2.line(img, (x,y), (x, y+l), (255,0,0), t)
        # Top rigth x,y
        cv2.line(img, (x1,y), (x1-l, y), (255,0,0), t)
        cv2.line(img, (x1,y), (x1, y+l), (255,0,0), t)
        # Bottom left x,y
        cv2.line(img, (x,y1), (x+l, y1), (255,0,0), t)
        cv2.line(img, (x,y1), (x, y1-l), (255,0,0), t)
        # Bottom right x,y
        cv2.line(img, (x1,y1), (x1-l, y1), (255,0,0), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (255,0,0), t)

        return img
        

def main():
    cap = cv2.VideoCapture("video1.mp4")
    pTime = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()
        if success == True:
            img, bboxs = detector.findFaces(img)
                
            cTime = time.time()
            fps = 1/ (cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0) , 2)
                
            cv2.imshow("FaceDetection",img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()