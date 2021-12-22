import cv2 
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands 
hands = mpHands.Hands(min_detection_confidence=0.75)

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
            mp_draw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        if len(results.multi_handedness)==2:
            cv2.putText(img,'Both Hands',(250,50),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),2)
        else:
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label=='Left':
                    cv2.putText(img,label+' Hand',(20,50),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),2)
                if label=='Right':
                    cv2.putText(img,label+' Hand',(460,50),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,255,0),2)
    
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break