import cv2
import HandDetection as hd
handDetect = hd.handDetection(detection_confident=0.8)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = handDetect.findHands(frame,draw_landmark=False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()