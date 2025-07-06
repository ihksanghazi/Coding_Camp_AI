import cv2
from mediapipe.python.solutions import hands, drawing_utils

class handDetection:
    def __init__(self, static_mode:bool = False, maxhands:int = 2, detection_confident:float = 0.5, tracking_confident:float = 0.5):
        self.static_mode = static_mode
        self.maxhands = maxhands
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        self.mphands = hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.maxhands,
            min_detection_confidence=self.detection_confident,
            min_tracking_confidence=self.tracking_confident)
        self.mpdraw = drawing_utils
    
    def findHands(self, frame, draw_landmark:bool=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
    
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)

        return frame