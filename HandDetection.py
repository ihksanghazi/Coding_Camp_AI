# Mengimpor pustaka OpenCV untuk pengolahan gambar
import cv2

# Mengimpor modul hands dan drawing_utils dari mediapipe untuk deteksi dan visualisasi tangan
from mediapipe.python.solutions import hands, drawing_utils

# Membuat class bernama handDetection untuk mendeteksi tangan menggunakan MediaPipe
class handDetection:
    # Konstruktor untuk inisialisasi objek deteksi tangan
    def __init__(self, static_mode:bool = False, maxhands:int = 2, detection_confident:float = 0.5, tracking_confident:float = 0.5):
        # Menyimpan parameter mode statis (True jika gambar tunggal, False untuk video/live)
        self.static_mode = static_mode

        # Jumlah maksimum tangan yang akan dideteksi
        self.maxhands = maxhands

        # Tingkat kepercayaan minimum untuk mendeteksi tangan pertama kali
        self.detection_confident = detection_confident

        # Tingkat kepercayaan minimum untuk melacak tangan setelah terdeteksi
        self.tracking_confident = tracking_confident

        # Menyimpan referensi ke modul hands dari mediapipe
        self.mphands = hands

        # Membuat objek `Hands` dari mediapipe dengan parameter yang telah ditentukan
        self.hands = self.mphands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.maxhands,
            min_detection_confidence=self.detection_confident,
            min_tracking_confidence=self.tracking_confident)

        # Menyimpan referensi ke modul untuk menggambar landmark tangan di atas gambar
        self.mpdraw = drawing_utils

    # Fungsi untuk mendeteksi tangan dari frame (gambar) yang diberikan
    def findHands(self, frame, draw_landmark:bool=True):
        # Mengubah format warna gambar dari BGR (OpenCV) ke RGB (MediaPipe)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Memproses gambar untuk mendeteksi tangan
        self.results = self.hands.process(img)

        # Jika terdapat hasil deteksi tangan (lebih dari 0)
        if self.results.multi_hand_landmarks:
            # Untuk setiap tangan yang terdeteksi
            for handlms in self.results.multi_hand_landmarks:
                # Jika parameter draw_landmark = True, maka gambar landmark dan koneksi tangan
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)

        # Mengembalikan frame yang sudah diproses (dengan atau tanpa landmark tergantung parameter)
        return frame

    def getHandLocation(self, frame, handNo:int = 0, draw: bool = True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for idx,lm in enumerate(myHand.landmark):
                    
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append([idx,cx,cy])
                    
                if draw:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return lmList 