from mediapipe.python.solutions import face_mesh, drawing_utils 
import numpy as np
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_keras

from collections import deque

model = load_model('models/_trained.hdf5', compile=False)
Q = deque(maxlen=10) #Inisialisasi deque dengan nama Q, dengan maximum 10 element
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral") #Inisialisasi tuple macam-macam emosi apa saja yang bisa di classification model !

# 
# Inisialisasi kamera
cap = cv2.VideoCapture(0)
writer = None
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model = load_model("models/_trained.hdf5", compile=False) 

# Inisialisasi modul FaceMesh dan drawing
mp_face_mesh = face_mesh
mp_drawing = drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)

# Gunakan FaceMesh sebagai konteks
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        check, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # Penjelasan A
        results = face_mesh.process(frame)  # Penjelasan B
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detected_face = np.zeros((64, 64), dtype=np.uint8)  # default agar tidak error saat wajah tidak terdeteksi

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:  # Penjelasan C
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                h, w, c = frame.shape
                cx_min, cy_min = w, h
                cx_max, cy_max = 0, 0

                # Loop semua landmark untuk mencari batas minimum dan maksimum dari wajah
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx < cx_min: cx_min = cx
                    if cy < cy_min: cy_min = cy
                    if cx > cx_max: cx_max = cx
                    if cy > cy_max: cy_max = cy

                # Pastikan bounding box valid (tidak negatif atau terbalik)
                cx_min = max(0, cx_min)
                cy_min = max(0, cy_min)
                cx_max = min(w, cx_max)
                cy_max = min(h, cy_max)

                if cx_max > cx_min and cy_max > cy_min:
                    face_crop = frame[cy_min:cy_max, cx_min:cx_max]
                    if face_crop.size > 0:
                        detected_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)  # Penjelasan b
                        detected_face = cv2.resize(detected_face, (64, 64))  # Penjelasan c
                        frame_pixels = img_keras.img_to_array(detected_face)
                        frame_pixels = np.expand_dims(frame_pixels, axis=0)
                        frame_pixels /= 255 #Scale image dalam numpy array memiliki range 0-255, dqilakukan normalisasi menjadi 0-1 karena lebih cocok untuk model yang kita gunakan
                        emotion = model.predict(frame_pixels)[0]
                        Q.append(emotion) #Memasukan hasil prediksi ke dalam Deque Q yang sebelumnya sudah kita inisialisasi
                        # print(Q)
                        results = np.array(Q).mean(axis=0)
                        i = np.argmax(results)
                        label = emotions[i]
                        # print(label)
                        cv2.putText(frame, label, (cx_min, cy_min),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)
        
        if writer is None:
            h, w, c = frame.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            writer = cv2.VideoWriter('output.avi', fourcc, 20, 
            (w, h), True)
            
        # Tampilkan frame utama dan wajah yang terdeteksi
        cv2.imshow('frame', frame)
        cv2.imshow('detected_face', detected_face)

        # Tekan tombol 'q' untuk keluar
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
