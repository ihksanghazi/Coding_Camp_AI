# Import modul FaceMesh dan utilitas gambar dari MediaPipe (meskipun sebaiknya pakai import mediapipe as mp)
from mediapipe.python.solutions import face_mesh, drawing_utils  
import numpy as np
import cv2
from tensorflow.compat.v1 import ConfigProto  # Konfigurasi GPU agar tidak langsung menghabiskan memori
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model  # Untuk memuat model deep learning Keras
from tensorflow.keras.preprocessing import image as img_keras  # Untuk memproses gambar jadi input model
from collections import deque  # Untuk menyimpan hasil prediksi terakhir (agar lebih stabil)

# Memuat model dari file .hdf5 tanpa compile ulang (karena hanya untuk prediksi)
model = load_model('models/_trained.hdf5', compile=False)

# Inisialisasi deque Q dengan panjang maksimum 10 untuk menyimpan prediksi emosi sebelumnya
Q = deque(maxlen=10)

# Daftar label emosi yang dikenali oleh model
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral")

# Inisialisasi kamera webcam
cap = cv2.VideoCapture(0)

# Inisialisasi VideoWriter (nanti akan dibuat di dalam loop)
writer = None

# Konfigurasi GPU agar alokasi memori bertahap (tidak langsung penuh)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Duplikasi pemanggilan load_model (sebaiknya dihapus salah satunya)
model = load_model("models/_trained.hdf5", compile=False)

# Inisialisasi modul FaceMesh dan utilitas menggambar landmark
mp_face_mesh = face_mesh
mp_drawing = drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)  # Pengaturan tampilan titik-titik wajah

# Menggunakan FaceMesh sebagai context manager
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,  # Kepercayaan minimum saat deteksi wajah
    min_tracking_confidence=0.5    # Kepercayaan minimum saat pelacakan wajah
) as face_mesh:

    while True:
        check, frame = cap.read()  # Membaca frame dari kamera
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # Penjelasan A: Flip horizontal dan ubah ke RGB
        results = face_mesh.process(frame)  # Penjelasan B: Proses wajah menggunakan FaceMesh
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Ubah kembali ke format BGR untuk ditampilkan dengan OpenCV

        # Default wajah kosong agar tidak error saat tidak ada wajah
        detected_face = np.zeros((64, 64), dtype=np.uint8)

        # Jika berhasil mendeteksi wajah
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:  # Penjelasan C: Loop semua wajah yang terdeteksi
                mp_drawing.draw_landmarks(  # Gambar landmark di wajah
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                h, w, c = frame.shape
                cx_min, cy_min = w, h
                cx_max, cy_max = 0, 0

                # Mencari bounding box dari wajah berdasarkan titik landmark
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx < cx_min: cx_min = cx
                    if cy < cy_min: cy_min = cy
                    if cx > cx_max: cx_max = cx
                    if cy > cy_max: cy_max = cy

                # Pastikan bounding box valid (tidak keluar dari frame)
                cx_min = max(0, cx_min)
                cy_min = max(0, cy_min)
                cx_max = min(w, cx_max)
                cy_max = min(h, cy_max)

                if cx_max > cx_min and cy_max > cy_min:
                    # Ambil wajah dari frame berdasarkan bounding box
                    face_crop = frame[cy_min:cy_max, cx_min:cx_max]
                    if face_crop.size > 0:
                        # Penjelasan b: Ubah ke grayscale
                        detected_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        # Penjelasan c: Resize ke 64x64 agar sesuai input model
                        detected_face = cv2.resize(detected_face, (64, 64))
                        # Ubah ke format array dan ubah dimensi agar cocok untuk model
                        frame_pixels = img_keras.img_to_array(detected_face)
                        frame_pixels = np.expand_dims(frame_pixels, axis=0)
                        frame_pixels /= 255  # Normalisasi pixel dari 0-255 menjadi 0-1
                        # Prediksi emosi dari wajah
                        emotion = model.predict(frame_pixels)[0]
                        Q.append(emotion)  # Tambahkan ke antrian Q
                        # Hitung rata-rata dari prediksi emosi sebelumnya
                        results = np.array(Q).mean(axis=0)
                        i = np.argmax(results)  # Ambil indeks emosi tertinggi
                        label = emotions[i]  # Ambil label emosi
                        # Tampilkan label emosi di frame
                        cv2.putText(frame, label, (cx_min, cy_min),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Gambar bounding box di wajah
                        cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)

        # Inisialisasi VideoWriter hanya sekali (saat pertama kali loop)
        if writer is None:
            h, w, c = frame.shape
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # Codec DIVX (hati-hati, tidak semua OS mendukung)
            writer = cv2.VideoWriter('output.avi', fourcc, 20, (w, h), True)

        writer.write(frame)  # Simpan frame ke video output

        # Tampilkan frame utama dan wajah yang terdeteksi
        cv2.imshow('frame', frame)
        cv2.imshow('detected_face', detected_face)

        # Tekan tombol 'q' untuk keluar dari loop
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# Bebaskan resource setelah selesai
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
