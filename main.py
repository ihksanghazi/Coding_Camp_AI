import cv2  # Mengimpor pustaka OpenCV untuk pengolahan gambar dan video
import HandDetection as hd  # Mengimpor modul deteksi tangan yang disingkat sebagai hd
import time  # Mengimpor modul time untuk menghitung frame rate (fps)

# Membuat objek deteksi tangan dengan confidence minimal 0.8
handDetect = hd.handDetection(detection_confident=0.8)

# Membuka kamera (0 = kamera default seperti webcam laptop)
cap = cv2.VideoCapture(0)

# Daftar indeks titik teratas jari-jari (thumb, index, middle, ring, pinky)
top_idx = [4, 8, 12, 16, 20]

# Variabel untuk menghitung FPS
previous_time = 0
current_time = 0

# Perulangan utama program
while True:
    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    # Membalik frame secara horizontal supaya tampak seperti pantulan cermin
    frame = cv2.flip(frame, 1)

    # Mendeteksi tangan dan menggambar landmark tangan di frame
    frame = handDetect.findHands(frame)

    # Mendapatkan daftar landmark tangan dalam format [id, x, y]
    lmlist = handDetect.getHandLocation(frame, draw=True)

    # Jika tangan terdeteksi (lmlist tidak kosong)
    if len(lmlist) != 0:
        fingers = []  # Menyimpan status masing-masing jari (1 = terbuka, 0 = tertutup)

        # Cek jempol (thumb): posisi X jempol harus lebih kecil dari X titik di sampingnya
        if lmlist[top_idx[0]][1] < lmlist[top_idx[0] - 1][1]:
            fingers.append(1)  # Jempol terbuka
        else:
            fingers.append(0)  # Jempol tertutup

        # Cek 4 jari lainnya (index hingga pinky)
        for idx in range(1, 5):
            # Jika ujung jari (y) lebih tinggi dari sendi bawahnya, maka jari dianggap terbuka
            if lmlist[top_idx[idx]][2] < lmlist[top_idx[idx] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Cetak status jari ke konsol (misalnya: [0, 1, 1, 0, 0])
        print(fingers)

        # Hitung berapa banyak jari yang terbuka
        openFingers = fingers.count(1)

        # Gambar kotak putih untuk menampilkan jumlah jari yang terbuka
        cv2.rectangle(frame, (20, 20), (200, 200), (255, 255, 255), cv2.FILLED)

        # Tampilkan angka jumlah jari yang terbuka di layar
        cv2.putText(frame, str(int(openFingers)), (50, 170), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        # Jika hanya jari telunjuk dan jari tengah yang terbuka, tampilkan teks "Peace"
        if fingers[1] and fingers[2] and fingers.count(1) == 2:
            cv2.putText(frame, "Peace", (50, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Penjelasan A

        # Jika hanya jempol yang terbuka, tampilkan teks "Nice"
        elif fingers[0] and fingers.count(1) == 1:
            cv2.putText(frame, "Nice", (50, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Penjelasan B

    # Hitung FPS berdasarkan waktu antar frame
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Tampilkan nilai FPS di layar
    cv2.putText(frame, "frame rate: " + str(int(fps)), (350, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Tampilkan frame akhir ke jendela
    cv2.imshow('frame', frame)

    # Tekan tombol 'q' untuk keluar dari loop dan menutup program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepas akses kamera
cap.release()

# Menutup semua jendela OpenCV
cv2.destroyAllWindows()
