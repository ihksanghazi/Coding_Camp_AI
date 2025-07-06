import cv2  # Mengimpor pustaka OpenCV untuk pengolahan gambar dan video
import HandDetection as hd  # Mengimpor modul deteksi tangan yang dinamai HandDetection dan disingkat sebagai hd

# Membuat objek deteksi tangan dengan tingkat kepercayaan (confidence) minimal 0.8
handDetect = hd.handDetection(detection_confident=0.8)

# Mengakses kamera (0 berarti kamera default laptop/PC)
cap = cv2.VideoCapture(0)

# Perulangan utama program untuk membaca dan memproses setiap frame dari kamera
while True:
    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    # Membalik gambar secara horizontal agar seperti cermin (mirror view)
    frame = cv2.flip(frame, 1)

    # Mendeteksi tangan pada frame
    frame = handDetect.findHands(frame)

    # Menampilkan frame hasil deteksi ke layar
    cv2.imshow('frame', frame)

    # Jika tombol 'q' ditekan, keluar dari perulangan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Memberhentikan penggunaan kamera
cap.release()

# Menutup semua jendela tampilan OpenCV
cv2.destroyAllWindows()
