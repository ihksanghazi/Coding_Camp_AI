import cv2  # Mengimpor pustaka OpenCV untuk pengolahan gambar dan video

# Memuat model deteksi wajah berbasis Haar Cascade (pra-latih) dari file XML
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Mengakses kamera (0 berarti kamera default/laptop)
cap = cv2.VideoCapture(0)

# Memulai perulangan untuk membaca frame secara terus-menerus dari kamera
while True:
    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    # Mengubah frame menjadi grayscale agar deteksi wajah lebih cepat dan akurat
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah pada gambar grayscale
    # Parameter 1.1 adalah faktor skala, dan 5 adalah jumlah neighbor untuk validasi deteksi
    face = face_cascade.detectMultiScale(gray_image, 1.1, 5)

    # Melakukan perulangan untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in face:
        # Menggambar kotak hijau di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Menambahkan teks "Person" di atas kotak wajah
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)

    # Menampilkan frame hasil deteksi dengan kotak dan teks
    cv2.imshow('frame', frame)

    # Menunggu tombol keyboard ditekan. Jika 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera
cap.release()

# Menutup semua jendela OpenCV
cv2.destroyAllWindows()
