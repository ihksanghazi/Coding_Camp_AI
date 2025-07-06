import cv2  # Mengimpor pustaka OpenCV untuk pemrosesan gambar

# Membaca gambar dari file 'photo.jpg' dan menyimpannya ke variabel img
img = cv2.imread('photo.jpg')

# Menampilkan dimensi gambar: (tinggi, lebar, jumlah kanal warna)
print(img.shape)

# Mengaburkan gambar dengan Gaussian Blur untuk mengurangi noise
img = cv2.GaussianBlur(img, (7, 7), 0)

# Mendefinisikan warna dalam format BGR (Blue, Green, Red)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Menggambar garis hijau dari pojok kiri atas ke pojok kanan bawah gambar
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), GREEN, 3)

# Menggambar garis merah horizontal di posisi y=20 dari kiri ke kanan
cv2.line(img, (0, 20), (img.shape[1], 20), RED, 3)

# Menggambar persegi panjang biru dari titik kiri atas (300, 60) ke kanan bawah (450, 200)
cv2.rectangle(img, (300, 60), (450, 200), BLUE, 3)

# Menggambar lingkaran putih dengan pusat di (200, 250) dan radius 100
cv2.circle(img, (200, 250), 100, (255, 255, 255), 3)

# Menambahkan teks "Person" di atas kotak (titik mulai di (300, 50)) dengan font dan ukuran tertentu
cv2.putText(img, "Person", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)

# Menambahkan teks "Laptop" di tengah lingkaran dengan warna hitam
cv2.putText(img, "Laptop", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Menampilkan gambar yang telah dimodifikasi dalam jendela berjudul 'Image'
cv2.imshow('Image', img)

# Menunggu penekanan tombol dari pengguna (dalam milidetik, 0 berarti tunggu selamanya)
cv2.waitKey(0)

# Menutup semua jendela yang dibuat oleh OpenCV
cv2.destroyAllWindows()
