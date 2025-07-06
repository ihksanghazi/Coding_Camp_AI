import cv2  # Mengimpor pustaka OpenCV untuk pengolahan gambar dan video
import HandDetection as hd  # Mengimpor modul deteksi tangan yang dinamai HandDetection dan disingkat sebagai hd
import time

# Membuat objek deteksi tangan dengan tingkat kepercayaan (confidence) minimal 0.8
handDetect = hd.handDetection(detection_confident=0.8)

# Mengakses kamera (0 berarti kamera default laptop/PC)
cap = cv2.VideoCapture(0)
top_idx = [4,8,12,16,20]
previous_time = 0
current_time = 0

# Perulangan utama program untuk membaca dan memproses setiap frame dari kamera
while True:
    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    # Membalik gambar secara horizontal agar seperti cermin (mirror view)
    frame = cv2.flip(frame, 1)

    # Mendeteksi tangan pada frame
    frame = handDetect.findHands(frame)

    lmlist = handDetect.getHandLocation(frame,draw=True)

    if len(lmlist) != 0:
        fingers = [] #Untuk menyimpan informasi apakah jari dibuka atau tidak 
        if lmlist[top_idx[0]][1] < lmlist[top_idx[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for idx in range(1,5):
            if lmlist[top_idx[idx]][2] < lmlist[top_idx[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        openFingers = fingers.count(1)
        cv2.rectangle(frame, (20,20),(200,200),(255,255,255), cv2.FILLED)
        cv2.putText(frame, str(int(openFingers)), (50,170), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
        if fingers[1] and fingers[2] and fingers.count(1) == 2:
            cv2.putText(frame, "Peace", (50,270),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3) #Penjelasan A
        elif fingers[0] and fingers.count(1) == 1:	
            cv2.putText(frame, "Nice", (50,270), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3) #Penjelasan B

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, "frame rate: "+str(int(fps)),(350,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    # Menampilkan frame hasil deteksi ke layar
    cv2.imshow('frame', frame)

    # Jika tombol 'q' ditekan, keluar dari perulangan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Memberhentikan penggunaan kamera
cap.release()

# Menutup semua jendela tampilan OpenCV
cv2.destroyAllWindows()
