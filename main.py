import cv2

img = cv2.imread('photo.jpg')
print(img.shape)
img = cv2.GaussianBlur(img, (7, 7), 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),GREEN,3)
cv2.line(img, (0, 20), (img.shape[1], 20), RED, 3)

cv2.rectangle(img,(300,60),(450,200),BLUE,3)

cv2.circle(img,(200,250),100,(255,255,255),3)

cv2.putText(img, "Person",(300,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,150,0),2)

cv2.putText(img, "Laptop",(200,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()