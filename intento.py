import cv2
import numpy as np

img= cv2.imread("my_dress.jpg", cv2.IMREAD_GRAYSCALE)

cap=cv2.VideoCapture(1)

#captura
surfeo = cv2.SIFT_create()
kp_image, desc_image = surfeo.detectAndCompute(img, None)
img =cv2.drawKeypoints(img, kp_image, img)


while True:
    _, frame= cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_gris, desc_gris = surfeo.detectAndCompute(gris, None)

    cv2.imshow("imagen", img)
    cv2.imshow("Camara", frame)
    key = cv2.waitKey (1)
    if key == 27:
        break

cap.release ()
cv2.destroyAllWindows ()
