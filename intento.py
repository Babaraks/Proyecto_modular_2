import cv2
import numpy as np

img= cv2.imread("my_dress.jpg", cv2.IMREAD_GRAYSCALE)#imagen comparativa

cap=cv2.VideoCapture(1)

#captura
surfeo = cv2.SIFT_create()
kp_image, desc_image = surfeo.detectAndCompute(img, None)
img =cv2.drawKeypoints(img, kp_image, img)

#comparador
index_parametro=dict(algorithm=0, trees=5)
search_parametro=dict()
flans=cv2.FlannBasedMatcher(index_parametro,search_parametro)

while True:
    _, frame= cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_gris, desc_gris = surfeo.detectAndCompute(gris, None)
    comparar= flans.knnMatch(desc_image, desc_gris, k=2)

    puntos_gd=[]
    for m,n in comparar:
        if m.distance < 0.55*n.distance:
            puntos_gd.append(m)

    #homolografia
    if len(puntos_gd)>10:
        query_pts=np.float32([kp_image[m.queryIdx].pt for m in puntos_gd]).reshape(-1, 1, 2)
        entreno_pts=np.float32([kp_gris[m.trainIdx].pt for m in puntos_gd]).reshape(-1, 1, 2)

        matrix, mascara= cv2.findHomography(query_pts, entreno_pts,cv2.RANSAC,5.0)
        comparar_mascara=mascara.ravel().tolist()
        
        # transforma prespectiva
        h, w, c = img.shape
        pts=np.float32 ([[0, 0], [0, h], [w, h], [w, 0]]) .reshape (-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines (frame, [np.int32 (dst) ], True, (255, 0, 0), 3)

        cv2.imshow("homalografia", homography)
        print("libro dectectado")
    else:
        cv2.imshow("homalografia", gris)

    key = cv2.waitKey (1)
    if key == 27:
        break

cap.release ()
cv2.destroyAllWindows ()
