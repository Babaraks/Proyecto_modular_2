import cv2
import numpy as np

# List of image filenames to compare
image_filenames = ["my_dress.jpg", "ragnarok.jpg"]

# Function to perform the image comparison
def compare_images(img1, img2, surfeo, flans):
    kp_image, desc_image = surfeo.detectAndCompute(img1, None)
    img1 = cv2.drawKeypoints(img1, kp_image, img1)

    kp_gris, desc_gris = surfeo.detectAndCompute(img2, None)
    comparar = flans.knnMatch(desc_image, desc_gris, k=2)

    puntos_gd = []
    for m, n in comparar:
        if m.distance < 0.4 * n.distance:
            puntos_gd.append(m)

    if len(puntos_gd) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in puntos_gd]).reshape(-1, 1, 2)
        entreno_pts = np.float32([kp_gris[m.trainIdx].pt for m in puntos_gd]).reshape(-1, 1, 2)

        matrix, mascara = cv2.findHomography(query_pts, entreno_pts, cv2.RANSAC, 5.0)
        comparar_mascara = mascara.ravel().tolist()

        h, w, c = img1.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        try:
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("homalografia", homography)
            print("Image detected",image_filename)
        except cv2.error as e:
            print(f"Error: {e}")

    else:
        cv2.imshow("homalografia", img2)

# Initialize SIFT and Flann matcher
surfeo = cv2.SIFT_create()
index_parametro = dict(algorithm=0, trees=5)
search_parametro = dict()
flans = cv2.FlannBasedMatcher(index_parametro, search_parametro)

# Capture from camera
cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    for image_filename in image_filenames:
        reference_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        compare_images(reference_image, gray_frame, surfeo, flans)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
