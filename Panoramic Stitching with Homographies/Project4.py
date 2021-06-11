import cv2
import numpy as np

img1 = cv2.imread('./set1/1.jpg')
img2 = cv2.imread('./set1/2.jpg')
img3 = cv2.imread('./set1/3.jpg')


# 2  Intro to homographies

t1 = np.array([[np.cos(10 * np.pi/180), -np.sin(10 * np.pi/180),0], [np.sin(10 * np.pi/180), np.cos(10 * np.pi/180),0], [0, 0,1]])
t2 = np.float32([[1, 0, 100], [0, 1,0], [0, 0,1]])
t3 = np.array([[1/2, 0, 0], [0, 1/2,0], [0, 0, 1]])


new_img1 = cv2.warpPerspective(img1,t1,(1000,800))
new_img2 = cv2.warpPerspective(img2,t2,(1000,800))
new_img3 = cv2.warpPerspective(img3,t3,(1000,800))



cv2.imwrite("rotate.jpg", new_img1)
cv2.imwrite("translate.jpg", new_img2)
cv2.imwrite("scale.jpg", new_img3)


# 3.1  Compute SIFT features

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)

# 3.2  Match features
from scipy.spatial import distance_matrix

def computeMatrix(kp1,des1,kp2,des2):
    d1 = distance_matrix(des2, des1)
    threshold = np.sort(d1.flatten())[100]

    kp1_indexes = []
    kp2_indexes = []
    for i in range(len(d1)):
        for j in range(len(d1[i])):
            if d1[i][j] < threshold:
                kp1_indexes.append(j)
                kp2_indexes.append(i)

    best_kp1 = []
    best_des1 = []
    for value in kp1_indexes:
        best_kp1.append(kp1[value])
        best_des1.append(des1[value])

    best_kp2 = []
    best_des2 = []
    for value in kp2_indexes:
        best_kp2.append(kp2[value])
        best_des2.append(des2[value])
    # 3.3 Estimate the homographies
    src_pts = np.float32([ m.pt for m in best_kp1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ m.pt for m in best_kp2 ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
    matchesMask = mask.ravel().tolist()
    return M

M1 = computeMatrix(kp1,des1,kp2,des2)
M2 = computeMatrix(kp3,des3,kp2,des2)

# 3.4 Warp and translate
t1 = np.float32([[1, 0, 350], [0, 1,300], [0, 0,1]])
t2 = np.dot(t1, M1)
t3 = np.dot(t1, M2)


new_img1 = cv2.warpPerspective(img1,t2,(1000,800))
new_img2 = cv2.warpPerspective(img2,t1,(1000,800))
new_img3 = cv2.warpPerspective(img3,t3,(1000,800))

result = np.maximum(np.maximum(new_img1, new_img2), new_img3)
cv2.imwrite("result.jpg", result)
cv2.imshow('img',result)
cv2.waitKey(0)
cv2.destroyAllWindows()