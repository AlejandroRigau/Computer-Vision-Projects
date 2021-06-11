from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

fx, fy, cx, cy, _, LUT = ReadCameraModel("./Oxford_dataset_reduced/model")
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

positions = []
cur_T = np.array([[0.0],[0.0],[0.0]])
cur_R = np.eye(3)

positions.append(cur_T.tolist())
images = os.listdir("./Oxford_dataset_reduced/images")
for i in range(len(images)-1):
    # print(i)
    img1 = cv2.imread(f"./Oxford_dataset_reduced/images/{images[i]}",flags=-1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    img1 = UndistortImage(img1, LUT)
    img2 = cv2.imread(f"./Oxford_dataset_reduced/images/{images[i+1]}",flags=-1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    img2 = UndistortImage(img2, LUT)

    # Source for Keypoint correspondences: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    E = np.matmul(np.matmul(K.T, F), K)

    points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    cur_T = cur_T + cur_R.dot(t) 
    cur_R = R.dot(cur_R)
    
    
    positions.append(cur_T)


positions = np.array(positions)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(positions[:,0].flatten(), positions[:,2].flatten())
plt.savefig('2d reconstruction.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:,0].flatten(), positions[:,2].flatten(), positions[:,1].flatten())
plt.savefig('3d reconstruction.png')
plt.show()
