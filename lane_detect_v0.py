import cv2 as cv
import numpy as np
from math import *

fileName = "laneimage"
img = cv.imread(fileName+'.jpg')
scale_percent = 30
w = int(img.shape[1] * scale_percent / 100)
h = int(img.shape[0] * scale_percent / 100)
print(w,h)
img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

def drawLine(rho, theta, img, color, thick):
    x0 = rho * np.cos(theta)
    y0 = rho * np.sin(theta)
    x1 = int(x0 - 1000*np.sin(theta))
    y1 = int(y0 + 1000*np.cos(theta))
    x2 = int(x0 + 1000*np.sin(theta))
    y2 = int(y0 - 1000*np.cos(theta))
    cv.line(img, (x1, y1), (x2, y2), color, thick)
    return [x1, y1], [x2, y2]

def myShow(title,  img):
    cv.imshow(title, img)
    key = cv.waitKey(0)
    if key & 0xFF == 27:
        cv.destroyAllWindows()
        exit(0)
    elif key == 'd' or key == 'D':
        cv.destroyWindow(title)
    else:
        pass

# Gray
imgGray = cv.imread(fileName+'.jpg', 0)
imgGray = cv.resize(imgGray, (w, h), interpolation=cv.INTER_AREA)
cv.imwrite("imgGray.jpg", imgGray)

imgRoi = np.zeros((h, w, 1), np.uint8)
imgOut = np.zeros(img.shape, np.uint8)
# Region Of Interest

pointRoi = np.array(
    [[w * 0.3, h * 0.67], [w * 0.45, h * 0.4], [w * 0.6, h * 0.4], [w * 0.8, h * 0.67]],
    np.int32,
)
syROI = pointRoi[1][1]
eyROI = pointRoi[0][1]
cv.fillConvexPoly(imgRoi, pointRoi, 255)
imgRoiDisp = cv.addWeighted(imgGray, 0.8, imgRoi, 0.2, 0)
myShow(fileName+'_Roi', imgRoiDisp)

# Threshold
ret, thresh = cv.threshold(imgGray, 180, 255, cv.THRESH_BINARY)
myShow(fileName+'_Thresh', thresh)
for r in range(0, img.shape[0]):
    for c in range(0, img.shape[1]):
        if(imgRoi[r, c] == 0):
            thresh[r, c] = 0
myShow(fileName+'_roiThresh', thresh)

# Canny
edges = cv.Canny(thresh, 50, 150)
myShow(fileName+'_Canny', edges)
cv.imwrite("edges.jpg", edges)

# Hough Line
lines = cv.HoughLines(edges, 1, np.pi/180, 40)

imgHough = np.copy(img)

thetaQ = np.array([0.0, 0.0])
rhoQ = np.array([0.0, 0.0])
countQ = np.array([0.0, 0.0])

pt1 = np.array([[0, 0], [0, 0]])
pt2 = np.array([[0, 0], [0, 0]])

for line in lines:
    rho, theta = line[0]
# Angle of Lines
    if(theta < np.pi*40/180 and theta > np.pi*30/180):
        thetaQ[0] = thetaQ[0] + theta
        rhoQ[0] = rhoQ[0] + rho
        countQ[0] = countQ[0] + 1
    if(theta < np.pi*130/180 and theta > np.pi*120/180):
        thetaQ[1] = thetaQ[1] + theta
        rhoQ[1] = rhoQ[1] + rho
        countQ[1] = countQ[1] + 1
myShow(fileName+'_Hough', imgHough)

# Average lines
for i in range(2):
    thetaQ[i] = thetaQ[i] / countQ[i]
    rhoQ[i] = rhoQ[i] / countQ[i]
    p1, p2 = drawLine(rhoQ[i], thetaQ[i], imgHough, (0, 0, 255), 1)
    
    # Detecting the lane
    retval, pt1[i], pt2[i] = cv.clipLine((0, syROI, w, eyROI-syROI), p1, p2)
    cv.rectangle(imgHough, [0, syROI], [w, eyROI], (0, 128, 0), 2)
    print(p1,p2)
    print(retval, pt1[i], pt2[i])

cv.rectangle(imgHough, pt1[0], pt2[0], (0, 128, 256), 2)
cv.rectangle(imgHough, pt1[1], pt2[1], (128, 0, 0), 2)
myShow(fileName+'_Hough', imgHough)

cv.imwrite("imgHough.jpg", imgHough)
# Displaying the lane
point = np.array([pt2[0],  pt1[0], pt2[1], pt1[1]], np.int32)
cv.fillConvexPoly(imgOut, point, (255, 0, 0))
overlapImage = cv.addWeighted(img, 0.6, imgOut, 0.4, 0)
myShow(fileName+'_Lane', overlapImage)
cv.imwrite("overlapImage.jpg", overlapImage)

# Topview of the lane
pts2 = np.float32([[0, 0], [0, 512], [512, 512], [512, 0]])
pts1 = np.float32([pt2[0],  pt1[0], pt2[1], pt1[1]])

M = cv.getPerspectiveTransform(pts1, pts2)
perspectiveImg = cv.warpPerspective(img, M, (512, 512))
myShow(fileName+'_TopView', perspectiveImg)

cv.imwrite("perspectiveImg.jpg", perspectiveImg)
