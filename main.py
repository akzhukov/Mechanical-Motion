import sys
import cv2 as cv
import numpy as np
from matplotlib.pyplot import pause
from time import time


cap = cv.VideoCapture("D:\\diplom\\python scripts\\video1.avi")

# по нескольким первым кадрам необходимо определить большую окружность и искать маленькие только в этой области
# обрезать квадрат величиной радиуса
center_radius = []
radiuses = []
start = time()
for i in range(5):
    res, src = cap.read()
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                           param1=10, param2=50,
                           minRadius=120, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_radius.append([i[0], i[1], i[2]])

center_radius.sort() #мб надо какую-то сортировку поумнее

cap = cv.VideoCapture("D:\\diplom\\python scripts\\video1.avi")



half_side = center_radius[2][2]
center_x = center_radius[2][0]
center_y = center_radius[2][1]
fourcc = cv.VideoWriter_fourcc(*'mp4v')
video = cv.VideoWriter('video.avi', fourcc, 30, (272, 272))
# height = img.size[1]
# img3 = img.crop( (0,0,width,height-20) )
finish = time()
print(str((finish-start)))



start = time()
while True:
    res, src = cap.read()
    if res == False:
        break
    src_cropped = src[center_y-half_side:center_y+half_side,center_x-half_side:center_x+half_side]
    src = src_cropped.copy()
    src_cropped[75:200,75:200] = (255,255,255)
    gray = cv.cvtColor(src_cropped, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    # cv.imshow("frame", gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                              param1=100, param2=17,
                              minRadius=8, maxRadius=14)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = 12
            cv.circle(src, center, radius, (255, 0, 255), 3)
    video.write(src)

video.release()
finish = time()
print(str((finish-start)))
