import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny

big_number = 100000000

fourcc = cv.VideoWriter_fourcc(*'mp4v')
videoname = "Test2-185c.mp4"

cap = cv.VideoCapture("D:\\diplom\\curtain\\" + videoname)
frame_rate = cap.get(cv.CAP_PROP_FPS)
# video = cv.VideoWriter('video4.avi', fourcc, 25, (1920, 108))
i = 0
distance = []
prev_dist = 0
resize = 0.1
res, src = cap.read()
if not res:
    exit()
src = cv.resize(src, (0, 0), fx=1, fy=resize)

start_motion = big_number
motion_frames = 0
start_position = 0
num_motion_frames = 100
max_dist_per_frame = 100
while True:
    i += 1
    res1, src1 = cap.read()
    if not res1:
        break
    src1 = cv.resize(src1, (0, 0), fx=1, fy=resize)
    #
    # if i == 500:
    #     break
    print(i)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(gray, gray1)
    angle = 0.1
    h, theta, d = hough_line(canny(diff), theta=np.linspace(-angle, angle, 100))
    nearest = big_number

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - gray.shape[1] * np.cos(angle)) / np.sin(angle)
        # if abs(angle) < 0.1:  # проверка на почти вертикальность
        cv.line(src, (0, int(y0)), (int(gray.shape[1]), int(y1)), (0, 100, 100), 3)
        curr_dist = gray.shape[1] * (180 + math.fabs(y0)) / (math.fabs(y0) + math.fabs(y1))
        if abs(prev_dist - curr_dist) < nearest:
            nearest = curr_dist
    if nearest == big_number or (i > start_motion and math.fabs(prev_dist - nearest) > max_dist_per_frame):
        distance.append(prev_dist)
    else:
        distance.append(nearest)
        prev_dist = nearest
    src = src1
    if math.fabs(prev_dist - nearest) < max_dist_per_frame and prev_dist != 0:
        motion_frames += 1
    if motion_frames > num_motion_frames and start_motion == big_number:
        start_motion = i
    # cv.imwrite("curt\\"+str(i)+"img.jpg", src)
#     video.write(src)
# video.release()

video_length = len(distance) - 1
for i in range(video_length):
    if math.fabs(distance[video_length - i] - distance[video_length - i - 1]) > max_dist_per_frame:
        distance[video_length - i - 1] = distance[video_length - i]


y = []
z = []
for i in range(len(distance)):
    y.append(i / frame_rate)
for i in range(len(distance) - 1):
    z.append(i / frame_rate)


plt.xlabel("время, с")
plt.ylabel("расстояние, пиксель")
plt.grid()
plt.plot(y, distance)

plt.savefig("figures\\" + videoname + "_dist_per_frame_" + str(resize) + "_4.png")

n_iter = len(distance)
sz = (n_iter,)

Q = 1e-4  # process variance

# allocate space for arrays
xhat = np.zeros(sz)  # a posteri estimate of x
P = np.zeros(sz)  # a posteri error estimate
xhatminus = np.zeros(sz)  # a priori estimate of x
Pminus = np.zeros(sz)  # a priori error estimate
K = np.zeros(sz)  # gain or blending factor

R = 0.1 ** 4  # estimate of measurement variance, change to see effect
R = 0.001# intial guesses
xhat[0] = 0.0
P[0] = 1.0
K_value = 0.1
for k in range(1, n_iter):
    # time update
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q

    # measurement update
    # K[k] = Pminus[k] / (Pminus[k] + R)
    K[k] = K_value
    xhat[k] = xhatminus[k] + K[k] * (distance[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]

plt.figure()
plt.plot(y, distance, 'r-', label='noisy measurements')
plt.plot(y, xhat, 'b-', label='a posteri estimate')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('время')
plt.ylabel('расстояние')
plt.show()
