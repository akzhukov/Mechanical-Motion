import sys
import cv2 as cv
import numpy as np
from matplotlib.pyplot import pause
import matplotlib.pyplot as plt
from time import time
import math

cap = cv.VideoCapture("D:\\diplom\\v2.3gp")
cap = cv.VideoCapture("D:\\diplom\\python scripts\\video1.avi")
frame_rate = cap.get(cv.CAP_PROP_FPS)

radius_coeff_down = 0.05
radius_coeff_up = 0.11


def dist(p1, p2):
    dx = int(p1[0]) - int(p2[0])
    dy = int(p1[1]) - int(p2[1])
    return np.sqrt(dx ** 2 + dy ** 2)


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def vec_coord(p1, p2):
    return [p2[0] - p1[0], p2[1] - p1[1]]


def obj_border(diff):
    mask = diff > 25
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    top = vertical_indices[0]
    bottom = vertical_indices[-1]

    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    left = horizontal_indices[0]
    right = horizontal_indices[-1]
    return (top, bottom), (left, right)


def search_motion_element_diff():
    res1, src1 = cap.read()
    gray1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    res2, src2 = cap.read()
    gray2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(gray1, gray2)
    vert, hor = obj_border(diff)
    vert_len = math.fabs(vert[1] - vert[0])
    hor_len = math.fabs(hor[1] - hor[0])
    if vert_len > hor_len:
        radius = vert_len / 2
    else:
        radius = hor_len / 2
    center_x = (hor[1] + hor[0]) / 2
    center_y = (vert[1] + vert[0]) / 2
    possible_error = 1.1
    return [int(center_x), int(center_y)], int(radius * possible_error)


def search_motion_element_hough():
    center_radius = []
    for i in range(5):
        res, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        # TODO задавать как-то эти параметры
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                                  param1=10, param2=50,
                                  minRadius=120, maxRadius=150)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center_radius.append([i[0], i[1], i[2]])
                cv.circle(src, (int(i[0]), int(i[1])), i[2], (0, 100, 100), 3)
    center_radius.sort()
    return [center_radius[2][0], center_radius[2][1]], center_radius[2][2]


def search_motion_center(cut_center, cut_radius):
    shift_x = cut_center[0] - cut_radius
    shift_y = cut_center[1] - cut_radius
    crop_center_rad = int(cut_radius * 0.4)
    for i in range(5):
        res, src = cap.read()
        src[cut_center[1] - crop_center_rad:cut_center[1] + crop_center_rad,
        cut_center[0] - crop_center_rad:cut_center[0] + crop_center_rad] = (255, 255, 255)
        src_cropped = src[shift_y:shift_y + 2 * cut_radius, shift_x:shift_x + 2 * cut_radius]
        gray = cv.cvtColor(src_cropped, cv.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                                  param1=100, param2=17,
                                  minRadius=int(cut_radius * radius_coeff_down),
                                  maxRadius=int(cut_radius * radius_coeff_up))
        if len(circles[0, :]) != 4:
            continue
        else:
            circles = np.uint16(np.around(circles))
            circles = circles[0, :]
            max_dist = 0
            point = 0
            for j in range(1, 4):
                cur_dist = dist(circles[0], circles[j])
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    point = j
            break
    center_x = int((circles[0][0] + circles[point][0]) / 2) + shift_x
    center_y = int((circles[0][1] + circles[point][1]) / 2) + shift_y
    return [center_x, center_y], int(max_dist / 2)


center, radius = search_motion_element_diff()
motion_center, motion_radius = search_motion_center(center, radius)


res, src = cap.read()
src_cropped = src[motion_center[1] - radius:motion_center[1] + radius,
              motion_center[0] - radius:motion_center[0] + radius]
crop_center_rad = int(radius * 0.4)
src[motion_center[1] - crop_center_rad:motion_center[1] + crop_center_rad,
    motion_center[0] - crop_center_rad:motion_center[0] + crop_center_rad] = (255, 255, 255)
gray = cv.cvtColor(src_cropped, cv.COLOR_BGR2GRAY)
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                          param1=100, param2=17,
                          minRadius=int(radius * radius_coeff_down), maxRadius=int(radius * radius_coeff_up))

first_circle_x = circles[0][0][0] + motion_center[0] - radius
first_circle_y = circles[0][0][1] + motion_center[1] - radius
area_size = radius/2
shift_x = int(first_circle_x - area_size)
shift_y = int(first_circle_y - area_size)
start = time()
angles = []
i = 0
turns = []
while True:
    i += 1
    res, src = cap.read()
    if not res:
        break

    src[motion_center[1] - crop_center_rad:motion_center[1] + crop_center_rad,
    motion_center[0] - crop_center_rad:motion_center[0] + crop_center_rad] = (255, 255, 255)
    src_cropped = src[int(shift_y):int(shift_y + 2 * area_size), int(shift_x):int(shift_x + 2 * area_size)]

    gray = cv.cvtColor(src_cropped, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                              param1=100, param2=17,
                              minRadius=int(radius * radius_coeff_down), maxRadius=int(radius * radius_coeff_up))
    # if circles is not None:
    #     for j in circles[0, :]:
    #         cv.circle(gray, (int(j[0]), int(j[1])), j[2], (255, 255, 255), 3)
    if circles is not None:
        circle_center_x = circles[0][0][0] + shift_x
        circle_center_y = circles[0][0][1] + shift_y
        circle_center = (int(circle_center_x), int(circle_center_y))
        shift_x = (circle_center_x - area_size)
        shift_y = (circle_center_y - area_size)
        vec_len = dist(motion_center, circle_center)
        vec1 = vec_coord(motion_center, circle_center)
        vec2 = vec_coord(motion_center, [motion_center[0] + motion_radius, motion_center[1]])
        angle_cos = (dot(vec_coord(motion_center, circle_center),
                         vec_coord(motion_center, [motion_center[0] + motion_radius, motion_center[1]])) / (
                             motion_radius * vec_len))
        if circle_center[1] >= motion_center[1]:
            above_axis_x = 1
        else:
            above_axis_x = -1
        if 1 >= angle_cos >= -1:
            angles.append((math.acos(angle_cos), above_axis_x))
        elif angle_cos > 1:
            angles.append((math.acos(1), above_axis_x))
        else:
            angles.append((math.acos(-1), above_axis_x))

for i in range(len(angles) - 1):
    if angles[i + 1][1] == angles[i][1]:
        turns.append(math.fabs(angles[i + 1][0] - angles[i][0]))
    elif angles[i][0] > math.pi / 2:
        turns.append(2 * math.pi - angles[i + 1][0] - angles[i][0])
    else:
        turns.append(angles[i + 1][0] + angles[i][0])
curr_angle = 0
num_frames = []
curr_num_frames = 0
for i in range(len(turns)):
    curr_angle += turns[i]
    curr_num_frames += 1
    if curr_angle > 2 * math.pi:
        print(str(i) + " " + str(num_frames))
        num_frames.append(curr_num_frames)
        curr_num_frames = 0
        curr_angle -= 2 * math.pi
finish = time()
plt.xlabel("количество сделанных оборотов")
plt.ylabel("количество кадров на оборот")
plt.grid()
plt.plot(num_frames)
plt.savefig("figures\\numframes.png")
# plt.show()
print(finish - start)
# video.release()
