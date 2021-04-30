######################################
# air-paint  >  main.py
# Created by Uygur Kiran on 2021/04/30
######################################

######################################
import os
import cv2 as cv
import numpy as np
from src.HandTracker import HandTracker
from enum import Enum
######################################

class Res(Enum):
    p1080 = (1920, 1080)
    p720 = (1280, 720)

## CURRENT RESOLUTION
res: Res = Res.p720

######################################
# HAND TRACKER
######################################
DEBUG = False
t = HandTracker(cam_num=1, min_detection_confidence=0.85)
# t.highlight([8])
cam = t.get_capture()

# FRAME WIDTH & HEIGHT
cam.set(3, res.value[0])
cam.set(4, res.value[1])

######################################
# UI
######################################
def get_header():
    path = "./src/headers/720" if res == Res.p720 else "./src/headers/1080"
    img_file_list = os.listdir(path)
    if ".DS_Store" in img_file_list: img_file_list.remove(".DS_Store")
    img_list = []
    for i in sorted(img_file_list):
        img = cv.imread(f"{path}/{i}")
        img_list.append(img)
    return img_list

headers = get_header()
header = headers[0]

# BGR COLOR
class Color(Enum):
    red = (60,76,231)
    green = (113,204,46)
    blue = (219,152,52)
    black = (0,0,0)

selected_color = Color.red.value
brush_radius = 20 if res == Res.p720 else 25
eraser_radius = 80 if res == Res.p720 else 100

def select_brush(x1):
    global header, selected_color
    if res == Res.p720:
        if 350 < x1 < 500:
            header = headers[0]
            selected_color = Color.red.value
        elif 600 < x1 < 750:
            header = headers[1]
            selected_color = Color.green.value
        elif 800 < x1 < 950:
            header = headers[2]
            selected_color = Color.blue.value
        elif 1050 < x1 < 1200:
            header = headers[3]
            selected_color = Color.black.value
    else:
        if 620 < x1 < 850:
            header = headers[0]
            selected_color = Color.red.value
        elif 990 < x1 < 1200:
            header = headers[1]
            selected_color = Color.green.value
        elif 1300 < x1 < 1500:
            header = headers[2]
            selected_color = Color.blue.value
        elif 1620 < x1 < 1810:
            header = headers[3]
            selected_color = Color.black.value


######################################
# MAINLOOP
######################################
drw_layer = np.zeros((res.value[1], res.value[0], 3), np.uint8)
fps_pos = (20, 710) if res == Res.p720 else (20, 1050)
xp, yp = 0, 0
while True:
    success, img = cam.read()
    img = cv.flip(img, 1)

    ## DETECT & GET LM POSs
    lms = t.get_hand_landmarks(img)

    ## GET TIP POSs
    if len(lms) != 0:
        # tip of the index finger
        x1, y1 = lms[8][1:]
        # tip of the middle finger
        x2, y2 = lms[12][1:]

    ## FIND WHICH FINGERS UP
    updown = t.fingers_up_or_down()
    if updown and DEBUG: print("***\n" + f"UP OR DOWN: {updown}")

    ## SELECTION MODE ##################
    # if index and middle fingers are up
    if updown is not None and len(updown) != 0:
        if updown[1] == 1 and updown[2] == 1:
            xp, yp = 0, 0
            if DEBUG: print("MODE: Selection mode" + "\n***")
            if len(lms) != 0:
                if y1 < 130:
                    select_brush(x1)

                t.draw_rect((x1, y1-15), (x2, y2+25), selected_color, "select")

    ## DRAWING MODE ####################
    # if only index finger is up
    if updown is not None and len(updown) != 0:
        if updown[1] == 1 and updown[2] == 0:
            if DEBUG: print("MODE: Drawing mode" + "\n***")
            if len(lms) != 0:
                t.draw_circle(center=(x1,y1),
                              color=selected_color,
                              radius=brush_radius)
                # 1st iter
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if selected_color == Color.black.value:
                    cv.line(img, (xp, yp), (x1, y1), selected_color, brush_radius)
                    cv.line(drw_layer, (xp, yp), (x1, y1), selected_color, eraser_radius)
                else:
                    cv.line(img, (xp, yp), (x1, y1), selected_color, brush_radius)
                    cv.line(drw_layer, (xp, yp), (x1, y1), selected_color, brush_radius)

                xp, yp = x1, y1

    ## MASK NEG SPACE
    gray_img = cv.cvtColor(drw_layer, cv.COLOR_BGR2GRAY)
    _, img_inv = cv.threshold(gray_img, 50, 255, cv.THRESH_BINARY_INV)
    img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)

    ## MERGE
    img = cv.bitwise_and(img, img_inv)
    img = cv.bitwise_or(img, drw_layer)

    ## SET HEADER
    # by reassigning top of the img matrix
    img[0:130, 0:res.value[0]] = header

    ## FPS
    t.show_fps(img=img, pos=fps_pos)

    cv.imshow("Image", img)
    cv.waitKey(1)