######################################
# air-paint  >  HandTracker.py
# Created by Uygur Kiran on 2021/04/29
######################################

######################################
import cv2 as cv
import mediapipe as mp
import time
######################################

class HandTracker:
    def __init__(self,
                 cam_num: int = 0,
                 static_img_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 show_points: bool = True,
                 show_lines: bool = False):

        ### INIT ##########################
        self.cam_num = cam_num
        self.show_points = show_points
        self.show_lines = show_lines

        ### PRIVATE ########################
        self.__cam = cv.VideoCapture(self.cam_num)

        ## DETECTION
        self.__hands_kit = mp.solutions.hands
        self.__hands = self.__hands_kit.Hands(static_img_mode, max_num_hands,
                                              min_detection_confidence,
                                              min_tracking_confidence)
        self.__drawer = mp.solutions.drawing_utils

        ## FPS TRACKING
        self.__prev_time = 0
        self.__curr_time = 0

        ## CACHE
        self.__landmarks = None
        self.__highlighted_points = []
        self.__lms_for_hand = None
        self.__img = None

        ### OUTPUT VALUES ###################
        self.fps = 0

    def get_capture(self):
        """
        get hold of the video session

        :return: VideoCapture
        """
        return self.__cam

    def highlight(self, lm_idx = []):
        """
        adds bigger circles to desired hand landmarks

        lm_idx: landmark index [int]
        """
        self.__highlighted_points = lm_idx


    def get_hand_landmarks(self,img):
        """
        use this in a capture loop when not using the <>.mainloop()
        """
        self.__detect_hands(img)
        lms = self.__get_pos(img)

        if lms is not None and len(lms) != 0:
            for mark in lms:
                if mark[0] in self.__highlighted_points:
                    cv.circle(img, (mark[1], mark[2]),
                              10, (192, 57, 43), cv.FILLED)

        self.__lms_for_hand = lms
        return lms

    def fingers_up_or_down(self):
        """
        compares the tip position with the other landmarks of the finger.
        eg. finger is up if the tip of the finger is above the other landmarks of the same finger
        :return: [0 | 1]
        """
        lms = self.__lms_for_hand
        tip_ids = [4,8,12,16,20]
        fingers = []    # 1 -> up, 0 -> not up

        if lms is not None and len(lms) != 0:
            # THUMB (check for horizontal diff. / center_x)
            # > for right hand, < for left hand
            if lms[tip_ids[0]][1] > lms[tip_ids[0] - 1][1]:
                fingers.append(1)
            else: fingers.append(0)

            # OTHER FINGERS (check for vertical diff. / center_y)
            for id in range(1,5):
                if lms[tip_ids[id]][2] < lms[tip_ids[id]-2][2]:
                    fingers.append(1)
                else: fingers.append(0)
        return fingers


    def draw_circle(self, center, color,
                    radius: int = 20):
        """
        :param center: (x, y)
        :param color: (b, g, r)
        :param radius: defs. to 20

        """
        if self.__img is not None:
            cv.circle(self.__img, center, radius, color, cv.FILLED)

    def draw_rect(self, pt1, pt2, color, text=None):
        if self.__img is not None:
            cv.rectangle(self.__img, pt1, pt2, color, cv.FILLED)

        if text is not None and isinstance(text, str):
            cv.putText(self.__img, text, (pt1[0], pt1[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def draw_line(self, pt1, pt2, color, thickness=20):
        cv.line(self.__img, pt1, pt2, color, thickness)

    def show_fps(self, img, pos = (20, 70)):
        self.__curr_time = time.time()
        self.fps = 1 / (self.__curr_time - self.__prev_time)
        self.__prev_time = self.__curr_time

        cv.putText(img=img, text=f"{int(self.fps)} fps",
                   org=pos,
                   fontFace=cv.FONT_HERSHEY_DUPLEX,
                   fontScale=1,
                   color=(0, 0, 0),
                   thickness=5)

    def mainloop(self):
        while True:
            success, img = self.__cam.read()
            self.__detect_hands(img)
            self.show_fps(img)

            def_hand_marks = self.__get_pos(img)

            if def_hand_marks and len(def_hand_marks) != 0:
                for mark in def_hand_marks:
                    if mark[0] in self.__highlighted_points:
                        cv.circle(img, (mark[1], mark[2]),
                                  10, (34, 126,230), cv.FILLED)

            cv.imshow("Image", img)
            cv.waitKey(1)


    ######################################
    # PRIVATE METHODS
    ######################################
    def __detect_hands(self, img):
        ## CACHE
        self.__img = img

        ## TO RGB_IMG & DETECT
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.__hands.process(rgb_img)
        self.__landmarks = res.multi_hand_landmarks

        ## TRACK RESULTS
        if self.__landmarks:
            for hand in self.__landmarks:
                if self.show_points and self.show_lines:
                    self.__drawer.draw_landmarks(img, hand, self.__hands_kit.HAND_CONNECTIONS)
                elif self.show_points:
                    self.__drawer.draw_landmarks(img, hand)


    def __get_pos(self, from_img, for_hand: int = 0):
        positions = []
        if self.__landmarks:
            hand = self.__landmarks[for_hand]
            for id, landmark in enumerate(hand.landmark):
                height, width, channels = from_img.shape

                center_x = int(landmark.x * width)
                center_y = int(landmark.y * height)
                positions.append((id, center_x, center_y))

        return positions

