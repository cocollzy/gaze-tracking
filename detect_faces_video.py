# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from __future__ import division
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
from numpy import asarray
from numpy import savetxt
import argparse
import imutils
import time

import datetime
import pygame
import cv2
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import dlib
from eye import Eye
from calibration import Calibration
from testmouse import Mouse
import autopy


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



object_pts = np.float32([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Mouth left corner
    (150.0, -150.0, -125.0)      # Mouth right corner
])/45

"""
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])"""
point_3d = []
rear_size = 7.5
rear_depth = 0
point_3d.append((-rear_size, -rear_size, rear_depth))
point_3d.append((-rear_size, rear_size, rear_depth))
point_3d.append((rear_size, rear_size, rear_depth))
point_3d.append((rear_size, -rear_size, rear_depth))
point_3d.append((-rear_size, -rear_size, rear_depth))

front_size = 10
front_depth = 10
point_3d.append((-front_size, -front_size, front_depth))
point_3d.append((-front_size, front_size, front_depth))
point_3d.append((front_size, front_size, front_depth))
point_3d.append((front_size, -front_size, front_depth))
point_3d.append((-front_size, -front_size, front_depth))
point_3d = np.float32(point_3d).reshape(-1, 3)


"""reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])"""



line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



class FaceDetector:

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        DNN = "CAFFE"
        if DNN == "CAFFE":
            modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
            configFile = "deploy.prototxt"
            # load our serialized model from disk
            print("[INFO] loading model...")
            self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)


        #net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


    # loop over the frames from the video stream
    def detect_face(self, frame):
        # grab the frame from the threaded video stream and resize it


        # grab the frame dimensions and convert it to a blob
        self.frame = frame

        (h, w) = frame.shape[:2]
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        self.detections = detections


        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            face_rect = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

            #dlib_rectangle = dlib.rectangle(left=0, top=int(frameW), right=int(frameW), bottom=int(frameH))
            try:
                landmarks = self._predictor(frame, face_rect)
                self.landmarks = landmarks

                self.eye_left = Eye(frame, landmarks, 0, self.calibration)
                self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            except IndexError:
                self.eye_left = None
                self.eye_right = None
            landmarks_np = face_utils.shape_to_np(landmarks)
            #for (x, y) in landmarks_np:
                #cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            """cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)"""


            #frame = self.annotated_frame()

            shape = landmarks_np

            self.get_head_pose(frame,shape,h,w)


        return frame
    def collecthead(self, frame):
        (h, w) = frame.shape[:2]
        shape = face_utils.shape_to_np(self.landmarks)
        return self.get_head_pose(frame, shape, h,w)


    def collectdata(self, frame):
        LEFT_EYE_POINTS = [36, 37,  39, 41]
        RIGHT_EYE_POINTS = [42, 43,  45, 47]
        padding_px = 15
        startX = self.landmarks.part(LEFT_EYE_POINTS[0]).x - padding_px
        startY = self.landmarks.part(LEFT_EYE_POINTS[1]).y - padding_px
        endX = self.landmarks.part(LEFT_EYE_POINTS[2]).x + padding_px
        endY = self.landmarks.part(LEFT_EYE_POINTS[3]).y + padding_px
        eye_left_coords = (startX, startY, endX, endY)
        self.eye_left_coords = eye_left_coords

        startX = self.landmarks.part(RIGHT_EYE_POINTS[0]).x - padding_px
        startY = self.landmarks.part(RIGHT_EYE_POINTS[1]).y - padding_px
        endX = self.landmarks.part(RIGHT_EYE_POINTS[2]).x + padding_px
        endY = self.landmarks.part(RIGHT_EYE_POINTS[3]).y + padding_px
        eye_right_coords = (startX, startY, endX, endY)
        self.eye_right_coords = eye_right_coords

        eye_frame_l = frame[eye_left_coords[1]:eye_left_coords[3], eye_left_coords[0]:eye_left_coords[2]]
        eye_frame_r = frame[eye_right_coords[1]:eye_right_coords[3], eye_right_coords[0]:eye_right_coords[2]]

        eye_frame_l_resized = cv2.resize(eye_frame_l,(80, 40))
        eye_frame_r_resized = cv2.resize(eye_frame_r, (80, 40))

        return eye_frame_l_resized, eye_frame_r_resized

    def show_gaze(self,pywindow,frame,gaze_model,app_resolution):

        try:
            LEFT_EYE_POINTS = [36, 37,  39, 41]
            RIGHT_EYE_POINTS = [42, 43,  45, 47]
            padding_px = 15
            startX = self.landmarks.part(LEFT_EYE_POINTS[0]).x - padding_px
            startY = self.landmarks.part(LEFT_EYE_POINTS[1]).y - padding_px
            endX = self.landmarks.part(LEFT_EYE_POINTS[2]).x + padding_px
            endY = self.landmarks.part(LEFT_EYE_POINTS[3]).y + padding_px
            eye_left_coords = (startX, startY, endX, endY)
            self.eye_left_coords = eye_left_coords

            startX = self.landmarks.part(RIGHT_EYE_POINTS[0]).x - padding_px
            startY = self.landmarks.part(RIGHT_EYE_POINTS[1]).y - padding_px
            endX = self.landmarks.part(RIGHT_EYE_POINTS[2]).x + padding_px
            endY = self.landmarks.part(RIGHT_EYE_POINTS[3]).y + padding_px
            eye_right_coords = (startX, startY, endX, endY)
            self.eye_right_coords = eye_right_coords

            eye_frame_l = frame[eye_left_coords[1]:eye_left_coords[3], eye_left_coords[0]:eye_left_coords[2]]
            eye_frame_r = frame[eye_right_coords[1]:eye_right_coords[3], eye_right_coords[0]:eye_right_coords[2]]

            eye_frame_l_resized = cv2.resize(eye_frame_l, (80, 40))
            eye_frame_r_resized = cv2.resize(eye_frame_r, (80, 40))
            gaze_input_l = cv2.cvtColor(eye_frame_l_resized, cv2.COLOR_BGR2GRAY)
            gaze_input_r = cv2.cvtColor(eye_frame_r_resized, cv2.COLOR_BGR2GRAY)
            self.show_gaze_tracking(pywindow, gaze_model, [gaze_input_l, gaze_input_r], app_resolution)

        except Exception as error:
            print(error)




    def show_gaze_tracking(self, pywindow, model, model_inputs, app_resolution, color=(255, 0, 0), radius=20):

         # print(model_input.shape)
         # mirror effect
         #model_input_l = cv2.flip(model_inputs[0], 1)
         model_input_l = model_inputs[0]
         model_input_r = model_inputs[1]

         model_input_l = np.expand_dims(np.expand_dims(model_input_l, axis=0), axis=3)
         #model_input_r = cv2.flip(model_inputs[1], 1)
         model_input_r = np.expand_dims(np.expand_dims(model_input_r, axis=0), axis=3)

         pos = model.predict([model_input_l, model_input_r])
         pos = pos.astype(int).tolist()[0]

         if pos[0] <= 0:
             pos[0] = 0
         if pos[0] >= app_resolution[0]:
             pos[0] = app_resolution[0]
         if pos[1] <=0:
             pos[1] = 0
         if pos[1] >= app_resolution[1]:
             pos[1] = app_resolution[1]
         print(pos)

         pygame.draw.circle(pywindow, color, pos, radius)
         return pos



    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def get_head_pose(self, frame,shape,h,w):

        size = (h,w)
        #camera_intrinsic
        center = (size[1]/2, size[0]/2)
        focal_length = size[1]
        #focal_length = center[0] / np.tan(60/2 * np.pi / 180)


        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        cam_matrix = np.float32(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]] )
        #print("Camera Matrix :\n {0}".format(cam_matrix))

        #image_points

        """image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])"""

        image_pts = np.float32([shape[30], shape[8], shape[36], shape[45], shape[48],
                                shape[54]])

        #compute r and v, project 3d to 2d, reshape
        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #point_3d = np.float32(reprojectsrc.reshape(-1, 3))
        point_2d, _ = cv2.projectPoints(point_3d, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)
        #point_2d = tuple(map(tuple, point_2d.reshape(8, 2)))
        point_2d = np.int32(point_2d.reshape(-1, 2)) #to draw polylines

        #get axispoints for drawing axis
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0,0,0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(
            points, rotation_vec, translation_vec, cam_matrix, dist_coeffs)



        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))


        """drawing"""
        #draw features
        for p in image_pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        #eyePoints, _ = cv2.projectPoints(
            #points, r, v, cam_matrix, dist_coeffs)
        #eye_pts2 = ((x_left+x_right)/2, (y_left+y_right)/2)
        #print(eye_pts2)
        #print(eye_pts)
        #autopy.mouse.move((x_left+x_right)/2, (y_left+y_right)/2)
        #Mouse(eye_pts2)
        #drawing face box
        for start, end in line_pairs:
            #cv2.line(frame, point_2d[start], point_2d[end], (0, 170, 255), 2, cv2.LINE_AA)
            # Draw all the lines
            #cv2.polylines(frame, [point_2d], True, (0,170, 255), 2, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[start]), tuple(point_2d[end]), (0, 170, 255), 2, cv2.LINE_AA)
       #drawing face box 2
        color =(0, 170, 255)
        line_width = 2
        cv2.polylines(frame, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(frame, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(frame, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(frame, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

        #drawing three axis
        frame = cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        frame = cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        frame = cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

        #def draw_axes(self, frame, R, t):
        #frame   = cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs,rotation_vec, translation_vec, 30)"""


        cv2.putText(frame, "Pitch: " + "{:7.2f}".format(float(pitch)), (100, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        cv2.putText(frame, "Yaw: " + "{:7.2f}".format(float(yaw)), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        cv2.putText(frame, "Roll: " + "{:7.2f}".format(float(roll)), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        return [pitch, roll, yaw]








    """def detect_eyes(self,frame):
        #Detects the face and initialize Eye objects
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = self._face_detector(frame)
        #(frameH, frameW) = self.detections.shape[:2]
        # compute the (x, y) coords of face bounding box
        face_coords = (startX, startY, endX, endY)
        face_rect = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)

        #dlib_rectangle = dlib.rectangle(left=0, top=int(frameW), right=int(frameW), bottom=int(frameH))

        try:
            landmarks = self._predictor(frame, dlib_rectangle)
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        #Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze

        #self.frame = frame
        self.detect_eyes(frame)"""


    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
