from imutils.video import VideoStream
import pygame
import numpy as np
from numpy import asarray
from numpy import savetxt
import argparse
import imutils
import time
import datetime
from keras.models import load_model
import cv2
from pygame.locals import *
from eye import Eye
from detect_faces_video import FaceDetector


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)
GREEN = (0,255,0)
def main():
    """MAIN"""

    pygame.init()

    app_resolution = ([1200,800])
    pywindow = pygame.display.set_mode([1200, 800])
    pygame.display.set_caption('Draw Circle at Cursor!')



    face = FaceDetector()







    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)

    # loop over the frames from the video stream
    while True:

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        _,frame = vs.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1200,800))



        mouse = pygame.mouse.get_pos()


        frame = face.detect_face(frame)


        for event in pygame.event.get():

            ### trigger gaze tracking analysis ###


            """gaze_model = load_model("/Users/liuziyiliu/Desktop/gaze_tracking/model/train.h5")


            face.show_gaze(pywindow,frame,gaze_model,app_resolution)"""



            if event.type == pygame.MOUSEBUTTONDOWN: #This checks for the mouse press event
                print(u'button {} pressed in the position {}'.format(event.button, event.pos))
                mouse_pos = event.pos
                circ = pygame.mouse.get_pos() #Gets the mouse position
                pygame.draw.circle(pywindow, WHITE, (circ), 4, 4) #Draws a circle at the mouse position!
                headpose = face.collecthead(frame)

                dataset_dir = "/Users/liuziyiliu/Desktop/gaze_tracking/data"
                f_index = datetime.datetime.now().strftime("%m%d_%H%M%S")
                fname_l = dataset_dir + '/left_h/{}-{}_{}-{}_{}_{}.png'.format(f_index, mouse_pos[0], mouse_pos[1],headpose[0], headpose[1], headpose[2])
                fname_r = dataset_dir + '/right_h/{}-{}_{}-{}_{}_{}.png'.format(f_index, mouse_pos[0], mouse_pos[1],headpose[0], headpose[1], headpose[2])



                eye_frame_l, eye_frame_r = face.collectdata(frame)
                head_frame = frame

                cv2.imwrite(fname_l, eye_frame_l)
                cv2.imwrite(fname_r, eye_frame_r)

                print('Frame saved to', fname_l, 'and', fname_r)




        """text = ""

        if face.is_blinking():
            text = "Blinking"
        elif face.is_right():
            text = "Looking right"
        elif face.is_left():
            text = "Looking left"
        elif face.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)"""

        left_pupil = face.pupil_left_coords()
        right_pupil = face.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (100, 180), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (100, 210), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)


        # show the output frame
        cv2.imshow("Frame", frame)

        surface = cv2.flip(frame, 1)
        surface = np.rot90(surface)
        surface = cv2.cvtColor(surface, cv2.COLOR_RGB2BGR)
        surface = pygame.surfarray.make_surface(surface)
        surface = pygame.transform.scale(surface, (600, 400))


        pywindow.blit(surface,(300,0))
        key = cv2.waitKey(1) & 0xFF
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        pygame.display.update()
        pywindow.fill(BLUE)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    pygame.quit()

"""
def draw_button(self, pywindow, button, button_coords, mouse_pos, text, color=GREEN, active_color=RED, text_color=BLACK):
    ""
    args:
        pywindow: pygame pywindow object
        button: pygame.Rect object
        button_coords: list of button coordinates - [x,y,w,h]
        mouse_pos: coords of mouse - (x,y)
        text: text to be displayed on the button
        color: default color of button
        active_color: color when mouse hovers on the button
        text_color: color of button text
    ""
    if button_coords[0] < mouse_pos[0] < button_coords[0]+button_coords[2] and button_coords[1] < mouse_pos[1] < button_coords[1]+button_coords[3]: # mouse hover the button
        pygame.draw.rect(pywindow, active_color, button)
        draw_text(pywindow, (button_coords[0]+(button_coords[2]//2), button_coords[1]+(button_coords[3]//2)), text, int(button_coords[3]/2.5), color=text_color)
    else:
        pygame.draw.rect(pywindow, color, button)  # button
        draw_text(pywindow, (button_coords[0]+(button_coords[2]//2), button_coords[1]+(button_coords[3]//2)), text, button_coords[3]//3, color=text_color)
"""

if __name__ == '__main__':
    main()
