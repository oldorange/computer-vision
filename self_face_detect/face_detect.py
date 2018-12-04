# -*- coding: utf-8 -*-

import dlib
import cv2

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()
	        
# Wait until the user hits <enter> to close the window	        
dlib.hit_enter_to_continue()



def detect(gray, frame):
    detected_faces = face_detector(frame, 1)
    print("I found {} faces".format(len(detected_faces)))
    return detected_faces



video_capture = cv2.VideoCapture(0)
try:
    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detect(gray, frame)
        # Open a window on the desktop showing the image
        win.clear_overlay()
        win.set_image(frame)
        for i, face_rect in enumerate(detected_faces):
            win.add_overlay(face_rect)
except KeyboardInterrupt:
       pass
    
video_capture.release()
cv2.destroyAllWindows()