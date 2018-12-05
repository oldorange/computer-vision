# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import dlib
import cv2

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()


def detect(gray, frame):
    detected_faces = face_detector(gray, 1)
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
            pose_landmarks = face_pose_predictor(frame, face_rect)
            win.add_overlay(pose_landmarks)
except KeyboardInterrupt:
       pass
    
video_capture.release()
cv2.destroyAllWindows()