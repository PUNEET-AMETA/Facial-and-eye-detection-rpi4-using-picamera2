# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:29:32 2024

@author: ameta
"""

import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import time
from picamera2 import Picamera2
from playsound import playsound
import mediapipe as ap
import pygame
pygame.mixer.init()
sound_file = '/home/pi/wake up 3.wav'
pygame.mixer.music.load(sound_file)
pygame.mixer.music.set_volume(0.5)


engine = pyttsx3.init()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'/home/pi/haarcascades/haarcascade_frontalcatface_extended.xml’)

# Initialize the PiCamera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()

time.sleep(0.1)  # Allow the camera to warm up

face_detector = dlib.get_frontal_face_detector()


dlib_facelandmark = dlib.shape_predictor('/home/pi/shape_predictor_68_face_landmarks.dat')

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

eyes_closed = False

closed_eye_timer_start = 0

eye_closure_threshold = 0.25

max_closed_eye_duration = 0.55


while True:
        cam = camera.capture_array()

        grey = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

        faces = face_detector(grey)
        

        for face in faces:
            face_landmarks = dlib_facelandmark(grey, face)
            #faces12 = face_detector.detectMultiScale(grey, 1.1, 5)
            leftEye = []
            rightEye = []

            for n in range(42, 48):
                x = face_landmarks.part(n).xy = face_landmarks.part(n).y

rightEye.append((x, y))
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(cam, (x, y), (x2, y2), (0, 255, 0), 1)

            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(cam, (x, y), (x2, y2), (255, 255, 0), 1)


right_Eye = Detect_Eye(rightEye)
            left_Eye = Detect_Eye(leftEye)
            Eye_Rat = (left_Eye + right_Eye) / 2
            Eye_Rat = round(Eye_Rat, 2)

            if Eye_Rat < eye_closure_threshold:
                if not eyes_closed:
                    closed_eye_timer_start = time.time()
                    eyes_closed = True
            else:
                eyes_closed = False

            if eyes_closed:
                time_passed = time.time() - closed_eye_timer_start cv2.putText(cam, f"Sleeping- Eyes closed for {int(time_passed)} seconds", (50, 100),
                            cv2.FONT_HERSHEY_PLAIN, 1, (21, 56, 210), 2)


if time_passed > max_closed_eye_duration:
                    cv2.putText(cam, "Alert!!!! WAKE UP DUDE", (50, 450),
                                cv2.FONT_HERSHEY_PLAIN, 1, (21, 56, 212), 2)
                    pygame.mixer.music.play()
                    time.sleep(5)
                    pygame.mixer.music.stop()



                    engine.say("Alert!!!!")
                    engine.runAndWait()

        cv2.imshow(" Sleeping timecounter",cam )
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

cv2.destroyAllWindows()

