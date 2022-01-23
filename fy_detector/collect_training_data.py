import cv2
import mediapipe as mp
import numpy as np
import training_utils

MAX_NUM_HANDS = 1
TRAINING_FILE_NAME = 'train.csv'

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition data
file = np.genfromtxt(TRAINING_FILE_NAME, delimiter=',')

cap = cv2.VideoCapture(0)

# TODO: Choose a new pose (auto coding), or use the exist code.
LABEL = input("Please input the label code : ")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    data = np.array([])

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:

            joint, data = training_utils.get_angle_from_gesture(res.landmark)
            data = np.append(data, 11)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('collect_training_data', img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('a'):
        try:
            file = np.vstack((file, data))
            np.savetxt('train.csv', file, delimiter=',')
            print("Saved pose to csv file.")
        except:
            print("No data.")


