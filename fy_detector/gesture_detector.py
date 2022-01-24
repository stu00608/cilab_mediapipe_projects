import cv2
import mediapipe as mp
import numpy as np
import datetime
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

# Gesture recognition model
file = np.genfromtxt(TRAINING_FILE_NAME, delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            
            joint, data = training_utils.get_angle_from_gesture(res.landmark)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
            x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

            cv2.putText(img, str(idx), (x2, y1), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 0, 255), 3)

            # Uncomment the line below to see the skeleton.
            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('gesture_detector', img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('img_'+datetime.datetime.now().strftime('%m%d%H%M')+'.jpg', img)
        print("Saved image.")
    
