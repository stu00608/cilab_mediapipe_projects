import cv2
import mediapipe as mp


if __name__ == '__main__':

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()

            result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.detections:
                for detection in result.detections:
                    mp_drawing.draw_detection(frame, detection)
            
            cv2.imshow("Face detection using mediapipe", cv2.flip(frame, 1))

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        
        cv2.destroyAllWindows()
        cap.release()
