import cv2

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        
        ret, frame = cap.read()

        frame_x = cv2.Sobel(frame, cv2.CV_16S, 1, 0)
        frame_y = cv2.Sobel(frame, cv2.CV_16S, 1, 0)

        frame_x = cv2.convertScaleAbs(frame_x)
        frame_y = cv2.convertScaleAbs(frame_y)

        img = cv2.addWeighted(frame_x, 0.1, frame_y, 0.1, 0)

        cv2.imshow('Sobel example', img)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()