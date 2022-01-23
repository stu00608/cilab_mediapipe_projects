import numpy as np
import cv2

def get_angle_from_gesture(landmark):
    # Input result.multi_hand_landmarks[0].landmark.
    joint = np.zeros((21, 3))
    for j, lm in enumerate(landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
    v = v2 - v1 # [20,3]
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    angle = np.degrees(angle) # Convert radian to degree
    angle = np.array([angle], dtype=np.float32)

    return joint, angle

##馬賽克
def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    馬賽克的實現原理是把圖像上某個像素點一定範圍鄰域內的所有點用鄰域內左上像素點的顏色代替，這樣可以模糊細節，但是可以保留大體的輪廓。
    :param frame: opencv frame
    :param int x :  馬賽克左頂點
    :param int y:  馬賽克右頂點
    :param int w:  馬賽克寬
    :param int h:  馬賽克高
    :param int neighbor:  馬賽克每一塊的寬
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 關鍵點0 減去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 關鍵點1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 關鍵點2 減去一個像素
            cv2.rectangle(frame, left_up, right_down, color, -1)