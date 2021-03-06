import dlib
import cv2
import datetime
import argparse
import numpy as np
from collections import Counter

con_dict = {0:'RIGHT', 1:'CENTER_Hon', 2:'LEFT', 3:'UP', 4:'CENTER_Vertical', 5:'Down'}

parser = argparse.ArgumentParser()
parser.add_argument("-numpy_path", help='numpy path', type=str, default= "")
parser.add_argument("-landmarks_path", help='landmarks path', type=str, default= "")
args = parser.parse_args()

numpy_path = args.numpy_path
landmarks_path = args.landmarks_path

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_path)
time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 

def get_gaze_ratio(frame, gray, eye_points, facial_landmarks, eyename, time_now):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x, max_x = np.min(left_eye_region[:, 0]), np.max(left_eye_region[:, 0])
    min_y, max_y = np.min(left_eye_region[:, 1]), np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    cv2.imwrite('./content/'+eyename+time_now+'.png', gray_eye)
    
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY) 
    height, width = threshold_eye.shape

    ########## Left ##########
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    ########## Right ##########
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
     ########## Top ##########
    top_side_threshold = threshold_eye[0: int(height/2) , 0: width]
    top_side_white = cv2.countNonZero(top_side_threshold)
    ########## Bottom ##########
    bot_side_threshold = threshold_eye[int(height/2) : height, 0: width]
    bot_side_white = cv2.countNonZero(bot_side_threshold)

    ########## Right & Left ##########
    if left_side_white == 0:
      gaze_ratio = 1
    elif right_side_white == 0:
      gaze_ratio = 5
    else:
      gaze_ratio = round((left_side_white / right_side_white), 2)

    ########## Top & Bot ##########
    if top_side_white == 0:
      gaze_ratio_ver = 0.9
    elif bot_side_white == 0:
      gaze_ratio_ver = 5
    else:
      gaze_ratio_ver = round((top_side_white / bot_side_white), 2)
    return gaze_ratio, gaze_ratio_ver

def smile_point(parts):
    smile_lv = abs(parts[48].x - parts[54].x)
    mouthopen =  abs(parts[57].y - parts[51].y)
    return smile_lv, mouthopen

def eye_point(parts, left=True):
    if left:
        eyes = [ parts[36], min(parts[37], parts[38], key=lambda x: x.y), max(parts[40], parts[41], key=lambda x: x.y), parts[39],]
    else:
        eyes = [ parts[42], min(parts[43], parts[44], key=lambda x: x.y), max(parts[46], parts[47], key=lambda x: x.y), parts[45], ]
    org_y_top = eyes[1].y  # Top eyes,
    org_y_bot = eyes[2].y  # Down eye
    distancee = abs(org_y_top-org_y_bot)
    return distancee

def eyeclsration(landmarks):
  dis_smile, mouthopen = smile_point(landmarks.parts())
  dis_left_eye = eye_point(landmarks.parts())
  dis_right_eye = eye_point(landmarks.parts(), False)
  return dis_left_eye, dis_right_eye, dis_smile, mouthopen

def eyemove(numpy_path, detector, predictor, time_now):
  imgs = np.load(numpy_path)
  i = 0
  hons_idx, vers_idx = [], []
  for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        ####### Cal Gaze Ratio #######
        gaze_Hon_left_eye, gaze_vertical_left = get_gaze_ratio(img, gray, [36, 37, 38, 39, 40, 41], landmarks, 'Left eye', time_now)
        gaze_Hon_right_eye, gaze_vertical_right = get_gaze_ratio(img, gray, [42, 43, 44, 45, 46, 47], landmarks, 'Right eye', time_now)
        ####### cal openclose ratio #######
        dis_left_eye, dis_right_eye, dis_smile, mouthopen = eyeclsration(landmarks)
        gaze_ratio = (gaze_Hon_left_eye + gaze_Hon_right_eye)/2
        gaze_vertical = (gaze_vertical_left + gaze_vertical_right)/2
        ####### Horizontal #######
        if gaze_ratio <= 1:
          pos_h, pos_h_idx = "RIGHT", 0
        elif 1 < gaze_ratio < 3:
          pos_h, pos_h_idx = "CENTER", 1
        else:
          pos_h, pos_h_idx = "LEFT", 2
        ####### Vertical #######
        if gaze_vertical <= 0.4:
          pos_v, pos_v_idx = "UP", 3
        elif 0.4 < gaze_vertical < 1:
          pos_v, pos_v_idx = "CENTER", 4
        else:
          pos_v, pos_v_idx = "Down", 5
        
        hons_idx.append(pos_h_idx)
        vers_idx.append(pos_v_idx)
        print('Sample:', str(i), ' Horizontal: ', pos_h, ' Vertical: ', pos_v)
        i+=1
  hon, ver = Counter(hons_idx), Counter(vers_idx)
  result_hon = hon.most_common(1)[0][0]
  result_ver = ver.most_common(1)[0][0]
  dic_h, dic_v = con_dict[result_hon], con_dict[result_ver]
  print('Horizontal: ', dic_h, 'Vertical: ', dic_v)
  return dic_h, dic_v

pos_h, pos_v = eyemove(numpy_path, detector, predictor, time_now)