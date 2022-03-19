import copy
import argparse
import cv2
import numpy as np
from collections import Counter
import datetime
from utils import CvFpsCalc
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark

time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
con_dict = {0:'RIGHT', 1:'LEFT', 2:'UP', 3:'Down'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy_path", help='numpy path', type=str, default="")
    parser.add_argument("--fix_size", help='fix size', type=bool, default=False)
    parser.add_argument("--img_size", help='image size', type=int, default=72)
    parser.add_argument("--scale_percent", help='scale_percent', type=int, default=80)
    parser.add_argument("--out_path", help='output path', type=str, default="")
    args = parser.parse_args()
    return args

def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,left_eye_x1:left_eye_x2])
    eye_contour, iris = iris_detector(left_eye_image)
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)

    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,right_eye_x1:right_eye_x2])
    eye_contour, iris = iris_detector(right_eye_image)
    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)
    return left_iris, right_iris, left_eye_image, right_eye_image

def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] * ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] * ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]
        iris_list.append((point_x, point_y))
    return iris_list

def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    return center, radius

def draw_debug_image( debug_image, left_iris, right_iris, left_center, left_radius, right_center, right_radius,):
  cv2.circle(debug_image, left_center, left_radius, (0, 255, 0), 2)
  cv2.circle(debug_image, right_center, right_radius, (0, 255, 0), 2)

  for point in left_iris:
      cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)
  for point in right_iris:
      cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)

  cv2.putText(debug_image, 'r:' + str(left_radius) + 'px', (left_center[0] + int(left_radius * 1.5), left_center[1] + int(left_radius * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
  cv2.putText(debug_image, 'r:' + str(right_radius) + 'px', (right_center[0] + int(right_radius * 1.5), right_center[1] + int(right_radius * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
  return debug_image

def eyemove(img_path, fix_size, img_size, scale_percent, out_path):
  max_num_faces = 2
  min_detection_confidence, min_tracking_confidence = 0.6, 0.6
  face_mesh = FaceMesh( max_num_faces, min_detection_confidence, min_tracking_confidence,)
  iris_detector = IrisLandmark()
  cvFpsCalc = CvFpsCalc(buffer_len=10)
  display_fps = cvFpsCalc.get()
  imgs = np.load(img_path)
  i = 0
  pos_h_idxs, pos_v_idxs = [], []
  for img in imgs:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    image = cv2.flip(image, 1) 
    debug_image = copy.deepcopy(image)

    face_results = face_mesh(image)
    for face_result in face_results:
        left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)
        left_iris, right_iris, left_eye_img, right_eye_img = detect_iris(image, iris_detector, left_eye, right_eye)
        L_x1 = max(left_iris[0][0], left_iris[1][0], left_iris[2][0], left_iris[3][0], left_iris[4][0], 0)
        L_y1 = max(left_iris[0][1], left_iris[1][1], left_iris[2][1], left_iris[3][1], left_iris[4][1], 0)
        L_x2 = min(left_iris[0][0], left_iris[1][0], left_iris[2][0], left_iris[3][0], left_iris[4][0], image.shape[0])
        L_y2 = min(left_iris[0][1], left_iris[1][1], left_iris[2][1], left_iris[3][1], left_iris[4][1], image.shape[1])
        # iris_left = image[L_y2:L_y1, L_x2:L_x1]
        w_L,  h_L = left_eye_img.shape[0], left_eye_img.shape[1]

        ######### Left #########
        iris_left_Left = left_eye_img[0: h_L, 0: int(w_L / 2)]
        _, IMG_THD_L1 = cv2.threshold(iris_left_Left, 70, 255, cv2.THRESH_BINARY) 
        THD_L1, CZ_L1 = cv2.countNonZero(IMG_THD_L1), np.sum(IMG_THD_L1 == 0) 
        #print('Left => CountNonZero: ', THD_L1, ' Countzero: ', CZ_L1)
        ######### Right #########
        iris_left_Right = left_eye_img[0: h_L, int(w_L / 2): w_L]
        _, IMG_THD_L2 = cv2.threshold(iris_left_Right, 70, 255, cv2.THRESH_BINARY) 
        THD_L2, CZ_L2 = cv2.countNonZero(IMG_THD_L2), np.sum(IMG_THD_L2 == 0)
        #print('Right => CountNonZero: ', THD_L2,' Countzero: ', CZ_L2)
        ######### Top #########
        iris_left_Top = left_eye_img[0: int(h_L/2) , 0: w_L]
        _, IMG_THD_L3 = cv2.threshold(iris_left_Top, 70, 255, cv2.THRESH_BINARY) 
        THD_L3, CZ_L3 = cv2.countNonZero(IMG_THD_L3), np.sum(IMG_THD_L3 == 0)
        #print('Top => CountNonZero: ', THD_L3, 'Countzero: ', CZ_L3)
        ######### Bottom #########
        iris_left_Bottom = left_eye_img[int(h_L/2) : h_L, 0: w_L]
        _, IMG_THD_L4 = cv2.threshold(iris_left_Bottom, 70, 255, cv2.THRESH_BINARY) 
        THD_L4, CZ_L4 = cv2.countNonZero(IMG_THD_L4), np.sum(IMG_THD_L4 == 0)
        #print('Bottom => CountNonZero: ', THD_L4, 'Countzero: ', CZ_L4)
        ######### Ratio L #########
        UD_ratio_L, RL_ratio_L = CZ_L3/CZ_L4, CZ_L2/CZ_L1
        #print('UP_down_ratio: ', UD_ratio_L, ' RL_ratio: ', RL_ratio_L)
        
        R_x1 = max(right_iris[0][0], right_iris[1][0], right_iris[2][0], right_iris[3][0], right_iris[4][0], 0)
        R_y1 = max(right_iris[0][1], right_iris[1][1], right_iris[2][1], right_iris[3][1], right_iris[4][1], 0)
        R_x2 = min(right_iris[0][0], right_iris[1][0], right_iris[2][0], right_iris[3][0], right_iris[4][0], image.shape[0])
        R_y2 = min(right_iris[0][1], right_iris[1][1], right_iris[2][1], right_iris[3][1], right_iris[4][1], image.shape[1])
        # iris_right = image[R_y2:R_y1, R_x2:R_x1]
        w_R,  h_R = right_eye_img.shape[0], right_eye_img.shape[1]

        ######### Left #########
        R1 = right_eye_img[0: h_R, 0: int(w_R / 2)]
        _, IMG_THD_R1 = cv2.threshold(R1, 70, 255, cv2.THRESH_BINARY) 
        THD_R1, CZ_R1 = cv2.countNonZero(IMG_THD_R1), np.sum(IMG_THD_R1 == 0) 
        #print('Left => CountNonZero: ', THD_R1, ' Countzero: ', CZ_R1)
        ######### Right #########
        R2 = right_eye_img[0: h_R, int(w_R / 2): w_R]
        _, IMG_THD_R2 = cv2.threshold(R2, 70, 255, cv2.THRESH_BINARY) 
        THD_R2, CZ_R2 = cv2.countNonZero(IMG_THD_R2), np.sum(IMG_THD_R2 == 0)
        #print('Right => CountNonZero: ', THD_R2, ' Countzero: ', CZ_R2)
        ######### Top #########
        R3 = right_eye_img[0: int(h_R/2) , 0: w_R]
        _, IMG_THD_R3 = cv2.threshold(R3, 70, 255, cv2.THRESH_BINARY) 
        THD_R3, CZ_R3 = cv2.countNonZero(IMG_THD_R3), np.sum(IMG_THD_R3 == 0)
        #print('Top => CountNonZero: ', THD_R3, 'Countzero: ', CZ_R3)
        ######### Bottom #########
        R4 = right_eye_img[int(h_R/2) : h_R, 0: w_R]
        _, IMG_THD_R4 = cv2.threshold(R4, 70, 255, cv2.THRESH_BINARY) 
        THD_R4, CZ_R4 = cv2.countNonZero(IMG_THD_R4), np.sum(IMG_THD_R4 == 0)
        #print('Bottom => CountNonZero: ', THD_R4, 'Countzero: ', CZ_R4)
        ######### Ratio L #########
        UD_ratio_R, RL_ratio_R = CZ_R3/CZ_R4, CZ_R2/CZ_R1
        #print('UP_down_ratio: ', UD_ratio_R, ' RL_ratio: ', RL_ratio_R)

        Vertical_ratio = (UD_ratio_L+UD_ratio_R)/2
        Horizontal_ratio = (RL_ratio_L+RL_ratio_R)/2

        ####### Horizontal #######
        if Horizontal_ratio > 1:
          pos_h, pos_h_idx = "RIGHT", 0
        else:
          pos_h, pos_h_idx = "LEFT", 1
        ####### Vertical #######
        if Vertical_ratio > 1:
          pos_v, pos_v_idx = "UP", 2
        else:
          pos_v, pos_v_idx = "Down", 3

        print('sample:', str(i), ' Horizontal: ', pos_h, ' Vertical: ', pos_v)
        pos_h_idxs.append(pos_h_idx)
        pos_v_idxs.append(pos_v_idx)
        i+=1

        left_center, left_radius = calc_min_enc_losingCircle(left_iris)
        right_center, right_radius = calc_min_enc_losingCircle(right_iris)
        debug_image = draw_debug_image( debug_image, left_iris, right_iris, left_center, left_radius, right_center, right_radius,)
        if fix_size == True:
          resized_img = cv2.resize(debug_image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        else:
          width, height = int(debug_image.shape[1] * scale_percent / 100), int(debug_image.shape[0] * scale_percent / 100)
          dim = (width, height)
          resized_img = cv2.resize(debug_image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(out_path+time_now+'.png', resized_img)
  hon, ver = Counter(pos_h_idxs), Counter(pos_v_idxs)
  result_hon = hon.most_common(1)[0][0]
  result_ver = ver.most_common(1)[0][0]

  dic_h, dic_v = con_dict[result_hon], con_dict[result_ver]
  print('Result Horizontal: ', dic_h, 'Result Vertical: ', dic_v)
  return resized_img, dic_h, dic_v
  
args = get_args()
numpy_path = args.numpy_path
fix_size = args.fix_size
img_size = args.img_size
scale_percent = args.scale_percent
out_path = args.out_path

resized_img, pos_h, pos_v = eyemove(numpy_path, fix_size, img_size, scale_percent, out_path)
