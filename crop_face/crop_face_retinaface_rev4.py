import cv2
import argparse
import numpy as np
import pandas as pd
from feat import Detector
import datetime
time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 

face_model = "retinaface"
landmark_model = "MobileNet"
au_model = "svm"
emotion_model = "resmasknet" 
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)

parser = argparse.ArgumentParser()
parser.add_argument("-numpy_path", help='numpy path',  type=str, default= "")
parser.add_argument("-scale", help='image scale', type=int, default= 50)
parser.add_argument("-out_path", help='output path', type=str, default="")
args = parser.parse_args()

bboxs, facial_img = [], []
def crop_face(numpy_path, scale_percent):
  imgs = np.load(numpy_path)
  for img in imgs:
    face_img = detector.detect_faces(img)
    pos = face_img[0][0][:4] 
    bbox = [pos[1], pos[3], pos[0], pos[2]]
    face =img[int(pos[1]):int(pos[3]),int(pos[0]):int(pos[2])] 
    try:
      width = int(face.shape[1] * scale_percent / 100)
      height = int(face.shape[0] * scale_percent / 100)
      resized_img = cv2.resize(face, (width, height), interpolation = cv2.INTER_AREA)
    except:
      print("No face detected")
      return 'No face detected'
    bboxs.append(bbox)
    facial_img.append(resized_img)
  bboxs_result = np.array(bboxs)
  facial_img_result = np.array(facial_img)
  return facial_img_result, bboxs_result

numpy_path = args.numpy_path
scale = args.scale
out_path = args.out_path

facial_img, bboxs = crop_face(numpy_path, scale)
print('facial_img: ', facial_img.shape)
print('bboxs: ', bboxs.shape, ' bboxs: ', bboxs)






