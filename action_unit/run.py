import cv2
import json
import dlib
import numpy as np
import argparse
from feat import Detector
import matplotlib.pyplot as plt
from au_modelm.download_model import *
from help_module.plot_aus import *
from feat.au_detectors.StatLearning.SL_test import RandomForestClassifier, LogisticClassifier, SVMClassifier
from feat.au_detectors.JAANet.JAA_test import JAANet
from feat.au_detectors.DRML.DRML_test import DRMLNet
from feat.utils import (jaanet_AU_presence,RF_AU_presence,)

face_model = "retinaface"
landmark_model = "MobileNet"
au_model = "svm"
emotion_model = "resmasknet" 
#detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)

parser = argparse.ArgumentParser()
parser.add_argument("-img_path", help='img_path', type=str, default= "")
args = parser.parse_args()

detectordlib = dlib.get_frontal_face_detector()
predictorlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def action_unit(img):
  for k in ["logistic", "svm", "rf", "JAANET", "DRML"]:
    downloadau(k) 

  #detector = Detector()
  detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
  au_model_svm = SVMClassifier() 
  au_model_rf = RandomForestClassifier() 
  au_model_logistic = LogisticClassifier() 
  au_model_JAANet = JAANet()
  au_model_DRMLNet = DRMLNet()

  img = cv2.imread(img)
  img_show = img.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  faces = detectordlib(gray)
  landmark_listxy = []
  for face in faces:
      x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
      detected_faces1 = [[x1, y1, x2, y2, 1]]
      cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
      landmarks = predictorlib(gray, face)
      for n in range(0, 68):
          x, y = landmarks.part(n).x, landmarks.part(n).y
          temp = [x, y]
          landmark_listxy.append(temp)
          cv2.circle(img_show, (x, y), 1, (255, 255, 0), -1) 
      landmark_listxy = [np.array(landmark_listxy)]

      #convex_hull, new_lands = detector.extract_face(frame=img, detected_faces=detected_faces1, landmarks=landmark_listxy, size_output=112)
      convex_hull, new_lands = detector.extract_face(frame=img, detected_faces=detected_faces1, landmarks=landmark_listxy, size_output=112)
      hogs = extract_hog(frame=convex_hull, visualize=False)
      au_svm = detect_aus(frame=hogs, landmarks=new_lands, au_model=au_model_svm)[0]
      au_logistic = detect_aus(frame=hogs, landmarks=new_lands, au_model=au_model_logistic)[0]
      au_rf = detect_aus(frame=hogs, landmarks=new_lands, au_model=au_model_rf)[0]
      au_JAANET = detect_aus(frame=img, landmarks=new_lands, au_model=au_model_JAANet)[0] #jaanet
      au_DRML = detect_aus(frame=img, landmarks=new_lands, au_model=au_model_DRMLNet)[0] #drml
      auoccur_col = jaanet_AU_presence
      
      imgs, result = plotau(au_svm, au_logistic, au_rf, au_JAANET, au_DRML, auoccur_col)
  return img_show, imgs, result
  


img = args.img_path
img_show, imgs, result = action_unit(img)

print(result)
print(result.keys())
print("")
print(result['svm'].keys())
print(result['svm'])
print("")
print(result['logistic'].keys())
print(result['logistic'])
print("")
print(result['rf'].keys())
print(result['rf'])
print("")
print(result['jaanet'].keys())
print(result['jaanet'])
print("")
print(result['DRML'][0].keys())
print(result['DRML'])

cv2.imshow('img_show', img_show)
cv2.imshow('imgs', imgs)
plt.plot()
plt.show()