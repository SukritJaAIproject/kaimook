# -*- coding: utf-8 -*-
"""crop_face_mediapipe.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1dfkPBecPXcOoX1o8pzHQmaDxvc4aK_2C
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import datetime
# from google.colab.patches import cv2_imshow
time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 

max_num_faces = 1
refine_landmarks = True
min_detection_confidence = 0.5
min_tracking_confidence = 0.5 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = max_num_faces,
    refine_landmarks = refine_landmarks,
    min_detection_confidence = min_detection_confidence,
    min_tracking_confidence = min_tracking_confidence,)

parser = argparse.ArgumentParser()
parser.add_argument("-numpy_path", help='numpy path', type=str, default= "")
parser.add_argument("-img_size", help='(height, width) size', type=int, default= 72)
parser.add_argument("-scale", help='image scale', type=int, default=50)
parser.add_argument("-out_path", help='output path', type=str, default="")

args = parser.parse_args()
#print("img_path:", args.img_path)
#print("img_size:", args.img_size)
#print("output:", args.output)

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def crop_face_med(numpy_path, img_size):
  imgs = np.load(numpy_path)
  print(imgs.shape)
  cropped_face_numpy = []
  for img in imgs:
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
          try:
            brect = calc_bounding_rect(img, face_landmarks)
            img = img[brect[1]:brect[3],brect[0]:brect[2]]
            cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
          except:
            print('Error')
            return 'No face detected'
    cropped_face_numpy.append(cropped)
  resized_imgs = np.array(cropped_face_numpy)
  return resized_imgs

#img_path = '/content/anger.jpg'
#img_size = 72 
#resized_img = crop_face_med(img_path, img_size)
#cv2_imshow(resized_img)

resized_img = crop_face_med(args.numpy_path, args.img_size)
out_path = args.out_path
#cv2.imwrite(out_path+time_now+'.png', resized_img)
print(resized_img.shape)
np.save(out_path+time_now+'.npy', resized_img)

