import cv2
import numpy as np
import mediapipe as mp
import argparse
import datetime
from tqdm import tqdm
import tensorflow as tf
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
parser.add_argument("-vdo_path", help='vdo_path path', type=str, default= "")
parser.add_argument("-img_size", help='(height, width) size', type=int, default= 72)
parser.add_argument("-scale", help='image scale', type=int, default=50)
parser.add_argument("-out_path", help='output path', type=str, default="")
args = parser.parse_args()

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
    
def caminfo(vdo_name):
  cap = cv2.VideoCapture(vdo_name)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = cap.get(cv2.CAP_PROP_FPS)
  return total_frames, fps

def crop_face_med(vdo_path, img_size, out_path):
  T_frames, fps = caminfo(vdo_path)
  print('FPS = ', fps, ' Total Frames = ', T_frames)
  cropped_face_numpy = []
  
  for j in tqdm(range(T_frames)):
    cap = cv2.VideoCapture(vdo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, j)
    success, img = cap.read()
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    try:
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
              try:
                brect = calc_bounding_rect(img, face_landmarks)
                img = img[brect[1]:brect[3],brect[0]:brect[2]]
                cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
              except:
                print('Error: No face detected')
                #return 'No face detected'
        cropped_face_numpy.append(cropped)
      resized_imgs = np.array(cropped_face_numpy)
      frameSize = (img_size, img_size) 
      out = cv2.VideoWriter(out_path+'Crop_Video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, frameSize)
    except:
        print('main error')
     
  for i in range(resized_imgs.shape[0]):
      img = tf.image.resize_with_pad(resized_imgs[i], target_height=img_size, target_width=img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False).numpy().astype('uint8')
      out.write(img)
  out.release()  
  return resized_imgs

resized_img = crop_face_med(args.vdo_path, args.img_size, args.out_path)
out_path = args.out_path
#cv2.imwrite(out_path+time_now+'.png', resized_img)
print(resized_img.shape)
np.save(out_path+time_now+'.npy', resized_img)

