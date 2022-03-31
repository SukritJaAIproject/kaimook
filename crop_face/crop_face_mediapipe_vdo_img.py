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
parser.add_argument("-img_path", help='img path', type=str, default= "")
parser.add_argument("-input_type", help='input type', type=str, default= "img")
parser.add_argument("-vdo_path", help='vdo_path path', type=str, default= "")
parser.add_argument("-img_size", help='(height, width) size', type=int, default= 224)
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
  cap.set(cv2.CAP_PROP_FPS, 5)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = cap.get(cv2.CAP_PROP_FPS)
  return total_frames, fps

def crop_face_med(img_path, vdo_path, img_size, out_path, input_type):
  if input_type == 'img':
    print("##########  img ##########")
    img = cv2.imread(img_path)
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
          try:
            brect = calc_bounding_rect(img, face_landmarks)
            #[y:y+h, x:x+w]
            y, y_h, x, x_w = brect[1], brect[3], brect[0], brect[2]
            img = img[brect[1]:brect[3],brect[0]:brect[2]]
            #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)    
          except:
            print('Error')
    return (cropped, [y, y_h, x, x_w])
  else:
    print("########## video ##########")
    T_frames, fps = caminfo(vdo_path)
    print('FPS = ', fps, ' Total Frames = ', T_frames)
    cropped_face_numpy = []    
    for j in tqdm(range(0, T_frames, 5)):
      cap = cv2.VideoCapture(vdo_path)
      cap.set(cv2.CAP_PROP_FPS, 5)
      cap.set(cv2.CAP_PROP_POS_FRAMES, j)
      success, img = cap.read()    
      try:
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
                  #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                  #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
                  #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                  #cropped = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)           
                except:
                  print('Error: No face detected')
          cropped_face_numpy.append(cropped)
      except:
          print('flags')
    resized_imgs = np.array(cropped_face_numpy)
    frameSize = (img_size, img_size)
    out = cv2.VideoWriter(out_path+'Crop_Video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, frameSize)
    for i in range(resized_imgs.shape[0]):
        img = tf.image.resize_with_pad(resized_imgs[i], target_height=img_size, target_width=img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False).numpy().astype('uint8')
        out.write(img)
    out.release()  
    return resized_imgs

out_path = args.out_path
input_type = args.input_type

if input_type == 'vdo':
    resized_img = crop_face_med(args.img_path, args.vdo_path, args.img_size, args.out_path, args.input_type)
    print(resized_img.shape)
    np.save(out_path+time_now+'.npy', resized_img)
else :
    result = crop_face_med(args.img_path, args.vdo_path, args.img_size, args.out_path, args.input_type)
    cv2.imwrite(out_path+time_now+'.png', result[0])