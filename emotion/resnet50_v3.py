import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from collections import Counter
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import argparse

convert_dic = {0:'angry', 1:'happy', 2:'neutral', 3:'relax', 4:'sad'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy_path", help='numpy path', type=str, default="")
    parser.add_argument("--model_path", help='model_path', type=str, default="model/")
    args = parser.parse_args()
    return args

def call_model():
  model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(72, 72, 3))
  x = model1.output
  x= Flatten()(x)
  x = Dense(5, activation='softmax')(x)
  model = Model(inputs=model1.input, outputs=x)
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  return model

def emtion_pilot(numpy_path, model_path):
  imgs = np.load(numpy_path)
  model = call_model()
  #model.load_weights(model_path)
  #model = tf.saved_model.load(model_path)
  model = tf.keras.models.load_model(model_path)
  result = []
  for img in imgs:
    x_test  = cv2.resize(img, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)
    x_test =  x_test[np.newaxis,:,:,:]
    x_pred =  model.predict(x_test.astype(float))
    x_pred = np.argmax(x_pred)
    result.append(x_pred)
  final_pred = Counter(result)
  final_val = final_pred.most_common(1)[0][0]
  final_val_txt =  convert_dic[final_val]
  print('Predicted emotion: ', final_val,', idx: ', final_val_txt)
  return final_val, final_val_txt
  
args = get_args()
numpy_path = args.numpy_path
model_path = args.model_path
res, img = emtion_pilot(numpy_path, model_path)