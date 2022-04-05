import os
import json
from torchvision.datasets.utils import download_url
from feat.utils import (get_resource_path,align_face_68pts,BBox)
from skimage.feature import hog
import numpy as np
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
import cv2
import torch
from feat.landmark_detectors.basenet_test import MobileNet_GDConv

landmark_detector = MobileNet_GDConv(136)

def downloadau(au_model):
    # download au model
    with open(os.path.join(get_resource_path(), "model_list.json"), "r") as f:
        model_urls = json.load(f)

    if au_model:
        for url in model_urls["au_detectors"][au_model.lower()]["urls"]:
            download_url(url, get_resource_path())
            if ".zip" in url:
                import zipfile

                with zipfile.ZipFile(os.path.join(get_resource_path(), "JAANetparams.zip"), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(get_resource_path()))
            if au_model.lower() in ['logistic', 'svm', 'rf']:
                download_url(
                    model_urls["au_detectors"]['hog-pca']['urls'][0], get_resource_path())
                download_url(
                    model_urls["au_detectors"]['au_scalar']['urls'][0], get_resource_path())

def detect_aus(frame, landmarks, au_model):
    return au_model.detect_au(frame, landmarks)

def extract_hog(frame, orientation=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False):
    hog_output = hog(frame, orientations=orientation, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, visualize=visualize, multichannel=True)
    if visualize:
        return (hog_output[0], hog_output[1])
    else:
        return hog_output

def extract_face(frame, detected_faces, landmarks, size_output=112):
    detected_faces = np.array(detected_faces)
    landmarks = np.array(landmarks)
    detected_faces = detected_faces.astype(int)

    aligned_img, new_landmarks = align_face_68pts(frame, landmarks.flatten(), 2.5, img_size=size_output)

    hull = ConvexHull(new_landmarks)
    mask = grid_points_in_poly(shape=np.array(aligned_img).shape,
                               verts=list(zip(new_landmarks[hull.vertices][:, 1],
                                              new_landmarks[hull.vertices][:, 0])))

    mask[0:np.min([new_landmarks[0][1], new_landmarks[16][1]]),
    new_landmarks[0][0]:new_landmarks[16][0]] = True
    aligned_img[~mask] = 0
    resized_face_np = aligned_img
    resized_face_np = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)
    return resized_face_np, new_landmarks


def detect_landmarks(frame, detected_faces):
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    landmark_detector.eval()
    out_size = 224

    height, width, _ = frame.shape
    landmark_list = []

    for k, face in enumerate(detected_faces):
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h]) * 1.2)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = frame[
                  new_bbox.top: new_bbox.bottom, new_bbox.left: new_bbox.right
                  ]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                int(dy),
                int(edy),
                int(dx),
                int(edx),
                cv2.BORDER_CONSTANT,
                0,
            )
        cropped_face = cv2.resize(cropped, (out_size, out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            continue
        test_face = cropped_face.copy()
        test_face = test_face / 255.0
        test_face = (test_face - mean) / std
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)

        input = torch.from_numpy(test_face).float()
        input = torch.autograd.Variable(input)

        landmark = landmark_detector(input).cpu().data.numpy()
        landmark = landmark.reshape(-1, 2)
        landmark = new_bbox.reprojectLandmark(landmark)
        landmark_list.append(landmark)

    return landmark_list