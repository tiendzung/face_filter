import cv2
import os
from imutils import face_utils, rotate_bound
import math
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import pyrootutils

import hydra
from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
from omegaconf import DictConfig

__file__ = '/Users/tiendzung/Project/facial_landmarks-wandb/src/app/run.py'
path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root")
config_path = str(path / "configs")
output_path = path / "outputs"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 
           'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png', 
           'images/santa_filter.png', 'images/hat.png', 'images/hat2.png',
           'images/glasses.png', 'images/glasses1.png']
filterIndex = 0

transform = Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

from src.models.components.simple_resnet import SimpleResnet
from src.models.dlib_module import DlibLitModule


def calculate_inclination(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl

@hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
def main(cfg: DictConfig):
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        mps_device = torch.device("cpu")
    else:
        mps_device = torch.device("mps")

    print(mps_device)

    net = SimpleResnet()
    net.to(mps_device)

    model = DlibLitModule.load_from_checkpoint(checkpoint_path=cfg.ckpt_path, net = net)
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = mps_device)

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/tiendzung/Downloads/record-webcam.mov')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    import logging
    import time
    prev_time = time.time()
    first = True
    boxes = None
    faces = None
    while cap.isOpened():
        cur_time = time.time()
        logging.info(1 / (cur_time - prev_time))
        prev_time = cur_time
        isSuccess, frame = cap.read()
        frame = np.pad(frame, ((200, 200), (200, 200), (0, 0)), mode='constant', constant_values=0)
        if isSuccess:
            # if first is True:
            #     boxes, _ = mtcnn.detect(frame)
            #     faces = mtcnn(frame)
            #     if boxes is not None:
            #         first = False
            
            boxes, _ = mtcnn.detect(frame)
            faces = mtcnn(frame)
            
            if boxes is not None:
                face_box = []
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face_box.append(bbox)
                    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)

                for j, face in enumerate(faces):
                    face = face.permute(1, 2, 0).numpy()*255
                    h = face_box[j][3] - face_box[j][1]
                    w = face_box[j][2] - face_box[j][0]
                    landmarks = model.forward(transform(image = face)["image"].unsqueeze(0).to(mps_device))[0].to("cpu")
                    landmarks = (landmarks + 0.5) * torch.Tensor([w, h])
                    landmarks = landmarks + torch.Tensor([face_box[j][0], face_box[j][1]])
                    landmarks = landmarks.detach().numpy()

                    for i in range (landmarks.shape[0]):
                        frame = cv2.circle(frame, (int(landmarks[i, 0]),int(landmarks[i, 1])), radius=1, color=(255, 255, 0), thickness= 1)
                    
                    incl =  calculate_inclination(landmarks[17], landmarks[26])

                    filterIndex = 4
                    # print(filters[filterIndex])
                    # Add FILTER to the frame
                    sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
                    sunglasses = rotate_bound(sunglasses, incl)
                    # sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGR2BGRA)
                    # print(sunglasses.shape)
                    sunglass_width = int(landmarks[16][0]-landmarks[0][0]) #26 17
                    sunglass_height = int(landmarks[29][1]-landmarks[21][1])
                    sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height))
                    transparent_region = sunglass_resized[:,:,:3] != (0, 0 ,0)
                    # print(sunglass_resized[:,:,:3][transparent_region])
                    s_point = 17
                    frame[int(landmarks[s_point][1]):int(landmarks[s_point][1])+sunglass_height, 
                        int(landmarks[0][0]):int(landmarks[0][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]


                    beard = cv2.imread(filters[6], cv2.IMREAD_UNCHANGED)
                    # print(beard.shape)
                    beard_width = int(landmarks[14][0]-landmarks[2][0]) #26 17
                    beard_height = int(beard_width*1.4)
                    beard = cv2.resize(beard, (beard_width, beard_height))
                    transparent_region = beard[:,:,3] != 0
                    # print(transparent_region)
                    s_point = 2
                    frame[int(landmarks[s_point][1]):int(landmarks[s_point][1])+beard_height, 
                        int(landmarks[s_point][0]):int(landmarks[s_point][0])+beard_width,:][transparent_region] = beard[:,:,:3][transparent_region]


        frame = frame[200:-200, 200:-200, :]
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1)&0xFF == 27:
            print("YES")
            break

    cap.release()
    cv2.destroyAllWindows()
    for i in range(30):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()