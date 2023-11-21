import os
import time
os.path.abspath(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
import pyrootutils
# _file__ = '/Users/tiendzung/Project/facial_landmarks-wandb/src/app/app.py'
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
config_path = str(path / "configs")
output_path = path / "outputs"
data_dir = path / "data"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
from omegaconf import DictConfig

from src.app import filter_processor as fp

from src.models.components.simple_resnet import SimpleResnet
from src.models.dlib_module import DlibLitModule
from facenet_pytorch import MTCNN

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2


transform = Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

# config for filter
filters_config = {
    'anonymous':
        [{'path': "filters/anonymous.png",
          'anno_path': "filters/anonymous_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'squid_game_front_man':
        [{'path': "filters/squid_game_front_man.png",
          'anno_path': "filters/squid_game_front_man.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "filters/dog-ears.png",
          'anno_path': "filters/dog-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/dog-nose.png",
          'anno_path': "filters/dog-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "filters/cat-ears.png",
          'anno_path': "filters/cat-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/cat-nose.png",
          'anno_path': "filters/cat-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "filters/jason-joker.png",
          'anno_path': "filters/jason-joker_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "filters/flower-crown.png",
          'anno_path': "filters/flower-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}

# create the output video file
def init_video_writer(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(os.path.join(output_path,'output.mp4'), fourcc, fps, frame_size)


@hydra.main(version_base="1.3", config_path=config_path, config_name="app")
def main(cfg: DictConfig):
    filter = filters_config[cfg.filter_name]
    print(cfg.filter_name)
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        mps_device = torch.device("cpu")
    else:
        mps_device = torch.device("mps") ##change to torch.device("mps") if you want to use Apple Silicon GPU

    print(mps_device)

    net = SimpleResnet()
    net.to(mps_device)

    model = DlibLitModule.load_from_checkpoint(checkpoint_path=cfg.ckpt_path, net = net)
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = mps_device)
    # return

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/tiendzung/Downloads/record-webcam.mov')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,700) #640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,500) #480
    # print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) # 1920
    # print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 1080
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width/2) #960
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height/2) #540
    video_writer = init_video_writer(cap)
    import logging
    # filter = []
    prev_time = time.time()
    isFirstFrame = True
    while cap.isOpened():
        # cur_time = time.time()
        # print(1 / (cur_time - prev_time))
        # prev_time = cur_time
        isSuccess, frame = cap.read()

        if isSuccess:
            # print(filter[0]['path'])
            frame = np.pad(frame, ((200, 200), (200, 200), (0, 0)), mode='constant', constant_values=0)
            boxes, _ = mtcnn.detect(frame)
            faces = mtcnn(frame)
            
            if boxes is not None:
                face_box = []
                for box in boxes:
                    bbox = list(map(int,box.tolist())) ##x1, y1, x2, y2
                    face_box.append(bbox)
                    # frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3) ##Draw Bounding box with the coordinates

                for j, face in enumerate(faces):
                    face = face.permute(1, 2, 0).numpy()*255
                    h = face_box[j][3] - face_box[j][1]
                    w = face_box[j][2] - face_box[j][0]
                    landmarks = model.forward(transform(image = face)["image"].unsqueeze(0).to(mps_device))[0].to("cpu")
                    landmarks = (landmarks + 0.5) * torch.Tensor([w, h])
                    landmarks = landmarks + torch.Tensor([face_box[j][0], face_box[j][1]])
                    landmarks = landmarks.detach().numpy()

                    ## Add 2 points to landmarks
                    landmarks = np.append(landmarks, [[face_box[j][0], face_box[j][1]]], axis = 0)
                    landmarks = np.append(landmarks, [[face_box[j][0] + w, face_box[j][1]]], axis = 0)

                    # points2 = landmarks.tolist()
                    ################ Optical Flow and Stabilization Code #####################
                    img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if isFirstFrame:
                        points2Prev = np.array(landmarks, np.float32)
                        img2GrayPrev = np.copy(img2Gray)
                        isFirstFrame = False

                    lk_params = dict(winSize=(101, 101), maxLevel=15,
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
                    points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                                    np.array(landmarks, np.float32),
                                                                    **lk_params)

                    # Final landmark points are a weighted average of detected landmarks and tracked landmarks

                    for k in range(0, len(landmarks)):
                        d = cv2.norm(np.array(landmarks[k]) - points2Next[k])
                        alpha = np.exp(-d * d / 100)
                        landmarks[k] = (1 - alpha) * np.array(landmarks[k]) + alpha * points2Next[k]
                        landmarks[k] = fp.constrainPoint(landmarks[k], frame.shape[1], frame.shape[0])
                        landmarks[k] = (int(landmarks[k][0]), int(landmarks[k][1]))

                    # Update variables for next pass
                    points2Prev = np.array(landmarks, np.float32)
                    img2GrayPrev = img2Gray
                    ################ End of Optical Flow and Stabilization Code ###############

                    # for i in range (landmarks.shape[0]):
                    #     frame = cv2.circle(frame, (int(landmarks[i, 0]),int(landmarks[i, 1])), radius=1, color=(255, 255, 0), thickness= 1) ##Draw landmarks
                    
                    for i in range(len(filter)):
                        filter_img = cv2.imread(filter[i]['path'], cv2.IMREAD_UNCHANGED)
                        filter_points = fp.load_filter_points(filter[i]['anno_path'])

                        ## Get alpha channel
                        alpha = []
                        if filter[i]['has_alpha']:
                            b, g, r, alpha = cv2.split(filter_img)
                            filter_img = cv2.merge((b, g, r))

                        wrapImage = frame.copy()
                        mask = np.zeros((wrapImage.shape[0], wrapImage.shape[1], 3), dtype=np.float32)
                        alpha = cv2.merge((alpha, alpha, alpha))

                        if filter[i]['morph']:
                            filter_points = np.array(list(filter_points.values()))
                            # fp.wrap_affine(filter_img, wrapImage, filter_points, landmarks)
                            # fp.wrap_affine(alpha, mask, filter_points, landmarks)

                            frame = fp.apply_filter(frame, landmarks, filter_img, filter_points, alpha)
                        else:
                            dst_points = np.array((landmarks[int(list(filter_points.keys())[0])], landmarks[int(list(filter_points.keys())[1])]))
                            src_points = np.array(list(filter_points.values()))
                            affine_matrix = fp.calculate_affine_matrix_for_2_points(src_points, dst_points)

                            wrapImage = cv2.warpAffine(filter_img, affine_matrix, (wrapImage.shape[1], wrapImage.shape[0]))
                            mask = cv2.warpAffine(alpha, affine_matrix, (mask.shape[1], mask.shape[0]))
                            mask = cv2.GaussianBlur(mask, (3, 3), 10)

                            mask_r = (255., 255., 255.) - mask

                            mask = mask * (1.0/255)
                            mask_r = mask_r * (1.0/255)

                            frame = np.uint8(frame * mask_r + wrapImage * mask) ##if we dont have np.uint8, the image will be black
                            # frame[:,:,:] = frame * mask_r + wrapImage * mask ##i dont know why but if we use frame[:,:,:] then it will normal
                        # frame = output
            frame = frame[200:-200, 200:-200, :]
        # frame = frame[200:-200, 200:-200, :]

        video_writer.write(frame)
        cv2.imshow('Face Filter', frame)
        if cv2.waitKey(1)&0xFF == 27:
            print("YES")
            break
        if cv2.waitKey(1)&0xFF == ord('f'):
            try:
                filter = filters_config[next(iter_filters_config)]
            except:
                iter_filters_config = iter(filters_config)
                filter_name = next(iter_filters_config)
                filter = filters_config[filter_name]
                print(filter_name)

    cap.release()
    cv2.destroyAllWindows()
    for i in range(30):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

    # filter_points = fp.load_filter_points('filters/squid_game_front_man.csv')
    # print(filter_points)
    # img = cv2.imread('filters/squid_game_front_man.png')
    # cv2.imshow('Face Detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()