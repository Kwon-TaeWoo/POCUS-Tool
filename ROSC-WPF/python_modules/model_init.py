"""
Code Name  : semifinal_model_init.py
Author     : Subin Park (subinn.park@gmail.com)
Created on : 24. 2. 9. 오후 1:36
Desc       : 
"""
import albumentations as A
import cv2
import os
import pandas as pd
import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
import multiprocessing

from segmentation_models_pytorch import utils
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from model import VisionTransformer, CONFIGS
from utils import *
from calculate_CAC import process_images_realtime



def model_load_ViT(weight_save_path = 'checkpoint240225__output_noising_CLAHE_transunet_256_iter1.pth', target_size = (512, 512, 3)):
        
    TRAIN_IMG_SIZE = target_size #(256, 256, 3)
    ORIGINAL_IMG_SIZE = (500, 460)

    VAL_IMG_SIZE = TRAIN_IMG_SIZE
    TEST_IMG_SIZE = TRAIN_IMG_SIZE
    N_CLASSES = 3
    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = TRAIN_BATCH_SIZE
    TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
    NUM_EPOCHS = 100
    TRAIN_NUM_WORKERS = 3
    VAL_NUM_WORKERS = 3
    TEST_NUM_WORKERS = 3
    PIN_MEMORY = True
    LEARNING_RATE = 0.001
    TRAIN_NUM_WORKERS = 2
    DEVICE = 'cuda'

    LOAD_MODEL = True
    START_EPOCH = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zzroot_path = '../data/Synapse/train_npz'
    
    zzdataset = 'Synapse'
    zzlist_dir = './lists/lists_Synapse'
    zznum_classes = N_CLASSES
    zzmax_iterations = 30000
    zzmax_epochs = NUM_EPOCHS
    zzbatch_size = TRAIN_BATCH_SIZE
    zzn_gpu = 1
    zzdeterministic = 1
    zzbase_lr = LEARNING_RATE
    zzimg_size = TRAIN_IMG_SIZE[0]
    zzseed = 1234
    zzn_skip = 3
    zzvit_name = 'R50-ViT-B_16'
    zzvit_patches_size = 16
        
    test_transforms = A.Compose([
        #A.Normalize(),
        #A.Resize(TEST_IMG_SIZE[0], TEST_IMG_SIZE[1]),
        #A.PadIfNeeded(min_height=TEST_IMG_SIZE[0], min_width=TEST_IMG_SIZE[1]),
        A.Resize(TEST_IMG_SIZE[0], TEST_IMG_SIZE[1]),
    ])


    if not zzdeterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(zzseed)
    np.random.seed(zzseed)
    torch.manual_seed(zzseed)
    torch.cuda.manual_seed(zzseed)
    dataset_name = zzdataset
    dataset_config = {
        'Synapse': {
            'root_path': '../input/project-transunet/project_TransUNet/data/Synapse/train_npz',
            'list_dir': '../input/project-transunet/project_TransUNet/TransUNet/lists/lists_Synapse',
            'num_classes': N_CLASSES,
        },
    }
    zznum_classes = N_CLASSES
    zzroot_path = dataset_config[dataset_name]['root_path']
    zzlist_dir = dataset_config[dataset_name]['list_dir']
    zzis_pretrain = True
    zzexp = 'TU_' + dataset_name + str(zzimg_size)
    snapshot_path = "../model/{}/{}".format(zzexp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if zzis_pretrain else snapshot_path
    snapshot_path += '_' + zzvit_name
    snapshot_path = snapshot_path + '_skip' + str(zzn_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        zzvit_patches_size) if zzvit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(zzmax_iterations)[
                                            0:2] + 'k' if zzmax_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(zzmax_epochs) if zzmax_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(zzbatch_size)
    snapshot_path = snapshot_path + '_lr' + str(zzbase_lr) if zzbase_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(zzimg_size)
    snapshot_path = snapshot_path + '_s' + str(zzseed) if zzseed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS[zzvit_name]
    config_vit.n_classes = zznum_classes
    config_vit.n_skip = zzn_skip
    if zzvit_name.find('R50') != -1:
        config_vit.patches.grid = (int(zzimg_size / zzvit_patches_size), int(zzimg_size / zzvit_patches_size))

    # VisionTransformer 모델 초기화
    model = VisionTransformer(config_vit, img_size=zzimg_size, num_classes=config_vit.n_classes).cuda()

    # 모델 체크포인트 로드
    # weight_save_path = './checkpoint240225__output_noising_CLAHE_transunet_256_iter1.pth' # 모델 체크포인트 파일 경로
    model.load_state_dict(torch.load(weight_save_path))

    model.eval() # 평가 모드로 설정
    return model, test_transforms

def predict_image(self, image, model):
    """
    model prediction 함수
    input : 
        image, model
    return : 
        prediction mask
    """    

    # 이미지 예측 로직
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = self.test_transforms(image=img)
    img = transformed['image']

    img = img / 255
    img = img.astype('float32')

    img = np.transpose(img, (2, 0, 1))
    
    imgs = torch.from_numpy(img)
    imgs = imgs.unsqueeze(0)
    preds = model(imgs.to('cuda'))

    data____ = np.argmax(preds[0].cpu().detach().numpy(), axis=0) * 120
    datad = cv2.resize(data____.astype('uint8'), (500, 460))
    #cv2.imwrite(image_path + os.path.basename(img_path).replace('.png', '_mask.jpg'), ddd)
    return datad
