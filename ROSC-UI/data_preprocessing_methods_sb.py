
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import torch.nn as nn

import numpy as np
import skimage
import cv2

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import albumentations
import albumentations.pytorch

import torch.nn.functional as F


import atexit

from utils import *


def image_preprocessing_normalize(image, save_as=False):
    # save_as는 file path임
    original_image = image
    try :
        _,_,_ = original_image.shape
    except:
        image = PIL_to_CV_func(image)


    # Normalize the image to the range [0, 255]
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    if save_as != False:
        # output path
        # cv2.imwrite(output_path_original+os.path.basename(png_file), image_file)
        cv2.imwrite(save_as, img_norm)

    try :
        _,_,_ = original_image.shape
        return img_norm
    except:
        norm_output = CV_to_PIL_func(img_norm)
        return norm_output


def image_preprocessing_CLAHE(image, save_as=False):
    # save_as는 file path임
    original_image = image
    try :
        _,_,_ = original_image.shape
    except:
        image = PIL_to_CV_func(image)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    eq = clahe.apply(gray)
    BGR_eq = CV_2_to_3ch(eq)

    if save_as != False:
        # output path
        # cv2.imwrite(output_path_original+os.path.basename(png_file), image_file)
        cv2.imwrite(save_as, BGR_eq)

    try :
        _,_,_ = original_image.shape
        return BGR_eq
    except:
        norm_output = CV_to_PIL_func(BGR_eq)
        return norm_output


def image_preprocessing_HQ(image, save_as=False):
    #histogram equalization
    # save_as는 file path임
    original_image = image
    try :
        _,_,_ = original_image.shape
    except:
        image = PIL_to_CV_func(image)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    img_eq = cv2.equalizeHist(gray)
    BGR_eq = CV_2_to_3ch(img_eq)

    if save_as != False:
        # output path
        # cv2.imwrite(output_path_original+os.path.basename(png_file), image_file)
        cv2.imwrite(save_as, BGR_eq)

    try :
        _,_,_ = original_image.shape
        return BGR_eq
    except:
        norm_output = CV_to_PIL_func(BGR_eq)
        return norm_output
