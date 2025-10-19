from calendar import c
import torch
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
from enum import Enum, auto

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import albumentations
import albumentations.pytorch

from sklearn.metrics import mean_squared_error
import math
import shutil

def folder_make_func(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    elif "temp" in folder_name:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)


def CV_2_to_3ch(gray_image):
    # convert from BGR to RGB
    color_coverted = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return color_coverted



from data_preprocessing_methods_sb import *
import torch.nn.functional as F


import atexit


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(obj=state, f=filename)


def load_checkpoint(path, model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(path))

def PIL_to_CV_func(PIL_image):

    # use numpy to convert the pil_image into a numpy array
    numpy_image = np.array(PIL_image)
    # convert to a openCV2 image and convert from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def CV_to_PIL_func(cv_image):
    # convert from BGR to RGB
    color_coverted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # convert from openCV2 to PIL
    pil_image = Image.fromarray(color_coverted)

    return pil_image


# 데이터를 zip해서 모델 로드시 에러 안나게 하는 함수
def collate_fn(batch):
    return tuple(zip(*batch))

def get_JI(pred_m, gt_m):
    pred_m[pred_m > 0]=1
    gt_m[gt_m > 0] = 1

    pred_m = np.asarray(pred_m).astype(np.bool)
    gt_m = np.asarray(gt_m).astype(np.bool)

    intersection = np.logical_and(gt_m, pred_m)

    true_sum= gt_m[:,:].sum()
    pred_sum= pred_m[:,:].sum()
    intersection_sum = intersection[:,:].sum()

    ji = (intersection_sum + 1.) / (true_sum + pred_sum - intersection_sum + 1.)

    return ji

def histo_bland(error_diff,error_mean, save_path,category_name):
    # Classify the data into overestimate, underestimate, and accurate for all images
    overestimate_all = error_diff > 0.01
    underestimate_all = error_diff < -0.01
    accurate_all = np.logical_and(~overestimate_all, ~underestimate_all)

    # Plot the histogram of errors for each category for all images
    plt.hist(error_diff[overestimate_all], bins=50, alpha=0.5, label='Overestimate')
    plt.hist(error_diff[underestimate_all], bins=50, alpha=0.5, label='Underestimate')
    plt.hist(error_diff[accurate_all], bins=50, alpha=0.5, label='Accurate')
    plt.legend(loc='upper right')
    plt.xlabel('Error')

    plt.ylabel('Frequency')
    plt.ylim(0, 150)

    plt.savefig(save_path + '/{}_error_histogram.jpg'.format(category_name))
    plt.show()
    plt.close()

    # Bland-Altman plot analysis for all images
    mean_all = error_mean
    diff_all = error_diff
    plt.scatter(mean_all, diff_all)
    plt.axhline(y=np.mean(diff_all), color='black', linestyle='--')
    plt.xlabel('Mean')
    plt.ylim(-0.2, 0.2)
    plt.ylabel('Difference')
    plt.savefig(save_path + '/{}_bland.jpg'.format(category_name))
    plt.show()
    plt.close()

def RMSE_calc(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)

    return rmse


def BinaryConfusionMatrix(prediction, groundtruth):
    """Computes scores:
    TP = True Positives
    FP = False Positives
    FN = False Negatives
    TN = True Negatives
    return: TP, FP, FN, TN"""
    prediction = np.asarray(prediction).astype(np.bool)
    groundtruth = np.asarray(groundtruth).astype(np.bool)

    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return TN, FP, FN, TP


def get_precision(prediction, groundtruth):
    _, FP, _, TP = BinaryConfusionMatrix(prediction, groundtruth)
    precision = float(TP) / (float(TP + FP) + 1e-6)
    return precision


def get_recall(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    recall = float(TP) / (float(TP + FN) + 1e-6)
    return recall


def get_accuracy(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    accuracy = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
    return accuracy


def get_sensitivity(prediction, groundtruth):
    return get_recall(prediction, groundtruth)


def get_specificity(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    specificity = float(TN) / (float(TN + FP) + 1e-6)
    return specificity


def get_f1_score(prediction, groundtruth):
    precision = get_precision(prediction, groundtruth)
    recall = get_recall(prediction, groundtruth)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return f1_score


def get_dice(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    dice = 2 * float(TP) / (float(FP + 2 * TP + FN) + 1e-6)
    return dice


def get_iou1(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TP) / (float(FP + TP + FN) + 1e-6)
    return iou


def get_iou0(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TN) / (float(FP + TN + FN) + 1e-6)
    return iou


def get_mean_iou(prediction, groundtruth):
    iou0 = get_iou0(prediction, groundtruth)
    iou1 = get_iou1(prediction, groundtruth)
    mean_iou = (iou1 + iou0) / 2
    return mean_iou




def get_mcc(prediction, groundtruth):
    from math import sqrt
    tn, fp, fn, tp = BinaryConfusionMatrix(prediction, groundtruth)
    # https://stackoverflow.com/a/56875660/992687
    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return ((tp * tn) - (fp * fn)) / (sqrt(x) + 1e-6)


def thick_saveas(img, thick_as=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(img, kernel, iterations=thick_as)

    return dilate


def dice_calc(pred_mask, true_mask, non_seg_score=1.0):
    """
        Computes the Dice coefficient.
        Args:
            true_mask : Array of arbitrary shape.
            pred_mask : Array with the same shape than true_mask.

        Returns:
            A scalar representing the Dice coefficient between the two segmentations.

    """
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum



def plot_image_from_output(img, annotation, save_as):
    img = img*255 #0-1 to 0-255
    img = img.permute(1, 2, 0)
    img = np.array(img, np.int32)

    fig, ax = plt.subplots(1)
    ax.imshow((img), vmin=0, vmax=255)

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.annotate('artery', xy=(xmax - 40, ymin - 20),color='red')

        elif annotation['labels'][idx] == 2:

            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.annotate('vein', xy=(xmax - 40, ymin - 20),color='blue')

        else:

            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')

        ax.add_patch(rect)

    plt.savefig(save_as, bbox_inches='tight')
    plt.close(fig)
    #plt.show()



def plot_image_from_output_withmask(img, annotation, save_as=False):
    img = img*255
    img = img.permute(1, 2, 0)
    img = np.array(img, np.int32)

    fig, ax = plt.subplots(1)

    masks = np.transpose(annotation["masks"].detach().cpu().numpy(), (1, 2, 0))

    _,_,channel_num= masks.shape
    for numm in range(channel_num):
        img[:, :, numm][masks[:,:,numm] > 0.5] = 255


    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.annotate('artery', xy=(xmax - 40, ymin + 20))

        elif annotation['labels'][idx] == 2:

            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.annotate('vein', xy=(xmax - 40, ymin + 20))


        else:

            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')

        ax.add_patch(rect)

    ax.imshow(img, vmin=0, vmax=255)
    if save_as == False:
        plt.show()
    else :
        plt.savefig(save_as, bbox_inches='tight')
        plt.close(fig)



def plot_image_from_output_withmask_inference(img, annotation, save_as = False,inferen=True, evaluate=False):
    img = img*255
    img = img.permute(1, 2, 0)
    img = np.array(img, np.int32)

    fig, ax = plt.subplots(1)



    if inferen == True:
        #print(annotation)
        pred_masks = annotation["masks"].detach().cpu().numpy()
        pred_labels = annotation["labels"].detach().cpu().numpy()
        pred_boxes = annotation["boxes"].detach().cpu().numpy()
        pred_scores = annotation["scores"].detach().cpu().numpy()

        #print('pred_mask>shape:', pred_masks.shape)
        #print('pred_mask_best_artery?>shape:', pred_masks[pred_labels.tolist().index(1)].shape)

        if evaluate != False :
            width, height = pred_masks[0, 0].shape
            mask_output = np.zeros((width, height ,evaluate))

        try:
            best_artery_index = pred_labels.tolist().index(1)
            artery_pred_box = pred_boxes[best_artery_index]
            artery_pred_score = pred_scores[best_artery_index]
            artery_pred_mask = pred_masks[best_artery_index, 0]

            if artery_pred_score > 0.8:

                #print('artery best score :', artery_pred_score)
                #print('artery best score :', artery_pred_mask.max())

                xmin, ymin, xmax, ymax = artery_pred_box
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.annotate('artery, ' + str(artery_pred_score), xy=(xmax - 40, ymin - 20), color='red')
                ax.add_patch(rect)

                #masks = np.transpose(artery_pred_mask, (1, 2, 0))
                img[:, :, 0][artery_pred_mask > 0.5] = 255

                if evaluate :
                    artery_pred_mask[artery_pred_mask > 0.5] = 1
                    artery_pred_mask[artery_pred_mask <= 0.5] = 0
                    mask_output[:,:,0] = artery_pred_mask


        except:
            pass
        try:
            best_vein_index = pred_labels.tolist().index(2)
            vein_pred_box = pred_boxes[best_vein_index]
            vein_pred_score = pred_scores[best_vein_index]
            vein_pred_mask = pred_masks[best_vein_index, 0]
            #print(vein_pred_mask.shape)

            if vein_pred_score > 0.8:

                xmin, ymin, xmax, ymax = vein_pred_box
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='b',
                                         facecolor='none')
                #print('vein best score :', vein_pred_score)
                #print('artery best score :', vein_pred_mask.max())

                ax.annotate('vein, ' + str(vein_pred_score), xy=(xmax - 40, ymin - 20), color='blue')
                ax.add_patch(rect)

                #masks = np.transpose(vein_pred_mask, (1, 2, 0))
                img[:, :, 2][vein_pred_mask > 0.5] = 255


                if evaluate :
                    vein_pred_mask[vein_pred_mask > 0.5] = 1
                    vein_pred_mask[vein_pred_mask > 0.5] != 0
                    mask_output[:,:,1] = vein_pred_mask


        except:
            pass


    ax.imshow(img, vmin=0, vmax=255)
    if save_as == False:
        plt.show()
    else :
        plt.savefig(save_as, bbox_inches='tight')
        plt.close(fig)

    return mask_output


class State(Enum):
    Ready = auto()
    Load = auto()
    Play = auto()
    Calc = auto()
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name



#CV 버전
class dodraw_rectangle_ROI():
    def __init__(self):
        # 초기 설정
        self.setROI_x1, self.setROI_y1, self.setROI_x2, self.setROI_y2 = 0, 0, 0, 0
        self.drawing = False
        self.roi_selected = False # ROI가 선택되었는지 확인하는 플래그
        self.setROI_coordinate = None

    def draw_rectangle_ROI(self, event, setROI_x, setROI_y, flags, param):
    
        #global setROI_x1, setROI_y1, setROI_x2, setROI_y2, drawing, setROI_image
        if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 누름
            self.drawing = True
            self.setROI_x1, self.setROI_y1 = setROI_x, setROI_y

        elif event == cv2.EVENT_MOUSEMOVE: # 마우스를 움직임
            if self.drawing == True:
                temp_setROI_image = param['image_before'].copy()
                cv2.rectangle(temp_setROI_image, (self.setROI_x1, self.setROI_y1), (setROI_x, setROI_y), (0, 255, 0), 2)
                cv2.imshow("setROI_image", temp_setROI_image)

        elif event == cv2.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼을 뗌
            self.drawing = False
            self.setROI_x2, self.setROI_y2 = setROI_x, setROI_y
            
            #cv2.rectangle(param['image'], (self.setROI_x1, self.setROI_y1), (self.setROI_x2, self.setROI_y2), (0, 255, 0), 2) #rectbox를 계속 띄워둘때
            setROI_imag_final = param['image'].copy()    # 한번하고 rectbox초기화할때
            cv2.rectangle(setROI_imag_final, (self.setROI_x1, self.setROI_y1), (self.setROI_x2, self.setROI_y2), (0, 255, 0), 2)
            cv2.imshow("setROI_image", setROI_imag_final)

            # 사용자가 선택한 영역의 좌표 출력
            print("Selected ROI Coordinates: setROI_x1: {}, setROI_y1: {}, setROI_x2: {}, setROI_y2: {}".format(self.setROI_x1, self.setROI_y1, self.setROI_x2, self.setROI_y2))
            self.setROI_coordinate = (self.setROI_x1, self.setROI_y1, self.setROI_x2, self.setROI_y2)
            # 선택한 영역을 이미지로 저장
            #roi = param['image'][self.setROI_y1:self.setROI_y2, self.setROI_x1:self.setROI_x2]

            #cv2.imwrite('selected_roi.png', roi)
        elif event == cv2.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼을 누름
            if self.setROI_x2 and self.setROI_y2: # 유효한 ROI가 있는 경우
                if self.roi_selected: # ROI 선택 플래그를 참으로 설정
                    self.setROI_coordinate = self.setROI_x1, self.setROI_y1, self.setROI_x2, self.setROI_y2
                cv2.destroyAllWindows() # 창을 닫음

                
        

import sys
# from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QVBoxLayout, QWidget
# from PyQt5.QtCore import Qt, QPoint
# from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage

# class CustomLabel(QLabel):
#     def __init__(self, cvImage, parent=None):
#         super(CustomLabel, self).__init__(parent)
#         self.startPoint = QPoint()
#         self.endPoint = QPoint()
#         self.isDrawing = False
        
#         self.cvImage = cvImage
#         self.qImage = self.pyqt_display_func(self.cvImage)
#         self.display_position =  self.labelImage
#         self.display_position.setPixmap(QPixmap.fromImage( self.qImage).scaled(self.display_position.width(), self.display_position.height(), aspectRatioMode=Qt.KeepAspectRatio))
#         #self.qImage = self.convertCvToQt(self.cvImage)
#         #self.setPixmap(QPixmap.fromImage(self.qImage))

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.startPoint = event.pos()
#             self.endPoint = self.startPoint
#             self.isDrawing = True

#     def mouseMoveEvent(self, event):
#         if self.isDrawing:
#             self.endPoint = event.pos()
#             self.update()

#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton and self.isDrawing:
#             self.endPoint = event.pos()
#             self.isDrawing = False
#             self.update()
#             self.cropImage(self.startPoint, self.endPoint)

#     def paintEvent(self, event):
#         super().paintEvent(event)
#         if not self.startPoint.isNull() and not self.endPoint.isNull():
#             painter = QPainter(self)
#             painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
#             painter.drawRect(self.startPoint.x(), self.startPoint.y(), self.endPoint.x() - self.startPoint.x(), self.endPoint.y() - self.startPoint.y())

#     def convertCvToQt_fix(self, cvImg):
#         rgbImage = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgbImage.shape
#         bytesPerLine = ch * w
#         convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
#         return convertToQtFormat.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
    
#     def pyqt_display_func(self, before_img):
#         img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
#         height, width, channel = img.shape
#         bytesPerLine = 3*width
#         qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#         return qImg


#     def cropImage(self, startPoint, endPoint):
#         # Crop the selected area
#         x1, y1 = startPoint.x(), startPoint.y()
#         x2, y2 = endPoint.x(), endPoint.y()
#         croppedImg = self.cvImage[y1:y2, x1:x2]
#         croppedQImg = self.pyqt_display_func(croppedImg)
#         self.display_position.setPixmap(QPixmap.fromImage(croppedQImg).scaled(self.display_position.width(), self.display_position.height(), aspectRatioMode=Qt.KeepAspectRatio))
        
