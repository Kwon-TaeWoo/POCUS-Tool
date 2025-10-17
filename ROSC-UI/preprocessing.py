import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import os

INTER_AREA = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
def image_crop(img, up_margin=80, down_margin=310, left_margin=100, right_margin=240):
    """
    화면에서 inference할 image area crop하는 function.
    상하 여백 값을 설정할 수 있으며, 좌우 여백은 동일한 값으로 crop됨.

    Args: 
        img(numpy array): 이미지 array
        up_offset: 위쪽 여백
        down_offset: 아래쪽 여백
        width_offset: 좌우 여백
    
    Return:
        crop_img(numpy array)
    """

    if len(img.shape) != 2:
        print("image is not a grayscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #return False

    height, width = img.shape

    if up_margin + down_margin >= height or left_margin+right_margin >= width:
        print("여백이 이미지 크기를 초과")
        return False

    # crop_width = width - (width_offset * 2)
    # crop_height = height - (up_offset + down_offset)
    
    crop_img = img[up_margin : height - down_margin, left_margin: width - right_margin]
    return crop_img
# image crop
def image_crop_3ch(img, ROI_x1= 450, ROI_y1= 40, ROI_x2=1500, ROI_y2=950):
    """
    화면에서 inference할 image area crop하는 function.
    상하 여백 값을 설정할 수 있으며, 좌우 여백은 동일한 값으로 crop됨.

    Args: 
        img(numpy array): 이미지 array
        up_offset: 위쪽 여백
        down_offset: 아래쪽 여백
        width_offset: 좌우 여백
    
    Return:
        crop_img(numpy array)
    """
    '''
    if len(img.shape) != 2:
        print("image is not a grayscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #return False
    '''    
    # Selected ROI Coordinates: setROI_x1: 468, setROI_y1: 39, setROI_x2: 1448, setROI_y2: 912
    height, width, channel = img.shape
    '''
    if up_margin + down_margin >= height or left_margin+right_margin >= width:
        print("여백이 이미지 크기를 초과")
        return False

    # crop_width = width - (width_offset * 2)
    # crop_height = height - (up_offset + down_offset)
    '''
    crop_img = img[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    return crop_img


# image resize
def image_scaling(ori_img, img_size = (640,480), interpolation = INTER_AREA, crop = False):
    """
    inference에 필요한 input size를 반환하기 위해 화면에서 영역을 crop하고 scaling하는 function
    resize interpolation 방법을 선택할 수 있음.

    Args:
        ori_img(numpy array): 이미지 array
        img_size(tuple): 반환할 이미지 크기 length: 2
        interpolation(const): default = INTER_AREA (0), 보간법 설정

    Return:
        resized_img(numpy array)
    """
    if interpolation == 0:
        # 영상 축소에 효과적인 방법
        resized_img = cv2.resize(ori_img, dsize=img_size, interpolation = cv2.INTER_AREA)
    elif interpolation == 1:
        # 양선형 보간법, 4개의  픽셀을 이용하며, 효율성이 가장 좋음. (cv2.resize interpolation argument의 default)
       resized_img = cv2.resize(ori_img, dsize=img_size, interpolation = cv2.INTER_LINEAR)
    elif interpolation == 2:
        # 3차회선 보간법, 이웃 16개의 픽셀 이용하여 선형 보간법보다 느리지만 quality good
       resized_img = cv2.resize(ori_img, dsize=img_size, interpolation = cv2.INTER_CUBIC)
    else:
        return False
    
    return resized_img
    

def video_init(video_path):
    """
    영상의 현재 프레임 이미지 (numpy array)로 반환하는 함수
    
    Args:
        video_path(string): 영상 경로
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("영상 열리지 않음")
        return False
    
    return video

def load_videos_from_folder(self, folder_path):
    """
    주어진 폴더의 모든 동영상 파일을 로드하는 함수
    """
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')  # 필요한 확장자를 추가할 수 있음
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            self.video_files.append(os.path.join(folder_path, filename))


def video_read(video):
    ret, cur_img = video.read()

    if not ret:
        return False
    
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    print(cur_img.shape)
    
    return cur_img


    
if __name__ == "__main__":
    print('hello')