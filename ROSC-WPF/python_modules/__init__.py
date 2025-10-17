"""
ROSC Python Modules
원본 ROSC-UI Python 코드를 최적화하여 C# WPF에서 사용
"""

from .model_init import model_load_ViT, predict_image
from .calculate_CAC import process_images_realtime
from .preprocessing import image_crop_3ch, image_scaling, video_init, video_read
from .utils import *

__version__ = "1.0.0"
__author__ = "ROSC-WPF Team"

# 모델 포맷 지원
SUPPORTED_MODEL_FORMATS = ['.pth', '.onnx', '.npz', '.pt']

# 기본 설정
DEFAULT_CONFIG = {
    'model_size': (256, 256, 3),
    'num_classes': 3,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
}
