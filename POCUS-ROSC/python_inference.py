#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python PyTorch 추론 스크립트
C# WPF에서 호출하여 AI 추론 수행
다양한 모델 포맷 지원: PTH, ONNX, NPZ, PT
"""

import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time
import traceback
import os

# 최적화된 Python 모듈 import
try:
    # 현재 디렉토리의 python_modules에서 import
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from python_modules.model_loader import ModelLoader
    from python_modules.calculate_CAC import process_images_realtime
    from python_modules.preprocessing import image_crop_3ch, image_scaling
    from python_modules.cuda_checker import check_cuda_status, is_cuda_available
    MODEL_AVAILABLE = True
    print("Optimized Python model modules loaded successfully")
    
    # CUDA 상태 확인
    if is_cuda_available():
        print("✅ CUDA GPU 사용 가능")
    else:
        print("⚠️ CUDA GPU 사용 불가 - CPU 모드로 실행")
        
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"Warning: Python model not available: {e}")
    print("Using dummy inference mode")

class PythonInference:
    def __init__(self):
        self.model_loader = ModelLoader() if MODEL_AVAILABLE else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.model_info = {}

    def load_model(self, model_path):
        try:
            if not MODEL_AVAILABLE or self.model_loader is None:
                print('Model not available, using dummy mode')
                return True
                
            print(f'Loading model from: {model_path}')
            
            # 모델 크기 결정
            if '256' in model_path or 'small' in model_path.lower():
                target_size = (256, 256, 3)
            else:
                target_size = (512, 512, 3)
            
            # 통합 모델 로더 사용
            success = self.model_loader.load_model(model_path, target_size)
            
            if success:
                self.model_info = self.model_loader.get_model_info()
                print('Model loaded successfully')
                print(f'Model type: {self.model_info.get("model_type", "unknown")}')
                print(f'Model device: {self.model_info.get("device", "unknown")}')
                return True
            else:
                print('Model loading failed')
                return False
                
        except Exception as e:
            print(f'Model loading error: {e}')
            traceback.print_exc()
            return False

    def predict(self, image_path, output_path):
        try:
            start_time = time.time()
            
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print('Failed to load image')
                return False

            print(f'Processing image: {image_path}')
            print(f'Image shape: {image.shape}')

            if not MODEL_AVAILABLE or self.model_loader is None or not self.model_loader.is_loaded():
                # 더미 추론 (개발용)
                mask = self.create_dummy_mask(image)
            else:
                # 실제 추론 (통합 모델 로더 사용)
                mask = self.model_loader.predict(image)

            # 결과 저장
            success = cv2.imwrite(output_path, mask)
            if success:
                inference_time = (time.time() - start_time) * 1000  # ms
                self.inference_count += 1
                self.total_inference_time += inference_time
                avg_time = self.total_inference_time / self.inference_count
                
                print(f'Prediction completed in {inference_time:.2f}ms (avg: {avg_time:.2f}ms)')
                return True
            else:
                print('Failed to save prediction result')
                return False
                
        except Exception as e:
            print(f'Prediction error: {e}')
            traceback.print_exc()
            return False

    def run_inference(self, image):
        try:
            # 기존 Python 추론 로직 사용
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.test_transforms(image=img)
            img = transformed['image']
            img = img / 255
            img = img.astype('float32')
            img = np.transpose(img, (2, 0, 1))
            
            imgs = torch.from_numpy(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(imgs)
            
            # Argmax 계산
            data = np.argmax(preds[0].cpu().detach().numpy(), axis=0) * 120
            mask = cv2.resize(data.astype('uint8'), (500, 460))
            
            return mask
        except Exception as e:
            print(f'Inference error: {e}')
            traceback.print_exc()
            return self.create_dummy_mask(image)

    def create_dummy_mask(self, image):
        try:
            # 개발용 더미 마스크 생성
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 원형 마스크 생성 (테스트용)
            center = (w//2, h//2)
            radius = min(w, h) // 4
            cv2.circle(mask, center, radius, 120, -1)
            
            # 추가로 타원형 마스크도 생성
            cv2.ellipse(mask, center, (radius//2, radius), 0, 0, 360, 240, -1)
            
            return cv2.resize(mask, (500, 460))
        except Exception as e:
            print(f'Dummy mask creation error: {e}')
            return np.zeros((460, 500), dtype=np.uint8)

    def convert_to_onnx(self, output_path, input_size=(256, 256, 3)):
        """PyTorch 모델을 ONNX로 변환"""
        try:
            if not MODEL_AVAILABLE or self.model_loader is None:
                print('Model not available for conversion')
                return False
                
            if not self.model_loader.is_loaded():
                print('No model loaded for conversion')
                return False
                
            success = self.model_loader.convert_to_onnx(output_path, input_size)
            if success:
                print(f'ONNX conversion completed: {output_path}')
                return True
            else:
                print('ONNX conversion failed')
                return False
                
        except Exception as e:
            print(f'ONNX conversion error: {e}')
            traceback.print_exc()
            return False

    def get_stats(self):
        stats = {
            'total_inferences': self.inference_count,
            'average_time': self.total_inference_time / max(1, self.inference_count),
            'device': str(self.device),
            'model_available': MODEL_AVAILABLE
        }
        
        # 모델 정보 추가
        if self.model_info:
            stats.update(self.model_info)
            
        return stats

def main():
    if len(sys.argv) != 2:
        print('Usage: python python_inference.py <config.json>')
        return

    config_path = sys.argv[1]
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f'Config loading error: {e}')
        return

    inference = PythonInference()
    
    try:
        if config['action'] == 'load_model':
            success = inference.load_model(config['model_path'])
            print('Model loaded successfully' if success else 'Model loading failed')
            
        elif config['action'] == 'predict':
            success = inference.predict(config['image_path'], config['output_path'])
            print('Prediction completed' if success else 'Prediction failed')
            
        elif config['action'] == 'convert_to_onnx':
            output_path = config.get('output_path', 'converted_model.onnx')
            input_size = tuple(config.get('input_size', [256, 256, 3]))
            success = inference.convert_to_onnx(output_path, input_size)
            print('ONNX conversion completed' if success else 'ONNX conversion failed')
            
        elif config['action'] == 'get_stats':
            stats = inference.get_stats()
            print(json.dumps(stats))
            
        elif config['action'] == 'check_cuda':
            cuda_status = check_cuda_status()
            print(cuda_status)
            
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
