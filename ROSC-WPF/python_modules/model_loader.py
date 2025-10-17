"""
다양한 모델 포맷 지원 로더
PTH, ONNX, NPZ, PT 파일을 지원
"""

import os
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model import VisionTransformer, CONFIGS
from .model_init import model_load_ViT
from .gpu_optimizer import get_gpu_optimizer, optimize_model_for_inference


class ModelLoader:
    """다양한 모델 포맷을 지원하는 통합 로더"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_path = None
        self.transforms = None
        self.gpu_optimizer = get_gpu_optimizer()
        self.device = self.gpu_optimizer.device
        self.target_size = (256, 256, 3)
        
        # GPU 최적화 리포트 출력
        print(self.gpu_optimizer.get_optimization_report())
        
    def load_model(self, model_path: str, target_size: Tuple[int, int, int] = (256, 256, 3)) -> bool:
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로
            target_size: 입력 이미지 크기 (H, W, C)
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            self.model_path = model_path
            self.target_size = target_size
            
            if not os.path.exists(model_path):
                print(f"모델 파일이 존재하지 않습니다: {model_path}")
                return False
                
            file_ext = Path(model_path).suffix.lower()
            
            if file_ext == '.pth' or file_ext == '.pt':
                return self._load_pytorch_model(model_path, target_size)
            elif file_ext == '.onnx':
                return self._load_onnx_model(model_path, target_size)
            elif file_ext == '.npz':
                return self._load_numpy_model(model_path, target_size)
            else:
                print(f"지원하지 않는 모델 포맷: {file_ext}")
                return False
                
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
    
    def _load_pytorch_model(self, model_path: str, target_size: Tuple[int, int, int]) -> bool:
        """PyTorch 모델 로드 (.pth, .pt)"""
        try:
            # 기존 model_load_ViT 함수 사용
            self.model, self.transforms = model_load_ViT(model_path, target_size)
            self.model_type = 'pytorch'
            self.model.eval()
            
            # GPU 최적화 적용
            self.model = optimize_model_for_inference(self.model)
            
            print(f"PyTorch 모델 로드 완료: {model_path}")
            print(f"모델 디바이스: {next(self.model.parameters()).device}")
            print(f"GPU 최적화 적용: {self.device.type == 'cuda'}")
            return True
            
        except Exception as e:
            print(f"PyTorch 모델 로드 실패: {e}")
            return False
    
    def _load_onnx_model(self, model_path: str, target_size: Tuple[int, int, int]) -> bool:
        """ONNX 모델 로드"""
        try:
            # ONNX Runtime 세션 생성
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)
            self.model_type = 'onnx'
            
            # ONNX용 전처리 변환
            self.transforms = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            print(f"ONNX 모델 로드 완료: {model_path}")
            print(f"ONNX 제공자: {self.model.get_providers()}")
            return True
            
        except Exception as e:
            print(f"ONNX 모델 로드 실패: {e}")
            return False
    
    def _load_numpy_model(self, model_path: str, target_size: Tuple[int, int, int]) -> bool:
        """NumPy 모델 로드 (.npz)"""
        try:
            # NumPy 모델 로드 (가중치만 저장된 경우)
            weights = np.load(model_path)
            
            # VisionTransformer 모델 초기화
            config_vit = CONFIGS['R50-ViT-B_16']
            config_vit.n_classes = 3
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(target_size[0] / 16), int(target_size[0] / 16))
            
            self.model = VisionTransformer(config_vit, img_size=target_size[0], num_classes=3)
            
            # NumPy 가중치를 PyTorch 모델에 로드
            state_dict = {}
            for key, value in weights.items():
                state_dict[key] = torch.from_numpy(value)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            self.model_type = 'numpy'
            
            # 전처리 변환
            self.transforms = A.Compose([
                A.Resize(target_size[0], target_size[1]),
            ])
            
            print(f"NumPy 모델 로드 완료: {model_path}")
            return True
            
        except Exception as e:
            print(f"NumPy 모델 로드 실패: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 추론
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            np.ndarray: 예측 마스크
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None
            
        try:
            if self.model_type == 'pytorch':
                return self._predict_pytorch(image)
            elif self.model_type == 'onnx':
                return self._predict_onnx(image)
            elif self.model_type == 'numpy':
                return self._predict_pytorch(image)  # NumPy도 PyTorch와 동일한 방식
            else:
                print(f"알 수 없는 모델 타입: {self.model_type}")
                return None
                
        except Exception as e:
            print(f"추론 오류: {e}")
            return None
    
    def _predict_pytorch(self, image: np.ndarray) -> np.ndarray:
        """PyTorch 모델 추론"""
        # 이미지 전처리
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=img)
        img = transformed['image']
        img = img / 255
        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        
        # 텐서 변환
        imgs = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            preds = self.model(imgs)
        
        # 후처리
        data = np.argmax(preds[0].cpu().detach().numpy(), axis=0) * 120
        mask = cv2.resize(data.astype('uint8'), (500, 460))
        
        return mask
    
    def _predict_onnx(self, image: np.ndarray) -> np.ndarray:
        """ONNX 모델 추론"""
        # 이미지 전처리
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=img)
        img = transformed['image']
        
        # ONNX 입력 형식으로 변환
        input_name = self.model.get_inputs()[0].name
        input_data = img.unsqueeze(0).numpy()
        
        # 추론
        outputs = self.model.run(None, {input_name: input_data})
        preds = outputs[0]
        
        # 후처리
        data = np.argmax(preds[0], axis=0) * 120
        mask = cv2.resize(data.astype('uint8'), (500, 460))
        
        return mask
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'target_size': self.target_size,
            'device': str(self.device)
        }
        
        if self.model_type == 'onnx':
            info['providers'] = self.model.get_providers()
            info['input_names'] = [inp.name for inp in self.model.get_inputs()]
            info['output_names'] = [out.name for out in self.model.get_outputs()]
        elif self.model_type == 'pytorch':
            info['parameters'] = sum(p.numel() for p in self.model.parameters())
            info['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info
    
    def convert_to_onnx(self, output_path: str, input_size: Tuple[int, int, int] = (256, 256, 3)) -> bool:
        """
        PyTorch 모델을 ONNX로 변환
        
        Args:
            output_path: 출력 ONNX 파일 경로
            input_size: 입력 크기 (H, W, C)
            
        Returns:
            bool: 변환 성공 여부
        """
        if self.model_type != 'pytorch':
            print("PyTorch 모델만 ONNX로 변환할 수 있습니다.")
            return False
            
        try:
            # 더미 입력 생성
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            
            # ONNX로 변환
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"ONNX 변환 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"ONNX 변환 실패: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """모델 로드 여부 확인"""
        return self.model is not None
