#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 모델을 ONNX로 변환하는 스크립트
C# WPF 애플리케이션에서 사용할 수 있도록 최적화
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Tuple, Dict, Any

# 현재 디렉토리의 python_modules에서 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'python_modules'))

try:
    from python_modules.model import VisionTransformer, CONFIGS
    from python_modules.model_init import model_load_ViT
    from python_modules.gpu_optimizer import get_gpu_optimizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    from model import VisionTransformer, CONFIGS
    from model_init import model_load_ViT
    from gpu_optimizer import get_gpu_optimizer


class PyTorchToONNXConverter:
    """PyTorch 모델을 ONNX로 변환하는 클래스"""
    
    def __init__(self):
        self.gpu_optimizer = get_gpu_optimizer()
        self.device = self.gpu_optimizer.device
        print(f"사용 디바이스: {self.device}")
        print(self.gpu_optimizer.get_optimization_report())
    
    def convert_model(self, 
                     model_path: str, 
                     output_path: str,
                     input_size: Tuple[int, int, int] = (256, 256, 3),
                     opset_version: int = 11) -> bool:
        """
        PyTorch 모델을 ONNX로 변환
        
        Args:
            model_path: PyTorch 모델 파일 경로 (.pth)
            output_path: 출력 ONNX 파일 경로 (.onnx)
            input_size: 입력 이미지 크기 (H, W, C)
            opset_version: ONNX opset 버전
            
        Returns:
            bool: 변환 성공 여부
        """
        try:
            print(f"모델 변환 시작: {model_path} -> {output_path}")
            
            # 1. PyTorch 모델 로드
            model, transforms = self._load_pytorch_model(model_path, input_size)
            if model is None:
                return False
            
            # 2. 모델을 평가 모드로 설정
            model.eval()
            
            # 3. 더미 입력 생성
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            print(f"더미 입력 크기: {dummy_input.shape}")
            
            # 4. ONNX로 변환
            print("ONNX 변환 중...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
            
            # 5. 변환된 모델 검증
            if self._validate_onnx_model(output_path, dummy_input):
                print(f"✅ ONNX 변환 성공: {output_path}")
                
                # 6. 모델 정보 저장
                model_info = self._get_model_info(model, input_size, output_path)
                self._save_model_info(output_path.replace('.onnx', '_info.json'), model_info)
                
                return True
            else:
                print("❌ ONNX 모델 검증 실패")
                return False
                
        except Exception as e:
            print(f"❌ ONNX 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_pytorch_model(self, model_path: str, input_size: Tuple[int, int, int]):
        """PyTorch 모델 로드"""
        try:
            print(f"PyTorch 모델 로드 중: {model_path}")
            
            # 기존 model_load_ViT 함수 사용
            model, transforms = model_load_ViT(model_path, input_size)
            
            if model is None:
                print("모델 로드 실패")
                return None, None
            
            # 모델을 디바이스로 이동
            model = model.to(self.device)
            
            print(f"모델 로드 완료")
            print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, transforms
            
        except Exception as e:
            print(f"PyTorch 모델 로드 실패: {e}")
            return None, None
    
    def _validate_onnx_model(self, onnx_path: str, dummy_input: torch.Tensor) -> bool:
        """변환된 ONNX 모델 검증"""
        try:
            import onnxruntime as ort
            
            # ONNX Runtime으로 모델 로드
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            # 입력 이름 확인
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            print(f"ONNX 입력 이름: {input_name}")
            print(f"ONNX 출력 이름: {output_name}")
            
            # 더미 입력으로 추론 테스트
            input_data = dummy_input.cpu().numpy()
            outputs = session.run([output_name], {input_name: input_data})
            
            print(f"ONNX 출력 크기: {outputs[0].shape}")
            print("✅ ONNX 모델 검증 성공")
            
            return True
            
        except Exception as e:
            print(f"ONNX 모델 검증 실패: {e}")
            return False
    
    def _get_model_info(self, model: torch.nn.Module, input_size: Tuple[int, int, int], output_path: str) -> Dict[str, Any]:
        """모델 정보 수집"""
        info = {
            'model_type': 'VisionTransformer',
            'input_size': input_size,
            'output_path': output_path,
            'parameters_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(self.device),
            'opset_version': 11,
            'input_names': ['input'],
            'output_names': ['output'],
            'preprocessing': {
                'resize': input_size[:2],
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'postprocessing': {
                'resize_output': [500, 460],
                'multiply_factor': 120
            }
        }
        
        return info
    
    def _save_model_info(self, info_path: str, model_info: Dict[str, Any]):
        """모델 정보를 JSON 파일로 저장"""
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            print(f"모델 정보 저장: {info_path}")
        except Exception as e:
            print(f"모델 정보 저장 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환')
    parser.add_argument('--model_path', type=str, required=True,
                       help='PyTorch 모델 파일 경로 (.pth)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='출력 ONNX 파일 경로 (.onnx)')
    parser.add_argument('--input_size', type=int, nargs=3, default=[256, 256, 3],
                       help='입력 이미지 크기 (H W C)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset 버전')
    
    args = parser.parse_args()
    
    # 입력 크기 튜플로 변환
    input_size = tuple(args.input_size)
    
    # 변환기 생성 및 실행
    converter = PyTorchToONNXConverter()
    
    success = converter.convert_model(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=input_size,
        opset_version=args.opset_version
    )
    
    if success:
        print("\n🎉 변환 완료!")
        print(f"ONNX 모델: {args.output_path}")
        print(f"모델 정보: {args.output_path.replace('.onnx', '_info.json')}")
    else:
        print("\n❌ 변환 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main()
