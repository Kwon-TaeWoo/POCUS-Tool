"""
GPU 최적화 및 CUDA 관리 모듈
PyTorch CUDA 사용을 최적화하고 성능을 모니터링
"""

import torch
import torch.nn as nn
import psutil
import time
from typing import Dict, Any, Optional
import logging

class GPUOptimizer:
    """GPU 최적화 및 CUDA 관리 클래스"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.gpu_info = self._get_gpu_info()
        self.memory_info = self._get_memory_info()
        
        # GPU 최적화 설정
        if self.device.type == 'cuda':
            self._optimize_cuda_settings()
    
    def _get_optimal_device(self) -> torch.device:
        """최적의 디바이스 선택"""
        if torch.cuda.is_available():
            # CUDA 사용 가능한 경우
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # 여러 GPU가 있는 경우 가장 빠른 GPU 선택
                best_gpu = self._select_best_gpu()
                return torch.device(f'cuda:{best_gpu}')
            else:
                return torch.device('cuda:0')
        else:
            # CPU 사용
            return torch.device('cpu')
    
    def _select_best_gpu(self) -> int:
        """가장 성능이 좋은 GPU 선택"""
        best_gpu = 0
        best_score = 0
        
        for i in range(torch.cuda.device_count()):
            # GPU 메모리 크기와 계산 능력으로 점수 계산
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            compute_capability = props.major * 10 + props.minor
            
            # 점수 계산 (메모리 70%, 계산능력 30%)
            score = memory_gb * 0.7 + compute_capability * 0.3
            
            if score > best_score:
                best_score = score
                best_gpu = i
        
        return best_gpu
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 수집"""
        info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'compute_capability': None
        }
        
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current_device)
            
            info.update({
                'current_device': current_device,
                'device_name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory': props.total_memory,
                'multiprocessor_count': props.multi_processor_count
            })
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 수집"""
        info = {
            'system_ram': psutil.virtual_memory().total,
            'system_ram_available': psutil.virtual_memory().available,
            'gpu_memory': None,
            'gpu_memory_available': None
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_available': torch.cuda.memory_reserved(0)
            })
        
        return info
    
    def _optimize_cuda_settings(self):
        """CUDA 최적화 설정"""
        if not torch.cuda.is_available():
            return
        
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True  # 고정 크기 입력에 최적화
        torch.backends.cudnn.deterministic = False  # 성능 우선
        
        # 메모리 할당 최적화
        torch.cuda.empty_cache()  # 캐시 정리
        
        # 메모리 할당 전략 설정
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.9)  # GPU 메모리의 90% 사용
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """모델 최적화"""
        if self.device.type == 'cuda':
            # GPU로 모델 이동
            model = model.to(self.device)
            
            # 모델 최적화
            if hasattr(torch, 'compile'):
                # PyTorch 2.0+ 컴파일 최적화
                model = torch.compile(model, mode='max-autotune')
            
            # Mixed Precision 사용 (메모리 절약 및 속도 향상)
            if torch.cuda.is_available():
                model = model.half()  # FP16 사용
        
        return model
    
    def benchmark_inference(self, model: nn.Module, input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> Dict[str, float]:
        """추론 성능 벤치마크"""
        model.eval()
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # GPU 동기화
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 벤치마크 실행
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        
        return {
            'total_time': total_time,
            'average_time': avg_time,
            'fps': 1.0 / avg_time,
            'device': str(self.device)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량 조회"""
        usage = {
            'system_ram_used': psutil.virtual_memory().used,
            'system_ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            usage.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_reserved': torch.cuda.memory_reserved(),
                'gpu_memory_cached': torch.cuda.memory_cached()
            })
        
        return usage
    
    def clear_cache(self):
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def get_optimization_report(self) -> str:
        """최적화 리포트 생성"""
        report = f"""
=== GPU 최적화 리포트 ===
디바이스: {self.device}
CUDA 사용 가능: {self.gpu_info['available']}
GPU 개수: {self.gpu_info['device_count']}

현재 GPU 정보:
- 이름: {self.gpu_info.get('device_name', 'N/A')}
- 계산 능력: {self.gpu_info.get('compute_capability', 'N/A')}
- 총 메모리: {self.gpu_info.get('total_memory', 0) / (1024**3):.1f} GB
- 멀티프로세서: {self.gpu_info.get('multiprocessor_count', 0)}개

메모리 정보:
- 시스템 RAM: {self.memory_info['system_ram'] / (1024**3):.1f} GB
- GPU 메모리: {self.memory_info.get('gpu_memory', 0) / (1024**3):.1f} GB

최적화 설정:
- cuDNN 벤치마크: {torch.backends.cudnn.benchmark}
- cuDNN 결정적: {torch.backends.cudnn.deterministic}
- 메모리 할당 전략: 최적화됨
"""
        return report

# 전역 GPU 최적화 인스턴스
gpu_optimizer = GPUOptimizer()

def get_gpu_optimizer() -> GPUOptimizer:
    """전역 GPU 최적화 인스턴스 반환"""
    return gpu_optimizer

def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """추론용 모델 최적화"""
    return gpu_optimizer.optimize_model(model)

def benchmark_model_performance(model: nn.Module, input_size: tuple = (1, 3, 256, 256)) -> Dict[str, float]:
    """모델 성능 벤치마크"""
    device = gpu_optimizer.device
    input_tensor = torch.randn(input_size).to(device)
    
    return gpu_optimizer.benchmark_inference(model, input_tensor)
