"""
CUDA 사용 상태 확인 및 진단 도구
PyTorch CUDA 설정을 확인하고 문제를 진단
"""

import torch
import subprocess
import sys
import platform
from typing import Dict, Any, List, Tuple

class CUDAChecker:
    """CUDA 사용 상태 확인 클래스"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.cuda_info = self._get_cuda_info()
        self.pytorch_info = self._get_pytorch_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor()
        }
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """CUDA 정보 수집"""
        info = {
            'available': torch.cuda.is_available(),
            'version': None,
            'device_count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            info['version'] = torch.version.cuda
            info['device_count'] = torch.cuda.device_count()
            
            # 각 GPU 정보 수집
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multi_processor_count
                }
                info['devices'].append(device_info)
        
        return info
    
    def _get_pytorch_info(self) -> Dict[str, Any]:
        """PyTorch 정보 수집"""
        return {
            'version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'cudnn_available': torch.backends.cudnn.is_available(),
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic
        }
    
    def check_cuda_driver(self) -> Dict[str, Any]:
        """CUDA 드라이버 확인"""
        try:
            # nvidia-smi 명령어로 드라이버 정보 확인
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                drivers = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        drivers.append({
                            'version': parts[0],
                            'name': parts[1],
                            'memory_mb': int(parts[2])
                        })
                
                return {
                    'available': True,
                    'drivers': drivers,
                    'error': None
                }
            else:
                return {
                    'available': False,
                    'drivers': [],
                    'error': 'nvidia-smi 명령어 실행 실패'
                }
        except subprocess.TimeoutExpired:
            return {
                'available': False,
                'drivers': [],
                'error': 'nvidia-smi 명령어 타임아웃'
            }
        except FileNotFoundError:
            return {
                'available': False,
                'drivers': [],
                'error': 'nvidia-smi 명령어를 찾을 수 없음 (NVIDIA 드라이버 미설치)'
            }
        except Exception as e:
            return {
                'available': False,
                'drivers': [],
                'error': f'드라이버 확인 중 오류: {str(e)}'
            }
    
    def test_cuda_functionality(self) -> Dict[str, Any]:
        """CUDA 기능 테스트"""
        if not torch.cuda.is_available():
            return {
                'success': False,
                'error': 'CUDA를 사용할 수 없음',
                'tests': {}
            }
        
        tests = {}
        
        try:
            # 기본 CUDA 테스트
            device = torch.device('cuda:0')
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            tests['basic_operations'] = True
        except Exception as e:
            tests['basic_operations'] = False
            tests['basic_operations_error'] = str(e)
        
        try:
            # 메모리 할당 테스트
            large_tensor = torch.randn(1000, 1000).to(device)
            del large_tensor
            torch.cuda.empty_cache()
            tests['memory_allocation'] = True
        except Exception as e:
            tests['memory_allocation'] = False
            tests['memory_allocation_error'] = str(e)
        
        try:
            # cuDNN 테스트
            if torch.backends.cudnn.is_available():
                conv = torch.nn.Conv2d(3, 64, 3).to(device)
                input_tensor = torch.randn(1, 3, 224, 224).to(device)
                output = conv(input_tensor)
                tests['cudnn_operations'] = True
            else:
                tests['cudnn_operations'] = False
                tests['cudnn_operations_error'] = 'cuDNN을 사용할 수 없음'
        except Exception as e:
            tests['cudnn_operations'] = False
            tests['cudnn_operations_error'] = str(e)
        
        success = all(test for test in tests.values() if isinstance(test, bool))
        
        return {
            'success': success,
            'error': None if success else '일부 CUDA 테스트 실패',
            'tests': tests
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량 조회"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'error': 'CUDA를 사용할 수 없음'
            }
        
        usage = {}
        
        for i in range(torch.cuda.device_count()):
            usage[f'gpu_{i}'] = {
                'allocated': torch.cuda.memory_allocated(i),
                'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'reserved': torch.cuda.memory_reserved(i),
                'reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                'total': torch.cuda.get_device_properties(i).total_memory,
                'total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
        
        return {
            'available': True,
            'usage': usage
        }
    
    def generate_diagnostic_report(self) -> str:
        """진단 리포트 생성"""
        report = []
        report.append("=== CUDA 진단 리포트 ===")
        report.append("")
        
        # 시스템 정보
        report.append("시스템 정보:")
        report.append(f"  플랫폼: {self.system_info['platform']}")
        report.append(f"  Python 버전: {self.system_info['python_version']}")
        report.append(f"  아키텍처: {self.system_info['architecture']}")
        report.append("")
        
        # CUDA 드라이버 정보
        driver_info = self.check_cuda_driver()
        report.append("CUDA 드라이버:")
        if driver_info['available']:
            for driver in driver_info['drivers']:
                report.append(f"  GPU: {driver['name']}")
                report.append(f"  드라이버 버전: {driver['version']}")
                report.append(f"  메모리: {driver['memory_mb']} MB")
        else:
            report.append(f"  오류: {driver_info['error']}")
        report.append("")
        
        # PyTorch CUDA 정보
        report.append("PyTorch CUDA 정보:")
        report.append(f"  CUDA 사용 가능: {self.cuda_info['available']}")
        if self.cuda_info['available']:
            report.append(f"  CUDA 버전: {self.cuda_info['version']}")
            report.append(f"  GPU 개수: {self.cuda_info['device_count']}")
            for device in self.cuda_info['devices']:
                report.append(f"  GPU {device['id']}: {device['name']}")
                report.append(f"    메모리: {device['total_memory_gb']:.1f} GB")
                report.append(f"    계산 능력: {device['compute_capability']}")
        report.append("")
        
        # PyTorch 설정
        report.append("PyTorch 설정:")
        report.append(f"  PyTorch 버전: {self.pytorch_info['version']}")
        report.append(f"  cuDNN 사용 가능: {self.pytorch_info['cudnn_available']}")
        report.append(f"  cuDNN 버전: {self.pytorch_info['cudnn_version']}")
        report.append(f"  cuDNN 벤치마크: {self.pytorch_info['cudnn_benchmark']}")
        report.append("")
        
        # 기능 테스트
        test_result = self.test_cuda_functionality()
        report.append("CUDA 기능 테스트:")
        if test_result['success']:
            report.append("  모든 테스트 통과 ✅")
        else:
            report.append(f"  테스트 실패: {test_result['error']}")
            for test_name, result in test_result['tests'].items():
                if isinstance(result, bool):
                    status = "✅" if result else "❌"
                    report.append(f"    {test_name}: {status}")
                elif test_name.endswith('_error'):
                    report.append(f"    {test_name}: {result}")
        report.append("")
        
        # 메모리 사용량
        memory_info = self.get_memory_usage()
        if memory_info['available']:
            report.append("GPU 메모리 사용량:")
            for gpu_name, usage in memory_info['usage'].items():
                report.append(f"  {gpu_name}:")
                report.append(f"    할당됨: {usage['allocated_gb']:.2f} GB")
                report.append(f"    예약됨: {usage['reserved_gb']:.2f} GB")
                report.append(f"    총 메모리: {usage['total_gb']:.1f} GB")
        report.append("")
        
        # 권장사항
        report.append("권장사항:")
        if not self.cuda_info['available']:
            report.append("  ❌ CUDA를 사용할 수 없습니다.")
            report.append("  - NVIDIA GPU 드라이버 설치 확인")
            report.append("  - CUDA Toolkit 설치 확인")
            report.append("  - PyTorch CUDA 버전 설치 확인")
        else:
            report.append("  ✅ CUDA 사용 가능")
            if not self.pytorch_info['cudnn_benchmark']:
                report.append("  - cuDNN 벤치마크 활성화 권장 (성능 향상)")
            if self.cuda_info['device_count'] > 1:
                report.append("  - 다중 GPU 사용 고려")
        
        return "\n".join(report)

def check_cuda_status() -> str:
    """CUDA 상태 확인 및 리포트 반환"""
    checker = CUDAChecker()
    return checker.generate_diagnostic_report()

def is_cuda_available() -> bool:
    """CUDA 사용 가능 여부 확인"""
    return torch.cuda.is_available()

def get_cuda_device_info() -> Dict[str, Any]:
    """CUDA 디바이스 정보 반환"""
    checker = CUDAChecker()
    return checker.cuda_info

if __name__ == "__main__":
    # 직접 실행 시 진단 리포트 출력
    print(check_cuda_status())
