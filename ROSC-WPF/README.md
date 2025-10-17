# ROSC-WPF: Point-of-Care Ultrasound Tool 🩺

## 🎯 개요
ROSC-WPF는 C# WPF 기반의 POCUS (Point-of-Care Ultrasound) 도구로, 실시간 AI 분석을 통해 CAC (Carotid Artery Compression) 및 IJV (Internal Jugular Vein) 측정을 수행합니다.

## ✨ 주요 기능

### 🚀 듀얼 추론 시스템
- **ONNX Runtime**: C# 네이티브 추론 (프로덕션용)
- **Python PyTorch**: Python 기반 추론 (개발/연구용)
- **실시간 성능 비교**: 두 방식의 성능을 직접 비교
- **동적 전환**: 런타임에 추론 방식 변경 가능

### 🎮 직관적인 사용자 인터페이스
- **상태 기반 UI**: 명확한 상태 표시 (색상 표시등)
- **초기화 시스템**: 진행률 표시와 함께 체계적 초기화
- **실시간 피드백**: 모든 동작에 대한 즉시 피드백

### 🔧 고급 기능
- **HDMI 캡처보드 지원**: 실시간 비디오 입력
- **비디오 파일 재생**: 저장된 비디오 분석
- **ROI 설정**: 관심 영역 선택 및 조정
- **실시간 그래프**: CAC 및 ABP 데이터 시각화
- **데이터 내보내기**: Excel, 이미지, 비디오 저장

## 🏗️ 시스템 아키텍처

### 상태 관리
```
Initializing → Ready → Load → Play → Calc
   (주황)      (녹색)   (파랑)  (노랑) (빨강)
```

### 초기화 과정
1. **Starting (0%)**: 프로그램 초기화 시작
2. **LoadingConfig (10%)**: 설정 파일 로드
3. **InitializingServices (30%)**: 서비스 초기화
4. **LoadingModels (50%)**: AI 모델 로드
5. **PreparingPython (70%)**: Python 모듈 준비
6. **Finalizing (90%)**: 초기화 완료
7. **Completed (100%)**: 초기화 완료

### Python 통합
- **원본 코드 통합**: ROSC-UI Python 모듈 완전 통합
- **다중 모델 포맷**: PTH, ONNX, NPZ 지원
- **CUDA 최적화**: 자동 GPU 감지 및 최적화
- **PTH → ONNX 변환**: 원클릭 모델 변환

## 📊 성능 특징

### 추론 성능
- **ONNX Runtime**: ~45ms/프레임 (C# 네이티브)
- **Python PyTorch**: ~78ms/프레임 (Python IPC)
- **CUDA GPU**: 3-5배 성능 향상 (자동 감지)

### 실시간 처리
- **백그라운드 워커**: UI 블로킹 없는 실시간 처리
- **프레임 풀링**: 메모리 효율적인 프레임 관리
- **성능 모니터링**: FPS, 메모리, CPU 사용량 추적

## 🔒 라이선스 및 상업적 사용

### ✅ 모든 라이브러리 상업적 사용 가능
- **MIT License**: ONNX Runtime, OxyPlot, ClosedXML, Newtonsoft.Json
- **Apache 2.0**: OpenCvSharp4
- **PSF License**: Python 3.12

### 📈 최신 기술 스택
- **Python 3.12**: 최신 기능과 성능 향상
- **.NET Framework 4.8.1**: 안정적인 Windows 지원
- **최신 라이브러리**: 모든 의존성 최신 버전

## 📝 로깅 시스템

### 체계적인 로깅
- **로그 레벨**: INFO, WARNING, ERROR
- **날짜별 파일**: `POCUS_LOG_2025-01-15.log`
- **자동 관리**: 7일 이상 된 로그 파일 자동 정리

### 로그 메시지 예시
```
[INFO] [2025-01-15 09:00:00.123] ROSC-WPF 애플리케이션이 시작되었습니다.
[INFO] [2025-01-15 09:00:01.456] [UI 상태] 프로그램 초기화를 시작합니다.
[INFO] [2025-01-15 09:01:02.789] [UI 상태] HDMI 캡처보드가 연결되었습니다.
[INFO] [2025-01-15 09:02:00.123] [UI 상태] 추론 방식이 Python PyTorch로 변경되었습니다.
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Python 3.12 설치
python3.12 -m venv rosc_env
source rosc_env/bin/activate
pip install -r requirements.txt

# C# 프로젝트 빌드
# Visual Studio에서 ROSC.WPF.sln 열기
# 빌드 → 솔루션 빌드
```

### 2. 프로그램 실행
```
1. ROSC-WPF.exe 실행
2. 초기화 화면에서 진행률 확인 (Initializing → Ready)
3. Ready 상태에서 HDMI 연결 또는 비디오 로드 (Load 상태)
4. Play 버튼으로 비디오 재생 (Play 상태)
5. Calculate 버튼으로 AI 분석 시작 (Calc 상태)
```

### 3. Python 추론 사용
```
1. "Python Inference" 체크박스 체크
2. 자동으로 Python PyTorch 모드로 전환
3. 실시간 AI 분석 수행
```

## 📚 상세 문서

### 🏗️ [시스템 아키텍처](docs/SYSTEM_ARCHITECTURE.md)
- 초기화 시스템 상세 설명
- 상태 관리 및 전환 로직
- 사용자 워크플로우
- 시나리오 검증 결과

### 🐍 [Python 통합](docs/PYTHON_INTEGRATION.md)
- 듀얼 추론 시스템 구현
- Python 모듈 통합 방법
- CUDA GPU 최적화
- 성능 비교 및 벤치마크

### 📝 [로깅 및 라이선스](docs/LOGGING_AND_LICENSE.md)
- 로깅 시스템 구현
- 라이선스 분석 결과
- 상업적 사용 가능성
- 업그레이드 요약

### 📚 [개발 가이드](docs/DEVELOPMENT_GUIDE.md)
- Python 환경 설정
- 프로젝트 구조 설명
- 코드 리팩토링 결과
- 성능 최적화 방법

## 🛠️ 기술 스택

### C# WPF
- **.NET Framework 4.8.1**
- **ONNX Runtime 1.17.1** (MIT License)
- **OpenCvSharp4 4.9.0** (Apache 2.0 License)
- **OxyPlot 2.1.2** (MIT License)
- **ClosedXML 0.104.1** (MIT License)
- **Newtonsoft.Json 13.0.3** (MIT License)

### Python
- **Python 3.12** (PSF License)
- **PyTorch 2.1+** (BSD License)
- **OpenCV 4.8+** (Apache 2.0 License)
- **ONNX Runtime 1.16+** (MIT License)
- **NumPy 1.24+** (BSD License)

## 🎯 사용 시나리오

### 의료진용
- **실시간 POCUS 분석**: 환자 진료 중 즉시 분석
- **정확한 측정**: AI 기반 정밀한 CAC/IJV 측정
- **데이터 기록**: 측정 결과 자동 저장 및 관리

### 연구용
- **데이터 수집**: 대량 POCUS 데이터 분석
- **성능 비교**: ONNX vs Python 추론 성능 비교
- **모델 개발**: Python으로 새 모델 개발 및 테스트

### 교육용
- **학습 도구**: POCUS 해부학 및 측정 교육
- **시각화**: 실시간 그래프로 데이터 이해
- **기록 관리**: 학습 과정 및 결과 저장

## 🔧 문제 해결

### 일반적인 문제
1. **Python 모듈 로드 실패**: Python 환경 및 의존성 확인
2. **CUDA 사용 불가**: NVIDIA 드라이버 및 CUDA Toolkit 확인
3. **HDMI 연결 실패**: 캡처보드 드라이버 및 설정 확인

### 로그 확인
```
"View Logs" 버튼 클릭 → 로그 파일 확인
ERROR 레벨 로그에서 문제 원인 파악
```

## 📞 지원

### 로그 파일 위치
```
{프로그램파일}/Log/POCUS_LOG_{날짜}.log
```

### 설정 파일
```
{프로그램파일}/config.ini
```

## 🎉 결론

ROSC-WPF는 **최신 기술 스택**과 **완전한 로깅 시스템**으로 구축된 **상업적 사용이 안전한** POCUS 분석 도구입니다.

- ✅ **듀얼 추론**: 개발 편의성과 프로덕션 성능 모두 지원
- ✅ **완전한 추적**: 모든 동작 로그 기록으로 문제 해결 용이
- ✅ **안전한 라이선스**: 모든 라이브러리 상업적 사용 가능
- ✅ **최신 기술**: Python 3.12와 최신 라이브러리로 최적화

---

**완전한 POCUS 분석 솔루션으로 의료진, 연구자, 교육자 모두를 지원합니다!** 🎊