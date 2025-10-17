# 시스템 아키텍처 및 시나리오 🏗️

## 📋 목차
- [초기화 시스템](#초기화-시스템)
- [상태 관리](#상태-관리)
- [시나리오 검증](#시나리오-검증)
- [사용자 워크플로우](#사용자-워크플로우)

## 🚀 초기화 시스템

### 개요
프로그램 시작 시 모든 필요한 컴포넌트를 백그라운드에서 로드하고, 사용자에게 진행 상황을 보여주는 완전한 초기화 시스템입니다.

### 초기화 단계
1. **Starting (0%)**: 프로그램 초기화 시작
2. **LoadingConfig (10%)**: 설정 파일 로드
3. **InitializingServices (30%)**: 서비스 초기화
4. **LoadingModels (50%)**: AI 모델 로드
5. **PreparingPython (70%)**: Python 모듈 준비
6. **Finalizing (90%)**: 초기화 완료
7. **Completed (100%)**: 초기화 완료

### 구현 파일
- `Utilities/InitializationWorker.cs`: 백그라운드 초기화 워커
- `Views/InitializationWindow.xaml`: 초기화 화면 UI
- `Models/AppState.cs`: 애플리케이션 상태 정의

## 🔄 상태 관리

### 애플리케이션 상태
```csharp
public enum AppState
{
    Initializing,   // 초기화 중 - 프로그램 시작 시 모델 로드 등
    Ready,          // 대기 상태 - HDMI 연결 전
    Load,           // HDMI 연결됨 - 비디오 재생 전
    Play,           // 재생 중 - 계산 없이 비디오만 표시
    Calc            // 계산 중 - CAC 추론 및 측정 수행
}
```

### 상태 전환 흐름
```
Initializing → Ready → Load → Play → Calc
   (주황)      (녹색)   (파랑)  (노랑) (빨강)
```

### 상태별 UI 동작
| 상태 | Connect HDMI | Load Video | Play | Calculate |
|------|-------------|------------|------|-----------|
| **Ready** | ✅ Connect | ✅ Load | ❌ | ❌ |
| **Load** | ✅ Reconnect | ✅ Load | ✅ Play | ✅ Calculate |
| **Play** | ✅ Reconnect | ✅ Load | ✅ Stop | ✅ Calculate |
| **Calc** | ✅ Reconnect | ✅ Load | ✅ Stop | ✅ Stop Calculate |

## 🎯 시나리오 검증

### ✅ 구현 완료된 시나리오

#### 1. WPF 앱 실행 시 초기화
- **구현**: `InitializationWorker` 백그라운드 워커
- **UI**: `InitializationWindow`에서 진행률 표시
- **로깅**: 각 단계별 진행 상황 로그 기록

#### 2. 초기화 완료 후 Ready 상태
- **구현**: `OnInitializationCompleted`에서 `AppState.Ready`로 변경
- **UI**: 녹색 표시등으로 상태 표시
- **로깅**: "프로그램이 준비되었습니다" 로그 기록

#### 3. HDMI 연결 시 Load 상태
- **구현**: `ConnectHDMI()` 성공 시 `AppState.Load`로 변경
- **UI**: 파란색 표시등으로 상태 표시
- **로깅**: "HDMI 캡처보드 연결 완료" 로그 기록

#### 4. Play 버튼으로 Play 상태
- **구현**: `BtnPlay_Click`에서 `AppState.Play`로 변경
- **UI**: 노란색 표시등으로 상태 표시
- **로깅**: "비디오 재생 시작" 로그 기록

#### 5. Calculate로 Calc 상태
- **구현**: `BtnCalculateCAC_Click`에서 `AppState.Calc`로 변경
- **UI**: 빨간색 표시등으로 상태 표시
- **로깅**: "AI 분석 시작" 로그 기록

#### 6. Python 체크박스 로직
- **구현**: `CbPythonInference_Checked`로 추론 방식 변경
- **통신**: JSON 파일 기반 Python ↔ C# 통신
- **로깅**: "추론 방식 변경" 로그 기록

## 🎮 사용자 워크플로우

### 일반적인 사용 시나리오
```
1. 프로그램 시작
   ↓
2. 초기화 화면 (진행률 표시)
   ↓
3. Ready 상태 (녹색 표시등)
   ↓
4. HDMI 연결 → Load 상태 (파란색 표시등)
   ↓
5. Play 버튼 → Play 상태 (노란색 표시등)
   ↓
6. Calculate 버튼 → Calc 상태 (빨간색 표시등)
   ↓
7. 실시간 AI 분석 및 결과 표시
```

### Python 추론 사용 시나리오
```
1. Python Inference 체크박스 체크
   ↓
2. 추론 서비스 재초기화 (Python PyTorch)
   ↓
3. 이미지 → Python 모듈 전달
   ↓
4. Python에서 AI 추론 수행
   ↓
5. 결과 → C#으로 수신
   ↓
6. CAC 계산 및 UI 표시
```

## 🔧 기술적 구현

### 초기화 워커
```csharp
public class InitializationWorker : BackgroundWorker
{
    public enum InitializationStep
    {
        Starting, LoadingConfig, InitializingServices,
        LoadingModels, PreparingPython, Finalizing, Completed
    }
    
    // 각 단계별 진행률과 로깅
    private void ReportStep(BackgroundWorker worker, InitializationStep step, string message, int progress)
    {
        Logger.Info($"[초기화 {progress}%] {message}");
        // UI 업데이트
    }
}
```

### 상태 관리
```csharp
private void UpdateAppState(AppState newState)
{
    _currentState = newState;
    
    // UI 스레드에서 상태 표시 업데이트
    Dispatcher.Invoke(() =>
    {
        statusText.Text = newState.ToDisplayString();
        statusIndicator.Fill = GetStateColor(newState);
        UpdateButtonStates(newState);
    });
}
```

### Python 통신
```csharp
// C# → Python
string configJson = JsonConvert.SerializeObject(config);
File.WriteAllText(configPath, configJson);

// Python → C#
string output = process.StandardOutput.ReadToEnd();
```

## 📊 성능 및 안정성

### 초기화 성능
- **백그라운드 처리**: UI 블로킹 없이 초기화
- **진행률 표시**: 실시간 진행 상황 사용자에게 표시
- **에러 처리**: 초기화 실패 시 적절한 오류 처리

### 상태 관리 안정성
- **상태 검증**: 유효하지 않은 상태 전환 방지
- **UI 동기화**: 상태 변경 시 UI 자동 업데이트
- **로깅**: 모든 상태 변경 로그 기록

### Python 통신 안정성
- **타임아웃**: 60초 타임아웃으로 무한 대기 방지
- **에러 처리**: Python 스크립트 실행 실패 시 적절한 처리
- **로깅**: 모든 Python 통신 로그 기록

## 🎯 결론

### ✅ 완성된 시스템
- **완전한 초기화**: 모든 컴포넌트 체계적 로드
- **직관적 상태 관리**: 명확한 상태 전환과 UI 피드백
- **안정적인 Python 통신**: C# ↔ Python 원활한 통신
- **포괄적 로깅**: 모든 동작 추적 가능

### 🚀 사용자 경험
- **전문적인 초기화**: 진행률 표시로 사용자 안심
- **명확한 상태 표시**: 색상과 텍스트로 현재 상태 명확히 표시
- **직관적 조작**: 상태에 따른 버튼 활성화/비활성화
- **안정적인 동작**: 예외 상황 적절한 처리

---

**완전한 시스템 아키텍처로 안정적이고 직관적인 ROSC-WPF 애플리케이션이 완성되었습니다!** 🎊
