using System;

namespace POCUS.ROSC.Models
{
    /// <summary>
    /// 애플리케이션 상태를 나타내는 열거형
    /// Python의 State enum과 동일한 구조
    /// </summary>
    public enum AppState
    {
        Initializing,   // 초기화 중 - 프로그램 시작 시 모델 로드 등
        Ready,          // 대기 상태 - HDMI 연결 전
        Load,           // HDMI 연결됨 - 비디오 재생 전
        Play,           // 재생 중 - 계산 없이 비디오만 표시
        Calc            // 계산 중 - CAC 추론 및 측정 수행
    }

    /// <summary>
    /// 애플리케이션 상태 확장 메서드
    /// </summary>
    public static class AppStateExtensions
    {
        /// <summary>
        /// 상태를 문자열로 변환
        /// </summary>
        public static string ToDisplayString(this AppState state)
        {
            return state switch
            {
                AppState.Initializing => "Initializing",
                AppState.Ready => "Ready",
                AppState.Load => "Load",
                AppState.Play => "Play",
                AppState.Calc => "Calculate",
                _ => "Unknown"
            };
        }
    }
}
