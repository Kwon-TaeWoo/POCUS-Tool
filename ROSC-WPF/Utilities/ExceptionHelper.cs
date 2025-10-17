using System;
using System.IO;
using System.Windows;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 예외 처리 및 로깅을 위한 유틸리티
    /// Logger와 연동하여 체계적인 로깅 수행
    /// </summary>
    public static class ExceptionHelper
    {
        /// <summary>
        /// 안전한 실행 (예외 발생 시 기본값 반환)
        /// </summary>
        public static T SafeExecute<T>(Func<T> action, T defaultValue = default(T), string operationName = "")
        {
            try
            {
                return action();
            }
            catch (Exception ex)
            {
                LogError(ex, operationName);
                return defaultValue;
            }
        }

        /// <summary>
        /// 안전한 실행 (예외 발생 시 void)
        /// </summary>
        public static void SafeExecute(Action action, string operationName = "")
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                LogError(ex, operationName);
            }
        }

        /// <summary>
        /// 안전한 실행 (예외 발생 시 bool 반환)
        /// </summary>
        public static bool SafeExecute(Func<bool> action, string operationName = "")
        {
            try
            {
                return action();
            }
            catch (Exception ex)
            {
                LogError(ex, operationName);
                return false;
            }
        }

        /// <summary>
        /// 에러 로깅
        /// </summary>
        public static void LogError(Exception ex, string operationName = "")
        {
            string message = string.IsNullOrEmpty(operationName) 
                ? ex.Message 
                : $"{operationName} 중 오류 발생: {ex.Message}";
            
            Logger.Error(message, ex);
        }

        /// <summary>
        /// 경고 로깅
        /// </summary>
        public static void LogWarning(string message, Exception ex = null)
        {
            Logger.Warning(message, ex);
        }

        /// <summary>
        /// 정보 로깅
        /// </summary>
        public static void LogInfo(string message)
        {
            Logger.Info(message);
        }

        /// <summary>
        /// 사용자에게 오류 메시지 표시 및 로깅
        /// </summary>
        public static void ShowErrorAndLog(string message, Exception ex, string operationName = "Operation")
        {
            Logger.Error($"{operationName}: {message}", ex);
            
            // UI 스레드에서 메시지 박스 표시
            Application.Current?.Dispatcher.Invoke(() =>
            {
                MessageBox.Show($"{message}\n\n자세한 내용은 로그 파일을 확인하세요.", 
                              "오류", MessageBoxButton.OK, MessageBoxImage.Error);
            });
        }

        /// <summary>
        /// 사용자에게 경고 메시지 표시 및 로깅
        /// </summary>
        public static void ShowWarningAndLog(string message, Exception ex = null)
        {
            Logger.Warning(message, ex);
            
            // UI 스레드에서 메시지 박스 표시
            Application.Current?.Dispatcher.Invoke(() =>
            {
                MessageBox.Show(message, "경고", MessageBoxButton.OK, MessageBoxImage.Warning);
            });
        }
    }
}
