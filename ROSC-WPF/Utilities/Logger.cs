using System;
using System.IO;
using System.Text;
using System.Threading;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 로깅 레벨
    /// </summary>
    public enum LogLevel
    {
        INFO,
        WARNING,
        ERROR
    }

    /// <summary>
    /// 로깅 유틸리티 클래스
    /// POCUS_LOG_{datetime}.log 형식으로 로그 파일 생성
    /// </summary>
    public static class Logger
    {
        private static readonly object _lock = new object();
        private static string _logDirectory;
        private static string _logFilePath;
        private static bool _isInitialized = false;

        /// <summary>
        /// 로거 초기화
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized) return;

            try
            {
                // Log 디렉토리 생성
                _logDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Log");
                if (!Directory.Exists(_logDirectory))
                {
                    Directory.CreateDirectory(_logDirectory);
                }

                // 로그 파일 경로 생성 (POCUS_LOG_2025-10-11.log 형식)
                UpdateLogFilePath();

                _isInitialized = true;
                
                // 초기화 로그 기록
                Log(LogLevel.INFO, "로거가 초기화되었습니다.");
            }
            catch (Exception ex)
            {
                // 로거 초기화 실패 시 콘솔에 출력
                Console.WriteLine($"로거 초기화 실패: {ex.Message}");
            }
        }

        /// <summary>
        /// 로그 파일 경로 업데이트 (날짜 변경 시)
        /// </summary>
        private static void UpdateLogFilePath()
        {
            string dateString = DateTime.Now.ToString("yyyy-MM-dd");
            _logFilePath = Path.Combine(_logDirectory, $"POCUS_LOG_{dateString}.log");
        }

        /// <summary>
        /// 로그 기록
        /// </summary>
        /// <param name="level">로그 레벨</param>
        /// <param name="message">로그 메시지</param>
        /// <param name="exception">예외 정보 (선택사항)</param>
        public static void Log(LogLevel level, string message, Exception exception = null)
        {
            if (!_isInitialized)
            {
                Initialize();
            }

            try
            {
                lock (_lock)
                {
                    // 날짜가 변경되었는지 확인하고 로그 파일 경로 업데이트
                    string currentDateString = DateTime.Now.ToString("yyyy-MM-dd");
                    string currentLogFileName = $"POCUS_LOG_{currentDateString}.log";
                    string currentLogFilePath = Path.Combine(_logDirectory, currentLogFileName);
                    
                    if (_logFilePath != currentLogFilePath)
                    {
                        _logFilePath = currentLogFilePath;
                        // 새 날짜 로그 파일 시작 메시지
                        string newDayMessage = FormatLogMessage(LogLevel.INFO, $"새로운 로그 세션이 시작되었습니다. ({currentDateString})", null);
                        File.AppendAllText(_logFilePath, newDayMessage + Environment.NewLine, Encoding.UTF8);
                    }
                    
                    // 로그 메시지 포맷팅
                    string logMessage = FormatLogMessage(level, message, exception);
                    
                    // 파일에 로그 기록
                    File.AppendAllText(_logFilePath, logMessage + Environment.NewLine, Encoding.UTF8);
                    
                    // 콘솔에도 출력 (디버깅용)
                    Console.WriteLine(logMessage);
                }
            }
            catch (Exception ex)
            {
                // 로그 기록 실패 시 콘솔에 출력
                Console.WriteLine($"로그 기록 실패: {ex.Message}");
                Console.WriteLine($"원본 메시지: [{level}] {message}");
            }
        }

        /// <summary>
        /// INFO 레벨 로그
        /// </summary>
        public static void Info(string message)
        {
            Log(LogLevel.INFO, message);
        }

        /// <summary>
        /// WARNING 레벨 로그
        /// </summary>
        public static void Warning(string message, Exception exception = null)
        {
            Log(LogLevel.WARNING, message, exception);
        }

        /// <summary>
        /// ERROR 레벨 로그
        /// </summary>
        public static void Error(string message, Exception exception = null)
        {
            Log(LogLevel.ERROR, message, exception);
        }

        /// <summary>
        /// 로그 메시지 포맷팅
        /// [INFO] [2025-10-11 18:00:00.111] Log message
        /// </summary>
        private static string FormatLogMessage(LogLevel level, string message, Exception exception)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            string levelString = $"[{level}]";
            string timestampString = $"[{timestamp}]";
            
            StringBuilder logMessage = new StringBuilder();
            logMessage.Append(levelString);
            logMessage.Append(" ");
            logMessage.Append(timestampString);
            logMessage.Append(" ");
            logMessage.Append(message);

            // 예외 정보 추가
            if (exception != null)
            {
                logMessage.AppendLine();
                logMessage.Append("Exception: ");
                logMessage.Append(exception.Message);
                
                if (!string.IsNullOrEmpty(exception.StackTrace))
                {
                    logMessage.AppendLine();
                    logMessage.Append("StackTrace: ");
                    logMessage.Append(exception.StackTrace);
                }
            }

            return logMessage.ToString();
        }

        /// <summary>
        /// 로그 파일 경로 반환
        /// </summary>
        public static string GetLogFilePath()
        {
            return _logFilePath;
        }

        /// <summary>
        /// 로그 디렉토리 경로 반환
        /// </summary>
        public static string GetLogDirectory()
        {
            return _logDirectory;
        }

        /// <summary>
        /// 로그 파일 크기 확인 (MB)
        /// </summary>
        public static double GetLogFileSizeMB()
        {
            try
            {
                if (File.Exists(_logFilePath))
                {
                    FileInfo fileInfo = new FileInfo(_logFilePath);
                    return fileInfo.Length / (1024.0 * 1024.0);
                }
                return 0;
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// 오래된 로그 파일 정리 (7일 이상)
        /// </summary>
        public static void CleanOldLogs(int daysToKeep = 7)
        {
            try
            {
                if (!Directory.Exists(_logDirectory))
                    return;

                var files = Directory.GetFiles(_logDirectory, "POCUS_LOG_*.log");
                DateTime cutoffDate = DateTime.Now.AddDays(-daysToKeep);
                int deletedCount = 0;

                foreach (var file in files)
                {
                    FileInfo fileInfo = new FileInfo(file);
                    if (fileInfo.CreationTime < cutoffDate)
                    {
                        File.Delete(file);
                        deletedCount++;
                        Info($"오래된 로그 파일 삭제: {Path.GetFileName(file)}");
                    }
                }
                
                if (deletedCount > 0)
                {
                    Info($"총 {deletedCount}개의 오래된 로그 파일을 정리했습니다.");
                }
            }
            catch (Exception ex)
            {
                Error("오래된 로그 파일 정리 중 오류 발생", ex);
            }
        }

        /// <summary>
        /// 로그 파일 목록 조회
        /// </summary>
        public static string[] GetLogFiles()
        {
            try
            {
                if (!Directory.Exists(_logDirectory))
                    return new string[0];

                var files = Directory.GetFiles(_logDirectory, "POCUS_LOG_*.log");
                return files.OrderByDescending(f => File.GetCreationTime(f)).ToArray();
            }
            catch (Exception ex)
            {
                Error("로그 파일 목록 조회 중 오류 발생", ex);
                return new string[0];
            }
        }

        /// <summary>
        /// 특정 날짜의 로그 파일 경로 반환
        /// </summary>
        public static string GetLogFilePathForDate(DateTime date)
        {
            try
            {
                string dateString = date.ToString("yyyy-MM-dd");
                return Path.Combine(_logDirectory, $"POCUS_LOG_{dateString}.log");
            }
            catch (Exception ex)
            {
                Error("로그 파일 경로 생성 중 오류 발생", ex);
                return null;
            }
        }

        /// <summary>
        /// 로그 파일 압축 (선택사항)
        /// </summary>
        public static void CompressLogFile()
        {
            try
            {
                if (!File.Exists(_logFilePath))
                    return;

                double fileSizeMB = GetLogFileSizeMB();
                if (fileSizeMB > 10) // 10MB 이상일 때 압축
                {
                    string compressedPath = _logFilePath + ".gz";
                    // 실제 압축 구현은 필요시 추가
                    Info($"로그 파일 압축 권장: {fileSizeMB:F2}MB");
                }
            }
            catch (Exception ex)
            {
                Error("로그 파일 압축 중 오류 발생", ex);
            }
        }
    }
}
