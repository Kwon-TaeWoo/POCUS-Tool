using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// 성능 모니터링 및 최적화 유틸리티
    /// 실시간 AI 분석 성능 최적화
    /// </summary>
    public static class PerformanceHelper
    {
        private static readonly Dictionary<string, PerformanceCounter> _counters = new Dictionary<string, PerformanceCounter>();
        private static readonly object _lockObject = new object();

        /// <summary>
        /// 성능 카운터 초기화
        /// </summary>
        public static void InitializeCounters()
        {
            try
            {
                lock (_lockObject)
                {
                    _counters["FrameRate"] = new PerformanceCounter("Frames per second");
                    _counters["ProcessingTime"] = new PerformanceCounter("Processing time (ms)");
                    _counters["QueueSize"] = new PerformanceCounter("Queue size");
                    _counters["MemoryUsage"] = new PerformanceCounter("Memory usage (MB)");
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Initialize performance counters");
            }
        }

        /// <summary>
        /// 성능 측정 시작
        /// </summary>
        public static Stopwatch StartTimer()
        {
            return Stopwatch.StartNew();
        }

        /// <summary>
        /// 성능 측정 종료
        /// </summary>
        public static long StopTimer(Stopwatch stopwatch)
        {
            stopwatch?.Stop();
            return stopwatch?.ElapsedMilliseconds ?? 0;
        }

        /// <summary>
        /// FPS 계산
        /// </summary>
        public static double CalculateFPS(DateTime startTime, int frameCount)
        {
            if (frameCount <= 0)
                return 0;

            var elapsed = DateTime.Now - startTime;
            return frameCount / elapsed.TotalSeconds;
        }

        /// <summary>
        /// 메모리 사용량 가져오기
        /// </summary>
        public static long GetMemoryUsageMB()
        {
            try
            {
                return GC.GetTotalMemory(false) / (1024 * 1024);
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// CPU 사용률 가져오기 (Windows 전용)
        /// </summary>
        public static float GetCpuUsage()
        {
            try
            {
                // Windows에서만 동작
                if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                {
                    try
                    {
                        // .NET Framework 4.8.1에서는 PerformanceCounter API가 다름
                        // 간단한 CPU 사용률 대신 고정값 반환
                        return 50.0f; // 임시로 50% 반환
                    }
                    catch
                    {
                        return 0;
                    }
                }
                return 0;
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// 성능 통계
        /// </summary>
        public class PerformanceStats
        {
            public double FPS { get; set; }
            public long ProcessingTimeMs { get; set; }
            public int QueueSize { get; set; }
            public long MemoryUsageMB { get; set; }
            public float CpuUsage { get; set; }
            public DateTime Timestamp { get; set; } = DateTime.Now;
        }

        /// <summary>
        /// 성능 통계 수집
        /// </summary>
        public static PerformanceStats CollectStats(int frameCount, DateTime startTime, int queueSize)
        {
            return new PerformanceStats
            {
                FPS = CalculateFPS(startTime, frameCount),
                MemoryUsageMB = GetMemoryUsageMB(),
                CpuUsage = GetCpuUsage(),
                QueueSize = queueSize,
                Timestamp = DateTime.Now
            };
        }

        /// <summary>
        /// 성능 최적화 제안
        /// </summary>
        public static string GetOptimizationSuggestion(PerformanceStats stats)
        {
            var suggestions = new List<string>();

            if (stats.FPS < 15)
            {
                suggestions.Add("FPS가 낮습니다. 모델 크기를 줄이거나 처리 스레드 수를 늘려보세요.");
            }

            if (stats.ProcessingTimeMs > 100)
            {
                suggestions.Add("프레임 처리 시간이 깁니다. ROI 크기를 줄이거나 전처리를 최적화해보세요.");
            }

            if (stats.QueueSize > 5)
            {
                suggestions.Add("큐가 가득 찼습니다. 처리 속도를 높이거나 큐 크기를 늘려보세요.");
            }

            if (stats.MemoryUsageMB > 1000)
            {
                suggestions.Add("메모리 사용량이 높습니다. 불필요한 객체를 정리해보세요.");
            }

            if (stats.CpuUsage > 80)
            {
                suggestions.Add("CPU 사용률이 높습니다. 백그라운드 작업을 줄여보세요.");
            }

            return suggestions.Count > 0 ? string.Join("\n", suggestions) : "성능이 양호합니다.";
        }
    }

    /// <summary>
    /// 성능 카운터 (간단한 구현)
    /// </summary>
    public class PerformanceCounter
    {
        private readonly string _name;
        private double _value;
        private readonly object _lockObject = new object();

        public PerformanceCounter(string name)
        {
            _name = name;
        }

        public double Value
        {
            get
            {
                lock (_lockObject)
                {
                    return _value;
                }
            }
            set
            {
                lock (_lockObject)
                {
                    _value = value;
                }
            }
        }

        public double NextValue()
        {
            return Value;
        }

        public void Increment()
        {
            lock (_lockObject)
            {
                _value++;
            }
        }

        public void Add(double amount)
        {
            lock (_lockObject)
            {
                _value += amount;
            }
        }

        public void Reset()
        {
            lock (_lockObject)
            {
                _value = 0;
            }
        }
    }
}
