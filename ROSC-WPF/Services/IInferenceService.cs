using System;
using System.Threading.Tasks;
using OpenCvSharp;
using ROSC.WPF.Models;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// AI 추론 서비스 인터페이스
    /// ONNX Runtime과 Python PyTorch 두 가지 구현을 지원
    /// </summary>
    public interface IInferenceService : IDisposable
    {
        /// <summary>
        /// 모델 로드
        /// </summary>
        Task<bool> LoadModelAsync(string modelPath);

        /// <summary>
        /// 이미지 추론
        /// </summary>
        Task<Mat> PredictImageAsync(Mat inputImage);

        /// <summary>
        /// 모델 사용 가능 여부
        /// </summary>
        bool IsModelAvailable { get; }

        /// <summary>
        /// 현재 모델 정보
        /// </summary>
        string ModelInfo { get; }

        /// <summary>
        /// 추론 타입
        /// </summary>
        InferenceType Type { get; }

        /// <summary>
        /// 추론 성능 통계
        /// </summary>
        InferenceStats Stats { get; }
    }

    /// <summary>
    /// 추론 타입
    /// </summary>
    public enum InferenceType
    {
        ONNXRuntime,    // C# ONNX Runtime
        PythonPyTorch   // Python PyTorch
    }

    /// <summary>
    /// 추론 성능 통계
    /// </summary>
    public class InferenceStats
    {
        public int TotalInferences { get; set; } = 0;
        public double AverageInferenceTime { get; set; } = 0.0;
        public double LastInferenceTime { get; set; } = 0.0;
        public DateTime LastInferenceTime { get; set; } = DateTime.Now;
        public long TotalMemoryUsage { get; set; } = 0;
        public bool IsGPUEnabled { get; set; } = false;
    }
}
