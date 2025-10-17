using System;
using System.Threading.Tasks;
using ROSC.WPF.Models;
using ROSC.WPF.Utilities;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// 추론 서비스 팩토리
    /// ONNX Runtime과 Python PyTorch 중 선택하여 추론 서비스 생성
    /// </summary>
    public static class InferenceServiceFactory
    {
        /// <summary>
        /// 추론 서비스 생성
        /// </summary>
        public static IInferenceService CreateInferenceService(InferenceType type, ImageProcessingService imageProcessor, string pythonExecutable = "python")
        {
            return type switch
            {
                InferenceType.ONNXRuntime => new ONNXInferenceService(imageProcessor),
                InferenceType.PythonPyTorch => new PythonInferenceService(imageProcessor, pythonExecutable),
                _ => throw new ArgumentException($"Unsupported inference type: {type}")
            };
        }

        /// <summary>
        /// 추론 서비스 생성 (비동기)
        /// </summary>
        public static async Task<IInferenceService> CreateInferenceServiceAsync(InferenceType type, ImageProcessingService imageProcessor, string modelPath, string pythonExecutable = "python")
        {
            var service = CreateInferenceService(type, imageProcessor, pythonExecutable);
            
            if (!string.IsNullOrEmpty(modelPath))
            {
                bool loaded = await service.LoadModelAsync(modelPath);
                if (!loaded)
                {
                    service.Dispose();
                    throw new InvalidOperationException($"Failed to load model: {modelPath}");
                }
            }
            
            return service;
        }

        /// <summary>
        /// 추론 타입 자동 감지
        /// </summary>
        public static InferenceType DetectInferenceType(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
                return InferenceType.ONNXRuntime; // 기본값

            string extension = PathHelper.GetExtension(modelPath).ToLower();
            
            return extension switch
            {
                ".onnx" => InferenceType.ONNXRuntime,
                ".pth" or ".pt" => InferenceType.PythonPyTorch,
                _ => InferenceType.ONNXRuntime // 기본값
            };
        }

        /// <summary>
        /// 추론 서비스 성능 비교
        /// </summary>
        public static async Task<InferenceComparison> CompareInferenceServices(
            ImageProcessingService imageProcessor, 
            string onnxModelPath, 
            string pythonModelPath,
            Mat testImage,
            int testIterations = 10)
        {
            var comparison = new InferenceComparison();

            try
            {
                // ONNX Runtime 테스트
                if (!string.IsNullOrEmpty(onnxModelPath) && PathHelper.FileExists(onnxModelPath))
                {
                    var onnxService = new ONNXInferenceService(imageProcessor);
                    await onnxService.LoadModelAsync(onnxModelPath);
                    
                    var onnxStopwatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < testIterations; i++)
                    {
                        await onnxService.PredictImageAsync(testImage);
                    }
                    onnxStopwatch.Stop();
                    
                    comparison.ONNXStats = new InferenceComparisonStats
                    {
                        AverageTime = onnxStopwatch.ElapsedMilliseconds / (double)testIterations,
                        TotalTime = onnxStopwatch.ElapsedMilliseconds,
                        Iterations = testIterations,
                        IsAvailable = true
                    };
                    
                    onnxService.Dispose();
                }

                // Python PyTorch 테스트
                if (!string.IsNullOrEmpty(pythonModelPath) && PathHelper.FileExists(pythonModelPath))
                {
                    var pythonService = new PythonInferenceService(imageProcessor);
                    await pythonService.LoadModelAsync(pythonModelPath);
                    
                    var pythonStopwatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < testIterations; i++)
                    {
                        await pythonService.PredictImageAsync(testImage);
                    }
                    pythonStopwatch.Stop();
                    
                    comparison.PythonStats = new InferenceComparisonStats
                    {
                        AverageTime = pythonStopwatch.ElapsedMilliseconds / (double)testIterations,
                        TotalTime = pythonStopwatch.ElapsedMilliseconds,
                        Iterations = testIterations,
                        IsAvailable = true
                    };
                    
                    pythonService.Dispose();
                }

                // 성능 비교 결과
                if (comparison.ONNXStats?.IsAvailable == true && comparison.PythonStats?.IsAvailable == true)
                {
                    comparison.Winner = comparison.ONNXStats.AverageTime < comparison.PythonStats.AverageTime 
                        ? InferenceType.ONNXRuntime 
                        : InferenceType.PythonPyTorch;
                    
                    comparison.PerformanceDifference = Math.Abs(comparison.ONNXStats.AverageTime - comparison.PythonStats.AverageTime);
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Compare inference services");
            }

            return comparison;
        }
    }

    /// <summary>
    /// 추론 서비스 성능 비교 결과
    /// </summary>
    public class InferenceComparison
    {
        public InferenceComparisonStats ONNXStats { get; set; }
        public InferenceComparisonStats PythonStats { get; set; }
        public InferenceType Winner { get; set; }
        public double PerformanceDifference { get; set; }
        public DateTime TestTime { get; set; } = DateTime.Now;
    }

    /// <summary>
    /// 추론 서비스 성능 통계
    /// </summary>
    public class InferenceComparisonStats
    {
        public double AverageTime { get; set; }
        public double TotalTime { get; set; }
        public int Iterations { get; set; }
        public bool IsAvailable { get; set; }
        public string ErrorMessage { get; set; }
    }
}
