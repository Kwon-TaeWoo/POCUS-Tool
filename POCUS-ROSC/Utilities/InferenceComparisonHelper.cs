using System;
using System.Threading.Tasks;
using System.Windows;
using OpenCvSharp;
using POCUS.ROSC.Models;
using POCUS.ROSC.Services;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// 추론 서비스 성능 비교 유틸리티
    /// ONNX Runtime vs Python PyTorch 성능 비교
    /// </summary>
    public static class InferenceComparisonHelper
    {
        /// <summary>
        /// 성능 비교 실행
        /// </summary>
        public static async Task<InferenceComparison> RunPerformanceComparison(
            ImageProcessingService imageProcessor,
            string onnxModelPath,
            string pythonModelPath,
            Mat testImage,
            int testIterations = 10)
        {
            var comparison = new InferenceComparison();

            try
            {
                // 테스트 이미지 준비
                if (!MatHelper.IsValid(testImage))
                {
                    testImage = CreateTestImage();
                }

                // ONNX Runtime 테스트
                if (!string.IsNullOrEmpty(onnxModelPath) && PathHelper.FileExists(onnxModelPath))
                {
                    comparison.ONNXStats = await TestONNXInference(imageProcessor, onnxModelPath, testImage, testIterations);
                }

                // Python PyTorch 테스트
                if (!string.IsNullOrEmpty(pythonModelPath) && PathHelper.FileExists(pythonModelPath))
                {
                    comparison.PythonStats = await TestPythonInference(imageProcessor, pythonModelPath, testImage, testIterations);
                }

                // 성능 비교 결과
                if (comparison.ONNXStats?.IsAvailable == true && comparison.PythonStats?.IsAvailable == true)
                {
                    comparison.Winner = comparison.ONNXStats.AverageTime < comparison.PythonStats.AverageTime 
                        ? InferenceType.ONNXRuntime 
                        : InferenceType.PythonPyTorch;
                    
                    comparison.PerformanceDifference = Math.Abs(comparison.ONNXStats.AverageTime - comparison.PythonStats.AverageTime);
                }

                comparison.TestTime = DateTime.Now;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Run performance comparison");
            }

            return comparison;
        }

        /// <summary>
        /// ONNX Runtime 성능 테스트
        /// </summary>
        private static async Task<InferenceComparisonStats> TestONNXInference(
            ImageProcessingService imageProcessor, string modelPath, Mat testImage, int iterations)
        {
            var stats = new InferenceComparisonStats();
            
            try
            {
                var onnxService = new ONNXInferenceService(imageProcessor);
                bool loaded = await onnxService.LoadModelAsync(modelPath);
                
                if (!loaded)
                {
                    stats.IsAvailable = false;
                    stats.ErrorMessage = "Failed to load ONNX model";
                    return stats;
                }

                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                
                for (int i = 0; i < iterations; i++)
                {
                    await onnxService.PredictImageAsync(testImage);
                }
                
                stopwatch.Stop();
                
                stats.AverageTime = stopwatch.ElapsedMilliseconds / (double)iterations;
                stats.TotalTime = stopwatch.ElapsedMilliseconds;
                stats.Iterations = iterations;
                stats.IsAvailable = true;
                
                onnxService.Dispose();
            }
            catch (Exception ex)
            {
                stats.IsAvailable = false;
                stats.ErrorMessage = ex.Message;
            }

            return stats;
        }

        /// <summary>
        /// Python PyTorch 성능 테스트
        /// </summary>
        private static async Task<InferenceComparisonStats> TestPythonInference(
            ImageProcessingService imageProcessor, string modelPath, Mat testImage, int iterations)
        {
            var stats = new InferenceComparisonStats();
            
            try
            {
                var pythonService = new PythonInferenceService(imageProcessor);
                bool loaded = await pythonService.LoadModelAsync(modelPath);
                
                if (!loaded)
                {
                    stats.IsAvailable = false;
                    stats.ErrorMessage = "Failed to load Python model";
                    return stats;
                }

                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                
                for (int i = 0; i < iterations; i++)
                {
                    await pythonService.PredictImageAsync(testImage);
                }
                
                stopwatch.Stop();
                
                stats.AverageTime = stopwatch.ElapsedMilliseconds / (double)iterations;
                stats.TotalTime = stopwatch.ElapsedMilliseconds;
                stats.Iterations = iterations;
                stats.IsAvailable = true;
                
                pythonService.Dispose();
            }
            catch (Exception ex)
            {
                stats.IsAvailable = false;
                stats.ErrorMessage = ex.Message;
            }

            return stats;
        }

        /// <summary>
        /// 테스트 이미지 생성
        /// </summary>
        private static Mat CreateTestImage()
        {
            Mat testImage = new Mat(460, 500, MatType.CV_8UC3, Scalar.All(128));
            
            // 테스트용 패턴 추가
            Cv2.Circle(testImage, new OpenCvSharp.Point(250, 230), 100, Scalar.White, -1);
            Cv2.Circle(testImage, new OpenCvSharp.Point(250, 230), 50, Scalar.Black, -1);
            
            return testImage;
        }

        /// <summary>
        /// 성능 비교 결과 표시
        /// </summary>
        public static void ShowComparisonResults(InferenceComparison comparison)
        {
            UIHelper.SafeInvoke(() =>
            {
                string message = "=== 추론 서비스 성능 비교 ===\n\n";
                
                if (comparison.ONNXStats?.IsAvailable == true)
                {
                    message += $"ONNX Runtime:\n";
                    message += $"  평균 시간: {comparison.ONNXStats.AverageTime:F2}ms\n";
                    message += $"  총 시간: {comparison.ONNXStats.TotalTime:F2}ms\n";
                    message += $"  반복 횟수: {comparison.ONNXStats.Iterations}\n\n";
                }
                else
                {
                    message += $"ONNX Runtime: 사용 불가 ({comparison.ONNXStats?.ErrorMessage})\n\n";
                }

                if (comparison.PythonStats?.IsAvailable == true)
                {
                    message += $"Python PyTorch:\n";
                    message += $"  평균 시간: {comparison.PythonStats.AverageTime:F2}ms\n";
                    message += $"  총 시간: {comparison.PythonStats.TotalTime:F2}ms\n";
                    message += $"  반복 횟수: {comparison.PythonStats.Iterations}\n\n";
                }
                else
                {
                    message += $"Python PyTorch: 사용 불가 ({comparison.PythonStats?.ErrorMessage})\n\n";
                }

                if (comparison.Winner != InferenceType.ONNXRuntime && comparison.Winner != InferenceType.PythonPyTorch)
                {
                    message += "승자: " + comparison.Winner + "\n";
                    message += $"성능 차이: {comparison.PerformanceDifference:F2}ms\n";
                }

                UIHelper.ShowInfoMessage(message, "성능 비교 결과");
            });
        }
    }
}
