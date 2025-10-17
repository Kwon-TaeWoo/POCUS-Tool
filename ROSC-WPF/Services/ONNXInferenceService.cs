using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using ROSC.WPF.Models;
using ROSC.WPF.Utilities;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// ONNX Runtime 기반 추론 서비스
    /// 기존 ModelInferenceService를 인터페이스 기반으로 리팩토링
    /// </summary>
    public class ONNXInferenceService : IInferenceService
    {
        private InferenceSession _session256;
        private InferenceSession _session512;
        private readonly ImageProcessingService _imageProcessor;
        private bool _useSmallModel = true;
        private readonly InferenceStats _stats;
        private bool _isDisposed = false;

        public bool IsModelAvailable => (_useSmallModel && _session256 != null) || _session512 != null;
        public string ModelInfo => GetCurrentModelInfo();
        public InferenceType Type => InferenceType.ONNXRuntime;
        public InferenceStats Stats => _stats;

        public ONNXInferenceService(ImageProcessingService imageProcessor)
        {
            _imageProcessor = imageProcessor;
            _stats = new InferenceStats();
        }

        /// <summary>
        /// 모델 로드 (비동기)
        /// </summary>
        public async Task<bool> LoadModelAsync(string modelPath)
        {
            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    // GPU 사용 가능한지 확인
                    var providers = new List<string> { "CUDAExecutionProvider", "CPUExecutionProvider" };
                    var sessionOptions = new SessionOptions();

                    // 모델 크기에 따라 세션 로드
                    if (modelPath.Contains("256") || modelPath.Contains("small"))
                    {
                        if (System.IO.File.Exists(modelPath))
                        {
                            _session256 = new InferenceSession(modelPath, sessionOptions);
                            _stats.IsGPUEnabled = sessionOptions.Providers.Contains("CUDAExecutionProvider");
                            return true;
                        }
                    }
                    else if (modelPath.Contains("512") || modelPath.Contains("large"))
                    {
                        if (System.IO.File.Exists(modelPath))
                        {
                            _session512 = new InferenceSession(modelPath, sessionOptions);
                            _stats.IsGPUEnabled = sessionOptions.Providers.Contains("CUDAExecutionProvider");
                            return true;
                        }
                    }

                    return false;
                }, false, "Load ONNX model");
            });
        }

        /// <summary>
        /// 이미지 추론 (비동기)
        /// </summary>
        public async Task<Mat> PredictImageAsync(Mat inputImage)
        {
            if (!MatHelper.IsValid(inputImage))
                return null;

            var stopwatch = PerformanceHelper.StartTimer();

            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    // 모델 크기에 따라 세션과 타겟 사이즈 결정
                    InferenceSession session;
                    Size targetSize;
                    
                    if (_useSmallModel && _session256 != null)
                    {
                        session = _session256;
                        targetSize = new Size(256, 256);
                    }
                    else if (_session512 != null)
                    {
                        session = _session512;
                        targetSize = new Size(512, 512);
                    }
                    else
                    {
                        return null;
                    }

                    // 이미지 전처리
                    Mat processedImage = _imageProcessor.PreprocessForModel(inputImage, targetSize);
                    if (!MatHelper.IsValid(processedImage))
                        return null;

                    // 텐서 생성
                    var inputTensor = CreateInputTensor(processedImage);
                    if (inputTensor == null)
                        return null;

                    // 추론 실행
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input", inputTensor)
                    };

                    using (var results = session.Run(inputs))
                    {
                        var output = results.First().AsTensor<float>();
                        var result = ProcessOutput(output, targetSize);
                        
                        // 성능 통계 업데이트
                        UpdateStats(PerformanceHelper.StopTimer(stopwatch));
                        
                        return result;
                    }
                }, null, "ONNX inference");
            });
        }

        /// <summary>
        /// 입력 텐서 생성
        /// </summary>
        private Tensor<float> CreateInputTensor(Mat image)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                // Mat을 float 배열로 변환
                var imageData = new float[1, 3, image.Height, image.Width];
                
                // OpenCV Mat 데이터를 C# 배열로 복사
                unsafe
                {
                    float* ptr = (float*)image.DataPointer;
                    for (int c = 0; c < 3; c++)
                    {
                        for (int h = 0; h < image.Height; h++)
                        {
                            for (int w = 0; w < image.Width; w++)
                            {
                                imageData[0, c, h, w] = ptr[h * image.Width * 3 + w * 3 + c];
                            }
                        }
                    }
                }

                return new DenseTensor<float>(imageData, new[] { 1, 3, image.Height, image.Width });
            }, null, "Create input tensor");
        }

        /// <summary>
        /// 출력 후처리
        /// </summary>
        private Mat ProcessOutput(Tensor<float> output, Size originalSize)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                var outputArray = output.ToArray();
                int channels = output.Dimensions[1];
                int height = output.Dimensions[2];
                int width = output.Dimensions[3];

                // Argmax 계산하여 마스크 생성
                Mat mask = new Mat(height, width, MatType.CV_8UC1);
                
                unsafe
                {
                    byte* maskPtr = (byte*)mask.DataPointer;
                    
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            float maxVal = float.MinValue;
                            int maxIdx = 0;
                            
                            // 각 픽셀에서 최대값 인덱스 찾기
                            for (int c = 0; c < channels; c++)
                            {
                                float val = outputArray[c * height * width + h * width + w];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = c;
                                }
                            }
                            
                            // 인덱스를 픽셀 값으로 변환
                            maskPtr[h * width + w] = (byte)(maxIdx * 120);
                        }
                    }
                }

                // 원본 크기로 리사이즈
                Mat resizedMask = new Mat();
                Cv2.Resize(mask, resizedMask, new Size(500, 460), 0, 0, InterpolationFlags.Linear);

                return resizedMask;
            }, null, "Process output");
        }

        /// <summary>
        /// 성능 통계 업데이트
        /// </summary>
        private void UpdateStats(long inferenceTimeMs)
        {
            _stats.LastInferenceTime = inferenceTimeMs;
            _stats.TotalInferences++;
            
            // 평균 추론 시간 계산
            _stats.AverageInferenceTime = (_stats.AverageInferenceTime * (_stats.TotalInferences - 1) + inferenceTimeMs) / _stats.TotalInferences;
            
            // 메모리 사용량 업데이트
            _stats.TotalMemoryUsage = PerformanceHelper.GetMemoryUsageMB();
        }

        /// <summary>
        /// 현재 사용 중인 모델 크기 반환
        /// </summary>
        private string GetCurrentModelInfo()
        {
            if (_useSmallModel && _session256 != null)
                return "ONNX 256x256 (Lightweight)";
            else if (_session512 != null)
                return "ONNX 512x512 (Full)";
            else
                return "No ONNX model loaded";
        }

        /// <summary>
        /// 모델 크기 설정
        /// </summary>
        public void SetModelSize(bool useSmallModel)
        {
            _useSmallModel = useSmallModel;
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;
            _session256?.Dispose();
            _session512?.Dispose();
        }
    }
}
