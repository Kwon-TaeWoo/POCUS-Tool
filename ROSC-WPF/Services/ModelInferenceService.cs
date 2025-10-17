using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using ROSC.WPF.Models;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// ONNX 모델 추론 서비스
    /// Python의 model_init.py와 predict_image() 함수를 구현
    /// </summary>
    public class ModelInferenceService : IDisposable
    {
        private InferenceSession _session256;
        private InferenceSession _session512;
        private readonly ImageProcessingService _imageProcessor;
        private bool _useSmallModel = true;

        public ModelInferenceService(ImageProcessingService imageProcessor)
        {
            _imageProcessor = imageProcessor;
        }

        /// <summary>
        /// ONNX 모델 로드
        /// </summary>
        public bool LoadModel(string modelPath256, string modelPath512)
        {
            try
            {
                // GPU 사용 가능한지 확인
                var providers = new List<string> { "CUDAExecutionProvider", "CPUExecutionProvider" };
                var sessionOptions = new SessionOptions();

                // 256x256 모델 로드
                if (!string.IsNullOrEmpty(modelPath256) && System.IO.File.Exists(modelPath256))
                {
                    _session256 = new InferenceSession(modelPath256, sessionOptions);
                    Console.WriteLine("256x256 model loaded successfully");
                }

                // 512x512 모델 로드
                if (!string.IsNullOrEmpty(modelPath512) && System.IO.File.Exists(modelPath512))
                {
                    _session512 = new InferenceSession(modelPath512, sessionOptions);
                    Console.WriteLine("512x512 model loaded successfully");
                }

                return _session256 != null || _session512 != null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Model loading error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 모델 크기 선택
        /// </summary>
        public void SetModelSize(bool useSmallModel)
        {
            _useSmallModel = useSmallModel;
        }

        /// <summary>
        /// 이미지 추론
        /// Python의 predict_image() 함수와 동일
        /// </summary>
        public Mat PredictImage(Mat inputImage)
        {
            if (inputImage == null || inputImage.Empty())
                return null;

            try
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
                    Console.WriteLine("No model available");
                    return null;
                }

                // 이미지 전처리
                Mat processedImage = _imageProcessor.PreprocessForModel(inputImage, targetSize);
                if (processedImage == null)
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
                    return ProcessOutput(output, targetSize);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Prediction error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 입력 텐서 생성
        /// </summary>
        private Tensor<float> CreateInputTensor(Mat image)
        {
            try
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
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Tensor creation error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 출력 후처리
        /// Python의 후처리와 동일
        /// </summary>
        private Mat ProcessOutput(Tensor<float> output, Size originalSize)
        {
            try
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
                            
                            // 인덱스를 픽셀 값으로 변환 (Python과 동일)
                            maskPtr[h * width + w] = (byte)(maxIdx * 120);
                        }
                    }
                }

                // 원본 크기로 리사이즈
                Mat resizedMask = new Mat();
                Cv2.Resize(mask, resizedMask, new Size(500, 460), 0, 0, InterpolationFlags.Linear);

                return resizedMask;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Output processing error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 모델 사용 가능 여부 확인
        /// </summary>
        public bool IsModelAvailable()
        {
            return (_useSmallModel && _session256 != null) || _session512 != null;
        }

        /// <summary>
        /// 현재 사용 중인 모델 크기 반환
        /// </summary>
        public string GetCurrentModelInfo()
        {
            if (_useSmallModel && _session256 != null)
                return "256x256 (Lightweight)";
            else if (_session512 != null)
                return "512x512 (Full)";
            else
                return "No model loaded";
        }

        public void Dispose()
        {
            _session256?.Dispose();
            _session512?.Dispose();
        }
    }
}
