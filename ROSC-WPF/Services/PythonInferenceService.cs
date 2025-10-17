using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Text;
using OpenCvSharp;
using ROSC.WPF.Models;
using ROSC.WPF.Utilities;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// Python PyTorch 기반 추론 서비스
    /// Python 스크립트를 실행하여 AI 추론 수행
    /// </summary>
    public class PythonInferenceService : IInferenceService
    {
        private readonly ImageProcessingService _imageProcessor;
        private readonly InferenceStats _stats;
        private readonly string _pythonScriptPath;
        private readonly string _pythonExecutable;
        private bool _isModelLoaded = false;
        private bool _isDisposed = false;

        public bool IsModelAvailable => _isModelLoaded;
        public string ModelInfo => "Python PyTorch (Development Mode)";
        public InferenceType Type => InferenceType.PythonPyTorch;
        public InferenceStats Stats => _stats;

        public PythonInferenceService(ImageProcessingService imageProcessor, string pythonExecutable = "python")
        {
            _imageProcessor = imageProcessor;
            _stats = new InferenceStats();
            _pythonExecutable = pythonExecutable;
            _pythonScriptPath = Path.Combine(Directory.GetCurrentDirectory(), "python_inference.py");
            
            // Python 추론 스크립트 생성
            CreatePythonInferenceScript();
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
                    Logger.Info($"Python 모델 로드 시작: {modelPath}");
                    
                    // Python 스크립트에 모델 경로 전달
                    var config = new
                    {
                        ModelPath = modelPath,
                        Action = "load_model"
                    };

                    string result = ExecutePythonScript(config);
                    _isModelLoaded = result.Contains("Model loaded successfully");
                    
                    if (_isModelLoaded)
                    {
                        _stats.IsGPUEnabled = result.Contains("CUDA available");
                        Logger.Info($"Python 모델 로드 완료: {modelPath}");
                        Logger.Info($"GPU 사용 가능: {_stats.IsGPUEnabled}");
                    }
                    else
                    {
                        Logger.Warning($"Python 모델 로드 실패: {modelPath}");
                    }

                    return _isModelLoaded;
                }, false, "Load Python model");
            });
        }

        /// <summary>
        /// 이미지 추론 (비동기)
        /// </summary>
        public async Task<Mat> PredictImageAsync(Mat inputImage)
        {
            if (!MatHelper.IsValid(inputImage) || !_isModelLoaded)
                return null;

            var stopwatch = PerformanceHelper.StartTimer();

            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    // 이미지를 임시 파일로 저장
                    string tempImagePath = PathHelper.GetTempFilePath(".png");
                    inputImage.SaveImage(tempImagePath);

                    try
                    {
                        // Python 스크립트에 추론 요청
                        var config = new
                        {
                            Action = "predict",
                            ImagePath = tempImagePath,
                            OutputPath = PathHelper.GetTempFilePath("_mask.png")
                        };

                        string result = ExecutePythonScript(config);
                        
                        if (result.Contains("Prediction completed"))
                        {
                            // 결과 마스크 로드
                            Mat mask = Cv2.ImRead(config.OutputPath, ImreadModes.Grayscale);
                            
                            // 성능 통계 업데이트
                            UpdateStats(PerformanceHelper.StopTimer(stopwatch));
                            
                            return mask;
                        }

                        return null;
                    }
                    finally
                    {
                        // 임시 파일 정리
                        if (File.Exists(tempImagePath))
                            File.Delete(tempImagePath);
                    }
                }, null, "Python inference");
            });
        }

        /// <summary>
        /// Python 스크립트 실행
        /// </summary>
        private string ExecutePythonScript(object config)
        {
            try
            {
                string configJson = Newtonsoft.Json.JsonConvert.SerializeObject(config);
                string configPath = Path.GetTempFileName();
                
                File.WriteAllText(configPath, configJson);

                var processStartInfo = new ProcessStartInfo
                {
                    FileName = _pythonExecutable,
                    Arguments = $"\"{_pythonScriptPath}\" \"{configPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                using (var process = Process.Start(processStartInfo))
                {
                    if (process != null)
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        
                        process.WaitForExit(30000); // 30초 타임아웃

                        if (process.ExitCode == 0)
                        {
                            return output;
                        }
                        else
                        {
                            ExceptionHelper.LogError(new Exception($"Python script error: {error}"), "Python inference");
                            return "";
                        }
                    }
                }

                return "";
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Execute Python script");
                return "";
            }
        }

        /// <summary>
        /// 모델을 ONNX로 변환
        /// </summary>
        public async Task<bool> ConvertToOnnxAsync(string modelPath, string outputPath)
        {
            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    var config = new
                    {
                        Action = "convert_to_onnx",
                        ModelPath = modelPath,
                        OutputPath = outputPath,
                        InputSize = new[] { 256, 256, 3 }
                    };

                    string result = ExecutePythonScript(config);
                    return result.Contains("ONNX conversion completed");
                }, false, "Convert to ONNX");
            });
        }

        /// <summary>
        /// 모델 정보 가져오기
        /// </summary>
        public async Task<Dictionary<string, object>> GetModelInfoAsync()
        {
            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    var config = new
                    {
                        Action = "get_stats"
                    };

                    string result = ExecutePythonScript(config);
                    if (!string.IsNullOrEmpty(result))
                    {
                        return Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(result);
                    }
                    return new Dictionary<string, object>();
                }, new Dictionary<string, object>(), "Get model info");
            });
        }

        /// <summary>
        /// Python 추론 스크립트 생성
        /// </summary>
        private void CreatePythonInferenceScript()
        {
            try
            {
                if (!File.Exists(_pythonScriptPath))
                {
                    string pythonScript = GeneratePythonInferenceScript();
                    File.WriteAllText(_pythonScriptPath, pythonScript);
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Create Python inference script");
            }
        }

        /// <summary>
        /// Python 추론 스크립트 내용 생성
        /// </summary>
        private string GeneratePythonInferenceScript()
        {
            return @"
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 기존 Python 모델 import
sys.path.append('.')
try:
    from model import VisionTransformer, CONFIGS
    from model_init import model_load_ViT
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print('Warning: Python model not available, using dummy inference')

class PythonInference:
    def __init__(self):
        self.model = None
        self.test_transforms = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

    def load_model(self, model_path):
        try:
            if not MODEL_AVAILABLE:
                print('Model not available, using dummy mode')
                return True
                
            # 모델 로드 (기존 Python 코드 사용)
            target_size = (256, 256, 3) if '256' in model_path else (512, 512, 3)
            self.model, self.test_transforms = model_load_ViT(model_path, target_size)
            self.model.eval()
            print('Model loaded successfully')
            return True
        except Exception as e:
            print(f'Model loading error: {e}')
            return False

    def predict(self, image_path, output_path):
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print('Failed to load image')
                return False

            if not MODEL_AVAILABLE or self.model is None:
                # 더미 추론 (개발용)
                mask = self.create_dummy_mask(image)
            else:
                # 실제 추론
                mask = self.run_inference(image)

            # 결과 저장
            cv2.imwrite(output_path, mask)
            print('Prediction completed')
            return True
        except Exception as e:
            print(f'Prediction error: {e}')
            return False

    def run_inference(self, image):
        # 기존 Python 추론 로직 사용
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.test_transforms(image=img)
        img = transformed['image']
        img = img / 255
        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        
        imgs = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(imgs)
        
        data = np.argmax(preds[0].cpu().detach().numpy(), axis=0) * 120
        mask = cv2.resize(data.astype('uint8'), (500, 460))
        return mask

    def create_dummy_mask(self, image):
        # 개발용 더미 마스크 생성
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 원형 마스크 생성 (테스트용)
        center = (w//2, h//2)
        radius = min(w, h) // 4
        cv2.circle(mask, center, radius, 120, -1)
        
        return cv2.resize(mask, (500, 460))

def main():
    if len(sys.argv) != 2:
        print('Usage: python python_inference.py <config.json>')
        return

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    inference = PythonInference()
    
    if config['action'] == 'load_model':
        success = inference.load_model(config['model_path'])
        print('Model loaded successfully' if success else 'Model loading failed')
    elif config['action'] == 'predict':
        success = inference.predict(config['image_path'], config['output_path'])
        print('Prediction completed' if success else 'Prediction failed')

if __name__ == '__main__':
    main()
";
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

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;
            
            // Python 스크립트 파일 정리 (선택적)
            // File.Delete(_pythonScriptPath);
        }
    }
}
