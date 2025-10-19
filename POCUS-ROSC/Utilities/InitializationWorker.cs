using System;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows;
using POCUS.ROSC.Models;
using POCUS.ROSC.Services;
using POCUS.ROSC.Utilities;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// 프로그램 초기화를 담당하는 백그라운드 워커
    /// 모델 로드, Python 모듈 준비, 설정 로드 등을 수행
    /// </summary>
    public class InitializationWorker : BackgroundWorker
    {
        private readonly ConfigService _configService;
        private readonly ImageProcessingService _imageProcessor;
        private IInferenceService _inferenceService;
        private VideoProcessingService _videoProcessor;
        private CACCalculationService _cacCalculator;
        private BackgroundWorkerHelper _backgroundWorker;

        // 초기화 단계 정의
        public enum InitializationStep
        {
            Starting = 0,
            LoadingConfig = 1,
            InitializingServices = 2,
            LoadingModels = 3,
            PreparingPython = 4,
            Finalizing = 5,
            Completed = 6
        }

        public event EventHandler<InitializationStep> StepChanged;
        public event EventHandler<string> StatusChanged;
        // public event EventHandler<Exception> ErrorOccurred; // 현재 사용되지 않음
        public event EventHandler<InitializationResult> Completed;

        public InitializationWorker()
        {
            _configService = new ConfigService();
            _imageProcessor = new ImageProcessingService();
            
            WorkerReportsProgress = true;
            WorkerSupportsCancellation = true;
            DoWork += OnDoWork;
            ProgressChanged += OnProgressChanged;
            RunWorkerCompleted += OnRunWorkerCompleted;
        }

        /// <summary>
        /// 초기화 시작
        /// </summary>
        public void StartInitialization()
        {
            if (IsBusy)
            {
                throw new InvalidOperationException("Initialization is already in progress");
            }

            RunWorkerAsync();
        }

        /// <summary>
        /// 백그라운드 작업 실행
        /// </summary>
        private void OnDoWork(object sender, DoWorkEventArgs e)
        {
            var worker = sender as BackgroundWorker;
            var result = new InitializationResult();

            try
            {
                // 1. 시작
                ReportStep(worker, InitializationStep.Starting, "프로그램 초기화를 시작합니다...", 0);
                System.Threading.Thread.Sleep(500); // UI 업데이트를 위한 짧은 지연

                // 2. 설정 로드
                ReportStep(worker, InitializationStep.LoadingConfig, "설정 파일을 로드하고 있습니다...", 10);
                var config = LoadConfigurationAsync().Result;
                result.Config = config;

                // 3. 서비스 초기화
                ReportStep(worker, InitializationStep.InitializingServices, "서비스를 초기화하고 있습니다...", 30);
                InitializeServicesAsync(config).Wait();

                // 4. 모델 로드
                ReportStep(worker, InitializationStep.LoadingModels, "AI 모델을 로드하고 있습니다...", 50);
                LoadModelsAsync(config).Wait();

                // 5. Python 모듈 준비
                ReportStep(worker, InitializationStep.PreparingPython, "Python 모듈을 준비하고 있습니다...", 70);
                PreparePythonModulesAsync().Wait();

                // 6. 최종화
                ReportStep(worker, InitializationStep.Finalizing, "초기화를 완료하고 있습니다...", 90);
                FinalizeInitializationAsync().Wait();

                // 7. 완료
                ReportStep(worker, InitializationStep.Completed, "초기화가 완료되었습니다!", 100);
                result.Success = true;
                result.Services = new InitializedServices
                {
                    ConfigService = _configService,
                    ImageProcessor = _imageProcessor,
                    InferenceService = _inferenceService,
                    VideoProcessor = _videoProcessor,
                    CACCalculator = _cacCalculator,
                    BackgroundWorker = _backgroundWorker
                };

                e.Result = result;
            }
            catch (OperationCanceledException)
            {
                Logger.Info("초기화가 취소되었습니다.");
                result.Success = false;
                result.Error = new OperationCanceledException("초기화가 취소되었습니다.");
                e.Result = result;
            }
            catch (Exception ex)
            {
                Logger.Error($"초기화 중 오류 발생: {ex.Message}", ex);
                result.Success = false;
                result.Error = ex;
                e.Result = result;
            }
        }

        /// <summary>
        /// 단계 보고
        /// </summary>
        private void ReportStep(BackgroundWorker worker, InitializationStep step, string message, int progress)
        {
            if (worker.CancellationPending)
            {
                throw new OperationCanceledException("Initialization was cancelled");
            }

            // 로깅 추가
            Logger.Info($"[초기화 {progress}%] {message}");

            // BackgroundWorker가 아직 실행 중인지 확인하고 안전하게 진행률 보고
            try
            {
                if (worker.IsBusy)
                {
                    worker.ReportProgress(progress, new InitializationProgress
                    {
                        Step = step,
                        Message = message,
                        Progress = progress
                    });
                }
                else
                {
                    Logger.Warning($"BackgroundWorker가 이미 완료되었습니다. 단계 보고를 건너뜁니다: {message}");
                }
            }
            catch (InvalidOperationException ex)
            {
                Logger.Warning($"ReportProgress 호출 실패: {ex.Message}");
                // 예외를 무시하고 계속 진행
            }

            // 이벤트는 항상 발생시킴 (UI 업데이트를 위해)
            StepChanged?.Invoke(this, step);
            StatusChanged?.Invoke(this, message);
        }

        /// <summary>
        /// 설정 로드
        /// </summary>
        private async Task<ConfigSettings> LoadConfigurationAsync()
        {
            return await Task.Run(() =>
            {
                return ExceptionHelper.SafeExecute(() =>
                {
                    var config = _configService.LoadConfig();
                    return config;
                }, new ConfigSettings(), "Load configuration");
            });
        }

        /// <summary>
        /// 서비스 초기화
        /// </summary>
        private async Task InitializeServicesAsync(ConfigSettings config)
        {
            await Task.Run(() =>
            {
                ExceptionHelper.SafeExecute(() =>
                {
                    _videoProcessor = new VideoProcessingService(_imageProcessor);
                    _cacCalculator = new CACCalculationService(_imageProcessor);
                    return true; // 명시적으로 반환
                }, false, "Initialize services");
            });
        }

        /// <summary>
        /// 모델 로드
        /// </summary>
        private async Task LoadModelsAsync(ConfigSettings config)
        {
            try
            {
                // 추론 타입 결정 (ONNX 우선 사용)
                InferenceType inferenceType = InferenceType.ONNXRuntime;
                string modelPath = config.OnnxModelName;
                
                // ONNX 모델이 없으면 Python 모델 사용
                if (!System.IO.File.Exists(modelPath))
                {
                    Logger.Warning($"ONNX 모델을 찾을 수 없습니다: {modelPath}");
                    Logger.Info("Python 모델로 대체합니다.");
                    
                    inferenceType = InferenceType.PythonPyTorch;
                    modelPath = config.SmallModel ? config.SmallModelName : config.LargeModelName;
                }
                
                Logger.Info($"AI 모델 로드 시작 - 타입: {inferenceType}, 경로: {modelPath}");
                
                // 추론 서비스 생성
                _inferenceService = await InferenceServiceFactory.CreateInferenceServiceAsync(
                    inferenceType, _imageProcessor, modelPath);

                // 백그라운드 워커 초기화
                _backgroundWorker = new BackgroundWorkerHelper(_inferenceService, _cacCalculator, maxQueueSize: 5, maxConcurrentProcessing: 2);
                
                Logger.Info($"AI 모델 로드 완료 - {_inferenceService.ModelInfo}");
            }
            catch (Exception ex)
            {
                Logger.Error($"모델 로드 실패: {ex.Message}", ex);
                throw new Exception($"모델 로드 실패: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Python 모듈 준비
        /// </summary>
        private async Task PreparePythonModulesAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    Logger.Info("Python 모듈 가용성을 확인하고 있습니다.");
                    
                    // Python 모듈 가용성 확인
                    var testConfig = new
                    {
                        Action = "get_stats"
                    };

                    // Python 스크립트 실행 테스트
                    string result = ExecutePythonTest(testConfig);
                    if (string.IsNullOrEmpty(result))
                    {
                        Logger.Warning("Python 모듈을 사용할 수 없습니다. ONNX 모델을 사용합니다.");
                        // Python 모듈이 없어도 ONNX 모델을 사용하므로 정상 완료로 처리
                        return;
                    }
                    
                    Logger.Info("Python 모듈이 준비되었습니다.");
                }
                catch (Exception ex)
                {
                    Logger.Warning($"Python 모듈 준비 중 오류 발생: {ex.Message}. ONNX 모델을 사용합니다.");
                    // 예외가 발생해도 ONNX 모델을 사용하므로 정상 완료로 처리
                }
            });
        }

        /// <summary>
        /// Python 테스트 실행
        /// </summary>
        private string ExecutePythonTest(object config)
        {
            try
            {
                string configJson = Newtonsoft.Json.JsonConvert.SerializeObject(config);
                string configPath = System.IO.Path.GetTempFileName();
                
                System.IO.File.WriteAllText(configPath, configJson);

                var processStartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "python_inference.py")}\" \"{configPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };

                using (var process = System.Diagnostics.Process.Start(processStartInfo))
                {
                    if (process != null)
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit(5000); // 5초 타임아웃

                        if (process.ExitCode == 0)
                        {
                            return output;
                        }
                    }
                }

                return "";
            }
            catch
            {
                return "";
            }
        }

        /// <summary>
        /// 최종화
        /// </summary>
        private async Task FinalizeInitializationAsync()
        {
            await Task.Run(() =>
            {
                System.Threading.Thread.Sleep(500); // 최종화 시뮬레이션
            });
        }

        /// <summary>
        /// 진행률 변경 이벤트
        /// </summary>
        private void OnProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (e.UserState is InitializationProgress progress)
            {
                // UI 업데이트는 MainWindow에서 처리
            }
        }

        /// <summary>
        /// 작업 완료 이벤트
        /// </summary>
        private void OnRunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Result is InitializationResult result)
            {
                Completed?.Invoke(this, result);
            }
        }
    }

    /// <summary>
    /// 초기화 진행률 정보
    /// </summary>
    public class InitializationProgress
    {
        public InitializationWorker.InitializationStep Step { get; set; }
        public string Message { get; set; }
        public int Progress { get; set; }
    }

    /// <summary>
    /// 초기화 결과
    /// </summary>
    public class InitializationResult
    {
        public bool Success { get; set; }
        public Exception Error { get; set; }
        public ConfigSettings Config { get; set; }
        public InitializedServices Services { get; set; }
    }

    /// <summary>
    /// 초기화된 서비스들
    /// </summary>
    public class InitializedServices
    {
        public ConfigService ConfigService { get; set; }
        public ImageProcessingService ImageProcessor { get; set; }
        public IInferenceService InferenceService { get; set; }
        public VideoProcessingService VideoProcessor { get; set; }
        public CACCalculationService CACCalculator { get; set; }
        public BackgroundWorkerHelper BackgroundWorker { get; set; }
    }
}
