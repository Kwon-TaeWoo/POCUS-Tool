using System;
using System.ComponentModel;
using System.Threading.Tasks;
using System.Windows;
using ROSC.WPF.Models;
using ROSC.WPF.Services;
using ROSC.WPF.Utilities;

namespace ROSC.WPF.Utilities
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
        public event EventHandler<Exception> ErrorOccurred;
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
        private async void OnDoWork(object sender, DoWorkEventArgs e)
        {
            var worker = sender as BackgroundWorker;
            var result = new InitializationResult();

            try
            {
                // 1. 시작
                ReportStep(worker, InitializationStep.Starting, "프로그램 초기화를 시작합니다...", 0);
                await Task.Delay(500); // UI 업데이트를 위한 짧은 지연

                // 2. 설정 로드
                ReportStep(worker, InitializationStep.LoadingConfig, "설정 파일을 로드하고 있습니다...", 10);
                var config = await LoadConfigurationAsync();
                result.Config = config;

                // 3. 서비스 초기화
                ReportStep(worker, InitializationStep.InitializingServices, "서비스를 초기화하고 있습니다...", 30);
                await InitializeServicesAsync(config);

                // 4. 모델 로드
                ReportStep(worker, InitializationStep.LoadingModels, "AI 모델을 로드하고 있습니다...", 50);
                await LoadModelsAsync(config);

                // 5. Python 모듈 준비
                ReportStep(worker, InitializationStep.PreparingPython, "Python 모듈을 준비하고 있습니다...", 70);
                await PreparePythonModulesAsync();

                // 6. 최종화
                ReportStep(worker, InitializationStep.Finalizing, "초기화를 완료하고 있습니다...", 90);
                await FinalizeInitializationAsync();

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
            catch (Exception ex)
            {
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

            worker.ReportProgress(progress, new InitializationProgress
            {
                Step = step,
                Message = message,
                Progress = progress
            });

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
                }, null, "Initialize services");
            });
        }

        /// <summary>
        /// 모델 로드
        /// </summary>
        private async Task LoadModelsAsync(ConfigSettings config)
        {
            try
            {
                // 추론 타입 결정
                InferenceType inferenceType = config.UsePythonInference 
                    ? InferenceType.PythonPyTorch 
                    : InferenceType.ONNXRuntime;

                string modelPath = config.SmallModel ? config.SmallModelName : config.LargeModelName;
                
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
                ExceptionHelper.SafeExecute(() =>
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
                        Logger.Warning("Python 모듈을 사용할 수 없습니다. Python 환경을 확인해주세요.");
                        throw new Exception("Python 모듈을 사용할 수 없습니다. Python 환경을 확인해주세요.");
                    }
                    
                    Logger.Info("Python 모듈이 준비되었습니다.");
                }, null, "Prepare Python modules");
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
            await Task.Delay(500); // 최종화 시뮬레이션
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
