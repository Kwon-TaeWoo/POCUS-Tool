using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using System.Text;
using System.IO;
using System.Diagnostics;
using Microsoft.Win32;
using OpenCvSharp;
using OxyPlot;
using OxyPlot.Series;
using WpfPoint = System.Windows.Point;
using WpfSize = System.Windows.Size;
using WpfRect = System.Windows.Rect;
using WpfWindow = System.Windows.Window;
using WpfVisibility = System.Windows.Visibility;
using WpfWindowState = System.Windows.WindowState;
using WpfVerticalAlignment = System.Windows.VerticalAlignment;
using WpfHorizontalAlignment = System.Windows.HorizontalAlignment;
using WpfFontWeights = System.Windows.FontWeights;
using POCUS.ROSC.Models;
using POCUS.ROSC.Services;
using POCUS.ROSC.Utilities;

namespace POCUS.ROSC
{
    /// <summary>
    /// MainWindow.xaml에 대한 상호 작용 논리
    /// Python의 UI_transunet_inference_240717_abp.py와 동일한 기능
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        // 서비스들
        private ConfigService _configService;
        private ImageProcessingService _imageProcessor;
        private IInferenceService _inferenceService;
        private VideoProcessingService _videoProcessor;
        private CACCalculationService _cacCalculator;
        private ModelInferenceService _modelInference; // 블록 기반 추론

        // 설정 및 상태
        private ConfigSettings _config;
        private AppState _currentState = AppState.Initializing;
        private bool _isCalculating = false;
        
        // 초기화 관련
        private Views.InitializationWindow _initWindow;
        // private bool _isInitialized = false; // 현재 사용되지 않음
        private bool _isPlaying = false;
        private bool _isVideoMode = false;
        private bool _isHDMIConnected = false;

        // 타이머
        private DispatcherTimer _frameTimer;
        private DispatcherTimer _autoSaveTimer;
        
        // 백그라운드 워커
        private BackgroundWorkerHelper _backgroundWorker;
        
        // 성능 모니터링
        private DateTime _performanceStartTime;
        private int _processedFrameCount;
        private DispatcherTimer _performanceTimer;
        
        // 프레임 처리 최적화
        private int _frameSkipCounter = 0;
        private int _targetFPS = 15; // 목표 FPS (15fps로 설정)
        private DateTime _lastInferenceTime = DateTime.MinValue;

        // 데이터 저장
        private List<MeasurementData> _measurements;
        private List<string> _imageFileNames;
        private GraphHelper.SlidingWindowBuffer<double> _cacBuffer;
        private GraphHelper.SlidingWindowBuffer<double> _ijvBuffer;
        private GraphHelper.SlidingWindowBuffer<double> _abpBuffer;

        // ROI 선택
        private bool _isDrawingROI = false;
        private WpfPoint _roiStartPoint;
        private WpfPoint _roiEndPoint;

        // 그래프 데이터
        private List<DataPoint> _cacDataPoints;
        private List<DataPoint> _ijvDataPoints;
        private List<DataPoint> _abpDataPoints;


        public MainWindow()
        {
            InitializeComponent();
            
            // 로거 초기화
            Logger.Initialize();
            Logger.Info("ROSC-WPF 애플리케이션이 시작되었습니다.");
            
            // 오래된 로그 파일 정리 (백그라운드에서)
            Task.Run(() =>
            {
                Logger.CleanOldLogs(7); // 7일 이상 된 로그 파일 삭제
            });
            
            StartInitialization();
        }

        /// <summary>
        /// 초기화 시작
        /// </summary>
        public void StartInitialization()
        {
            try
            {
                Logger.Info("프로그램 초기화를 시작합니다.");
                
                // 초기화 윈도우 표시
                _initWindow = new Views.InitializationWindow();
                _initWindow.InitializationCompleted += OnInitializationCompleted;
                _initWindow.Show();
                
                // 메인 윈도우는 숨김
                this.Visibility = Visibility.Hidden;
                
                // 초기화 시작
                _initWindow.StartInitialization();
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("초기화 시작 중 오류가 발생했습니다.", ex, "StartInitialization");
            }
        }

        /// <summary>
        /// 초기화 완료 이벤트
        /// </summary>
        private void OnInitializationCompleted(object sender, InitializationResult result)
        {
            if (result.Success)
            {
                try
                {
                    Logger.Info("프로그램 초기화가 완료되었습니다.");
                    
                    // 초기화된 서비스들 받기
                    _config = result.Config;
                    _imageProcessor = result.Services.ImageProcessor;
                    _inferenceService = result.Services.InferenceService;
                    _videoProcessor = result.Services.VideoProcessor;
                    _cacCalculator = result.Services.CACCalculator;
                    _backgroundWorker = result.Services.BackgroundWorker;
                    
                    Logger.Info($"추론 서비스 초기화 완료: {_inferenceService?.ModelInfo}");
                    
                    // 이벤트 구독
                    // _videoProcessor.FrameCaptured += OnFrameCaptured; // 현재 사용하지 않음
                    _videoProcessor.StatusChanged += OnStatusChanged;
                    _backgroundWorker.FrameProcessed += OnFrameProcessed;
                    _backgroundWorker.ErrorOccurred += OnBackgroundError;
                    
                    // UI 초기화
                    InitializeData();
                    InitializeTimers();
                    InitializeGraphs();
                    
                    // 메인 윈도우 표시
                    this.Visibility = Visibility.Visible;
                    this.WindowState = WindowState.Maximized;
                    this.Activate();
                    
                    Logger.Info("메인 윈도우가 표시되었습니다.");
                    UpdateUIFromConfig();
                    
                    // 상태 변경
                    _currentState = AppState.Ready;
                    
                    // 초기화 윈도우 정리
                    _initWindow?.Close();
                    _initWindow = null;
                    
                    // 상태 UI 업데이트
                    UpdateAppState(AppState.Ready);
                    UpdateStatus("프로그램이 준비되었습니다.");
                    
                    Logger.Info("메인 윈도우가 준비되었습니다. 사용자 입력을 기다립니다.");
                }
                catch (Exception ex)
                {
                    Logger.Error($"초기화 완료 후 오류 발생: {ex.Message}", ex);
                    ExceptionHelper.ShowErrorAndLog("초기화 완료 후 오류가 발생했습니다.", ex, "OnInitializationCompleted");
                    
                    // 오류 발생 시 프로그램 종료
                    Application.Current.Shutdown();
                }
            }
            else
            {
                // 초기화 실패 시
                Logger.Error($"초기화 실패: {result.Error?.Message}", result.Error);
                ExceptionHelper.ShowErrorAndLog($"초기화에 실패했습니다: {result.Error?.Message}", result.Error, "OnInitializationCompleted");
                
                // 프로그램 종료
                Application.Current.Shutdown();
            }
        }

        /// <summary>
        /// 설정에서 UI 업데이트
        /// </summary>
        private void UpdateUIFromConfig()
        {
            if (_config == null) return;
            
            // UI 설정 반영
            cbAutoCalc.IsChecked = _config.AutoCalculate;
            cbAutoROI.IsChecked = _config.AutoROI;
            cbAutoSave.IsChecked = _config.AutoSave;
            cbAutoFolder.IsChecked = _config.AutoFolder;
            cbSmallModel.IsChecked = _config.SmallModel;
            cbDrawEllipse.IsChecked = _config.DrawEllipse;
            cbPythonInference.IsChecked = _config.UsePythonInference;
            
            // 초기화 완료 후 UI 활성화
            EnableUIAfterInitialization();
        }

        /// <summary>
        /// 초기화 완료 후 UI 활성화
        private void EnableUIAfterInitialization()
        {
            // 상태에 따른 버튼 활성화 (Ready 상태)
            UpdateButtonStates(AppState.Ready);
            
            // 기타 버튼들 활성화
            btnExportImage.IsEnabled = true;
            btnExportExcel.IsEnabled = true;
            btnCompareInference.IsEnabled = true;
            btnSelectModel.IsEnabled = true;
            btnConvertToOnnx.IsEnabled = true;
            btnCheckCUDA.IsEnabled = true;
            btnViewLogs.IsEnabled = true;
            
            // 체크박스 활성화
            cbAutoCalc.IsEnabled = true;
            cbAutoROI.IsEnabled = true;
            cbAutoSave.IsEnabled = true;
            cbAutoFolder.IsEnabled = true;
            cbSmallModel.IsEnabled = true;
            cbDrawEllipse.IsEnabled = true;
            cbPythonInference.IsEnabled = true;
            cbPerformanceMode.IsEnabled = true;
        }

        /// <summary>
        /// 애플리케이션 상태 업데이트
        /// </summary>
        private void UpdateAppState(AppState newState)
        {
            _currentState = newState;
            
            // UI 스레드에서 상태 표시 업데이트
            Dispatcher.Invoke(() =>
            {
                statusText.Text = newState.ToDisplayString();
                
                // 상태에 따른 색상 변경
                statusIndicator.Fill = newState switch
                {
                    AppState.Initializing => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 165, 0)), // 주황색
                    AppState.Ready => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0, 255, 0)), // 녹색
                    AppState.Load => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0, 122, 204)), // 파란색
                    AppState.Play => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 255, 0)), // 노란색
                    AppState.Calc => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(255, 0, 0)), // 빨간색
                    _ => new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(128, 128, 128)) // 회색
                };
                
                // 상태에 따른 버튼 활성화/비활성화
                UpdateButtonStates(newState);
            });
        }

        /// <summary>
        /// 상태에 따른 버튼 상태 업데이트
        /// </summary>
        private void UpdateButtonStates(AppState state)
        {
            switch (state)
            {
                case AppState.Ready:
                    btnConnectHDMI.IsEnabled = true;
                    btnLoadVideo.IsEnabled = true;
                    btnPlay.IsEnabled = false;
                    btnCalculateCAC.IsEnabled = false;
                    btnConnectHDMI.Content = "Connect HDMI";
                    btnPlay.Content = "Play";
                    btnCalculateCAC.Content = "Calculate";
                    break;
                    
                case AppState.Load:
                    btnConnectHDMI.IsEnabled = true;
                    btnLoadVideo.IsEnabled = !_isHDMIConnected; // HDMI 연결 시 비디오 로드 비활성화
                    btnPlay.IsEnabled = true;
                    btnCalculateCAC.IsEnabled = true;
                    btnConnectHDMI.Content = _isHDMIConnected ? "Disconnect HDMI" : "Connect HDMI";
                    btnPlay.Content = "Play";
                    btnCalculateCAC.Content = "Calculate";
                    break;
                    
                case AppState.Play:
                    btnConnectHDMI.IsEnabled = true;
                    btnLoadVideo.IsEnabled = !_isHDMIConnected; // HDMI 연결 시 비디오 로드 비활성화
                    btnPlay.IsEnabled = true;
                    btnCalculateCAC.IsEnabled = true;
                    btnConnectHDMI.Content = _isHDMIConnected ? "Disconnect HDMI" : "Connect HDMI";
                    btnPlay.Content = "Stop";
                    btnCalculateCAC.Content = "Calculate";
                    break;
                    
                case AppState.Calc:
                    btnConnectHDMI.IsEnabled = true;
                    btnLoadVideo.IsEnabled = !_isHDMIConnected; // HDMI 연결 시 비디오 로드 비활성화
                    btnPlay.IsEnabled = true;
                    btnCalculateCAC.IsEnabled = true;
                    btnConnectHDMI.Content = _isHDMIConnected ? "Disconnect HDMI" : "Connect HDMI";
                    btnPlay.Content = "Stop";
                    btnCalculateCAC.Content = "Stop Calculate";
                    break;
                    
                default:
                    btnConnectHDMI.IsEnabled = false;
                    btnLoadVideo.IsEnabled = false;
                    btnPlay.IsEnabled = false;
                    btnCalculateCAC.IsEnabled = false;
                    break;
            }
        }

        /// <summary>
        /// 서비스 초기화 (기존 메서드 - 초기화 워커에서 사용)
        /// </summary>
        private void InitializeServices()
        {
            _imageProcessor = new ImageProcessingService();
            _videoProcessor = new VideoProcessingService(_imageProcessor);
            _cacCalculator = new CACCalculationService(_imageProcessor);
            _configService = new ConfigService();
            
            // 추론 서비스는 설정 로드 후 초기화
            _inferenceService = null;

            // 이벤트 구독
            // _videoProcessor.FrameCaptured += OnFrameCaptured; // 현재 사용하지 않음
            _videoProcessor.StatusChanged += OnStatusChanged;
        }

        /// <summary>
        /// 추론 서비스 초기화
        private async Task InitializeInferenceServiceAsync()
        {
            try
            {
                // 추론 방식 결정 (설정에 따라 자동 감지)
                InferenceType inferenceType = _config.UsePythonInference 
                    ? InferenceType.PythonPyTorch 
                    : InferenceType.ONNXRuntime;

                string modelPath = _config.SmallModel ? _config.SmallModelName : _config.LargeModelName;
                
                // 추론 서비스 생성
                _inferenceService = await InferenceServiceFactory.CreateInferenceServiceAsync(
                    inferenceType, _imageProcessor, modelPath);

                // 백그라운드 워커 초기화 (AI 서비스들 전달)
                _backgroundWorker = new BackgroundWorkerHelper(_inferenceService, _cacCalculator, maxQueueSize: 5, maxConcurrentProcessing: 2);
                
                // 이벤트 구독
                _backgroundWorker.FrameProcessed += OnFrameProcessed;
                _backgroundWorker.ErrorOccurred += OnBackgroundError;

                UpdateStatus($"추론 서비스 초기화 완료: {_inferenceService.ModelInfo}");
            }
            catch (Exception ex)
            {
                UpdateStatus($"추론 서비스 초기화 실패: {ex.Message}");
            }
        }

        /// <summary>
        /// 추론 서비스 재초기화
        /// </summary>
        private async Task ReinitializeInferenceServiceAsync()
        {
            try
            {
                // 기존 서비스 정리
                _backgroundWorker?.Dispose();
                _inferenceService?.Dispose();

                // 새 서비스 초기화
                await InitializeInferenceServiceAsync();
            }
            catch (Exception ex)
            {
                UpdateStatus($"추론 서비스 재초기화 실패: {ex.Message}");
            }
        }

        /// <summary>
        /// 데이터 초기화
        private void InitializeData()
        {
            _measurements = new List<MeasurementData>();
            _imageFileNames = new List<string>();
            _cacBuffer = new GraphHelper.SlidingWindowBuffer<double>(10);
            _ijvBuffer = new GraphHelper.SlidingWindowBuffer<double>(10);
            _abpBuffer = new GraphHelper.SlidingWindowBuffer<double>(10);

            _cacDataPoints = new List<DataPoint>();
            _ijvDataPoints = new List<DataPoint>();
            _abpDataPoints = new List<DataPoint>();
            
            // CAC 그래프 PlotModel 생성
            InitializeCacPlotModel();
            
            // ABP 그래프 PlotModel 생성
            InitializeAbpPlotModel();
        }
        
        /// <summary>
        /// CAC 그래프 PlotModel 초기화
        private void InitializeCacPlotModel()
        {
            var plotModel = new PlotModel
            {
                TextColor = OxyColors.White,
                PlotAreaBorderColor = OxyColors.Gray,
                Background = OxyColors.Black
            };
            
            // X�?추�?
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = OxyPlot.Axes.AxisPosition.Bottom,
                Title = "Frame",
                TextColor = OxyColors.White,
                TicklineColor = OxyColors.Gray
            });
            
            // Y�?추�?
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = OxyPlot.Axes.AxisPosition.Left,
                Title = "CAC Value",
                Minimum = 0.2,
                Maximum = 1.0,
                TextColor = OxyColors.White,
                TicklineColor = OxyColors.Gray
            });
            
            // CAC 라인 시리즈
            var cacLineSeries = new LineSeries
            {
                Title = "CAC",
                Color = OxyColors.Red,
                StrokeThickness = 2
            };
            plotModel.Series.Add(cacLineSeries);
            
            // IJV 라인 시리즈
            var ijvLineSeries = new LineSeries
            {
                Title = "IJV",
                Color = OxyColors.Blue,
                StrokeThickness = 2
            };
            plotModel.Series.Add(ijvLineSeries);
            
            // Threshold 라인 시리즈
            var thresholdLineSeries = new LineSeries
            {
                Title = "Threshold 0.94",
                Color = OxyColors.Yellow,
                LineStyle = LineStyle.Dash,
                StrokeThickness = 2
            };
            plotModel.Series.Add(thresholdLineSeries);
            
            cacGraph.Model = plotModel;
        }
        
        /// <summary>
        /// ABP 그래프 PlotModel 초기화
        /// </summary>
        private void InitializeAbpPlotModel()
        {
            var plotModel = new PlotModel
            {
                TextColor = OxyColors.White,
                PlotAreaBorderColor = OxyColors.Gray,
                Background = OxyColors.Black
            };
            
            // X축 추가
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = OxyPlot.Axes.AxisPosition.Bottom,
                Title = "CAC Value",
                Minimum = 0.3,
                Maximum = 1.05,
                TextColor = OxyColors.White,
                TicklineColor = OxyColors.Gray
            });
            
            // Y축 추가
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis
            {
                Position = OxyPlot.Axes.AxisPosition.Left,
                Title = "Pred BP Value",
                Minimum = 0,
                Maximum = 180,
                TextColor = OxyColors.White,
                TicklineColor = OxyColors.Gray
            });
            
            // SBP 라인 시리즈
            var sbpLineSeries = new LineSeries
            {
                Title = "SBP Line",
                Color = OxyColors.Red,
                StrokeThickness = 1
            };
            plotModel.Series.Add(sbpLineSeries);
            
            // MBP 라인 시리즈
            var mbpLineSeries = new LineSeries
            {
                Title = "MBP Line",
                Color = OxyColors.Green,
                StrokeThickness = 1
            };
            plotModel.Series.Add(mbpLineSeries);
            
            // DBP 라인 시리즈
            var dbpLineSeries = new LineSeries
            {
                Title = "DBP Line",
                Color = OxyColors.Magenta,
                StrokeThickness = 1
            };
            plotModel.Series.Add(dbpLineSeries);
            
            // ABP Scatter 시리즈
            var abpScatterSeries = new ScatterSeries
            {
                Title = "Current ABP",
                MarkerType = MarkerType.Circle,
                MarkerSize = 8,
                MarkerFill = OxyColors.Yellow
            };
            plotModel.Series.Add(abpScatterSeries);
            
            abpGraph.Model = plotModel;
        }

        /// <summary>
        /// 타이머 초기화
        /// </summary>
        private void InitializeTimers()
        {
            _frameTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(33) // 약 30fps (33ms 간격)
            };
            _frameTimer.Tick += FrameTimer_Tick;

            _autoSaveTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMinutes(3) // 3분마다 자동저장
            };
            _autoSaveTimer.Tick += AutoSaveTimer_Tick;

            // 성능 모니터링 타이머
            _performanceTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1) // 1초마다 성능 체크
            };
            _performanceTimer.Tick += PerformanceTimer_Tick;
        }

        /// <summary>
        /// 그래프 초기화
        /// </summary>
        private void InitializeGraphs()
        {
            // CAC 그래프 초기화
            InitializeCACGraph();
            
            // ABP 그래프 초기화
            InitializeABPGraph();
        }

        /// <summary>
        /// CAC 그래프 초기화
        /// </summary>
        private void InitializeCACGraph()
        {
            // CAC 그래프는 이미 InitializeCacPlotModel에서 초기화됨
            // 추가 초기화가 필요한 경우 여기에 구현
        }

        /// <summary>
        /// ABP 그래프 초기화
        /// </summary>
        private void InitializeABPGraph()
        {
            // ABP 그래프는 이미 InitializeAbpPlotModel에서 초기화됨
            // 추가 초기화가 필요한 경우 여기에 구현
        }

        /// <summary>
        /// 설정 로드 (초기화 워커에서 처리됨)
        /// </summary>
        private void LoadConfiguration()
        {
            // 이 메서드는 현재 사용되지 않음
            // 초기화는 InitializationWorker에서 처리됨
        }

        /// <summary>
        /// 모델 로드
        /// </summary>
        private void LoadModels()
        {
            try
            {
                // ModelInferenceService 초기화
                if (_modelInference == null)
                {
                    _modelInference = new ModelInferenceService(_imageProcessor);
                }
                
                bool success = _modelInference.LoadModel(_config.SmallModelName, _config.LargeModelName);
                if (success)
                {
                    _modelInference.SetModelSize(_config.SmallModel);
                    UpdateStatus("모델 로드 ?�료");
                }
                else
                {
                    UpdateStatus("모델 로드 ?�패");
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"모델 로드 ?�류: {ex.Message}");
            }
        }

        /// <summary>
        /// UI 상태 업데이트
        /// Python의 check_state_Ui() 함수와 동일
        /// </summary>
        private void UpdateUIState()
        {
            switch (_currentState)
            {
                case AppState.Ready:
                    UpdateReadyState();
                    break;
                case AppState.Load:
                    UpdateLoadState();
                    break;
                case AppState.Play:
                    UpdatePlayState();
                    break;
                case AppState.Calc:
                    UpdateCalcState();
                    break;
            }
        }

        private void UpdateReadyState()
        {
            btnConnectHDMI.Content = "Reconnect";
            btnConnectHDMI.IsEnabled = true;
            btnLoadVideo.IsEnabled = true;
            btnCalculateCAC.IsEnabled = false;
            btnSetROI.IsEnabled = false;
            btnExportVideo.IsEnabled = false;
            btnExportImage.IsEnabled = false;
            btnExportExcel.IsEnabled = false;

            if (_isVideoMode)
            {
                btnLoadVideo.Content = "Stop Play";
                btnSetROI.IsEnabled = true;
            }
            else
            {
                btnLoadVideo.Content = "Load Video";
                btnSetROI.IsEnabled = false;
            }
        }

        private void UpdateLoadState()
        {
            if (_isPlaying)
            {
                UpdateStatus("Something wrong! Please check");
                StopSlideShow();
            }

            btnLoadVideo.Visibility = Visibility.Collapsed;
            btnConnectHDMI.Content = "Start HDMI";
            btnCalculateCAC.Content = "Calculate";
            btnSetROI.IsEnabled = true;
            btnConnectHDMI.IsEnabled = true;
            btnCalculateCAC.IsEnabled = true;
            btnSetROI.Visibility = Visibility.Visible;
            btnExportVideo.IsEnabled = false;
            btnExportImage.IsEnabled = false;
            btnExportExcel.IsEnabled = false;
            btnExportVideo.Visibility = Visibility.Collapsed;
            btnExportImage.Visibility = Visibility.Collapsed;
            btnExportExcel.Visibility = Visibility.Collapsed;
        }

        private void UpdatePlayState()
        {
            if (_isPlaying)
            {
                btnConnectHDMI.Content = "Stop HDMI";
                btnSetROI.IsEnabled = false;
            }
            else
            {
                btnConnectHDMI.Content = "Start HDMI";
                btnSetROI.IsEnabled = true;
            }

            btnConnectHDMI.IsEnabled = true;
            btnLoadVideo.IsEnabled = false;
            btnLoadVideo.Visibility = Visibility.Collapsed;
            btnCalculateCAC.IsEnabled = true;
            btnCalculateCAC.Visibility = Visibility.Visible;
            btnExportVideo.IsEnabled = false;
            btnExportImage.IsEnabled = false;
            btnExportExcel.IsEnabled = false;
            btnExportVideo.Visibility = Visibility.Collapsed;
            btnExportImage.Visibility = Visibility.Collapsed;
            btnExportExcel.Visibility = Visibility.Collapsed;
        }

        private void UpdateCalcState()
        {
            if (_isCalculating)
            {
                if (_isPlaying)
                {
                    btnConnectHDMI.Content = "Start HDMI";
                    btnConnectHDMI.IsEnabled = false;
                    btnCalculateCAC.Content = "Stop Calculate";
                }
                else
                {
                    btnCalculateCAC.Content = "Calculate";
                    btnConnectHDMI.IsEnabled = true;
                }

                btnLoadVideo.IsEnabled = false;
                btnCalculateCAC.IsEnabled = true;
                btnSetROI.IsEnabled = false;
                btnExportExcel.IsEnabled = true;
                btnExportVideo.IsEnabled = true;
                btnLoadVideo.Visibility = Visibility.Collapsed;
                btnSetROI.Visibility = Visibility.Visible;
                btnExportVideo.Visibility = Visibility.Visible;
                btnExportImage.Visibility = Visibility.Visible;
                btnExportExcel.Visibility = Visibility.Visible;
            }
            else
            {
                btnCalculateCAC.Content = "Calculate";
                btnSetROI.IsEnabled = true;
                btnExportVideo.IsEnabled = false;
                btnExportImage.IsEnabled = false;
                btnExportExcel.IsEnabled = false;
                btnSetROI.Visibility = Visibility.Visible;
                btnExportVideo.Visibility = Visibility.Visible;
                btnExportImage.Visibility = Visibility.Visible;
                btnExportExcel.Visibility = Visibility.Visible;
            }
        }

        #region 이벤트 핸들러들
        /// <summary>
        /// HDMI 연결 버튼 클릭
        /// </summary>
        private void BtnConnectHDMI_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Logger.Info($"HDMI 연결 버튼 클릭 - 현재 상태: {_currentState}");
                
                if (_isHDMIConnected)
                {
                    // HDMI 연결 해제
                    DisconnectHDMI();
                }
                else
                {
                    // HDMI 연결
                    ConnectHDMI();
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("HDMI 연결 버튼 처리 중 오류가 발생했습니다.", ex, "BtnConnectHDMI_Click");
            }
        }

        /// <summary>
        /// 비디오 로드 버튼 클릭
        /// </summary>
        private void BtnLoadVideo_Click(object sender, RoutedEventArgs e)
        {
            LoadVideoFile();
        }

        /// <summary>
        /// Play 버튼 ?�릭
        /// </summary>
        private void BtnPlay_Click(object sender, RoutedEventArgs e)
        {
            switch (_currentState)
            {
                case AppState.Load:
                    StartPlay();
                    break;
                case AppState.Play:
                    StopPlay();
                    break;
                case AppState.Calc:
                    StopInference();
                    break;
            }
        }

        /// <summary>
        /// Play ?�작
        /// </summary>
        private void StartPlay()
        {
            try
            {
                Logger.Info("비디오 재생을 시작합니다.");
                
                _isPlaying = true;
                UpdateAppState(AppState.Play);
                UpdateStatus("비디오 재생을 시작합니다.");
                
                if (_isHDMIConnected)
                {
                    Logger.Info("HDMI 캡처보드에서 비디오를 재생합니다.");
                    StartHDMIShow();
                }
                else if (_isVideoMode)
                {
                    Logger.Info("로드된 비디오 파일을 재생합니다.");
                    StartVideoPlayback();
                }
                else
                {
                    Logger.Warning("재생할 비디오 소스가 없습니다.");
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("비디오 재생 시작 중 오류가 발생했습니다.", ex, "StartPlay");
            }
        }

        /// <summary>
        /// Play 중�?
        /// </summary>
        private void StopPlay()
        {
            _isPlaying = false;
            UpdateAppState(AppState.Load);
            UpdateStatus("비디오 재생을 중지합니다.");
            
            StopSlideShow();
        }

        /// <summary>
        /// 추론 중지
        /// </summary>
        private void StopInference()
        {
            _isCalculating = false;
            UpdateAppState(AppState.Play);
            UpdateStatus("AI 분석을 중지합니다.");
            
            // 백그라운드 워커 중지
            _backgroundWorker?.StopProcessing();
            _frameTimer?.Stop();
            _performanceTimer?.Stop();
            _autoSaveTimer?.Stop();
        }

        /// <summary>
        /// 비디오 재생 시작
        /// </summary>
        private void StartVideoPlayback()
        {
            try
            {
                Logger.Info("비디오 파일 재생을 시작합니다.");
                
                _isPlaying = true;
                UpdateAppState(AppState.Play);
                UpdateStatus("비디오 파일을 재생합니다.");
                
                _frameTimer.Start();
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("비디오 재생 시작 중 오류가 발생했습니다.", ex, "StartVideoPlayback");
            }
        }

        /// <summary>
        /// CAC 계산 버튼 ?�릭
        /// </summary>
        private void BtnCalculateCAC_Click(object sender, RoutedEventArgs e)
        {
            switch (_currentState)
            {
                case AppState.Play:
                    StartCalculate();
                    break;
                case AppState.Calc:
                    StopCalculate();
                    break;
                case AppState.Load:
                    // Load 상태에서 Calculate는 Play 먼저 시작
                    StartPlay();
                    StartCalculate();
                    break;
            }
        }

        /// <summary>
        /// Calculate ?�작
        /// </summary>
        private void StartCalculate()
        {
            try
            {
                string inferenceType = _config.UsePythonInference ? "Python PyTorch" : "ONNX Runtime";
                Logger.Info($"AI 분석을 시작합니다. 추론 방식: {inferenceType}");
                
                _isCalculating = true;
                UpdateAppState(AppState.Calc);
                UpdateStatus("AI 분석을 시작합니다.");
                
                // 성능 모니터링 시작
                _performanceStartTime = DateTime.Now;
                _processedFrameCount = 0;
                
                // 백그라운드 워커 시작
                _backgroundWorker.StartProcessing();
                _frameTimer.Start();
                _performanceTimer.Start();
                
                if (_config.AutoSave)
                {
                    Logger.Info("자동 저장 모드가 활성화되었습니다.");
                    _autoSaveTimer.Start();
                }
                
                Logger.Info("실시간 AI 분석을 시작했습니다.");
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("AI 분석 시작 중 오류가 발생했습니다.", ex, "StartCalculate");
            }
        }

        /// <summary>
        /// Calculate 중지
        /// </summary>
        private void StopCalculate()
        {
            _isCalculating = false;
            UpdateAppState(AppState.Play);
            UpdateStatus("AI 분석을 중지합니다.");
            
            // 백그라운드 워커 중지
            _backgroundWorker?.StopProcessing();
            _frameTimer?.Stop();
            _performanceTimer?.Stop();
            _autoSaveTimer?.Stop();
        }

        /// <summary>
        /// ROI ?�정 버튼 ?�릭
        /// </summary>
        private void BtnSetROI_Click(object sender, RoutedEventArgs e)
        {
            StartROISelection();
        }

        /// <summary>
        /// ROI ?�택 ?�작
        /// </summary>
        private void StartROISelection()
        {
            try
            {
                Logger.Info("ROI 선택을 시작합니다.");
                
                roiCanvas.Visibility = Visibility.Visible;
                roiRectangle.Visibility = Visibility.Collapsed;
                
                UpdateStatus("ROI 영역을 선택하세요.");
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("ROI 선택 시작 중 오류가 발생했습니다.", ex, "StartROISelection");
            }
        }

        /// <summary>
        /// 비디오 내보내기 버튼 클릭
        /// </summary>
        private void BtnExportVideo_Click(object sender, RoutedEventArgs e)
        {
            ExportVideo();
        }

        /// <summary>
        /// 비디오 내보내기
        /// </summary>
        private void ExportVideo()
        {
            try
            {
                var saveFileDialog = new Microsoft.Win32.SaveFileDialog
                {
                    Filter = "AVI files (*.avi)|*.avi",
                    Title = "비디오 저장"
                };

                if (saveFileDialog.ShowDialog() == true)
                {
                    ExportVideoInternal(Path.GetFileNameWithoutExtension(saveFileDialog.FileName));
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("비디오 내보내기 중 오류가 발생했습니다.", ex, "ExportVideo");
            }
        }

        /// <summary>
        /// 이미지 내보내기 버튼 클릭
        /// </summary>
        private void BtnExportImage_Click(object sender, RoutedEventArgs e)
        {
            ExportImages();
        }

        /// <summary>
        /// Excel 내보내기 버튼 클릭
        /// </summary>
        private void BtnExportExcel_Click(object sender, RoutedEventArgs e)
        {
            ExportExcel();
        }

        /// <summary>
        /// Excel 내보내기
        /// </summary>
        private void ExportExcel()
        {
            try
            {
                var saveFileDialog = new Microsoft.Win32.SaveFileDialog
                {
                    Filter = "Excel files (*.xlsx)|*.xlsx",
                    Title = "Excel 저장"
                };

                if (saveFileDialog.ShowDialog() == true)
                {
                    ExportExcelInternal(Path.GetFileNameWithoutExtension(saveFileDialog.FileName));
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("Excel 내보내기 중 오류가 발생했습니다.", ex, "ExportExcel");
            }
        }

        /// <summary>
        /// Python 추론 체크박스 변�?        /// </summary>
        private async void CbPythonInference_Checked(object sender, RoutedEventArgs e)
        {
            try
            {
                _config.UsePythonInference = cbPythonInference.IsChecked == true;
                string inferenceType = _config.UsePythonInference ? "Python PyTorch" : "ONNX Runtime";
                
                Logger.Info($"추론 방식이 {inferenceType}로 변경되었습니다.");
                
                _configService.UpdateSetting("Environment", "UsePythonInference", _config.UsePythonInference.ToString());
                
                // 추론 서비스 재초기화
                await ReinitializeInferenceServiceAsync();
                
                // 상태 메시지 업데이트
                UpdateStatus($"추론 방식이 {inferenceType}로 변경되었습니다.");
                
                Logger.Info($"추론 서비스 재초기화 완료: {inferenceType}");
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("추론 방식 변경 중 오류가 발생했습니다.", ex, "CbPythonInference_Checked");
            }
        }

        /// <summary>
        /// FPS 설정 변경
        /// </summary>
        private void SetTargetFPS(int fps)
        {
            _targetFPS = Math.Max(5, Math.Min(30, fps)); // 5-30fps 범위로 제한
            UpdateStatus($"목표 FPS가 {_targetFPS}fps로 설정되었습니다.");
        }
        
        /// <summary>
        /// 성능 모드 체크박스 변경
        /// </summary>
        private void CbPerformanceMode_Checked(object sender, RoutedEventArgs e)
        {
            // 성능 모드 설정
            bool performanceMode = cbPerformanceMode.IsChecked == true;
            
            if (performanceMode)
            {
                // 성능 모드: 낮은 FPS로 설정
                SetTargetFPS(10);
                UpdateStatus("성능 모드: 10fps로 설정 (낮은 CPU 사용량)");
            }
            else
            {
                // 일반 모드: 높은 FPS로 설정
                SetTargetFPS(15);
                UpdateStatus("일반 모드: 15fps로 설정 (균형잡힌 성능)");
            }
        }

        /// <summary>
        /// 추론 비교 버튼 클릭
        /// </summary>
        private async void BtnCompareInference_Click(object sender, RoutedEventArgs e)
        {
            await RunInferenceComparison();
        }

        /// <summary>
        /// 모델 ?�일 ?�택 버튼 ?�릭
        /// </summary>
        private void BtnSelectModel_Click(object sender, RoutedEventArgs e)
        {
            SelectModelFile();
        }

        /// <summary>
        /// 모델 ?�일 ?�택
        /// </summary>
        private void SelectModelFile()
        {
            try
            {
                var openFileDialog = new Microsoft.Win32.OpenFileDialog
                {
                    Title = "모델 파일 선택",
                    Filter = "모델 파일|*.pth;*.pt;*.onnx;*.npz|PyTorch 모델|*.pth;*.pt|ONNX 모델|*.onnx|NumPy 모델|*.npz|모든 파일|*.*",
                    InitialDirectory = PathHelper.Combine(Directory.GetCurrentDirectory(), "checkpoint")
                };

                if (openFileDialog.ShowDialog() == true)
                {
                    string selectedModelPath = openFileDialog.FileName;
                    string modelType = Path.GetExtension(selectedModelPath).ToLower();
                    
                    // 설정에 모델 경로 저장
                    if (modelType == ".pth" || modelType == ".pt")
                    {
                        if (_config.SmallModel)
                            _config.SmallModelName = selectedModelPath;
                        else
                            _config.LargeModelName = selectedModelPath;
                    }
                    else if (modelType == ".onnx")
                    {
                        // ONNX 모델은 별도 경로에 저장
                        _config.OnnxModelName = selectedModelPath;
                    }

                    // 설정 저장
                    _configService.SaveConfig(_config);
                    
                    // 추론 서비스 재초기화
                    _ = ReinitializeInferenceServiceAsync();
                    
                    UpdateStatus($"모델 파일 선택됨: {Path.GetFileName(selectedModelPath)}");
                    
                    // ONNX 변환 버튼 활성화 (PTH/PT 파일인 경우)
                    btnConvertToOnnx.IsEnabled = modelType == ".pth" || modelType == ".pt";
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"모델 파일 선택 오류: {ex.Message}");
                UIHelper.ShowErrorMessage($"모델 파일 선택 중 오류가 발생했습니다: {ex.Message}");
            }
        }

        /// <summary>
        /// ONNX 변환 버튼 클릭
        /// </summary>
        private async void BtnConvertToOnnx_Click(object sender, RoutedEventArgs e)
        {
            await ConvertPthToOnnx();
        }

        /// <summary>
        /// CUDA 상태 확인 버튼 클릭
        /// </summary>
        private async void BtnCheckCUDA_Click(object sender, RoutedEventArgs e)
        {
            await CheckCUDAStatus();
        }

        /// <summary>
        /// 로그 보기 버튼 ?�릭
        /// </summary>
        private void BtnViewLogs_Click(object sender, RoutedEventArgs e)
        {
            ViewLogs();
        }

        /// <summary>
        /// 로그 보기
        /// </summary>
        private void ViewLogs()
        {
            try
            {
                // 로그 ?�일 목록 조회
                var logFiles = Logger.GetLogFiles();
                
                if (logFiles.Length == 0)
                {
                    Logger.Warning("로그 파일이 존재하지 않습니다.");
                    UIHelper.ShowWarningMessage("로그 파일이 존재하지 않습니다.");
                    return;
                }

                // 로그 파일 선택 창 생성
                var logWindow = new System.Windows.Window
                {
                    Title = "로그 파일 선택",
                    Width = 600,
                    Height = 400,
                    WindowStartupLocation = WindowStartupLocation.CenterOwner,
                    Owner = this
                };

                var stackPanel = new System.Windows.Controls.StackPanel
                {
                    Margin = new Thickness(20)
                };

                var titleText = new System.Windows.Controls.TextBlock
                {
                    Text = "보고 싶은 로그 파일을 선택하세요.",
                    FontSize = 14,
                    FontWeight = WpfFontWeights.Bold,
                    Margin = new Thickness(0, 0, 0, 10)
                };
                stackPanel.Children.Add(titleText);

                var listBox = new System.Windows.Controls.ListBox
                {
                    Height = 250,
                    Margin = new Thickness(0, 0, 0, 10)
                };

                // 로그 ?�일 목록 추�?
                foreach (var logFile in logFiles)
                {
                    var fileName = Path.GetFileName(logFile);
                    var fileInfo = new FileInfo(logFile);
                    var fileSize = fileInfo.Length / 1024.0; // KB
                    var creationTime = fileInfo.CreationTime;
                    
                    var item = new System.Windows.Controls.ListBoxItem
                    {
                        Content = $"{fileName} ({fileSize:F1} KB, {creationTime:yyyy-MM-dd HH:mm})",
                        Tag = logFile
                    };
                    listBox.Items.Add(item);
                }

                // 첫 번째 항목 선택
                if (listBox.Items.Count > 0)
                {
                    listBox.SelectedIndex = 0;
                }

                stackPanel.Children.Add(listBox);

                var buttonPanel = new System.Windows.Controls.StackPanel
                {
                    Orientation = System.Windows.Controls.Orientation.Horizontal,
                    HorizontalAlignment = WpfHorizontalAlignment.Right
                };

                var openButton = new System.Windows.Controls.Button
                {
                    Content = "?�기",
                    Width = 80,
                    Height = 30,
                    Margin = new Thickness(0, 0, 10, 0)
                };

                var cancelButton = new System.Windows.Controls.Button
                {
                    Content = "취소",
                    Width = 80,
                    Height = 30
                };

                openButton.Click += (s, e) =>
                {
                    if (listBox.SelectedItem is System.Windows.Controls.ListBoxItem selectedItem)
                    {
                        string selectedLogFile = selectedItem.Tag as string;
                        Process.Start("notepad.exe", selectedLogFile);
                        Logger.Info($"로그 파일을 열었습니다. {Path.GetFileName(selectedLogFile)}");
                        logWindow.Close();
                    }
                };

                cancelButton.Click += (s, e) => logWindow.Close();

                buttonPanel.Children.Add(openButton);
                buttonPanel.Children.Add(cancelButton);
                stackPanel.Children.Add(buttonPanel);

                logWindow.Content = stackPanel;
                logWindow.ShowDialog();
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("로그 파일을 여는 중 오류가 발생했습니다.", ex, "ViewLogs");
            }
        }

        /// <summary>
        /// 추론 서비스 성능 비교 실행
        /// </summary>
        private async Task RunInferenceComparison()
        {
            try
            {
                UpdateStatus("추론 서비스 성능 비교 시작...");
                UIHelper.ShowProgress(progressBar, true);

                // 테스트용 이미지 생성
                Mat testImage = CreateTestImage();
                
                // 성능 비교 실행
                var comparison = await InferenceComparisonHelper.RunPerformanceComparison(
                    _imageProcessor,
                    _config.SmallModelName, // ONNX 모델 경로
                    _config.LargeModelName, // Python 모델 경로
                    testImage,
                    5 // 테스트 반복 횟수
                );

                // 결과 표시
                InferenceComparisonHelper.ShowComparisonResults(comparison);
                
                UpdateStatus("추론 서비스 성능 비교 완료");
            }
            catch (Exception ex)
            {
                UpdateStatus($"성능 비교 오류: {ex.Message}");
                UIHelper.ShowErrorMessage($"성능 비교 중 오류가 발생했습니다: {ex.Message}");
            }
            finally
            {
                UIHelper.ShowProgress(progressBar, false);
            }
        }

        /// <summary>
        /// 테스트용 이미지 생성
        /// </summary>
        private Mat CreateTestImage()
        {
            Mat testImage = new Mat(460, 500, MatType.CV_8UC3, Scalar.All(128));
            
            // 테스트용 패턴 추가
            Cv2.Circle(testImage, new OpenCvSharp.Point(250, 230), 100, Scalar.White, -1);
            Cv2.Circle(testImage, new OpenCvSharp.Point(250, 230), 50, Scalar.Black, -1);
            Cv2.Rectangle(testImage, new OpenCvSharp.Rect(100, 100, 300, 260), Scalar.Gray, 2);
            
            return testImage;
        }

        /// <summary>
        /// PTH 모델을 ONNX로 변환
        private async Task ConvertPthToOnnx()
        {
            try
            {
                UpdateStatus("PTH를 ONNX로 변환 시작...");
                UIHelper.ShowProgress(progressBar, true);

                // 현재 선택된 모델 경로
                string modelPath = _config.SmallModel ? _config.SmallModelName : _config.LargeModelName;
                
                if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
                {
                    UIHelper.ShowErrorMessage("변환할 PTH 모델 파일을 선택하지 않았습니다.");
                    return;
                }

                // 출력 ONNX 파일 경로
                string outputPath = Path.ChangeExtension(modelPath, ".onnx");
                
                // Python 스크립트를 통해 변환 실행
                bool success = await ConvertModelToOnnxAsync(modelPath, outputPath);
                
                if (success)
                {
                    UpdateStatus($"ONNX 변환 완료: {Path.GetFileName(outputPath)}");
                    UIHelper.ShowInfoMessage($"ONNX 변환이 완료되었습니다.\n파일: {outputPath}", "변환 완료");
                    
                    // 변환된 ONNX 모델을 설정에 추가
                    _config.OnnxModelName = outputPath;
                    _configService.SaveConfig(_config);
                }
                else
                {
                    UpdateStatus("ONNX 변환 실패");
                    UIHelper.ShowErrorMessage("ONNX 변환에 실패했습니다.");
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"ONNX 변환 오류: {ex.Message}");
                UIHelper.ShowErrorMessage($"ONNX 변환 중 오류가 발생했습니다: {ex.Message}");
            }
            finally
            {
                UIHelper.ShowProgress(progressBar, false);
            }
        }

        /// <summary>
        /// Python을 통해 모델을 ONNX로 변환
        private async Task<bool> ConvertModelToOnnxAsync(string modelPath, string outputPath)
        {
            return await Task.Run(() =>
            {
                try
                {
                    // Python 스크립트로 변환 요청
                    var config = new
                    {
                        Action = "convert_to_onnx",
                        ModelPath = modelPath,
                        OutputPath = outputPath,
                        InputSize = new[] { 256, 256, 3 }
                    };

                    string result = ExecutePythonScript(config);
                    return result.Contains("ONNX conversion completed");
                }
                catch (Exception ex)
                {
                    ExceptionHelper.LogError(ex, "Convert model to ONNX");
                    return false;
                }
            });
        }

        /// <summary>
        /// Python을 통해 CUDA 상태 확인
        /// </summary>
        private async Task<string> CheckCUDAStatusAsync()
        {
            return await Task.Run(() =>
            {
                try
                {
                    // Python 스크립트로 CUDA 상태 확인 요청
                    var config = new
                    {
                        Action = "check_cuda"
                    };

                    string result = ExecutePythonScript(config);
                    return result;
                }
                catch (Exception ex)
                {
                    ExceptionHelper.LogError(ex, "Check CUDA status");
                    return "";
                }
            });
        }

        /// <summary>
        /// CUDA ?�태 ?�인
        /// </summary>
        private async Task CheckCUDAStatus()
        {
            try
            {
                UpdateStatus("CUDA 상태를 확인하고 있습니다...");
                UIHelper.ShowProgress(progressBar, true);

                // Python 스크립트를 통해 CUDA 상태 확인
                string cudaStatus = await CheckCUDAStatusAsync();
                
                if (!string.IsNullOrEmpty(cudaStatus))
                {
                    // CUDA 상태를 별도 창에 표시
                    var statusWindow = new System.Windows.Window
                    {
                        Title = "CUDA 상태 확인",
                        Width = 800,
                        Height = 600,
                        WindowStartupLocation = WindowStartupLocation.CenterOwner,
                        Owner = this
                    };

                    var textBlock = new System.Windows.Controls.TextBlock
                    {
                        Text = cudaStatus,
                        FontFamily = new System.Windows.Media.FontFamily("Consolas"),
                        FontSize = 12,
                        TextWrapping = TextWrapping.Wrap,
                        Margin = new Thickness(10),
                        VerticalAlignment = WpfVerticalAlignment.Top,
                        HorizontalAlignment = WpfHorizontalAlignment.Left
                    };

                    var scrollViewer = new System.Windows.Controls.ScrollViewer
                    {
                        Content = textBlock,
                        VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                        HorizontalScrollBarVisibility = ScrollBarVisibility.Auto
                    };

                    statusWindow.Content = scrollViewer;
                    statusWindow.ShowDialog();
                    
                    UpdateStatus("CUDA ?�태 ?�인 ?�료");
                }
                else
                {
                    UpdateStatus("CUDA 상태 확인 실패");
                    UIHelper.ShowErrorMessage("CUDA 상태를 확인할 수 없습니다.");
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"CUDA 상태 확인 오류: {ex.Message}");
                UIHelper.ShowErrorMessage($"CUDA 상태 확인 중 오류가 발생했습니다: {ex.Message}");
            }
            finally
            {
                UIHelper.ShowProgress(progressBar, false);
            }
        }

        /// <summary>
        /// Python ?�크립트 ?�행
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
                    FileName = "python",
                    Arguments = $"\"{Path.Combine(Directory.GetCurrentDirectory(), "python_inference.py")}\" \"{configPath}\"",
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
                        
                        process.WaitForExit(60000); // 60초 대기
                        if (process.ExitCode == 0)
                        {
                            return output;
                        }
                        else
                        {
                            ExceptionHelper.LogError(new Exception($"Python script error: {error}"), "Python conversion");
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
        /// 프레임 타이머 이벤트        /// </summary>
        private void FrameTimer_Tick(object sender, EventArgs e)
        {
            ProcessFrame();
        }

        /// <summary>
        /// 자동저장 타이머 이벤트        /// </summary>
        private void AutoSaveTimer_Tick(object sender, EventArgs e)
        {
            if (_currentState == AppState.Calc && _config.AutoSave)
            {
                AutoSave();
            }
        }

        /// <summary>
        /// 성능 모니터링 타이머 이벤트        /// </summary>
        private void PerformanceTimer_Tick(object sender, EventArgs e)
        {
            if (_isPlaying)
            {
                var stats = PerformanceHelper.CollectStats(_processedFrameCount, _performanceStartTime, _backgroundWorker.QueueCount);
                
                // 성능 정보를 상태바에 표시 (선택적)
                string perfInfo = $"FPS: {stats.FPS:F1} | Queue: {stats.QueueSize} | Memory: {stats.MemoryUsageMB}MB | Target: {_targetFPS}fps";
                
                // 성능이 저하된 경우 경고
                if (stats.FPS < 10 || stats.QueueSize > 5)
                {
                    var suggestion = PerformanceHelper.GetOptimizationSuggestion(stats);
                    UpdateStatus($"성능 경고: {suggestion}");
                }
            }
        }

        /// <summary>
        /// ROI Canvas 마우스 이벤트
        private void RoiCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _isDrawingROI = true;
            _roiStartPoint = e.GetPosition(roiCanvas);
            roiRectangle.Visibility = Visibility.Visible;
        }

        private void RoiCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isDrawingROI)
            {
                _roiEndPoint = e.GetPosition(roiCanvas);
                UpdateROIRectangle();
            }
        }

        private void RoiCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isDrawingROI)
            {
                _isDrawingROI = false;
                _roiEndPoint = e.GetPosition(roiCanvas);
                UpdateROIRectangle();
                ApplyROISelection();
            }
        }

        #endregion

        #region ?�심 기능 구현

        /// <summary>
        /// HDMI ?�결
        /// </summary>
        private void ConnectHDMI()
        {
            try
            {
                Logger.Info($"HDMI 캡처보드 ?�결 ?�도 - Device ID: {_config.DeviceID}");
                
                bool success = _videoProcessor.ConnectHDMI(_config.DeviceID);
                if (success)
                {
                    _isHDMIConnected = true;
                    _isVideoMode = false; // HDMI 연결 시 비디오 모드 비활성화 (우선순위)
                    UpdateAppState(AppState.Load);
                    UpdateStatus("HDMI 캡처보드가 연결되었습니다. Play 버튼을 눌러 재생하세요.");
                    
                    Logger.Info("HDMI 캡처보드 연결이 완료되었습니다.");
                    
                    // 자동 재생 비활성화 - Play 버튼을 눌러야만 재생됨
                    // HDMI가 우선순위이므로 비디오 파일 로드 상태를 덮어씀
                }
                else
                {
                    Logger.Warning("HDMI 캡처보드 연결에 실패했습니다.");
                    UpdateStatus("HDMI 캡처보드 연결에 실패했습니다.");
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("HDMI 연결 중 오류가 발생했습니다.", ex, "ConnectHDMI");
            }
        }

        /// <summary>
        /// HDMI 연결 해제
        /// </summary>
        private void DisconnectHDMI()
        {
            try
            {
                Logger.Info("HDMI 연결을 해제합니다.");
                
                // 재생 중이면 정지
                if (_isPlaying)
                {
                    StopSlideShow();
                }
                
                // HDMI 연결 해제
                _videoProcessor.DisconnectHDMI();
                _isHDMIConnected = false;
                
                // 상태 초기화
                UpdateAppState(AppState.Ready);
                UpdateStatus("HDMI 연결이 해제되었습니다.");
                
                Logger.Info("HDMI 연결 해제가 완료되었습니다.");
            }
            catch (Exception ex)
            {
                ExceptionHelper.ShowErrorAndLog("HDMI 연결 해제 중 오류가 발생했습니다.", ex, "DisconnectHDMI");
            }
        }

        /// <summary>
        /// HDMI ?�시 ?�작
        /// </summary>
        private void StartHDMIShow()
        {
            if (_isPlaying)
            {
                _isCalculating = false;
                StopSlideShow();
            }
            else
            {
                _isPlaying = true;
                _currentState = AppState.Play;
                _isCalculating = false;
                
                // 성능 모니터링 시작
                _performanceStartTime = DateTime.Now;
                _processedFrameCount = 0;
                
                // 백그라운드 워커 시작
                _backgroundWorker.StartProcessing();
                _frameTimer.Start();
                _performanceTimer.Start();
            }
        }

        /// <summary>
        /// 추론 ?�작
        /// </summary>
        private void StartInference()
        {
            _isCalculating = true;
            _isPlaying = true;
            _currentState = AppState.Calc;
            
            // ?�능 모니?�링 ?�작
            _performanceStartTime = DateTime.Now;
            _processedFrameCount = 0;
            
            // 백그라운드 워커 시작
            _backgroundWorker.StartProcessing();
            _frameTimer.Start();
            _performanceTimer.Start();
            
            if (_config.AutoSave)
            {
                _autoSaveTimer.Start();
            }
        }

        /// <summary>
        /// 비디오 파일 로드
        /// </summary>
        private void LoadVideoFile()
        {
            // HDMI가 연결된 상태에서는 비디오 로드 제한 (우선순위)
            if (_isHDMIConnected)
            {
                UpdateStatus("HDMI가 연결된 상태입니다. HDMI 연결을 해제한 후 비디오를 로드하세요.");
                UIHelper.ShowWarningMessage("HDMI가 연결된 상태입니다.\nHDMI 연결을 해제한 후 비디오를 로드하세요.");
                return;
            }

            var openFileDialog = new OpenFileDialog
            {
                Filter = "Video files (*.avi;*.mp4;*.mov;*.mkv)|*.avi;*.mp4;*.mov;*.mkv|All files (*.*)|*.*",
                Title = "비디오 파일 선택"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                bool success = _videoProcessor.LoadVideoFile(openFileDialog.FileName);
                if (success)
                {
                    _isVideoMode = true;
                    btnCalculateCAC.IsEnabled = true;
                    btnSetROI.IsEnabled = false;
                    btnExportExcel.IsEnabled = false;
                    btnExportVideo.IsEnabled = false;
                    
                    // 비디오 로드 완료 - Play 버튼을 눌러야 재생됨
                    Logger.Info($"비디오 파일 로드 완료: {openFileDialog.FileName}");
                    UpdateAppState(AppState.Load);
                    UpdateStatus("비디오 파일이 로드되었습니다. Play 버튼을 눌러 재생하세요.");
                }
            }
        }

        /// <summary>
        /// 프레임 처리 (백그라운드 워커 사용)
        /// </summary>
        private void ProcessFrame()
        {
            try
            {
                Mat frame = null;
                
                if (_isVideoMode)
                {
                    frame = _videoProcessor.ReadVideoFrame();
                    if (frame == null)
                    {
                        // 비디오 끝에 도달
                        StopSlideShow();
                        return;
                    }
                }
                else
                {
                    frame = _videoProcessor.CaptureHDMIFrame();
                    if (frame == null)
                    {
                        // HDMI 연결 끊어짐                        ConnectHDMI();
                        if (_isPlaying)
                        {
                            _isCalculating = false;
                            StopSlideShow();
                        }
                        _currentState = AppState.Ready;
                        UpdateUIState();
                        return;
                    }
                }

                // 프레임 스킵 로직 (AI 추론이 활성화된 경우)
                if (_isCalculating)
                {
                    _frameSkipCounter++;
                    var timeSinceLastInference = DateTime.Now - _lastInferenceTime;
                    var targetInterval = TimeSpan.FromMilliseconds(1000.0 / _targetFPS); // 15fps = 66.7ms
                    
                    // 목표 FPS에 맞춰 프레임 스킵
                    if (timeSinceLastInference < targetInterval)
                    {
                        // 아직 추론할 시간이 안 됨 - 프레임만 표시
                        Mat croppedFrame = _videoProcessor.CropWithROI(frame, _config, _isVideoMode);
                        Mat resizedFrame = _videoProcessor.ResizeForModel(croppedFrame, new OpenCvSharp.Size(500, 460));
                        DisplayFrame(resizedFrame);
                        
                        MatHelper.SafeDispose(croppedFrame);
                        MatHelper.SafeDispose(resizedFrame);
                        MatHelper.SafeDispose(frame);
                        return;
                    }
                    
                    // 추론 시간 업데이트
                    _lastInferenceTime = DateTime.Now;
                }

                // 백그라운드 워커에 프레임 전달 (비동기 처리)
                bool enqueued = _backgroundWorker.EnqueueFrame(frame, _isCalculating, _config, _isVideoMode);
                
                if (!enqueued)
                {
                    // 큐가 가득 찬 경우 직접 처리 (폴백)
                    ProcessFrameDirectly(frame);
                }

                // 원본 프레임 해제 (백그라운드에서 복사해서 사용)
                MatHelper.SafeDispose(frame);
            }
            catch (Exception ex)
            {
                UpdateStatus($"프레임 처리 오류: {ex.Message}");
            }
        }

        /// <summary>
        /// 프레임 직접 처리 (폴백) - 비동기
        /// </summary>
        private async void ProcessFrameDirectly(Mat frame)
        {
            try
            {
                // ROI 적용
                Mat croppedFrame = _videoProcessor.CropWithROI(frame, _config, _isVideoMode);
                Mat resizedFrame = _videoProcessor.ResizeForModel(croppedFrame, new OpenCvSharp.Size(500, 460));

                if (_isCalculating)
                {
                    ProcessFrameWithInference(resizedFrame, frame);
                }
                else
                {
                    DisplayFrame(resizedFrame);
                }

                MatHelper.SafeDispose(croppedFrame);
                MatHelper.SafeDispose(resizedFrame);
            }
            catch (Exception ex)
            {
                UpdateStatus($"직접 프레임 처리 오류: {ex.Message}");
            }
        }

        /// <summary>
        /// 추론과 함께 프레임 처리 (비동기)
        /// </summary>
        private async void ProcessFrameWithInference(Mat resizedFrame, Mat originalFrame)
        {
            try
            {
                // 모델 추론 (비동기)
                Mat mask = null;
                if (_inferenceService != null && _inferenceService.IsModelAvailable)
                {
                    mask = await _inferenceService.PredictImageAsync(resizedFrame);
                }
                if (mask == null)
                {
                    DisplayFrame(resizedFrame);
                    return;
                }

                // CAC 계산
                string fileName = $"Frame_{(_measurements?.Count ?? 0) + 1}.png";
                var (measurement, overlay) = _cacCalculator?.ProcessImagesRealtime(
                    fileName, resizedFrame, mask, 300, _config?.DrawEllipse ?? true) ?? 
                    (new MeasurementData { FileName = fileName }, resizedFrame);

                // 측정값 저장
                _measurements?.Add(measurement);
                _imageFileNames?.Add(fileName);

                // 그래프 업데이트
                UpdateGraphs(measurement);

                // UI 업데이트
                UpdateMeasurementDisplay(measurement);
                DisplayFrame(overlay);

                // 이미지 저장
                SaveFrameWithText(overlay, fileName, measurement);
            }
            catch (Exception ex)
            {
                UpdateStatus($"추론 처리 ?�류: {ex.Message}");
            }
        }

        /// <summary>
        /// 프레임 표시
        /// </summary>
        private void DisplayFrame(Mat frame)
        {
            if (frame == null || frame.Empty())
                return;

            try
            {
                var bitmapImage = ImageConverter.MatToBitmapImage(frame);
                if (bitmapImage != null)
                {
                    imageDisplay.Source = bitmapImage;
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"프레임 표시 오류: {ex.Message}");
            }
        }

        /// <summary>
        /// 그래프 업데이트
        /// </summary>
        private void UpdateGraphs(MeasurementData measurement)
        {
            // CAC 그래프 업데이트
            UpdateCACGraph(measurement.CACValue, measurement.IJVValue);
            
            // ABP 그래프 업데이트
            UpdateABPGraph(measurement.CACValue);
        }

        /// <summary>
        /// CAC 그래프 업데이트
        /// </summary>
        private void UpdateCACGraph(double cacValue, double ijvValue)
        {
            _cacBuffer.Add(cacValue);
            _ijvBuffer.Add(ijvValue);

            // 10개가 찼을 때만 그래프 업데이트
            if (_cacBuffer.IsFull)
            {
                var cacLineSeries = cacGraph.Model.Series[0] as LineSeries;
                var ijvLineSeries = cacGraph.Model.Series[1] as LineSeries;
                var thresholdLineSeries = cacGraph.Model.Series[2] as LineSeries;
                
                if (cacLineSeries != null && ijvLineSeries != null)
                {
                    GraphHelper.UpdateCACGraph(cacLineSeries, ijvLineSeries, _cacBuffer, _ijvBuffer);
                    GraphHelper.RefreshGraph(cacGraph.Model);
                }
            }
        }

        /// <summary>
        /// ABP 그래프 업데이트
        /// </summary>
        private void UpdateABPGraph(double cacValue)
        {
            if (_measurements.Count % 40 == 0 && cacValue > 0)
            {
                var abpScatterSeries = abpGraph.Model.Series[3] as ScatterSeries;
                
                if (abpScatterSeries != null)
                {
                    GraphHelper.UpdateABPGraph(abpScatterSeries, cacValue);
                    GraphHelper.RefreshGraph(abpGraph.Model);
                }
            }
        }

        /// <summary>
        /// 측정값 표시 업데이트
        /// </summary>
        private void UpdateMeasurementDisplay(MeasurementData measurement)
        {
            Dispatcher.Invoke(() =>
            {
                labelMeasureROSC.Text = measurement.State;
                labelMeasureCAC.Text = $"CAC: {measurement.CACValue:F2}";

                // 상태에 따른 색상 변경
                var color = measurement.State switch
                {
                    "ROSC" => Brushes.Blue,
                    "Arrest" => Brushes.Red,
                    "Not compressed" => Brushes.White,
                    "Invalid CAC" => Brushes.Gray,
                    _ => Brushes.Yellow
                };

                labelMeasureROSC.Foreground = color;
            });
        }

        /// <summary>
        /// 프레임 저장 (텍스트 포함)
        /// </summary>
        private void SaveFrameWithText(Mat overlay, string fileName, MeasurementData measurement)
        {
            ExceptionHelper.SafeExecute(() =>
            {
                string frameInfo = fileName;
                string classInfo = $"Class: {measurement.Class}";
                string cacInfo = $"CAC: {measurement.CACValue:F2}";

                Mat frameWithText = _imageProcessor.AddTextToImage(overlay, frameInfo, classInfo, cacInfo);
                
                string tempFolder = PathHelper.Combine(Directory.GetCurrentDirectory(), "temp");
                PathHelper.CreateDirectory(tempFolder);
                
                string filePath = PathHelper.Combine(tempFolder, fileName);
                frameWithText.SaveImage(filePath);
            }, "Save frame with text");
        }

        /// <summary>
        /// 슬라이드쇼 중지
        /// </summary>
        private void StopSlideShow()
        {
            _frameTimer?.Stop();
            _autoSaveTimer?.Stop();
            _performanceTimer?.Stop();
            
            // 백그라운드 워커 중지
            _backgroundWorker?.StopProcessing();
            
            _isPlaying = false;
            _isCalculating = false;
            _currentState = _isVideoMode ? AppState.Ready : AppState.Load;

            // 데이터 초기화
            _cacCalculator?.ResetWindows();
        }

        /// <summary>
        /// ROI 사각형 업데이트
        /// </summary>
        private void UpdateROIRectangle()
        {
            double x = Math.Min(_roiStartPoint.X, _roiEndPoint.X);
            double y = Math.Min(_roiStartPoint.Y, _roiEndPoint.Y);
            double width = Math.Abs(_roiEndPoint.X - _roiStartPoint.X);
            double height = Math.Abs(_roiEndPoint.Y - _roiStartPoint.Y);

            Canvas.SetLeft(roiRectangle, x);
            Canvas.SetTop(roiRectangle, y);
            roiRectangle.Width = width;
            roiRectangle.Height = height;
        }

        /// <summary>
        /// ROI ?�택 ?�용
        /// </summary>
        private void ApplyROISelection()
        {
            // Canvas 좌표를 이미지 좌표로 변환
            // 현재 구현에서는 이미지 크기와 Canvas 크기 비율을 고려해야 함
            roiCanvas.Visibility = Visibility.Collapsed;
            
            // ROI 좌표 업데이트
            if (_isVideoMode)
            {
                _config.ROI_X1_Video = (int)_roiStartPoint.X;
                _config.ROI_Y1_Video = (int)_roiStartPoint.Y;
                _config.ROI_X2_Video = (int)_roiEndPoint.X;
                _config.ROI_Y2_Video = (int)_roiEndPoint.Y;
            }
            else
            {
                _config.ROI_X1 = (int)_roiStartPoint.X;
                _config.ROI_Y1 = (int)_roiStartPoint.Y;
                _config.ROI_X2 = (int)_roiEndPoint.X;
                _config.ROI_Y2 = (int)_roiEndPoint.Y;
            }

            // 설정 저장
            _configService.SaveConfig(_config);
        }

        /// <summary>
        /// 자동저장        /// </summary>
        private void AutoSave()
        {
            ExceptionHelper.SafeExecute(() =>
            {
                string fileName = PathHelper.GenerateTimestampFileName();
                ExportVideoInternal(fileName);
                ExportExcelInternal(fileName);
            }, "Auto save");
        }

        /// <summary>
        /// 비디오 내보내기 (내부)
        /// </summary>
        private void ExportVideoInternal(string fileName)
        {
            ExceptionHelper.SafeExecute(() =>
            {
                string tempFolder = PathHelper.Combine(Directory.GetCurrentDirectory(), "temp");
                string outputPath = PathHelper.Combine(_config.SaveFolder, $"{fileName}_video.avi");
                
                PathHelper.CreateDirectory(_config.SaveFolder);
                var framePaths = _imageFileNames.Select(f => PathHelper.Combine(tempFolder, f)).ToList();
                FileHelpers.SaveFramesAsVideo(framePaths, outputPath);
                
                UpdateStatus($"비디오 저장 완료: {outputPath}");
            }, "Export video internal");
        }

        /// <summary>
        /// 이미지 내보내기
        /// </summary>
        private void ExportImages()
        {
            using (var folderDialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                if (folderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    try
                    {
                        string tempFolder = Path.Combine(Directory.GetCurrentDirectory(), "temp");
                        var imagePaths = _imageFileNames.Select(f => Path.Combine(tempFolder, f)).ToList();
                        
                        foreach (var imagePath in imagePaths)
                        {
                            if (File.Exists(imagePath))
                            {
                                string destPath = Path.Combine(folderDialog.SelectedPath, Path.GetFileName(imagePath));
                                File.Copy(imagePath, destPath, true);
                            }
                        }
                        
                        UpdateStatus($"이미지 저장 완료: {folderDialog.SelectedPath}");
                    }
                    catch (Exception ex)
                    {
                        UpdateStatus($"이미지 저장 오류: {ex.Message}");
                    }
                }
            }
        }

        /// <summary>
        /// Excel 내보내기 (내부)
        /// </summary>
        private void ExportExcelInternal(string fileName)
        {
            ExceptionHelper.SafeExecute(() =>
            {
                string outputPath = PathHelper.Combine(_config.SaveFolder, $"{fileName}_measurements.xlsx");
                
                PathHelper.CreateDirectory(_config.SaveFolder);
                FileHelpers.ExportToExcel(_measurements, outputPath);
                
                UpdateStatus($"Excel 저장 완료: {outputPath}");
            }, "Export Excel internal");
        }

        /// <summary>
        /// 상태 업데이트
        /// </summary>
        private void UpdateStatus(string message)
        {
            try
            {
                // 로그 파일에도 기록
                Logger.Info($"[UI 상태] {message}");
                
                // UI 스레드에서 상태 표시 업데이트
                Dispatcher.Invoke(() =>
                {
                    // 상태바나 라벨에 메시지 표시
                    Console.WriteLine($"[UI] {message}");
                });
            }
            catch (Exception ex)
            {
                // UpdateStatus 실패 시에도 로그 기록
                Logger.Error($"UpdateStatus 실패: {message}", ex);
            }
        }

        /// <summary>
        /// 프레임 캡처 이벤트 핸들러
        private void OnFrameCaptured(object sender, Mat frame)
        {
            // 필요시 추가 처리
        }

        /// <summary>
        /// 상태 변경 이벤트 핸들러
        private void OnStatusChanged(object sender, string status)
        {
            UpdateStatus(status);
        }

        /// <summary>
        /// 백그라운드에서 처리된 프레임 이벤트 핸들러        /// </summary>
        private void OnFrameProcessed(object sender, ProcessedFrameData processedData)
        {
            try
            {
                if (processedData?.DisplayFrame == null)
                    return;

                // UI 업데이트 (UI 스레드에서 실행)
                UIHelper.SafeInvoke(() =>
                {
                    // 이미지 표시
                    var bitmapImage = ImageConverter.MatToBitmapImage(processedData.DisplayFrame);
                    if (bitmapImage != null)
                    {
                        imageDisplay.Source = bitmapImage;
                    }

                    // 측정값이 있는 경우 추가 처리
                    if (processedData.Measurement != null)
                    {
                        // 측정값 저장
                        _measurements.Add(processedData.Measurement);
                        _imageFileNames.Add(processedData.Measurement.FileName);

                        // 성능 카운터 업데이트
                        _processedFrameCount++;

                        // 그래프 업데이트
                        UpdateGraphs(processedData.Measurement);

                        // UI 업데이트
                        UpdateMeasurementDisplay(processedData.Measurement);

                        // 프레임 저장                        if (processedData.OverlayFrame != null)
                        {
                            SaveFrameWithText(processedData.OverlayFrame, processedData.Measurement.FileName, processedData.Measurement);
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                UpdateStatus($"프레임 처리 결과 업데이트 오류: {ex.Message}");
            }
        }

        /// <summary>
        /// 백그라운드 워커 오류 이벤트 핸들러        /// </summary>
        private void OnBackgroundError(object sender, Exception ex)
        {
            UpdateStatus($"백그라운드 처리 오류: {ex.Message}");
        }

        /// <summary>
        /// 윈도우 닫기 이벤트        /// </summary>
        protected override void OnClosed(EventArgs e)
        {
            _frameTimer?.Stop();
            _autoSaveTimer?.Stop();
            _performanceTimer?.Stop();
            
            // 백그라운드 워커 정리
            _backgroundWorker?.Dispose();
            
            _modelInference?.Dispose();
            _videoProcessor?.Dispose();
            base.OnClosed(e);
        }
    }
}
#endregion
