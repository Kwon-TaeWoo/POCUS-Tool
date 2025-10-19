using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using POCUS.ROSC.Utilities;

namespace POCUS.ROSC.Views
{
    /// <summary>
    /// InitializationWindow.xaml에 대한 상호 작용 논리
    /// 프로그램 초기화 화면
    /// </summary>
    public partial class InitializationWindow : Window
    {
        private readonly InitializationWorker _initWorker;
        private readonly DispatcherTimer _animationTimer;
        private int _animationStep = 0;

        public event EventHandler<InitializationResult> InitializationCompleted;

        public InitializationWindow()
        {
            InitializeComponent();
            
            _initWorker = new InitializationWorker();
            _initWorker.StepChanged += OnStepChanged;
            _initWorker.StatusChanged += OnStatusChanged;
            _initWorker.ProgressChanged += OnProgressChanged;
            _initWorker.RunWorkerCompleted += OnRunWorkerCompleted;
            
            // 애니메이션 타이머
            _animationTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(100)
            };
            _animationTimer.Tick += OnAnimationTick;
        }

        /// <summary>
        /// 초기화 시작
        /// </summary>
        public void StartInitialization()
        {
            _animationTimer.Start();
            _initWorker.StartInitialization();
        }

        /// <summary>
        /// 단계 변경 이벤트
        /// </summary>
        private void OnStepChanged(object sender, InitializationWorker.InitializationStep step)
        {
            Dispatcher.Invoke(() =>
            {
                StepText.Text = GetStepDisplayText(step);
            });
        }

        /// <summary>
        /// 상태 변경 이벤트
        /// </summary>
        private void OnStatusChanged(object sender, string status)
        {
            Dispatcher.Invoke(() =>
            {
                StatusText.Text = status;
            });
        }

        /// <summary>
        /// 진행률 변경 이벤트
        /// </summary>
        private void OnProgressChanged(object sender, System.ComponentModel.ProgressChangedEventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                try
                {
                    if (e.UserState is InitializationProgress progress)
                    {
                        // IsIndeterminate를 항상 false로 유지하고 Value만 설정
                        InitProgressBar.IsIndeterminate = false;
                        InitProgressBar.Value = progress.Progress;
                        InitProgressBar.Maximum = 100; // 명시적으로 Maximum 설정
                        
                        ProgressText.Text = $"{progress.Progress}%";
                        
                        Logger.Info($"프로그레스바 업데이트: {progress.Progress}% (Value: {InitProgressBar.Value}, Maximum: {InitProgressBar.Maximum})");
                    }
                }
                catch (Exception ex)
                {
                    Logger.Error($"OnProgressChanged 오류: {ex.Message}", ex);
                }
            });
        }

        /// <summary>
        /// BackgroundWorker 완료 이벤트
        /// </summary>
        private void OnRunWorkerCompleted(object sender, System.ComponentModel.RunWorkerCompletedEventArgs e)
        {
            _animationTimer.Stop();
            
            Dispatcher.Invoke(() =>
            {
                if (e.Result is InitializationResult result)
                {
                    OnInitializationCompleted(sender, result);
                }
                else if (e.Error != null)
                {
                    // 예외 발생 시
                    StatusText.Text = $"초기화 중 오류 발생: {e.Error.Message}";
                    StepText.Text = "오류";
                    InitProgressBar.Foreground = System.Windows.Media.Brushes.Red;
                    
                    Logger.Error($"초기화 중 예외 발생: {e.Error.Message}", e.Error);
                    
                    // 3초 후 종료
                    var timer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromSeconds(3)
                    };
                    timer.Tick += (s, args) =>
                    {
                        timer.Stop();
                        Application.Current.Shutdown();
                    };
                    timer.Start();
                }
                else
                {
                    // 예상치 못한 상황
                    StatusText.Text = "초기화 중 예상치 못한 오류가 발생했습니다.";
                    StepText.Text = "오류";
                    InitProgressBar.Foreground = System.Windows.Media.Brushes.Red;
                    
                    Logger.Error("초기화 중 예상치 못한 오류 발생");
                    
                    // 3초 후 종료
                    var timer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromSeconds(3)
                    };
                    timer.Tick += (s, args) =>
                    {
                        timer.Stop();
                        Application.Current.Shutdown();
                    };
                    timer.Start();
                }
            });
        }

        /// <summary>
        /// 초기화 완료 이벤트
        /// </summary>
        private void OnInitializationCompleted(object sender, InitializationResult result)
        {
            Dispatcher.Invoke(() =>
            {
                if (result.Success)
                {
                    // 성공 시 잠시 완료 메시지 표시
                    StatusText.Text = "초기화가 완료되었습니다!";
                    StepText.Text = "완료";
                    InitProgressBar.Value = 100;
                    ProgressText.Text = "100%";
                    
                    Logger.Info("초기화가 성공적으로 완료되었습니다.");
                    
                    // 1초 후 메인 윈도우로 전환
                    var timer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromSeconds(1)
                    };
                    timer.Tick += (s, e) =>
                    {
                        timer.Stop();
                        InitializationCompleted?.Invoke(this, result);
                        Close();
                    };
                    timer.Start();
                }
                else
                {
                    // 실패 시 오류 메시지 표시
                    StatusText.Text = $"초기화 실패: {result.Error?.Message}";
                    StepText.Text = "오류";
                    InitProgressBar.Foreground = System.Windows.Media.Brushes.Red;
                    
                    Logger.Error($"초기화 실패: {result.Error?.Message}", result.Error);
                    
                    // 3초 후 종료
                    var timer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromSeconds(3)
                    };
                    timer.Tick += (s, e) =>
                    {
                        timer.Stop();
                        Application.Current.Shutdown();
                    };
                    timer.Start();
                }
            });
        }

        /// <summary>
        /// 애니메이션 틱
        /// </summary>
        private void OnAnimationTick(object sender, EventArgs e)
        {
            _animationStep++;
            
            // 하트 아이콘 회전 애니메이션
            var heartIcon = FindName("HeartIcon") as FrameworkElement;
            if (heartIcon != null)
            {
                var transform = new System.Windows.Media.RotateTransform(_animationStep * 2);
                heartIcon.RenderTransform = transform;
            }
        }

        /// <summary>
        /// 단계 표시 텍스트 반환
        /// </summary>
        private string GetStepDisplayText(InitializationWorker.InitializationStep step)
        {
            return step switch
            {
                InitializationWorker.InitializationStep.Starting => "시작 중...",
                InitializationWorker.InitializationStep.LoadingConfig => "설정 로드 중...",
                InitializationWorker.InitializationStep.InitializingServices => "서비스 초기화 중...",
                InitializationWorker.InitializationStep.LoadingModels => "모델 로드 중...",
                InitializationWorker.InitializationStep.PreparingPython => "Python 준비 중...",
                InitializationWorker.InitializationStep.Finalizing => "최종화 중...",
                InitializationWorker.InitializationStep.Completed => "완료",
                _ => "처리 중..."
            };
        }

        /// <summary>
        /// 창 닫기 방지 (초기화 중에는 닫을 수 없음)
        /// </summary>
        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            if (_initWorker.IsBusy)
            {
                e.Cancel = true;
            }
            base.OnClosing(e);
        }
    }
}
