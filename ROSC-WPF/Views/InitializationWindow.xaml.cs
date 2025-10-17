using System;
using System.Windows;
using System.Windows.Threading;
using ROSC.WPF.Utilities;

namespace ROSC.WPF.Views
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
            _initWorker.Completed += OnInitializationCompleted;
            
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
                if (e.UserState is InitializationProgress progress)
                {
                    InitProgressBar.Value = progress.Progress;
                    ProgressText.Text = $"{progress.Progress}%";
                    
                    // 특정 단계에서 프로그레스 바 스타일 변경
                    if (progress.Step == InitializationWorker.InitializationStep.LoadingModels)
                    {
                        InitProgressBar.IsIndeterminate = true;
                    }
                    else
                    {
                        InitProgressBar.IsIndeterminate = false;
                    }
                }
            });
        }

        /// <summary>
        /// 초기화 완료 이벤트
        /// </summary>
        private void OnInitializationCompleted(object sender, InitializationResult result)
        {
            _animationTimer.Stop();
            
            Dispatcher.Invoke(() =>
            {
                if (result.Success)
                {
                    // 성공 시 잠시 완료 메시지 표시
                    StatusText.Text = "초기화가 완료되었습니다!";
                    StepText.Text = "완료";
                    InitProgressBar.Value = 100;
                    ProgressText.Text = "100%";
                    
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
