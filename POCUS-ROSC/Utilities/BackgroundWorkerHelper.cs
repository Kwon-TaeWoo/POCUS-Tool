using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Windows.Threading;
using OpenCvSharp;
using POCUS.ROSC.Models;
using POCUS.ROSC.Services;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// 백그라운드 작업을 위한 유틸리티
    /// 실시간 AI 분석을 위한 멀티스레딩 지원
    /// </summary>
    public class BackgroundWorkerHelper : IDisposable
    {
        private readonly CancellationTokenSource _cancellationTokenSource;
        private readonly ConcurrentQueue<FrameData> _frameQueue;
        private readonly SemaphoreSlim _processingSemaphore;
        private readonly int _maxQueueSize;
        private readonly int _maxConcurrentProcessing;
        private bool _isDisposed = false;
        
        // AI 추론 서비스들
        private readonly IInferenceService _inferenceService;
        private readonly CACCalculationService _cacCalculator;

        public event EventHandler<ProcessedFrameData> FrameProcessed;
        // public event EventHandler<string> StatusChanged; // 현재 사용되지 않음
        public event EventHandler<Exception> ErrorOccurred;

        public BackgroundWorkerHelper(IInferenceService inferenceService, CACCalculationService cacCalculator, 
            int maxQueueSize = 10, int maxConcurrentProcessing = 2)
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _frameQueue = new ConcurrentQueue<FrameData>();
            _processingSemaphore = new SemaphoreSlim(maxConcurrentProcessing, maxConcurrentProcessing);
            _maxQueueSize = maxQueueSize;
            _maxConcurrentProcessing = maxConcurrentProcessing;
            
            _inferenceService = inferenceService;
            _cacCalculator = cacCalculator;
        }

        /// <summary>
        /// 프레임 데이터를 큐에 추가
        /// </summary>
        public bool EnqueueFrame(Mat frame, bool isCalculating, ConfigSettings config, bool isVideoMode = false)
        {
            if (_isDisposed || _cancellationTokenSource.Token.IsCancellationRequested)
                return false;

            try
            {
                // 큐가 가득 찬 경우 오래된 프레임 제거
                while (_frameQueue.Count >= _maxQueueSize)
                {
                    if (_frameQueue.TryDequeue(out FrameData oldFrame))
                    {
                        oldFrame.Dispose();
                    }
                }

                var frameData = new FrameData
                {
                    Frame = MatHelper.SafeClone(frame),
                    IsCalculating = isCalculating,
                    Config = config,
                    IsVideoMode = isVideoMode,
                    Timestamp = DateTime.Now,
                    FrameNumber = _frameQueue.Count + 1
                };

                _frameQueue.Enqueue(frameData);
                return true;
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(this, ex);
                return false;
            }
        }

        /// <summary>
        /// 백그라운드 처리 시작
        /// </summary>
        public void StartProcessing()
        {
            if (_isDisposed)
                return;

            Task.Run(async () => await ProcessFramesAsync(_cancellationTokenSource.Token));
        }

        /// <summary>
        /// 백그라운드 처리 중지
        /// </summary>
        public void StopProcessing()
        {
            _cancellationTokenSource?.Cancel();
        }

        /// <summary>
        /// 프레임 처리 (백그라운드)
        /// </summary>
        private async Task ProcessFramesAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (_frameQueue.TryDequeue(out FrameData frameData))
                    {
                        await _processingSemaphore.WaitAsync(cancellationToken);
                        
                        // 백그라운드에서 프레임 처리
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await ProcessSingleFrameAsync(frameData, cancellationToken);
                            }
                            finally
                            {
                                _processingSemaphore.Release();
                                frameData.Dispose();
                            }
                        }, cancellationToken);
                    }
                    else
                    {
                        // 큐가 비어있으면 잠시 대기
                        await Task.Delay(1, cancellationToken);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    ErrorOccurred?.Invoke(this, ex);
                }
            }
        }

        /// <summary>
        /// 단일 프레임 처리
        /// </summary>
        private async Task ProcessSingleFrameAsync(FrameData frameData, CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested || !MatHelper.IsValid(frameData.Frame))
                return;

            try
            {
                var processedData = new ProcessedFrameData
                {
                    OriginalFrame = frameData.Frame,
                    FrameNumber = frameData.FrameNumber,
                    Timestamp = frameData.Timestamp
                };

                // ROI 적용
                Mat croppedFrame = CropWithROI(frameData.Frame, frameData.Config, frameData.IsVideoMode);
                Mat resizedFrame = ResizeForModel(croppedFrame, new Size(500, 460));

                if (frameData.IsCalculating)
                {
                    // AI 추론 및 CAC 계산 (백그라운드)
                    await ProcessFrameWithInferenceAsync(resizedFrame, processedData, cancellationToken);
                }
                else
                {
                    // 단순 표시용
                    processedData.DisplayFrame = MatHelper.SafeClone(resizedFrame);
                }

                // UI 스레드로 결과 전달
                UIHelper.SafeInvokeAsync(() =>
                {
                    FrameProcessed?.Invoke(this, processedData);
                });

                // 리소스 정리
                MatHelper.SafeDispose(croppedFrame);
                MatHelper.SafeDispose(resizedFrame);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(this, ex);
            }
        }

        /// <summary>
        /// AI 추론과 함께 프레임 처리
        /// </summary>
        private async Task ProcessFrameWithInferenceAsync(Mat resizedFrame, ProcessedFrameData processedData, CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested)
                return;

            // AI 모델 추론 (CPU/GPU에서 실행)
            Mat mask = null;
            if (_inferenceService != null && _inferenceService.IsModelAvailable)
            {
                mask = await _inferenceService.PredictImageAsync(resizedFrame);
            }
            else
            {
                // 모델이 없는 경우 더미 마스크 생성
                mask = CreateDummyMask(resizedFrame);
            }
                
            if (MatHelper.IsValid(mask))
            {
                // CAC 계산
                string fileName = $"Frame_{processedData.FrameNumber}.png";
                var (measurement, overlay) = _cacCalculator?.ProcessImagesRealtime(
                    fileName, resizedFrame, mask, 300, true) ?? 
                    ProcessCACCalculation(fileName, resizedFrame, mask, 300, true);

                processedData.Measurement = measurement;
                processedData.OverlayFrame = MatHelper.SafeClone(overlay);
                processedData.DisplayFrame = MatHelper.SafeClone(overlay);
            }
            else
            {
                processedData.DisplayFrame = MatHelper.SafeClone(resizedFrame);
            }

            MatHelper.SafeDispose(mask);
        }

        /// <summary>
        /// ROI 적용
        /// </summary>
        private Mat CropWithROI(Mat frame, ConfigSettings config, bool isVideo)
        {
            if (!MatHelper.IsValid(frame))
                return null;

            int x1, y1, x2, y2;
            
            if (isVideo)
            {
                x1 = config.ROI_X1_Video;
                y1 = config.ROI_Y1_Video;
                x2 = config.ROI_X2_Video;
                y2 = config.ROI_Y2_Video;
            }
            else
            {
                x1 = config.ROI_X1;
                y1 = config.ROI_Y1;
                x2 = config.ROI_X2;
                y2 = config.ROI_Y2;
            }

            var rect = new Rect(x1, y1, x2 - x1, y2 - y1);
            return new Mat(frame, rect);
        }

        /// <summary>
        /// 모델 입력용 리사이즈
        /// </summary>
        private Mat ResizeForModel(Mat image, Size targetSize)
        {
            if (!MatHelper.IsValid(image))
                return null;

            Mat resized = new Mat();
            Cv2.Resize(image, resized, targetSize, 0, 0, InterpolationFlags.Linear);
            return resized;
        }

        /// <summary>
        /// CAC 계산 (임시 구현)
        /// </summary>
        private (MeasurementData measurement, Mat overlay) ProcessCACCalculation(
            string fileName, Mat inputImage, Mat maskImage, int size, bool drawEllipse)
        {
            var measurement = new MeasurementData
            {
                FileName = fileName,
                CACValue = new Random().NextDouble() * 0.5 + 0.5, // 임시 값
                IJVValue = new Random().NextDouble() * 0.3 + 0.7, // 임시 값
                State = "Measuring..."
            };

            Mat overlay = MatHelper.SafeClone(inputImage);
            return (measurement, overlay);
        }

        /// <summary>
        /// 더미 마스크 생성 (임시)
        /// </summary>
        private Mat CreateDummyMask(Mat inputImage)
        {
            if (!MatHelper.IsValid(inputImage))
                return null;

            Mat mask = new Mat(inputImage.Size(), MatType.CV_8UC1, Scalar.All(120));
            return mask;
        }

        /// <summary>
        /// 큐 상태 확인
        /// </summary>
        public int QueueCount => _frameQueue.Count;
        public bool IsProcessing => !_cancellationTokenSource.Token.IsCancellationRequested;

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;
            _cancellationTokenSource?.Cancel();

            // 큐에 남은 프레임들 정리
            while (_frameQueue.TryDequeue(out FrameData frameData))
            {
                frameData.Dispose();
            }

            _processingSemaphore?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    /// <summary>
    /// 프레임 데이터
    /// </summary>
    public class FrameData : IDisposable
    {
        public Mat Frame { get; set; }
        public bool IsCalculating { get; set; }
        public ConfigSettings Config { get; set; }
        public bool IsVideoMode { get; set; }
        public DateTime Timestamp { get; set; }
        public int FrameNumber { get; set; }

        public void Dispose()
        {
            MatHelper.SafeDispose(Frame);
        }
    }

    /// <summary>
    /// 처리된 프레임 데이터
    /// </summary>
    public class ProcessedFrameData
    {
        public Mat OriginalFrame { get; set; }
        public Mat DisplayFrame { get; set; }
        public Mat OverlayFrame { get; set; }
        public MeasurementData Measurement { get; set; }
        public int FrameNumber { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
