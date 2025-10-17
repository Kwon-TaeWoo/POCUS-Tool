using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using ROSC.WPF.Models;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 실시간 처리 최적화 유틸리티
    /// 캡처보드 실시간 AI 분석 최적화
    /// </summary>
    public class RealtimeOptimizer : IDisposable
    {
        private readonly ConcurrentQueue<Mat> _framePool;
        private readonly SemaphoreSlim _framePoolSemaphore;
        private readonly int _poolSize;
        private bool _isDisposed = false;

        public RealtimeOptimizer(int poolSize = 10)
        {
            _poolSize = poolSize;
            _framePool = new ConcurrentQueue<Mat>();
            _framePoolSemaphore = new SemaphoreSlim(poolSize, poolSize);
            
            // 프레임 풀 초기화
            InitializeFramePool();
        }

        /// <summary>
        /// 프레임 풀 초기화
        /// </summary>
        private void InitializeFramePool()
        {
            for (int i = 0; i < _poolSize; i++)
            {
                _framePool.Enqueue(new Mat());
            }
        }

        /// <summary>
        /// 프레임 풀에서 Mat 가져오기
        /// </summary>
        public Mat GetFrameFromPool()
        {
            if (_isDisposed)
                return null;

            _framePoolSemaphore.Wait();
            
            if (_framePool.TryDequeue(out Mat frame))
            {
                return frame;
            }
            
            _framePoolSemaphore.Release();
            return new Mat(); // 풀이 비어있으면 새로 생성
        }

        /// <summary>
        /// 프레임 풀에 Mat 반환
        /// </summary>
        public void ReturnFrameToPool(Mat frame)
        {
            if (_isDisposed || frame == null)
                return;

            try
            {
                // 프레임 초기화
                frame.SetTo(Scalar.All(0));
                _framePool.Enqueue(frame);
                _framePoolSemaphore.Release();
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Return frame to pool");
                MatHelper.SafeDispose(frame);
            }
        }

        /// <summary>
        /// 프레임 복사 (메모리 효율적)
        /// </summary>
        public Mat CopyFrame(Mat source)
        {
            if (!MatHelper.IsValid(source))
                return null;

            Mat pooledFrame = GetFrameFromPool();
            if (pooledFrame != null)
            {
                try
                {
                    // 크기가 다르면 재할당
                    if (pooledFrame.Size() != source.Size() || pooledFrame.Type() != source.Type())
                    {
                        MatHelper.SafeDispose(pooledFrame);
                        pooledFrame = source.Clone();
                    }
                    else
                    {
                        source.CopyTo(pooledFrame);
                    }
                    return pooledFrame;
                }
                catch (Exception ex)
                {
                    ExceptionHelper.LogError(ex, "Copy frame");
                    ReturnFrameToPool(pooledFrame);
                    return source.Clone();
                }
            }

            return source.Clone();
        }

        /// <summary>
        /// 프레임 풀 상태
        /// </summary>
        public int AvailableFrames => _framePool.Count;
        public int PoolSize => _poolSize;

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;

            // 풀의 모든 프레임 해제
            while (_framePool.TryDequeue(out Mat frame))
            {
                MatHelper.SafeDispose(frame);
            }

            _framePoolSemaphore?.Dispose();
        }
    }

    /// <summary>
    /// 실시간 처리 설정
    /// </summary>
    public class RealtimeSettings
    {
        public int MaxQueueSize { get; set; } = 5;
        public int MaxConcurrentProcessing { get; set; } = 2;
        public int FramePoolSize { get; set; } = 10;
        public int TargetFPS { get; set; } = 30;
        public bool EnableGPU { get; set; } = true;
        public bool EnableFrameSkipping { get; set; } = true;
        public int FrameSkipThreshold { get; set; } = 3; // 큐가 이 개수 이상이면 프레임 스킵
    }

    /// <summary>
    /// 실시간 처리 관리자
    /// </summary>
    public class RealtimeProcessor : IDisposable
    {
        private readonly RealtimeOptimizer _optimizer;
        private readonly RealtimeSettings _settings;
        private readonly ConcurrentQueue<Mat> _processingQueue;
        private readonly SemaphoreSlim _processingSemaphore;
        private readonly CancellationTokenSource _cancellationTokenSource;
        private bool _isDisposed = false;
        private int _skippedFrames = 0;

        public event EventHandler<Mat> FrameProcessed;
        public event EventHandler<string> StatusChanged;

        public RealtimeProcessor(RealtimeSettings settings = null)
        {
            _settings = settings ?? new RealtimeSettings();
            _optimizer = new RealtimeOptimizer(_settings.FramePoolSize);
            _processingQueue = new ConcurrentQueue<Mat>();
            _processingSemaphore = new SemaphoreSlim(_settings.MaxConcurrentProcessing, _settings.MaxConcurrentProcessing);
            _cancellationTokenSource = new CancellationTokenSource();

            // 백그라운드 처리 시작
            Task.Run(ProcessFramesAsync, _cancellationTokenSource.Token);
        }

        /// <summary>
        /// 프레임 추가
        /// </summary>
        public bool AddFrame(Mat frame)
        {
            if (_isDisposed || _cancellationTokenSource.Token.IsCancellationRequested)
                return false;

            try
            {
                // 프레임 스킵 로직
                if (_settings.EnableFrameSkipping && _processingQueue.Count >= _settings.FrameSkipThreshold)
                {
                    _skippedFrames++;
                    if (_skippedFrames % 10 == 0)
                    {
                        StatusChanged?.Invoke(this, $"프레임 스킵: {_skippedFrames}개");
                    }
                    return false;
                }

                // 큐가 가득 찬 경우 오래된 프레임 제거
                while (_processingQueue.Count >= _settings.MaxQueueSize)
                {
                    if (_processingQueue.TryDequeue(out Mat oldFrame))
                    {
                        _optimizer.ReturnFrameToPool(oldFrame);
                    }
                }

                // 프레임 복사하여 큐에 추가
                Mat copiedFrame = _optimizer.CopyFrame(frame);
                if (copiedFrame != null)
                {
                    _processingQueue.Enqueue(copiedFrame);
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Add frame to processing queue");
                return false;
            }
        }

        /// <summary>
        /// 백그라운드 프레임 처리
        /// </summary>
        private async Task ProcessFramesAsync()
        {
            while (!_cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    if (_processingQueue.TryDequeue(out Mat frame))
                    {
                        await _processingSemaphore.WaitAsync(_cancellationTokenSource.Token);
                        
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await ProcessSingleFrameAsync(frame);
                            }
                            finally
                            {
                                _processingSemaphore.Release();
                                _optimizer.ReturnFrameToPool(frame);
                            }
                        }, _cancellationTokenSource.Token);
                    }
                    else
                    {
                        await Task.Delay(1, _cancellationTokenSource.Token);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    ExceptionHelper.LogError(ex, "Process frames async");
                }
            }
        }

        /// <summary>
        /// 단일 프레임 처리
        /// </summary>
        private async Task ProcessSingleFrameAsync(Mat frame)
        {
            if (!MatHelper.IsValid(frame))
                return;

            try
            {
                // 여기서 실제 AI 추론 및 CAC 계산 수행
                // 현재는 단순히 프레임을 그대로 전달
                await Task.Delay(10); // 처리 시간 시뮬레이션
                
                FrameProcessed?.Invoke(this, frame);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Process single frame");
            }
        }

        /// <summary>
        /// 처리 중지
        /// </summary>
        public void Stop()
        {
            _cancellationTokenSource?.Cancel();
        }

        /// <summary>
        /// 상태 정보
        /// </summary>
        public int QueueCount => _processingQueue.Count;
        public int SkippedFrames => _skippedFrames;
        public int AvailableFrames => _optimizer.AvailableFrames;

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;
            Stop();

            // 큐 정리
            while (_processingQueue.TryDequeue(out Mat frame))
            {
                _optimizer.ReturnFrameToPool(frame);
            }

            _optimizer?.Dispose();
            _processingSemaphore?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }
}
