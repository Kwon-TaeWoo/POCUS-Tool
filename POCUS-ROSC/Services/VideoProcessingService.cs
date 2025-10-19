using System;
using System.Threading.Tasks;
using OpenCvSharp;
using POCUS.ROSC.Models;
using POCUS.ROSC.Utilities;

namespace POCUS.ROSC.Services
{
    /// <summary>
    /// 비디오 처리 서비스
    /// Python의 HDMI 캡처 및 비디오 재생 기능을 구현
    /// </summary>
    public class VideoProcessingService : IDisposable
    {
        private VideoCapture _hdmiCapture;
        private VideoCapture _videoFile;
        private readonly ImageProcessingService _imageProcessor;
        private bool _isHDMIConnected = false;
        private bool _isVideoLoaded = false;
        private string _currentVideoPath = "";

        // public event EventHandler<Mat> FrameCaptured; // 현재 사용되지 않음
        public event EventHandler<string> StatusChanged;

        public VideoProcessingService(ImageProcessingService imageProcessor)
        {
            _imageProcessor = imageProcessor;
        }

        /// <summary>
        /// HDMI 연결
        /// Python의 doConnectHDMI() 함수와 동일
        /// </summary>
        public bool ConnectHDMI(int deviceId = 0)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                DisconnectHDMI();

                _hdmiCapture = new VideoCapture(deviceId);
                
                if (!_hdmiCapture.IsOpened())
                {
                    StatusChanged?.Invoke(this, "HDMI 연결 실패");
                    return false;
                }

                // 카메라 설정
                _hdmiCapture.Set(VideoCaptureProperties.FrameWidth, 1920);
                _hdmiCapture.Set(VideoCaptureProperties.FrameHeight, 1080);
                
                _isHDMIConnected = true;
                StatusChanged?.Invoke(this, "HDMI 연결됨");
                
                return true;
            }, false, "HDMI connection");
        }

        /// <summary>
        /// HDMI 연결 해제
        /// </summary>
        public void DisconnectHDMI()
        {
            try
            {
                if (_hdmiCapture != null)
                {
                    _hdmiCapture.Release();
                    _hdmiCapture.Dispose();
                    _hdmiCapture = null;
                }
                _isHDMIConnected = false;
                StatusChanged?.Invoke(this, "HDMI 연결 해제됨");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"HDMI disconnect error: {ex.Message}");
            }
        }

        /// <summary>
        /// 비디오 파일 로드
        /// Python의 onStartvideoShow() 함수와 동일
        /// </summary>
        public bool LoadVideoFile(string videoPath)
        {
            try
            {
                if (string.IsNullOrEmpty(videoPath) || !System.IO.File.Exists(videoPath))
                {
                    StatusChanged?.Invoke(this, "비디오 파일을 찾을 수 없습니다");
                    return false;
                }

                _videoFile?.Release();
                _videoFile?.Dispose();

                _videoFile = _imageProcessor.VideoInit(videoPath);
                
                if (_videoFile == null || !_videoFile.IsOpened())
                {
                    StatusChanged?.Invoke(this, "비디오 파일 열기 실패");
                    return false;
                }

                _currentVideoPath = videoPath;
                _isVideoLoaded = true;
                StatusChanged?.Invoke(this, $"비디오 로드됨: {System.IO.Path.GetFileName(videoPath)}");
                
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video load error: {ex.Message}");
                StatusChanged?.Invoke(this, $"비디오 로드 오류: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 비디오 파일 해제
        /// </summary>
        public void UnloadVideoFile()
        {
            try
            {
                if (_videoFile != null)
                {
                    _videoFile.Release();
                    _videoFile.Dispose();
                    _videoFile = null;
                }
                _isVideoLoaded = false;
                _currentVideoPath = "";
                StatusChanged?.Invoke(this, "비디오 파일 해제됨");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video unload error: {ex.Message}");
            }
        }

        /// <summary>
        /// HDMI 프레임 캡처
        /// Python의 doShowHDMI() 함수와 동일
        /// </summary>
        public Mat CaptureHDMIFrame()
        {
            if (!_isHDMIConnected || _hdmiCapture == null)
                return null;

            try
            {
                Mat frame = new Mat();
                bool ret = _hdmiCapture.Read(frame);

                if (!ret || frame.Empty())
                {
                    StatusChanged?.Invoke(this, "HDMI 프레임 캡처 실패");
                    return null;
                }

                return frame;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"HDMI frame capture error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 비디오 파일 프레임 읽기
        /// Python의 doshow_next_video() 함수와 동일
        /// </summary>
        public Mat ReadVideoFrame()
        {
            if (!_isVideoLoaded || _videoFile == null)
                return null;

            try
            {
                Mat frame = new Mat();
                bool ret = _videoFile.Read(frame);

                if (!ret || frame.Empty())
                {
                    // 비디오 끝에 도달했으면 처음으로 되돌리기
                    _videoFile.Set(VideoCaptureProperties.PosFrames, 0);
                    return null;
                }

                return frame;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video frame read error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// ROI 적용하여 이미지 크롭
        /// </summary>
        public Mat CropWithROI(Mat frame, ConfigSettings config, bool isVideo = false)
        {
            if (frame == null || frame.Empty())
                return null;

            try
            {
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

                return _imageProcessor.ImageCrop3Ch(frame, x1, y1, x2, y2);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ROI crop error: {ex.Message}");
                return frame.Clone();
            }
        }

        /// <summary>
        /// 이미지 리사이즈 (모델 입력용)
        /// </summary>
        public Mat ResizeForModel(Mat image, Size targetSize)
        {
            if (image == null || image.Empty())
                return null;

            return _imageProcessor.ImageScaling(image, targetSize, InterpolationFlags.Linear);
        }

        /// <summary>
        /// 비디오 정보 가져오기
        /// </summary>
        public VideoInfo GetVideoInfo()
        {
            VideoCapture video = _isVideoLoaded ? _videoFile : _hdmiCapture;
            
            if (video == null || !video.IsOpened())
                return null;

            try
            {
                return new VideoInfo
                {
                    Width = (int)video.Get(VideoCaptureProperties.FrameWidth),
                    Height = (int)video.Get(VideoCaptureProperties.FrameHeight),
                    FPS = video.Get(VideoCaptureProperties.Fps),
                    FrameCount = _isVideoLoaded ? (int)video.Get(VideoCaptureProperties.FrameCount) : -1,
                    CurrentFrame = (int)video.Get(VideoCaptureProperties.PosFrames)
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Get video info error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 연결 상태 확인
        /// </summary>
        public bool IsHDMIConnected => _isHDMIConnected;
        public bool IsVideoLoaded => _isVideoLoaded;
        public string CurrentVideoPath => _currentVideoPath;

        public void Dispose()
        {
            DisconnectHDMI();
            UnloadVideoFile();
        }
    }

    /// <summary>
    /// 비디오 정보 클래스
    /// </summary>
    public class VideoInfo
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public double FPS { get; set; }
        public int FrameCount { get; set; }
        public int CurrentFrame { get; set; }
    }
}
