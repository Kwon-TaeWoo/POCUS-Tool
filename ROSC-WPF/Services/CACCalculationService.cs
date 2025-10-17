using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using OpenCvSharp;
using ROSC.WPF.Models;

namespace ROSC.WPF.Services
{
    /// <summary>
    /// CAC 계산 서비스
    /// Python의 calculate_CAC.py 함수들을 구현
    /// </summary>
    public class CACCalculationService
    {
        private readonly ImageProcessingService _imageProcessor;
        private readonly ConcurrentQueue<double> _cacWindow;
        private readonly ConcurrentQueue<double> _ijvWindow;
        private const int WINDOW_SIZE = 10;
        private string _lastState = "Measuring...";
        private int _invalidCount = 0;

        public CACCalculationService(ImageProcessingService imageProcessor)
        {
            _imageProcessor = imageProcessor;
            _cacWindow = new ConcurrentQueue<double>();
            _ijvWindow = new ConcurrentQueue<double>();
        }

        /// <summary>
        /// 실시간 이미지 처리 및 CAC 계산
        /// Python의 process_images_realtime() 함수와 동일
        /// </summary>
        public (MeasurementData measurement, Mat overlay) ProcessImagesRealtime(string fileName, Mat inputImage, Mat maskImage, int size = 300, bool drawEllipse = true)
        {
            var measurement = new MeasurementData
            {
                FileName = fileName,
                CACValue = 0,
                IJVValue = 0,
                MinMaxRatio = 0,
                Class = "",
                State = "Measuring..."
            };

            Mat overlay = inputImage?.Clone();

            if (inputImage == null || maskImage == null)
                return (measurement, overlay);

            try
            {
                // 120, 240 픽셀 값으로 바이너리 마스크 생성
                Mat binary120 = CreateBinaryMask(maskImage, 120);
                Mat binary240 = CreateBinaryMask(maskImage, 240);

                // Contour 검출
                var contours120 = Cv2.FindContoursAsArray(binary120, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                var contours240 = Cv2.FindContoursAsArray(binary240, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                // IJV 계산 (240 contour)
                if (contours240.Length > 0)
                {
                    var largestContour240 = GetLargestContour(contours240);
                    if (largestContour240.Length >= 5)
                    {
                        var ellipse = Cv2.FitEllipse(largestContour240);
                        var minorAxis = Math.Min(ellipse.Size.Width, ellipse.Size.Height);
                        var majorAxis = Math.Max(ellipse.Size.Width, ellipse.Size.Height);
                        
                        if (majorAxis > 0)
                        {
                            measurement.IJVValue = Math.Sqrt(1 - Math.Pow(minorAxis / majorAxis, 2));
                        }
                    }
                }

                // CAC 계산 (120 contour)
                if (contours120.Length > 0)
                {
                    var largestContour120 = GetLargestContour(contours120);
                    if (largestContour120.Length >= 5)
                    {
                        var ellipse = Cv2.FitEllipse(largestContour120);
                        var minorAxis = Math.Min(ellipse.Size.Width, ellipse.Size.Height);
                        var majorAxis = Math.Max(ellipse.Size.Width, ellipse.Size.Height);
                        
                        if (majorAxis > 0)
                        {
                            measurement.MinMaxRatio = majorAxis / minorAxis;
                            measurement.CACValue = Math.Sqrt(1 - Math.Pow(minorAxis / majorAxis, 2));
                        }
                    }
                }

                // 상태 머신 적용
                measurement = ApplyStateMachine(measurement, overlay, contours120, drawEllipse);

                return (measurement, overlay);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CAC calculation error: {ex.Message}");
                return (measurement, overlay);
            }
        }

        /// <summary>
        /// 바이너리 마스크 생성
        /// </summary>
        private Mat CreateBinaryMask(Mat maskImage, byte targetValue)
        {
            Mat binary = new Mat();
            Cv2.InRange(maskImage, new Scalar(targetValue), new Scalar(targetValue), binary);
            return binary;
        }

        /// <summary>
        /// 가장 큰 contour 찾기
        /// </summary>
        private Point[] GetLargestContour(Point[][] contours)
        {
            if (contours.Length == 0)
                return new Point[0];

            double maxArea = 0;
            Point[] largestContour = contours[0];

            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                if (area > maxArea)
                {
                    maxArea = area;
                    largestContour = contour;
                }
            }

            return largestContour;
        }

        /// <summary>
        /// 상태 머신 적용
        /// Python의 calc_CAC_print() 함수와 동일한 로직
        /// </summary>
        private MeasurementData ApplyStateMachine(MeasurementData measurement, Mat overlay, Point[][] contours120, bool drawEllipse)
        {
            double cacValue = measurement.CACValue;
            double ijvValue = measurement.IJVValue;

            // CAC == 0: invalid 체크
            if (cacValue == 0.0)
            {
                _invalidCount++;
                if (_invalidCount >= 3)
                {
                    measurement.State = "Invalid CAC";
                }
                else
                {
                    measurement.State = _lastState;
                }
                return measurement;
            }

            // 정상치 들어오면 invalid 카운터 초기화 & 윈도우에 추가
            _invalidCount = 0;
            AddToWindow(cacValue, ijvValue);

            var windowCac = GetWindowValues(_cacWindow);
            var windowIjv = GetWindowValues(_ijvWindow);

            string newState = _lastState;
            Scalar newColor = Scalar.White;

            // 즉시 ROSC 판단 (급격 감소)
            if (cacValue <= 0.80)
            {
                newState = "ROSC";
                newColor = Scalar.Blue;
            }
            // 윈도우가 채워졌을 때만 추가 판단
            else if (windowCac.Count == WINDOW_SIZE)
            {
                if (AllValuesGreaterEqual(windowCac, 0.94))
                {
                    newState = "Arrest";
                    newColor = Scalar.Red;
                }
                else if (AllValuesLess(windowCac, 0.70) && AllValuesGreater(windowIjv, 0.5))
                {
                    newState = "Not compressed";
                    newColor = Scalar.White;
                }
                else if (AllValuesLess(windowCac, 0.94))
                {
                    newState = "ROSC";
                    newColor = Scalar.Blue;
                }
                else
                {
                    newState = _lastState;
                }
            }
            else
            {
                newState = "Measuring...";
                newColor = Scalar.Yellow;
            }

            measurement.State = newState;
            _lastState = newState;

            // 오버레이에 그리기
            if (contours120.Length > 0)
            {
                var largestContour = GetLargestContour(contours120);
                if (largestContour.Length >= 5)
                {
                    var ellipse = Cv2.FitEllipse(largestContour);
                    
                    if (drawEllipse)
                    {
                        Cv2.Ellipse(overlay, ellipse, newColor, 2);
                    }
                    else
                    {
                        var rect = Cv2.BoundingRect(largestContour);
                        Cv2.Rectangle(overlay, rect, newColor, 2);
                    }
                }
            }

            return measurement;
        }

        /// <summary>
        /// 윈도우에 값 추가
        /// </summary>
        private void AddToWindow(double cacValue, double ijvValue)
        {
            _cacWindow.Enqueue(cacValue);
            _ijvWindow.Enqueue(ijvValue);

            // 윈도우 크기 유지
            while (_cacWindow.Count > WINDOW_SIZE)
            {
                _cacWindow.TryDequeue(out _);
            }
            while (_ijvWindow.Count > WINDOW_SIZE)
            {
                _ijvWindow.TryDequeue(out _);
            }
        }

        /// <summary>
        /// 윈도우 값들 가져오기
        /// </summary>
        private List<double> GetWindowValues(ConcurrentQueue<double> queue)
        {
            var values = new List<double>();
            foreach (var value in queue)
            {
                values.Add(value);
            }
            return values;
        }

        /// <summary>
        /// 모든 값이 임계값 이상인지 확인
        /// </summary>
        private bool AllValuesGreaterEqual(List<double> values, double threshold)
        {
            foreach (var value in values)
            {
                if (value < threshold)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// 모든 값이 임계값 미만인지 확인
        /// </summary>
        private bool AllValuesLess(List<double> values, double threshold)
        {
            foreach (var value in values)
            {
                if (value >= threshold)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// 모든 값이 임계값 초과인지 확인
        /// </summary>
        private bool AllValuesGreater(List<double> values, double threshold)
        {
            foreach (var value in values)
            {
                if (value <= threshold)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// 윈도우 초기화
        /// </summary>
        public void ResetWindows()
        {
            while (_cacWindow.TryDequeue(out _)) { }
            while (_ijvWindow.TryDequeue(out _)) { }
            _lastState = "Measuring...";
            _invalidCount = 0;
        }

        /// <summary>
        /// 현재 윈도우 상태 반환
        /// </summary>
        public (List<double> cacValues, List<double> ijvValues) GetCurrentWindows()
        {
            return (GetWindowValues(_cacWindow), GetWindowValues(_ijvWindow));
        }
    }
}
