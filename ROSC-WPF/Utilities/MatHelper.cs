using System;
using OpenCvSharp;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// OpenCV Mat 관련 유틸리티 함수들
    /// </summary>
    public static class MatHelper
    {
        /// <summary>
        /// Mat 유효성 검사
        /// </summary>
        public static bool IsValid(Mat mat)
        {
            return mat != null && !mat.Empty();
        }

        /// <summary>
        /// 안전한 Mat 복사
        /// </summary>
        public static Mat SafeClone(Mat mat)
        {
            if (!IsValid(mat))
                return null;

            try
            {
                return mat.Clone();
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Mat clone");
                return null;
            }
        }

        /// <summary>
        /// 안전한 Mat 해제
        /// </summary>
        public static void SafeDispose(Mat mat)
        {
            if (mat != null)
            {
                try
                {
                    mat.Dispose();
                }
                catch (Exception ex)
                {
                    ExceptionHelper.LogError(ex, "Mat dispose");
                }
            }
        }

        /// <summary>
        /// Mat 크기 정보 가져오기
        /// </summary>
        public static (int width, int height, int channels) GetMatInfo(Mat mat)
        {
            if (!IsValid(mat))
                return (0, 0, 0);

            return (mat.Width, mat.Height, mat.Channels());
        }

        /// <summary>
        /// Mat이 지정된 크기와 채널을 가지는지 확인
        /// </summary>
        public static bool IsMatSize(Mat mat, int expectedWidth, int expectedHeight, int expectedChannels = -1)
        {
            if (!IsValid(mat))
                return false;

            var (width, height, channels) = GetMatInfo(mat);
            
            bool sizeMatch = width == expectedWidth && height == expectedHeight;
            bool channelMatch = expectedChannels == -1 || channels == expectedChannels;
            
            return sizeMatch && channelMatch;
        }

        /// <summary>
        /// Mat을 그레이스케일로 변환
        /// </summary>
        public static Mat ToGrayscale(Mat mat)
        {
            if (!IsValid(mat))
                return null;

            try
            {
                if (mat.Channels() == 1)
                    return mat.Clone();

                Mat gray = new Mat();
                Cv2.CvtColor(mat, gray, ColorConversionCodes.BGR2GRAY);
                return gray;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Convert to grayscale");
                return null;
            }
        }

        /// <summary>
        /// Mat을 RGB로 변환
        /// </summary>
        public static Mat ToRGB(Mat mat)
        {
            if (!IsValid(mat))
                return null;

            try
            {
                if (mat.Channels() == 3)
                    return mat.Clone();

                Mat rgb = new Mat();
                Cv2.CvtColor(mat, rgb, ColorConversionCodes.BGR2RGB);
                return rgb;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Convert to RGB");
                return null;
            }
        }

        /// <summary>
        /// Mat 정규화 (0-1 범위)
        /// </summary>
        public static Mat Normalize(Mat mat, double alpha = 1.0 / 255.0, double beta = 0.0)
        {
            if (!IsValid(mat))
                return null;

            try
            {
                Mat normalized = new Mat();
                mat.ConvertTo(normalized, MatType.CV_32F, alpha, beta);
                return normalized;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Normalize Mat");
                return null;
            }
        }

        /// <summary>
        /// Mat 크기 조정 (비율 유지)
        /// </summary>
        public static Mat ResizeKeepAspectRatio(Mat mat, int maxWidth, int maxHeight, InterpolationFlags interpolation = InterpolationFlags.Linear)
        {
            if (!IsValid(mat))
                return null;

            try
            {
                int originalWidth = mat.Width;
                int originalHeight = mat.Height;

                double ratioX = (double)maxWidth / originalWidth;
                double ratioY = (double)maxHeight / originalHeight;
                double ratio = Math.Min(ratioX, ratioY);

                int newWidth = (int)(originalWidth * ratio);
                int newHeight = (int)(originalHeight * ratio);

                Mat resized = new Mat();
                Cv2.Resize(mat, resized, new Size(newWidth, newHeight), 0, 0, interpolation);
                
                return resized;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Resize Mat keep aspect ratio");
                return null;
            }
        }

        /// <summary>
        /// Mat에 텍스트 추가
        /// </summary>
        public static Mat AddText(Mat mat, string text, Point position, double fontScale = 0.5, Scalar color = default, int thickness = 2)
        {
            if (!IsValid(mat) || string.IsNullOrEmpty(text))
                return mat;

            try
            {
                Mat result = mat.Clone();
                if (color == default)
                    color = Scalar.Cyan;

                Cv2.PutText(result, text, position, HersheyFonts.HersheySimplex, fontScale, color, thickness);
                return result;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Add text to Mat");
                return mat.Clone();
            }
        }

        /// <summary>
        /// Mat에 여러 텍스트 추가
        /// </summary>
        public static Mat AddMultipleText(Mat mat, params (string text, Point position)[] texts)
        {
            if (!IsValid(mat) || texts == null || texts.Length == 0)
                return mat;

            try
            {
                Mat result = mat.Clone();
                
                foreach (var (text, position) in texts)
                {
                    if (!string.IsNullOrEmpty(text))
                    {
                        Cv2.PutText(result, text, position, HersheyFonts.HersheySimplex, 0.5, Scalar.Cyan, 2);
                    }
                }
                
                return result;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Add multiple text to Mat");
                return mat.Clone();
            }
        }
    }
}
