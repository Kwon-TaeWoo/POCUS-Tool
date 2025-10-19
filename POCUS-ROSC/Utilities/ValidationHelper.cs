using System;
using System.IO;
using OpenCvSharp;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// 유효성 검사 유틸리티
    /// </summary>
    public static class ValidationHelper
    {
        /// <summary>
        /// 문자열 유효성 검사
        /// </summary>
        public static bool IsValidString(string value, bool allowEmpty = false)
        {
            if (allowEmpty)
                return value != null;
            return !string.IsNullOrWhiteSpace(value);
        }

        /// <summary>
        /// 파일 경로 유효성 검사
        /// </summary>
        public static bool IsValidFilePath(string filePath, bool checkExists = true)
        {
            if (!IsValidString(filePath))
                return false;

            try
            {
                // 경로 형식 검사
                if (Path.GetInvalidPathChars().Length > 0)
                {
                    foreach (char c in Path.GetInvalidPathChars())
                    {
                        if (filePath.Contains(c.ToString()))
                            return false;
                    }
                }

                // 파일명 검사
                string fileName = Path.GetFileName(filePath);
                if (string.IsNullOrEmpty(fileName))
                    return false;

                // 존재 여부 검사
                if (checkExists && !File.Exists(filePath))
                    return false;

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// 디렉토리 경로 유효성 검사
        /// </summary>
        public static bool IsValidDirectoryPath(string directoryPath, bool checkExists = true)
        {
            if (!IsValidString(directoryPath))
                return false;

            try
            {
                // 경로 형식 검사
                if (Path.GetInvalidPathChars().Length > 0)
                {
                    foreach (char c in Path.GetInvalidPathChars())
                    {
                        if (directoryPath.Contains(c.ToString()))
                            return false;
                    }
                }

                // 존재 여부 검사
                if (checkExists && !Directory.Exists(directoryPath))
                    return false;

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Mat 유효성 검사
        /// </summary>
        public static bool IsValidMat(Mat mat, int? expectedWidth = null, int? expectedHeight = null, int? expectedChannels = null)
        {
            if (!MatHelper.IsValid(mat))
                return false;

            if (expectedWidth.HasValue && mat.Width != expectedWidth.Value)
                return false;

            if (expectedHeight.HasValue && mat.Height != expectedHeight.Value)
                return false;

            if (expectedChannels.HasValue && mat.Channels() != expectedChannels.Value)
                return false;

            return true;
        }

        /// <summary>
        /// ROI 좌표 유효성 검사
        /// </summary>
        public static bool IsValidROI(int x1, int y1, int x2, int y2, int imageWidth = 0, int imageHeight = 0)
        {
            // 기본 좌표 검사
            if (x1 >= x2 || y1 >= y2)
                return false;

            if (x1 < 0 || y1 < 0)
                return false;

            // 이미지 크기와 비교 (이미지 크기가 제공된 경우)
            if (imageWidth > 0 && x2 > imageWidth)
                return false;

            if (imageHeight > 0 && y2 > imageHeight)
                return false;

            return true;
        }

        /// <summary>
        /// 숫자 범위 검사
        /// </summary>
        public static bool IsInRange(double value, double min, double max)
        {
            return value >= min && value <= max;
        }

        /// <summary>
        /// 정수 범위 검사
        /// </summary>
        public static bool IsInRange(int value, int min, int max)
        {
            return value >= min && value <= max;
        }

        /// <summary>
        /// CAC 값 유효성 검사
        /// </summary>
        public static bool IsValidCACValue(double cacValue)
        {
            return IsInRange(cacValue, 0.0, 1.0);
        }

        /// <summary>
        /// IJV 값 유효성 검사
        /// </summary>
        public static bool IsValidIJVValue(double ijvValue)
        {
            return IsInRange(ijvValue, 0.0, 1.0);
        }

        /// <summary>
        /// ABP 값 유효성 검사
        /// </summary>
        public static bool IsValidABPValue(double abpValue)
        {
            return IsInRange(abpValue, 0.0, 300.0); // 혈압 범위
        }

        /// <summary>
        /// 디바이스 ID 유효성 검사
        /// </summary>
        public static bool IsValidDeviceID(int deviceId)
        {
            return deviceId >= 0 && deviceId <= 10; // 일반적인 카메라 디바이스 범위
        }

        /// <summary>
        /// 프레임 크기 유효성 검사
        /// </summary>
        public static bool IsValidFrameSize(int width, int height)
        {
            return width > 0 && height > 0 && width <= 7680 && height <= 4320; // 8K 해상도까지
        }

        /// <summary>
        /// FPS 값 유효성 검사
        /// </summary>
        public static bool IsValidFPS(double fps)
        {
            return IsInRange(fps, 1.0, 120.0);
        }

        /// <summary>
        /// 윈도우 크기 유효성 검사
        /// </summary>
        public static bool IsValidWindowSize(int windowSize)
        {
            return IsInRange(windowSize, 1, 100);
        }

        /// <summary>
        /// 임계값 유효성 검사
        /// </summary>
        public static bool IsValidThreshold(double threshold)
        {
            return IsInRange(threshold, 0.0, 1.0);
        }

        /// <summary>
        /// 설정 값 유효성 검사
        /// </summary>
        public static bool ValidateConfig(POCUS.ROSC.Models.ConfigSettings config)
        {
            if (config == null)
                return false;

            // 필수 설정 검사
            if (!IsValidString(config.ProgramName))
                return false;

            if (!IsValidString(config.Version))
                return false;

            // 모델 경로 검사 (파일이 존재하는 경우)
            if (!string.IsNullOrEmpty(config.LargeModelName) && !IsValidFilePath(config.LargeModelName, true))
                return false;

            if (!string.IsNullOrEmpty(config.SmallModelName) && !IsValidFilePath(config.SmallModelName, true))
                return false;

            // 저장 폴더 검사
            if (!IsValidDirectoryPath(config.SaveFolder, false))
                return false;

            // ROI 검사
            if (!IsValidROI(config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2))
                return false;

            // 디바이스 ID 검사
            if (!IsValidDeviceID(config.DeviceID))
                return false;

            return true;
        }
    }
}
