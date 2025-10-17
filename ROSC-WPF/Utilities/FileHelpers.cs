using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using ClosedXML.Excel;
using OpenCvSharp;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 파일 처리 유틸리티
    /// Excel 내보내기, 비디오 저장, 폴더 관리 등
    /// </summary>
    public static class FileHelpers
    {
        /// <summary>
        /// 폴더 생성
        /// Python의 folder_make_func() 함수와 동일
        /// </summary>
        public static bool CreateFolder(string folderPath)
        {
            try
            {
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                    return true;
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Create folder error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 임시 폴더 정리 및 재생성
        /// </summary>
        public static bool RecreateTempFolder(string folderPath)
        {
            try
            {
                if (Directory.Exists(folderPath))
                {
                    Directory.Delete(folderPath, true);
                }
                return CreateFolder(folderPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Recreate temp folder error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Excel 파일로 측정 데이터 내보내기
        /// Python의 onSaveExcel() 함수와 동일
        /// </summary>
        public static bool ExportToExcel(List<ROSC.WPF.Models.MeasurementData> measurements, string filePath)
        {
            try
            {
                using (var workbook = new XLWorkbook())
                {
                    var worksheet = workbook.Worksheets.Add("Measurements");
                    
                    // 헤더 추가
                    worksheet.Cell(1, 1).Value = "File Name";
                    worksheet.Cell(1, 2).Value = "CA Compression Value";
                    worksheet.Cell(1, 3).Value = "IJV Compression Value";
                    worksheet.Cell(1, 4).Value = "Min/Max";
                    worksheet.Cell(1, 5).Value = "Class";
                    worksheet.Cell(1, 6).Value = "State";

                    // 데이터 추가
                    for (int i = 0; i < measurements.Count; i++)
                    {
                        var measurement = measurements[i];
                        int row = i + 2;
                        
                        worksheet.Cell(row, 1).Value = measurement.FileName;
                        worksheet.Cell(row, 2).Value = measurement.CACValue;
                        worksheet.Cell(row, 3).Value = measurement.IJVValue;
                        worksheet.Cell(row, 4).Value = measurement.MinMaxRatio;
                        worksheet.Cell(row, 5).Value = measurement.Class;
                        worksheet.Cell(row, 6).Value = measurement.State;
                    }

                    // 헤더 스타일링
                    var headerRange = worksheet.Range(1, 1, 1, 6);
                    headerRange.Style.Font.Bold = true;
                    headerRange.Style.Fill.BackgroundColor = XLColor.LightGray;

                    workbook.SaveAs(filePath);
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Excel export error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 비디오 파일로 프레임들 저장
        /// Python의 onSaveVideo() 함수와 동일
        /// </summary>
        public static bool SaveFramesAsVideo(List<string> framePaths, string outputPath, double fps = 10.0, Size frameSize = default)
        {
            if (framePaths == null || framePaths.Count == 0)
                return false;

            try
            {
                if (frameSize == default)
                    frameSize = new Size(640, 480);

                using (var writer = new VideoWriter(outputPath, FourCC.XVID, fps, frameSize))
                {
                    foreach (var framePath in framePaths)
                    {
                        if (File.Exists(framePath))
                        {
                            using (var frame = Cv2.ImRead(framePath))
                            {
                                if (!frame.Empty())
                                {
                                    Mat resizedFrame = new Mat();
                                    Cv2.Resize(frame, resizedFrame, frameSize);
                                    writer.Write(resizedFrame);
                                }
                            }
                        }
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video save error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 이미지 파일들을 폴더에 저장
        /// </summary>
        public static bool SaveImagesToFolder(List<Mat> images, List<string> fileNames, string folderPath)
        {
            if (images == null || fileNames == null || images.Count != fileNames.Count)
                return false;

            try
            {
                CreateFolder(folderPath);

                for (int i = 0; i < images.Count; i++)
                {
                    if (images[i] != null && !images[i].Empty())
                    {
                        string filePath = Path.Combine(folderPath, fileNames[i]);
                        images[i].SaveImage(filePath);
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Save images to folder error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 파일 확장자로 비디오 파일인지 확인
        /// </summary>
        public static bool IsVideoFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;

            var extension = Path.GetExtension(filePath).ToLower();
            var videoExtensions = new[] { ".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm" };
            
            return videoExtensions.Contains(extension);
        }

        /// <summary>
        /// 파일 확장자로 이미지 파일인지 확인
        /// </summary>
        public static bool IsImageFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;

            var extension = Path.GetExtension(filePath).ToLower();
            var imageExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif" };
            
            return imageExtensions.Contains(extension);
        }

        /// <summary>
        /// 폴더에서 특정 확장자 파일들 찾기
        /// </summary>
        public static List<string> GetFilesByExtension(string folderPath, string[] extensions)
        {
            if (!Directory.Exists(folderPath))
                return new List<string>();

            try
            {
                var files = Directory.GetFiles(folderPath)
                    .Where(file => extensions.Contains(Path.GetExtension(file).ToLower()))
                    .OrderBy(file => file)
                    .ToList();

                return files;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Get files by extension error: {ex.Message}");
                return new List<string>();
            }
        }

        /// <summary>
        /// 파일 크기 가져오기 (MB 단위)
        /// </summary>
        public static double GetFileSizeInMB(string filePath)
        {
            if (!File.Exists(filePath))
                return 0;

            try
            {
                var fileInfo = new FileInfo(filePath);
                return fileInfo.Length / (1024.0 * 1024.0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Get file size error: {ex.Message}");
                return 0;
            }
        }

        /// <summary>
        /// 파일명에서 타임스탬프 생성
        /// Python의 time.strftime() 함수와 동일
        /// </summary>
        public static string GenerateTimestampFileName(string prefix = "", string extension = ".png")
        {
            var timestamp = DateTime.Now.ToString("yyMMdd_HH_mm_ss");
            return $"{prefix}{timestamp}{extension}";
        }

        /// <summary>
        /// 안전한 파일명 생성 (특수문자 제거)
        /// </summary>
        public static string SanitizeFileName(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
                return "unnamed";

            var invalidChars = Path.GetInvalidFileNameChars();
            var sanitized = new string(fileName.Where(ch => !invalidChars.Contains(ch)).ToArray());
            
            return string.IsNullOrEmpty(sanitized) ? "unnamed" : sanitized;
        }

        /// <summary>
        /// 파일 존재 여부 확인 및 백업 파일명 생성
        /// </summary>
        public static string GetUniqueFileName(string filePath)
        {
            if (!File.Exists(filePath))
                return filePath;

            var directory = Path.GetDirectoryName(filePath);
            var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePath);
            var extension = Path.GetExtension(filePath);

            int counter = 1;
            string newFilePath;
            
            do
            {
                newFilePath = Path.Combine(directory, $"{fileNameWithoutExtension}_{counter}{extension}");
                counter++;
            } while (File.Exists(newFilePath));

            return newFilePath;
        }
    }
}
