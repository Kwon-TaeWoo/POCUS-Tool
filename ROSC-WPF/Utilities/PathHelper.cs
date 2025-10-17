using System;
using System.IO;
using System.Linq;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 파일 경로 처리 유틸리티
    /// </summary>
    public static class PathHelper
    {
        /// <summary>
        /// 안전한 경로 결합
        /// </summary>
        public static string Combine(params string[] paths)
        {
            if (paths == null || paths.Length == 0)
                return string.Empty;

            try
            {
                return Path.Combine(paths.Where(p => !string.IsNullOrEmpty(p)).ToArray());
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Path combine");
                return string.Empty;
            }
        }

        /// <summary>
        /// 상대 경로를 절대 경로로 변환
        /// </summary>
        public static string ToAbsolutePath(string path)
        {
            if (string.IsNullOrEmpty(path))
                return string.Empty;

            try
            {
                if (Path.IsPathRooted(path))
                    return path;

                return Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), path));
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Convert to absolute path");
                return path;
            }
        }

        /// <summary>
        /// 파일 존재 여부 확인
        /// </summary>
        public static bool FileExists(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;

            try
            {
                return File.Exists(filePath);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Check file existence");
                return false;
            }
        }

        /// <summary>
        /// 디렉토리 존재 여부 확인
        /// </summary>
        public static bool DirectoryExists(string directoryPath)
        {
            if (string.IsNullOrEmpty(directoryPath))
                return false;

            try
            {
                return Directory.Exists(directoryPath);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Check directory existence");
                return false;
            }
        }

        /// <summary>
        /// 안전한 디렉토리 생성
        /// </summary>
        public static bool CreateDirectory(string directoryPath)
        {
            if (string.IsNullOrEmpty(directoryPath))
                return false;

            try
            {
                if (!Directory.Exists(directoryPath))
                {
                    Directory.CreateDirectory(directoryPath);
                }
                return true;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Create directory");
                return false;
            }
        }

        /// <summary>
        /// 파일 확장자 가져오기
        /// </summary>
        public static string GetExtension(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return string.Empty;

            try
            {
                return Path.GetExtension(filePath).ToLower();
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get file extension");
                return string.Empty;
            }
        }

        /// <summary>
        /// 파일명에서 확장자 제거
        /// </summary>
        public static string GetFileNameWithoutExtension(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return string.Empty;

            try
            {
                return Path.GetFileNameWithoutExtension(filePath);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get filename without extension");
                return filePath;
            }
        }

        /// <summary>
        /// 파일명만 가져오기
        /// </summary>
        public static string GetFileName(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return string.Empty;

            try
            {
                return Path.GetFileName(filePath);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get filename");
                return filePath;
            }
        }

        /// <summary>
        /// 디렉토리 경로만 가져오기
        /// </summary>
        public static string GetDirectoryName(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return string.Empty;

            try
            {
                return Path.GetDirectoryName(filePath);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get directory name");
                return string.Empty;
            }
        }

        /// <summary>
        /// 타임스탬프가 포함된 파일명 생성
        /// </summary>
        public static string GenerateTimestampFileName(string prefix = "", string extension = ".png")
        {
            try
            {
                var timestamp = DateTime.Now.ToString("yyMMdd_HH_mm_ss");
                return $"{prefix}{timestamp}{extension}";
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Generate timestamp filename");
                return $"file_{DateTime.Now.Ticks}{extension}";
            }
        }

        /// <summary>
        /// 고유한 파일명 생성 (중복 시 번호 추가)
        /// </summary>
        public static string GetUniqueFileName(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return string.Empty;

            if (!File.Exists(filePath))
                return filePath;

            try
            {
                string directory = Path.GetDirectoryName(filePath);
                string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePath);
                string extension = Path.GetExtension(filePath);

                int counter = 1;
                string newFilePath;
                
                do
                {
                    newFilePath = Path.Combine(directory, $"{fileNameWithoutExtension}_{counter}{extension}");
                    counter++;
                } while (File.Exists(newFilePath));

                return newFilePath;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get unique filename");
                return filePath;
            }
        }

        /// <summary>
        /// 안전한 파일명 생성 (특수문자 제거)
        /// </summary>
        public static string SanitizeFileName(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
                return "unnamed";

            try
            {
                var invalidChars = Path.GetInvalidFileNameChars();
                var sanitized = new string(fileName.Where(ch => !invalidChars.Contains(ch)).ToArray());
                
                return string.IsNullOrEmpty(sanitized) ? "unnamed" : sanitized;
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Sanitize filename");
                return "unnamed";
            }
        }

        /// <summary>
        /// 파일 크기 가져오기 (MB 단위)
        /// </summary>
        public static double GetFileSizeInMB(string filePath)
        {
            if (!FileExists(filePath))
                return 0;

            try
            {
                var fileInfo = new FileInfo(filePath);
                return fileInfo.Length / (1024.0 * 1024.0);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get file size");
                return 0;
            }
        }

        /// <summary>
        /// 임시 파일 경로 생성
        /// </summary>
        public static string GetTempFilePath(string extension = ".tmp")
        {
            try
            {
                string tempDir = Path.GetTempPath();
                string fileName = Path.GetRandomFileName() + extension;
                return Path.Combine(tempDir, fileName);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Get temp file path");
                return $"temp_{DateTime.Now.Ticks}{extension}";
            }
        }
    }
}
