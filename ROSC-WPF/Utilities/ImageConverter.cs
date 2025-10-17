using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Media.Imaging;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 이미지 변환 유틸리티
    /// OpenCV Mat, Bitmap, BitmapImage 간 변환
    /// </summary>
    public static class ImageConverter
    {
        /// <summary>
        /// Mat을 BitmapImage로 변환 (WPF용)
        /// </summary>
        public static BitmapImage MatToBitmapImage(Mat mat)
        {
            if (mat == null || mat.Empty())
                return null;

            try
            {
                using (var bitmap = mat.ToBitmap())
                {
                    return BitmapToBitmapImage(bitmap);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Mat to BitmapImage conversion error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Bitmap을 BitmapImage로 변환
        /// </summary>
        public static BitmapImage BitmapToBitmapImage(Bitmap bitmap)
        {
            if (bitmap == null)
                return null;

            try
            {
                using (var memory = new MemoryStream())
                {
                    bitmap.Save(memory, ImageFormat.Png);
                    memory.Position = 0;

                    var bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.StreamSource = memory;
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.EndInit();
                    bitmapImage.Freeze();

                    return bitmapImage;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Bitmap to BitmapImage conversion error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// BitmapImage를 Mat으로 변환
        /// </summary>
        public static Mat BitmapImageToMat(BitmapImage bitmapImage)
        {
            if (bitmapImage == null)
                return null;

            try
            {
                using (var memory = new MemoryStream())
                {
                    var encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                    encoder.Save(memory);

                    memory.Position = 0;
                    var bytes = memory.ToArray();

                    return Cv2.ImDecode(bytes, ImreadModes.Color);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"BitmapImage to Mat conversion error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Mat을 바이트 배열로 변환
        /// </summary>
        public static byte[] MatToBytes(Mat mat, string format = ".png")
        {
            if (mat == null || mat.Empty())
                return null;

            try
            {
                var ext = format.ToLower();
                ImwriteFlags flags = ImwriteFlags.PngCompression;

                if (ext == ".jpg" || ext == ".jpeg")
                {
                    flags = ImwriteFlags.JpegQuality;
                }

                return mat.ToBytes(format, flags);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Mat to bytes conversion error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 바이트 배열을 Mat으로 변환
        /// </summary>
        public static Mat BytesToMat(byte[] bytes, ImreadModes mode = ImreadModes.Color)
        {
            if (bytes == null || bytes.Length == 0)
                return null;

            try
            {
                return Cv2.ImDecode(bytes, mode);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Bytes to Mat conversion error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Mat을 파일로 저장
        /// </summary>
        public static bool SaveMatToFile(Mat mat, string filePath)
        {
            if (mat == null || mat.Empty() || string.IsNullOrEmpty(filePath))
                return false;

            try
            {
                return mat.SaveImage(filePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Save Mat to file error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 파일에서 Mat 로드
        /// </summary>
        public static Mat LoadMatFromFile(string filePath, ImreadModes mode = ImreadModes.Color)
        {
            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
                return null;

            try
            {
                return Cv2.ImRead(filePath, mode);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Load Mat from file error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 이미지 크기 조정 (비율 유지)
        /// </summary>
        public static Mat ResizeMatKeepAspectRatio(Mat mat, int maxWidth, int maxHeight)
        {
            if (mat == null || mat.Empty())
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
                Cv2.Resize(mat, resized, new Size(newWidth, newHeight));
                
                return resized;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Resize Mat keep aspect ratio error: {ex.Message}");
                return mat.Clone();
            }
        }

        /// <summary>
        /// 이미지 회전
        /// </summary>
        public static Mat RotateMat(Mat mat, double angle)
        {
            if (mat == null || mat.Empty())
                return null;

            try
            {
                var center = new Point2f(mat.Width / 2.0f, mat.Height / 2.0f);
                var rotationMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);
                
                Mat rotated = new Mat();
                Cv2.WarpAffine(mat, rotated, rotationMatrix, mat.Size());
                
                return rotated;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Rotate Mat error: {ex.Message}");
                return mat.Clone();
            }
        }

        /// <summary>
        /// 이미지 밝기/대비 조정
        /// </summary>
        public static Mat AdjustBrightnessContrast(Mat mat, double alpha = 1.0, int beta = 0)
        {
            if (mat == null || mat.Empty())
                return null;

            try
            {
                Mat adjusted = new Mat();
                mat.ConvertTo(adjusted, MatType.CV_8UC3, alpha, beta);
                return adjusted;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Adjust brightness/contrast error: {ex.Message}");
                return mat.Clone();
            }
        }
    }
}
