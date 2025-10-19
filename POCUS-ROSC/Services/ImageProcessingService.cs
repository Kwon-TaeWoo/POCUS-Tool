using System;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Drawing;
using System.Drawing.Imaging;
using POCUS.ROSC.Utilities;

namespace POCUS.ROSC.Services
{
    /// <summary>
    /// 이미지 전처리 서비스
    /// Python의 preprocessing.py 함수들을 구현
    /// </summary>
    public class ImageProcessingService
    {
        /// <summary>
        /// 이미지 크롭 (3채널)
        /// Python의 image_crop_3ch() 함수와 동일
        /// </summary>
        public Mat ImageCrop3Ch(Mat img, int roiX1, int roiY1, int roiX2, int roiY2)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(img))
                    return null;

                var rect = new Rect(roiX1, roiY1, roiX2 - roiX1, roiY2 - roiY1);
                return new Mat(img, rect);
            }, null, "Image crop 3ch");
        }

        /// <summary>
        /// 이미지 크롭 (그레이스케일)
        /// Python의 image_crop() 함수와 동일
        /// </summary>
        public Mat ImageCrop(Mat img, int upMargin = 80, int downMargin = 310, int leftMargin = 100, int rightMargin = 240)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(img))
                    return null;

                // 그레이스케일로 변환
                Mat gray = MatHelper.ToGrayscale(img);
                if (!MatHelper.IsValid(gray))
                    return null;

                int height = gray.Height;
                int width = gray.Width;

                if (upMargin + downMargin >= height || leftMargin + rightMargin >= width)
                {
                    ExceptionHelper.LogError(new InvalidOperationException("여백이 이미지 크기를 초과"), "Image crop validation");
                    return null;
                }

                var rect = new Rect(leftMargin, upMargin, width - leftMargin - rightMargin, height - upMargin - downMargin);
                return new Mat(gray, rect);
            }, null, "Image crop");
        }

        /// <summary>
        /// 이미지 스케일링
        /// Python의 image_scaling() 함수와 동일
        /// </summary>
        public Mat ImageScaling(Mat oriImg, OpenCvSharp.Size imgSize, InterpolationFlags interpolation = InterpolationFlags.Area)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(oriImg))
                    return null;

                Mat resizedImg = new Mat();
                Cv2.Resize(oriImg, resizedImg, imgSize, 0, 0, interpolation);
                return resizedImg;
            }, null, "Image scaling");
        }

        /// <summary>
        /// 비디오 초기화
        /// Python의 video_init() 함수와 동일
        /// </summary>
        public VideoCapture VideoInit(string videoPath)
        {
            try
            {
                var video = new VideoCapture(videoPath);
                if (!video.IsOpened())
                {
                    Console.WriteLine("영상 열리지 않음");
                    return null;
                }
                return video;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video init error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 비디오 프레임 읽기
        /// Python의 video_read() 함수와 동일
        /// </summary>
        public Mat VideoRead(VideoCapture video)
        {
            if (video == null || !video.IsOpened())
                return null;

            try
            {
                Mat frame = new Mat();
                bool ret = video.Read(frame);

                if (!ret || frame.Empty())
                    return null;

                // 그레이스케일로 변환
                Mat gray = new Mat();
                Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
                
                return gray;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Video read error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Mat을 Bitmap으로 변환 (ImageConverter 사용)
        /// </summary>
        public Bitmap MatToBitmap(Mat mat)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(mat))
                    return null;
                return mat.ToBitmap();
            }, null, "Mat to Bitmap conversion");
        }

        /// <summary>
        /// Bitmap을 Mat으로 변환 (ImageConverter 사용)
        /// </summary>
        public Mat BitmapToMat(Bitmap bitmap)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (bitmap == null)
                    return null;
                return bitmap.ToMat();
            }, null, "Bitmap to Mat conversion");
        }

        /// <summary>
        /// 이미지 전처리 (모델 입력용)
        /// Python의 전처리 파이프라인과 동일
        /// </summary>
        public Mat PreprocessForModel(Mat inputImage, OpenCvSharp.Size targetSize)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(inputImage))
                    return null;

                Mat processed = MatHelper.SafeClone(inputImage);
                if (!MatHelper.IsValid(processed))
                    return null;

                // BGR to RGB 변환
                if (processed.Channels() == 3)
                {
                    Mat rgb = MatHelper.ToRGB(processed);
                    MatHelper.SafeDispose(processed);
                    processed = rgb;
                }

                // 리사이즈
                Mat resized = ImageScaling(processed, targetSize, InterpolationFlags.Linear);
                MatHelper.SafeDispose(processed);

                // 정규화 (0-1 범위)
                Mat normalized = MatHelper.Normalize(resized);
                MatHelper.SafeDispose(resized);

                return normalized;
            }, null, "Preprocess for model");
        }

        /// <summary>
        /// CLAHE 적용
        /// </summary>
        public Mat ApplyCLAHE(Mat inputImage, double clipLimit = 2.0, OpenCvSharp.Size tileGridSize = default)
        {
            if (inputImage == null || inputImage.Empty())
                return null;

            try
            {
                if (tileGridSize == default)
                    tileGridSize = new OpenCvSharp.Size(8, 8);

                Mat claheResult = new Mat();
                using (var clahe = Cv2.CreateCLAHE(clipLimit, tileGridSize))
                {
                    clahe.Apply(inputImage, claheResult);
                }

                return claheResult;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CLAHE error: {ex.Message}");
                return inputImage.Clone();
            }
        }

        /// <summary>
        /// 이미지에 텍스트 추가
        /// Python의 image_save_with_text() 함수와 동일
        /// </summary>
        public Mat AddTextToImage(Mat image, string frameInfo, string classInfo, string cacInfo)
        {
            return ExceptionHelper.SafeExecute(() =>
            {
                if (!MatHelper.IsValid(image))
                    return null;

                var texts = new[]
                {
                    (frameInfo, new OpenCvSharp.Point(10, 20)),
                    (classInfo, new OpenCvSharp.Point(10, 40)),
                    (cacInfo, new OpenCvSharp.Point(10, 60))
                };

                return MatHelper.AddMultipleText(image, texts);
            }, MatHelper.SafeClone(image), "Add text to image");
        }
    }
}
