using System;

namespace ROSC.WPF.Models
{
    /// <summary>
    /// 애플리케이션 설정을 관리하는 클래스
    /// Python의 config.ini 설정과 동일한 구조
    /// </summary>
    public class ConfigSettings
    {
        // System 설정
        public string ProgramName { get; set; } = "POCUS-CAC GUI";
        public string Version { get; set; } = "1.00";
        public string LargeModelName { get; set; } = "./checkpoint/checkpoint230605__output_noising_CLAHE_transunet_withaug_various_iter1.pth";
        public string SmallModelName { get; set; } = "./checkpoint/checkpoint240305__output_noising_CLAHE_transunet_256_cutout_iter1.pth";
        public string OnnxModelName { get; set; } = "./checkpoint/model.onnx"; // ONNX 모델 경로
        public string SaveFolder { get; set; } = "./model_output";

        // Environment 설정
        public bool AutoCalculate { get; set; } = true;
        public bool AutoROI { get; set; } = true;
        public bool AutoSave { get; set; } = true;
        public bool AutoFolder { get; set; } = true;
        public bool SmallModel { get; set; } = true;
        public bool DrawEllipse { get; set; } = true;
        public bool UsePythonInference { get; set; } = false; // Python 추론 사용 여부

        // Parameter 설정
        public string ROI { get; set; } = "450|40|1500|950";
        public int DeviceID { get; set; } = 0;

        // ROI 좌표 (파싱된 값)
        public int ROI_X1 { get; set; } = 450;
        public int ROI_Y1 { get; set; } = 40;
        public int ROI_X2 { get; set; } = 1500;
        public int ROI_Y2 { get; set; } = 950;

        // 비디오용 ROI 좌표
        public int ROI_X1_Video { get; set; } = 70;
        public int ROI_Y1_Video { get; set; } = 15;
        public int ROI_X2_Video { get; set; } = 560;
        public int ROI_Y2_Video { get; set; } = 475;

        /// <summary>
        /// ROI 문자열을 파싱하여 좌표값 설정
        /// </summary>
        public void ParseROI()
        {
            var parts = ROI.Split('|');
            if (parts.Length == 4)
            {
                if (int.TryParse(parts[0], out int x1) &&
                    int.TryParse(parts[1], out int y1) &&
                    int.TryParse(parts[2], out int x2) &&
                    int.TryParse(parts[3], out int y2))
                {
                    ROI_X1 = x1;
                    ROI_Y1 = y1;
                    ROI_X2 = x2;
                    ROI_Y2 = y2;
                }
            }
        }

        /// <summary>
        /// 현재 좌표값을 ROI 문자열로 변환
        /// </summary>
        public void UpdateROIString()
        {
            ROI = $"{ROI_X1}|{ROI_Y1}|{ROI_X2}|{ROI_Y2}";
        }
    }
}
