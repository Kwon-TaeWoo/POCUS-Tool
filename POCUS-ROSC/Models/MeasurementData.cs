using System;
using System.Collections.Generic;

namespace POCUS.ROSC.Models
{
    /// <summary>
    /// 측정 데이터를 저장하는 클래스
    /// Python의 measurement 리스트와 동일한 구조
    /// </summary>
    public class MeasurementData
    {
        public string FileName { get; set; } = "";
        public double CACValue { get; set; } = 0.0;
        public double IJVValue { get; set; } = 0.0;
        public double MinMaxRatio { get; set; } = 0.0;
        public string Class { get; set; } = "";
        public string State { get; set; } = "Measuring...";

        /// <summary>
        /// Python의 measurement 리스트 형태로 변환
        /// [file_name, CAC_value, IJV_value, min/max, class, state]
        /// </summary>
        public List<object> ToList()
        {
            return new List<object>
            {
                FileName,
                CACValue,
                IJVValue,
                MinMaxRatio,
                Class,
                State
            };
        }

        /// <summary>
        /// 리스트에서 MeasurementData 객체 생성
        /// </summary>
        public static MeasurementData FromList(List<object> data)
        {
            if (data == null || data.Count < 6)
                return new MeasurementData();

            return new MeasurementData
            {
                FileName = data[0]?.ToString() ?? "",
                CACValue = Convert.ToDouble(data[1]),
                IJVValue = Convert.ToDouble(data[2]),
                MinMaxRatio = Convert.ToDouble(data[3]),
                Class = data[4]?.ToString() ?? "",
                State = data[5]?.ToString() ?? "Measuring..."
            };
        }
    }

    /// <summary>
    /// ABP 예측 데이터
    /// </summary>
    public class ABPData
    {
        public double CAC { get; set; }
        public double SBP { get; set; }  // 수축기 혈압
        public double MBP { get; set; }  // 평균 혈압
        public double DBP { get; set; }  // 이완기 혈압

        /// <summary>
        /// CAC 값으로부터 ABP 계산
        /// Python의 ABP 계산 공식과 동일
        /// </summary>
        public static ABPData CalculateFromCAC(double cac)
        {
            return new ABPData
            {
                CAC = cac,
                SBP = -215.2 * cac + 251.54,
                MBP = -135.6 * cac + 165.12,
                DBP = -105.3 * cac + 128.11
            };
        }
    }

    /// <summary>
    /// 그래프 데이터 포인트
    /// </summary>
    public class GraphDataPoint
    {
        public int FrameNumber { get; set; }
        public double Value { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.Now;
    }
}
