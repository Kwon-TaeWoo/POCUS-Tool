using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using OxyPlot;
using OxyPlot.Series;
using ROSC.WPF.Models;

namespace ROSC.WPF.Utilities
{
    /// <summary>
    /// 그래프 데이터 관리 및 업데이트 유틸리티
    /// </summary>
    public static class GraphHelper
    {
        /// <summary>
        /// 슬라이딩 윈도우 버퍼 관리
        /// </summary>
        public class SlidingWindowBuffer<T>
        {
            private readonly ConcurrentQueue<T> _buffer;
            private readonly int _maxSize;

            public SlidingWindowBuffer(int maxSize)
            {
                _maxSize = maxSize;
                _buffer = new ConcurrentQueue<T>();
            }

            public void Add(T item)
            {
                _buffer.Enqueue(item);
                while (_buffer.Count > _maxSize)
                {
                    _buffer.TryDequeue(out _);
                }
            }

            public List<T> GetValues()
            {
                return _buffer.ToList();
            }

            public void Clear()
            {
                while (_buffer.TryDequeue(out _)) { }
            }

            public int Count => _buffer.Count;
            public bool IsFull => _buffer.Count >= _maxSize;
        }

        /// <summary>
        /// CAC 그래프 데이터 업데이트
        /// </summary>
        public static void UpdateCACGraph(LineSeries cacSeries, LineSeries ijvSeries, 
            SlidingWindowBuffer<double> cacBuffer, SlidingWindowBuffer<double> ijvBuffer, 
            int maxPoints = 40)
        {
            if (cacSeries == null || ijvSeries == null || cacBuffer == null || ijvBuffer == null)
                return;

            try
            {
                var cacValues = cacBuffer.GetValues();
                var ijvValues = ijvBuffer.GetValues();

                if (cacValues.Count == 0)
                    return;

                // 최근 N개 포인트만 표시
                var recentCac = cacValues.TakeLast(Math.Min(cacValues.Count, maxPoints)).ToList();
                var recentIjv = ijvValues.TakeLast(Math.Min(ijvValues.Count, maxPoints)).ToList();

                cacSeries.Points.Clear();
                ijvSeries.Points.Clear();

                for (int i = 0; i < recentCac.Count; i++)
                {
                    cacSeries.Points.Add(new DataPoint(i, recentCac[i]));
                }

                for (int i = 0; i < recentIjv.Count; i++)
                {
                    ijvSeries.Points.Add(new DataPoint(i, recentIjv[i]));
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Update CAC graph");
            }
        }

        /// <summary>
        /// ABP 그래프 데이터 업데이트
        /// </summary>
        public static void UpdateABPGraph(ScatterSeries abpSeries, double cacValue, int maxPoints = 10)
        {
            if (abpSeries == null)
                return;

            try
            {
                var abpData = ABPData.CalculateFromCAC(cacValue);
                
                // 기존 포인트가 너무 많으면 제거
                while (abpSeries.Points.Count >= maxPoints)
                {
                    abpSeries.Points.RemoveAt(0);
                }

                // 새로운 ABP 포인트 추가
                abpSeries.Points.Add(new ScatterPoint(cacValue, abpData.SBP));
                abpSeries.Points.Add(new ScatterPoint(cacValue, abpData.MBP));
                abpSeries.Points.Add(new ScatterPoint(cacValue, abpData.DBP));
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Update ABP graph");
            }
        }

        /// <summary>
        /// 그래프 시리즈 초기화
        /// </summary>
        public static void InitializeCACGraph(LineSeries cacSeries, LineSeries ijvSeries, LineSeries thresholdSeries)
        {
            try
            {
                if (cacSeries != null)
                {
                    cacSeries.Title = "CA";
                    cacSeries.Color = OxyColors.Red;
                    cacSeries.StrokeThickness = 2;
                }

                if (ijvSeries != null)
                {
                    ijvSeries.Title = "IJV";
                    ijvSeries.Color = OxyColors.Blue;
                    ijvSeries.StrokeThickness = 2;
                }

                if (thresholdSeries != null)
                {
                    thresholdSeries.Title = "Threshold 0.94";
                    thresholdSeries.Color = OxyColors.Yellow;
                    thresholdSeries.LineStyle = LineStyle.Dash;
                    thresholdSeries.StrokeThickness = 2;
                    
                    // Threshold 선 초기화
                    for (int i = 0; i < 40; i++)
                    {
                        thresholdSeries.Points.Add(new DataPoint(i, 0.94));
                    }
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Initialize CAC graph");
            }
        }

        /// <summary>
        /// ABP 그래프 시리즈 초기화
        /// </summary>
        public static void InitializeABPGraph(LineSeries sbpSeries, LineSeries mbpSeries, LineSeries dbpSeries, ScatterSeries abpSeries)
        {
            try
            {
                if (sbpSeries != null)
                {
                    sbpSeries.Title = "SBP Line";
                    sbpSeries.Color = OxyColors.Red;
                    sbpSeries.StrokeThickness = 1;
                }

                if (mbpSeries != null)
                {
                    mbpSeries.Title = "MBP Line";
                    mbpSeries.Color = OxyColors.Green;
                    mbpSeries.StrokeThickness = 1;
                }

                if (dbpSeries != null)
                {
                    dbpSeries.Title = "DBP Line";
                    dbpSeries.Color = OxyColors.Magenta;
                    dbpSeries.StrokeThickness = 1;
                }

                if (abpSeries != null)
                {
                    abpSeries.Title = "Current ABP";
                    abpSeries.MarkerType = MarkerType.Circle;
                    abpSeries.MarkerSize = 8;
                    abpSeries.MarkerFill = OxyColors.Yellow;
                }

                // ABP 예측 선들 초기화
                if (sbpSeries != null && mbpSeries != null && dbpSeries != null)
                {
                    for (double cac = 0.3; cac <= 1.05; cac += 0.01)
                    {
                        sbpSeries.Points.Add(new DataPoint(cac, -215.2 * cac + 251.54));
                        mbpSeries.Points.Add(new DataPoint(cac, -135.6 * cac + 165.12));
                        dbpSeries.Points.Add(new DataPoint(cac, -105.3 * cac + 128.11));
                    }
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Initialize ABP graph");
            }
        }

        /// <summary>
        /// 그래프 범위 설정
        /// </summary>
        public static void SetGraphRange(PlotModel plotModel, double xMin, double xMax, double yMin, double yMax)
        {
            if (plotModel == null)
                return;

            try
            {
                var xAxis = plotModel.Axes.FirstOrDefault(a => a.Position == AxisPosition.Bottom);
                var yAxis = plotModel.Axes.FirstOrDefault(a => a.Position == AxisPosition.Left);

                if (xAxis != null)
                {
                    xAxis.Minimum = xMin;
                    xAxis.Maximum = xMax;
                }

                if (yAxis != null)
                {
                    yAxis.Minimum = yMin;
                    yAxis.Maximum = yMax;
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Set graph range");
            }
        }

        /// <summary>
        /// 그래프 새로고침
        /// </summary>
        public static void RefreshGraph(PlotModel plotModel)
        {
            if (plotModel == null)
                return;

            try
            {
                plotModel.InvalidatePlot(true);
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Refresh graph");
            }
        }

        /// <summary>
        /// 그래프 데이터 초기화
        /// </summary>
        public static void ClearGraphData(params Series[] series)
        {
            if (series == null)
                return;

            try
            {
                foreach (var s in series)
                {
                    s?.Points.Clear();
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "Clear graph data");
            }
        }
    }
}
