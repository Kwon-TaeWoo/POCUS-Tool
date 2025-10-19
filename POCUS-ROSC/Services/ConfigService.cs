using System;
using System.IO;
using System.Configuration;
using POCUS.ROSC.Models;

namespace POCUS.ROSC.Services
{
    /// <summary>
    /// 설정 파일 관리 서비스
    /// Python의 config.ini 읽기/쓰기 기능을 구현
    /// </summary>
    public class ConfigService
    {
        private readonly string _configFilePath;
        private ConfigSettings _settings;

        public ConfigService(string configFilePath = "config.ini")
        {
            _configFilePath = configFilePath;
            _settings = new ConfigSettings();
        }

        /// <summary>
        /// 설정 로드
        /// </summary>
        public ConfigSettings LoadConfig()
        {
            try
            {
                if (File.Exists(_configFilePath))
                {
                    var configMap = new ExeConfigurationFileMap
                    {
                        ExeConfigFilename = _configFilePath
                    };
                    var config = ConfigurationManager.OpenMappedExeConfiguration(configMap, ConfigurationUserLevel.None);

                    // System 섹션
                    if (config.AppSettings.Settings["ProgramName"] != null)
                        _settings.ProgramName = config.AppSettings.Settings["ProgramName"].Value;
                    if (config.AppSettings.Settings["Version"] != null)
                        _settings.Version = config.AppSettings.Settings["Version"].Value;
                    if (config.AppSettings.Settings["LargeModelName"] != null)
                        _settings.LargeModelName = config.AppSettings.Settings["LargeModelName"].Value;
                    if (config.AppSettings.Settings["SmallModelName"] != null)
                        _settings.SmallModelName = config.AppSettings.Settings["SmallModelName"].Value;
                    if (config.AppSettings.Settings["OnnxModelName"] != null)
                        _settings.OnnxModelName = config.AppSettings.Settings["OnnxModelName"].Value;
                    if (config.AppSettings.Settings["SaveFolder"] != null)
                        _settings.SaveFolder = config.AppSettings.Settings["SaveFolder"].Value;

                    // Environment 섹션
                    if (config.AppSettings.Settings["AutoCalculate"] != null)
                        _settings.AutoCalculate = bool.Parse(config.AppSettings.Settings["AutoCalculate"].Value);
                    if (config.AppSettings.Settings["AutoROI"] != null)
                        _settings.AutoROI = bool.Parse(config.AppSettings.Settings["AutoROI"].Value);
                    if (config.AppSettings.Settings["AutoSave"] != null)
                        _settings.AutoSave = bool.Parse(config.AppSettings.Settings["AutoSave"].Value);
                    if (config.AppSettings.Settings["AutoFolder"] != null)
                        _settings.AutoFolder = bool.Parse(config.AppSettings.Settings["AutoFolder"].Value);
                    if (config.AppSettings.Settings["SmallModel"] != null)
                        _settings.SmallModel = bool.Parse(config.AppSettings.Settings["SmallModel"].Value);
                    if (config.AppSettings.Settings["DrawEllipse"] != null)
                        _settings.DrawEllipse = bool.Parse(config.AppSettings.Settings["DrawEllipse"].Value);
                    if (config.AppSettings.Settings["UsePythonInference"] != null)
                        _settings.UsePythonInference = bool.Parse(config.AppSettings.Settings["UsePythonInference"].Value);

                    // Parameter 섹션
                    if (config.AppSettings.Settings["ROI"] != null)
                    {
                        _settings.ROI = config.AppSettings.Settings["ROI"].Value;
                        _settings.ParseROI();
                    }
                    if (config.AppSettings.Settings["DeviceID"] != null)
                        _settings.DeviceID = int.Parse(config.AppSettings.Settings["DeviceID"].Value);

                    // 버전 체크
                    if (_settings.Version != "1.00")
                    {
                        _settings = InitConfig();
                    }
                }
                else
                {
                    _settings = InitConfig();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Config load error: {ex.Message}");
                _settings = InitConfig();
            }

            return _settings;
        }

        /// <summary>
        /// 설정 저장
        /// </summary>
        public void SaveConfig(ConfigSettings settings)
        {
            try
            {
                _settings = settings;
                _settings.UpdateROIString();

                var configMap = new ExeConfigurationFileMap
                {
                    ExeConfigFilename = _configFilePath
                };
                var config = ConfigurationManager.OpenMappedExeConfiguration(configMap, ConfigurationUserLevel.None);

                // System 섹션
                config.AppSettings.Settings["ProgramName"].Value = _settings.ProgramName;
                config.AppSettings.Settings["Version"].Value = _settings.Version;
                config.AppSettings.Settings["LargeModelName"].Value = _settings.LargeModelName;
                config.AppSettings.Settings["SmallModelName"].Value = _settings.SmallModelName;
                config.AppSettings.Settings["OnnxModelName"].Value = _settings.OnnxModelName;
                config.AppSettings.Settings["SaveFolder"].Value = _settings.SaveFolder;

                // Environment 섹션
                config.AppSettings.Settings["AutoCalculate"].Value = _settings.AutoCalculate.ToString();
                config.AppSettings.Settings["AutoROI"].Value = _settings.AutoROI.ToString();
                config.AppSettings.Settings["AutoSave"].Value = _settings.AutoSave.ToString();
                config.AppSettings.Settings["AutoFolder"].Value = _settings.AutoFolder.ToString();
                config.AppSettings.Settings["SmallModel"].Value = _settings.SmallModel.ToString();
                config.AppSettings.Settings["DrawEllipse"].Value = _settings.DrawEllipse.ToString();
                config.AppSettings.Settings["UsePythonInference"].Value = _settings.UsePythonInference.ToString();

                // Parameter 섹션
                config.AppSettings.Settings["ROI"].Value = _settings.ROI;
                config.AppSettings.Settings["DeviceID"].Value = _settings.DeviceID.ToString();

                config.Save(ConfigurationSaveMode.Modified);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Config save error: {ex.Message}");
            }
        }

        /// <summary>
        /// 설정 값 업데이트
        /// </summary>
        public void UpdateSetting(string category, string key, string value)
        {
            try
            {
                var configMap = new ExeConfigurationFileMap
                {
                    ExeConfigFilename = _configFilePath
                };
                var config = ConfigurationManager.OpenMappedExeConfiguration(configMap, ConfigurationUserLevel.None);

                if (config.AppSettings.Settings[key] != null)
                {
                    config.AppSettings.Settings[key].Value = value;
                }
                else
                {
                    config.AppSettings.Settings.Add(key, value);
                }

                config.Save(ConfigurationSaveMode.Modified);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Config update error: {ex.Message}");
            }
        }

        /// <summary>
        /// 초기 설정 생성
        /// </summary>
        private ConfigSettings InitConfig()
        {
            var defaultSettings = new ConfigSettings();
            
            try
            {
                var configMap = new ExeConfigurationFileMap
                {
                    ExeConfigFilename = _configFilePath
                };
                var config = ConfigurationManager.OpenMappedExeConfiguration(configMap, ConfigurationUserLevel.None);

                // System 섹션
                config.AppSettings.Settings.Add("ProgramName", defaultSettings.ProgramName);
                config.AppSettings.Settings.Add("Version", defaultSettings.Version);
                config.AppSettings.Settings.Add("LargeModelName", defaultSettings.LargeModelName);
                config.AppSettings.Settings.Add("SmallModelName", defaultSettings.SmallModelName);
                config.AppSettings.Settings.Add("OnnxModelName", defaultSettings.OnnxModelName);
                config.AppSettings.Settings.Add("SaveFolder", defaultSettings.SaveFolder);

                // Environment 섹션
                config.AppSettings.Settings.Add("AutoCalculate", defaultSettings.AutoCalculate.ToString());
                config.AppSettings.Settings.Add("AutoROI", defaultSettings.AutoROI.ToString());
                config.AppSettings.Settings.Add("AutoSave", defaultSettings.AutoSave.ToString());
                config.AppSettings.Settings.Add("AutoFolder", defaultSettings.AutoFolder.ToString());
                config.AppSettings.Settings.Add("SmallModel", defaultSettings.SmallModel.ToString());
                config.AppSettings.Settings.Add("DrawEllipse", defaultSettings.DrawEllipse.ToString());
                config.AppSettings.Settings.Add("UsePythonInference", defaultSettings.UsePythonInference.ToString());

                // Parameter 섹션
                config.AppSettings.Settings.Add("ROI", defaultSettings.ROI);
                config.AppSettings.Settings.Add("DeviceID", defaultSettings.DeviceID.ToString());

                config.Save(ConfigurationSaveMode.Modified);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Config init error: {ex.Message}");
            }

            return defaultSettings;
        }

        /// <summary>
        /// 현재 설정 반환
        /// </summary>
        public ConfigSettings GetCurrentSettings()
        {
            return _settings;
        }
    }
}