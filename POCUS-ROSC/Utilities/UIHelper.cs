using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using POCUS.ROSC.Models;

namespace POCUS.ROSC.Utilities
{
    /// <summary>
    /// UI 관련 유틸리티 함수들
    /// </summary>
    public static class UIHelper
    {
        /// <summary>
        /// UI 스레드에서 안전하게 실행
        /// </summary>
        public static void SafeInvoke(Action action, Dispatcher dispatcher = null)
        {
            try
            {
                dispatcher = dispatcher ?? Application.Current?.Dispatcher;
                if (dispatcher == null)
                {
                    action?.Invoke();
                    return;
                }

                if (dispatcher.CheckAccess())
                {
                    action?.Invoke();
                }
                else
                {
                    dispatcher.Invoke(action);
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "UI safe invoke");
            }
        }

        /// <summary>
        /// UI 스레드에서 안전하게 실행 (비동기)
        /// </summary>
        public static void SafeInvokeAsync(Action action, Dispatcher dispatcher = null)
        {
            try
            {
                dispatcher = dispatcher ?? Application.Current?.Dispatcher;
                if (dispatcher == null)
                {
                    action?.Invoke();
                    return;
                }

                if (dispatcher.CheckAccess())
                {
                    action?.Invoke();
                }
                else
                {
                    dispatcher.BeginInvoke(action);
                }
            }
            catch (Exception ex)
            {
                ExceptionHelper.LogError(ex, "UI safe invoke async");
            }
        }

        /// <summary>
        /// 상태에 따른 색상 반환
        /// </summary>
        public static Brush GetStateColor(string state)
        {
            return state switch
            {
                "ROSC" => Brushes.Blue,
                "Arrest" => Brushes.Red,
                "Not compressed" => Brushes.White,
                "Invalid CAC" => Brushes.Gray,
                _ => Brushes.Yellow
            };
        }

        /// <summary>
        /// 상태에 따른 텍스트 색상 반환
        /// </summary>
        public static string GetStateTextColor(string state)
        {
            return state switch
            {
                "ROSC" => "rgb(0, 0, 255)",
                "Arrest" => "rgb(255, 0, 0)",
                "Not compressed" => "rgb(255, 255, 255)",
                "Invalid CAC" => "rgb(128, 128, 128)",
                _ => "rgb(255, 255, 0)"
            };
        }

        /// <summary>
        /// 버튼 상태 업데이트
        /// </summary>
        public static void UpdateButtonState(System.Windows.Controls.Button button, bool isEnabled, string content = null)
        {
            SafeInvoke(() =>
            {
                if (button != null)
                {
                    button.IsEnabled = isEnabled;
                    if (!string.IsNullOrEmpty(content))
                    {
                        button.Content = content;
                    }
                }
            });
        }

        /// <summary>
        /// 체크박스 상태 업데이트
        /// </summary>
        public static void UpdateCheckBoxState(System.Windows.Controls.CheckBox checkBox, bool isChecked)
        {
            SafeInvoke(() =>
            {
                if (checkBox != null)
                {
                    checkBox.IsChecked = isChecked;
                }
            });
        }

        /// <summary>
        /// 텍스트 블록 업데이트
        /// </summary>
        public static void UpdateTextBlock(System.Windows.Controls.TextBlock textBlock, string text, Brush foreground = null)
        {
            SafeInvoke(() =>
            {
                if (textBlock != null)
                {
                    textBlock.Text = text;
                    if (foreground != null)
                    {
                        textBlock.Foreground = foreground;
                    }
                }
            });
        }

        /// <summary>
        /// 이미지 소스 업데이트
        /// </summary>
        public static void UpdateImageSource(System.Windows.Controls.Image image, System.Windows.Media.ImageSource source)
        {
            SafeInvoke(() =>
            {
                if (image != null && source != null)
                {
                    image.Source = source;
                }
            });
        }

        /// <summary>
        /// 진행률 표시
        /// </summary>
        public static void ShowProgress(System.Windows.Controls.ProgressBar progressBar, bool show = true)
        {
            SafeInvoke(() =>
            {
                if (progressBar != null)
                {
                    progressBar.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
                    if (show)
                    {
                        progressBar.IsIndeterminate = true;
                    }
                }
            });
        }

        /// <summary>
        /// 상태 메시지 표시
        /// </summary>
        public static void ShowStatusMessage(string message, System.Windows.Controls.TextBlock statusLabel = null)
        {
            SafeInvoke(() =>
            {
                if (statusLabel != null)
                {
                    statusLabel.Text = message;
                }
                else
                {
                    Console.WriteLine($"Status: {message}");
                }
            });
        }

        /// <summary>
        /// 에러 메시지 표시
        /// </summary>
        public static void ShowErrorMessage(string message, string title = "Error")
        {
            SafeInvoke(() =>
            {
                MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Error);
            });
        }

        /// <summary>
        /// 확인 메시지 표시
        /// </summary>
        public static bool ShowConfirmMessage(string message, string title = "Confirm")
        {
            bool result = false;
            SafeInvoke(() =>
            {
                result = MessageBox.Show(message, title, MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes;
            });
            return result;
        }

        /// <summary>
        /// 경고 메시지 표시
        /// </summary>
        public static void ShowWarningMessage(string message, string title = "Warning")
        {
            SafeInvoke(() =>
            {
                MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Warning);
            });
        }

        /// <summary>
        /// 정보 메시지 표시
        /// </summary>
        public static void ShowInfoMessage(string message, string title = "Information")
        {
            SafeInvoke(() =>
            {
                MessageBox.Show(message, title, MessageBoxButton.OK, MessageBoxImage.Information);
            });
        }
    }
}
