using System.Windows;
using System.Globalization;
using System.Threading;

namespace POCUS.ROSC
{
    /// <summary>
    /// App.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            // 문화권 설정을 한국어로 설정
            CultureInfo.DefaultThreadCurrentCulture = new CultureInfo("ko-KR");
            CultureInfo.DefaultThreadCurrentUICulture = new CultureInfo("ko-KR");
            Thread.CurrentThread.CurrentCulture = new CultureInfo("ko-KR");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("ko-KR");
            
            // MainWindow를 생성하되 초기화 과정을 거치도록 함
            var mainWindow = new MainWindow();
            mainWindow.StartInitialization(); // 초기화 시작
            
            base.OnStartup(e);
        }
    }
}

