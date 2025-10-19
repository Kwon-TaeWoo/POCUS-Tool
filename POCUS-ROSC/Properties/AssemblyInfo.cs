using System.Reflection;
using System.Resources;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows;

// 어셈블리에 대한 일반 정보는 다음 특성 집합을 통해 
// 제어됩니다. 어셈블리와 관련된 정보를 수정하려면
// 이러한 특성 값을 변경하세요.
[assembly: AssemblyTitle("ROSC.WPF")]
[assembly: AssemblyDescription("POCUS-CAC GUI - Real-time ROSC Monitoring & Visualization")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("")]
[assembly: AssemblyProduct("ROSC.WPF")]
[assembly: AssemblyCopyright("Copyright © 2024")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]

// ComVisible을 false로 설정하면 이 어셈블리의 형식이 COM 구성 요소에 
// 표시되지 않습니다. COM에서 이 어셈블리의 형식에 액세스하려면
// 해당 형식에 대해 ComVisible 특성을 true로 설정하세요.
[assembly: ComVisible(false)]

//지역화 가능한 애플리케이션을 빌드하려면 다음을 설정하세요.
//<UICulture>CultureYouAreCodingWith</UICulture>
//App.xaml 파일 내에서 <Application.Resources>
//  <ResourceDictionary Source="Resources\Resources.resx"/>
//</Application.Resources>
//를 추가하세요.
//그러면 해당 파일에서 <TextBlock x:Uid="TextBlock1" Text="Hello World"/>와 같은 태그를 사용할 수 있습니다.
[assembly: NeutralResourcesLanguage("en-US", UltimateResourceFallbackLocation.Satellite)]


[assembly: ThemeInfo(
    ResourceDictionaryLocation.None, //테마별 리소스 사전의 위치
                                     //(페이지 또는 애플리케이션 리소스 사전에 
                                     // 리소스가 없는 경우에 사용됨)
    ResourceDictionaryLocation.SourceAssembly //제네릭 리소스 사전의 위치
                                              //(페이지 또는 애플리케이션 리소스 사전에 
                                              // 리소스가 없는 경우에 사용됨)
)]


// 어셈블리의 버전 정보는 다음 네 개의 값으로 구성됩니다.
//
//      주 버전
//      부 버전 
//      빌드 번호
//      수정 버전
//
// 모든 값을 지정하거나 아래와 같이 '*'를 사용하여 빌드 번호와 수정 번호를 
// 기본값으로 설정할 수 있습니다.
// [assembly: AssemblyVersion("1.0.*")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]
