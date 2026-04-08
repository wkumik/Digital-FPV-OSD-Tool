@echo off
:: Generates VueOSD.lnk in the project folder + a copy on the Desktop,
:: both pointing at VueOSD.bat with the app icon.
setlocal
cd /d "%~dp0"

set "TARGET=%~dp0VueOSD.bat"
set "ICON=%~dp0assets\icon.ico"
set "WORKDIR=%~dp0"
set "LNK_LOCAL=%~dp0VueOSD.lnk"
set "LNK_DESK=%USERPROFILE%\Desktop\VueOSD.lnk"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%LNK_LOCAL%');" ^
  "$s.TargetPath = '%TARGET%';" ^
  "$s.WorkingDirectory = '%WORKDIR%';" ^
  "$s.IconLocation = '%ICON%,0';" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'VueOSD - Digital FPV OSD Tool';" ^
  "$s.Save();" ^
  "$d = (New-Object -ComObject WScript.Shell).CreateShortcut('%LNK_DESK%');" ^
  "$d.TargetPath = '%TARGET%';" ^
  "$d.WorkingDirectory = '%WORKDIR%';" ^
  "$d.IconLocation = '%ICON%,0';" ^
  "$d.WindowStyle = 7;" ^
  "$d.Description = 'VueOSD - Digital FPV OSD Tool';" ^
  "$d.Save();"

if exist "%LNK_LOCAL%" (
    echo Created: %LNK_LOCAL%
    echo Created: %LNK_DESK%
    echo.
    echo Done. Double-click VueOSD.lnk in this folder, or use the
    echo new shortcut on your Desktop.
) else (
    echo Failed to create shortcut. Check that PowerShell is allowed to run.
)
pause
endlocal
