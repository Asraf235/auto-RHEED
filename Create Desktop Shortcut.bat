@echo off
setlocal
rem Folder this script lives in (the project root), without trailing backslash.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BAT=%ROOT%\Launch Auto RHEED.bat"
set "ICO=%ROOT%\auto_rheed.ico"

if not exist "%BAT%" (
  echo Could not find "Launch Auto RHEED.bat" next to this script.
  pause
  exit /b 1
)

echo Creating an "Auto RHEED" shortcut on your Desktop...
powershell -NoProfile -NonInteractive -Command "$d=[Environment]::GetFolderPath('Desktop'); $lnk=Join-Path $d 'Auto RHEED.lnk'; $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut($lnk); $s.TargetPath='%BAT%'; $s.WorkingDirectory='%ROOT%'; $s.Description='Launch Auto RHEED'; if(Test-Path '%ICO%'){$s.IconLocation='%ICO%,0'}; $s.Save(); Write-Host ('Done: '+$lnk)"

echo.
pause
