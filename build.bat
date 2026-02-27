@echo off
setlocal

echo [1/2] Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)

echo [2/2] Building wikicom.exe...
pyinstaller --onefile --name wikicom --distpath dist --workpath build\pyinstaller --specpath build\pyinstaller wikicom.py
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

echo.
echo Build complete: %~dp0dist\wikicom.exe
echo.

:: --- 선택적 설치 ---
set INSTALL_DIR=%USERPROFILE%\tools
set /p INSTALL="Install to %INSTALL_DIR% and add to PATH? [y/N] "
if /i not "%INSTALL%"=="y" goto :done

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
copy /y "%~dp0dist\wikicom.exe" "%INSTALL_DIR%\wikicom.exe" >nul

:: User PATH에 이미 있는지 확인 후 추가
echo %PATH% | find /i "%INSTALL_DIR%" >nul
if errorlevel 1 (
    setx PATH "%USERPROFILE%\tools;%PATH%"
    echo Added %INSTALL_DIR% to user PATH.
    echo Restart your terminal to apply.
) else (
    echo %INSTALL_DIR% is already in PATH.
)

:done
endlocal
