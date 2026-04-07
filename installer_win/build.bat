@echo off
pushd "%~dp0"

if "%~2" neq "" goto error_arg

set "deposit_dir=%1"
if "%deposit_dir%"=="" set "deposit_dir=..\..\..\LAP\deposit_gui"

if not exist "%deposit_dir%" goto error_deposit
if not exist "%deposit_dir%\dist\deposit_gui" goto error_deposit_dist

pushd "%deposit_dir%"
set "deposit_dir=%CD%"
popd

cd ..
set "venv_python=%CD%\.venv\Scripts\python.exe"
if not exist "%venv_python%" goto error_venv

git config --global --get-all safe.directory | findstr /I /X /C:"%CD%" >nul
if errorlevel 1 git config --global --add safe.directory "%CD%"

if exist build\ceramatch rmdir build\ceramatch /S /Q
if exist dist\ceramatch rmdir dist\ceramatch /S /Q

"%venv_python%" -m pip show PyInstaller >nul 2>&1
if errorlevel 1 "%venv_python%" -m pip install "PyInstaller==5.6.2"

"%venv_python%" -m pip install -e . --no-deps
if errorlevel 1 goto error_build

"%venv_python%" bin\update_imports.py installer_win
if errorlevel 1 (
    echo Warning: bin\update_imports.py failed; using existing installer_win\hiddenimports.py
)

"%venv_python%" -m PyInstaller installer_win\ceramatch.spec ""%deposit_dir%""
if errorlevel 1 goto error_build

"%venv_python%" installer_win\make_nsi.py %cd%
if errorlevel 1 goto error_build

goto end

:error_arg
echo Error: Only zero or one argument is allowed.
echo Usage: build.bat ^<deposit dir^>
popd
exit /b 1

:error_deposit
echo Error: The directory %deposit_dir% does not exist.
popd
exit /b 1

:error_deposit_dist
echo Error: The directory %deposit_dir%\dist\deposit_gui does not exist.
echo Build Deposit GUI first, then rerun this script.
popd
exit /b 1

:error_venv
echo Error: The virtual environment Python was not found at %venv_python%.
popd
exit /b 1

:error_build
popd
exit /b 1

:end
popd
