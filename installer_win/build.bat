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
if exist build\ceramatch rmdir build\ceramatch /S /Q
if exist dist\ceramatch rmdir dist\ceramatch /S /Q

call .venv\Scripts\activate.bat
python -m pip install -e . --no-deps

python bin\update_imports.py installer_win
if errorlevel 1 (
    echo Warning: bin\update_imports.py failed; using existing installer_win\hiddenimports.py
)

call pyinstaller installer_win\ceramatch.spec ""%deposit_dir%""
if errorlevel 1 goto error_build

python installer_win\make_nsi.py %cd%
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

:error_build
popd
exit /b 1

:end
popd
