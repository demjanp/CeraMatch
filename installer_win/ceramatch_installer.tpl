!define VERSION "%(display_version)s"
!define APP_NAME "CeraMatch"
!define ROOT_PATH "%(root_path)s"
!define INSTALLER_PATH "%(installer_path)s"

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

RequestExecutionLevel admin

OutFile "${INSTALLER_PATH}\%(filename)s"
InstallDir "$PROGRAMFILES\${APP_NAME}"
Icon "${ROOT_PATH}\cm_icon.ico"
UninstallIcon "${ROOT_PATH}\cm_icon.ico"
VIProductVersion "%(product_version)s"
Caption "${APP_NAME} Setup"

!define MUI_ABORTWARNING
!define MUI_ICON "${ROOT_PATH}\cm_icon.ico"
!define MUI_UNICON "${ROOT_PATH}\cm_icon.ico"
!define MUI_WELCOMEPAGE_TITLE "Welcome to the ${APP_NAME} Setup Wizard"
!define MUI_WELCOMEPAGE_TEXT "This wizard will guide you through the installation of ${APP_NAME}. Click 'Next' to continue."
!define MUI_LICENSEPAGE_TEXT_TOP "Please review the license terms before installing ${APP_NAME}."
!define MUI_DIRECTORYPAGE_TEXT_TOP "Choose the folder in which to install ${APP_NAME}."
!define MUI_DIRECTORYPAGE_TEXT_DESTINATION "Setup will install ${APP_NAME} in the following folder. To install in a different folder, click Browse and select another folder. Click Install to start the installation."
!define MUI_INSTFILESPAGE_TEXT "Please wait while ${APP_NAME} is being installed."
!define MUI_FINISHPAGE_TITLE "Installation Complete"
!define MUI_FINISHPAGE_TEXT "${APP_NAME} has been installed on your computer. Click 'Finish' to exit the Setup Wizard."
!define MUI_UNCONFIRMPAGE_TEXT "Are you sure you want to completely remove ${APP_NAME} and all of its components?"
!define MUI_UNINSTFILESPAGE_TEXT "Please wait while ${APP_NAME} is being uninstalled."

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${ROOT_PATH}\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section -PreInstall
  RMDir /r "$INSTDIR"
SectionEnd

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "${ROOT_PATH}\dist\ceramatch\*.*"

  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\ceramatch.exe" "" "$INSTDIR\cm_icon.ico"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\ceramatch.exe" "" "$INSTDIR\cm_icon.ico"

  IfFileExists "$INSTDIR\deposit_gui\deposit_gui.exe" 0 +3
  CreateShortCut "$DESKTOP\Deposit.lnk" "$INSTDIR\deposit_gui\deposit_gui.exe"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Deposit.lnk" "$INSTDIR\deposit_gui\deposit_gui.exe"

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "QuietUninstallString" "$INSTDIR\uninstall.exe /S"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayIcon" "$INSTDIR\cm_icon.ico"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "Peter Demjan"
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%%08X" $0
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "EstimatedSize" "$0"

  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$DESKTOP\Deposit.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Deposit.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd
