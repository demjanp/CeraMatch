rmdir dist\cm /S /Q

pyinstaller cm.spec

robocopy /e deposit dist\cm\deposit
robocopy /e data dist\cm\data
robocopy /e res dist\cm\res
copy cm_icon.ico dist\cm

