rmdir dist\cm /S /Q

pyinstaller cm.spec

robocopy /e deposit\res dist\cm\deposit\res
robocopy /e data dist\cm\data
copy deposit\deposit_icon.ico dist\cm

