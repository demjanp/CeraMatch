rmdir dist\cm /S /Q

pyinstaller cm.spec

robocopy /e res dist\cm\res
robocopy /e deposit dist\cm\deposit
robocopy /e graphviz dist\dc\graphviz
robocopy /e pygraphviz dist\dc\pygraphviz
copy cm_icon.ico dist\cm
copy LICENSE dist\cm
copy THIRDPARTY.TXT dist\cm
