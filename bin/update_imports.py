from ceramatch.controller.controller import Controller

from PySide6 import QtCore, QtWidgets
import os
import sys


class CeraMatchImportScanner(QtCore.QObject):
    def __init__(self):
        super().__init__()

        self.app = QtWidgets.QApplication(sys.argv)

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.on_timer)

        self.controller = Controller()
        self.timer.start(5000)

    def run(self):
        self.app.exec()

    @QtCore.Slot()
    def on_timer(self):
        QtWidgets.QApplication.closeAllWindows()
        self.app.quit()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        raise Exception("Invalid number of arguments.\nCorrect use: update_imports.py <installer dir>")

    installer_dir, = args

    gui = CeraMatchImportScanner()

    def process_remaining():
        results = [module.__name__ for module in sys.modules.values() if module]
        results = sorted(results)
        text = ",\n".join([f"'{name}'" for name in results])
        text = f"imports=[\n{text},\n]"
        output_path = os.path.join(installer_dir, "hiddenimports.py")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(text)

    QtCore.QTimer.singleShot(0, process_remaining)
    gui.run()
