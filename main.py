import sys
from PyQt5.QtWidgets import QApplication
from view.gui_view import OlhoDeThothGUI
from controller.app_controller import AppController

def main():
    app = QApplication(sys.argv)
    view = OlhoDeThothGUI()
    controller = AppController(view)
    view.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()