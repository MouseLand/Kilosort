import sys

import pyqtgraph as pg
from kilosort.gui import DarkPalette, KiloSortGUI
from PyQt5 import QtWidgets


def launcher():
    kilosort_application = QtWidgets.QApplication(sys.argv)
    kilosort_application.setStyle("Fusion")
    kilosort_application.setPalette(DarkPalette())
    kilosort_application.setStyleSheet(
        "QToolTip { color: #aeadac;"
        "background-color: #35322f;"
        "border: 1px solid #aeadac; }"
    )

    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")
    pg.setConfigOption("useOpenGL", True)

    kilosort_gui = KiloSortGUI(kilosort_application)
    kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())


if __name__ == "__main__":
    launcher()
