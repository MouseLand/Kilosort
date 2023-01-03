import sys, argparse

import pyqtgraph as pg
from kilosort.gui import DarkPalette, KiloSortGUI
from PyQt5 import QtWidgets


def launcher(filename=None):
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

    kilosort_gui = KiloSortGUI(kilosort_application, filename=filename)
    # kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())

