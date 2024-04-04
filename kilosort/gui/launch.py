import sys
from urllib.error import HTTPError

import pyqtgraph as pg
from kilosort.gui import DarkPalette, KiloSortGUI
from qtpy import QtWidgets, QtGui, QtCore
from kilosort.utils import DOWNLOADS_DIR, download_url_to_file

# TODO: figure out how to fix margin/padding around tooltip text.
#       property-margin, margin-right not working as expected.
_QSS = """
    QToolTip { 
        color: #aeadac;
        background-color: #35322f;
        border: 1px solid #aeadac;
    }
"""


def launcher(filename=None):
    kilosort_application = QtWidgets.QApplication(sys.argv)
    kilosort_application.setStyle("Fusion")
    kilosort_application.setPalette(DarkPalette())
    kilosort_application.setStyleSheet(_QSS)
    
    # get icon
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    icon_path = DOWNLOADS_DIR / "logo.png"
    if not icon_path.is_file():
        print("downloading logo...")
        try:
            download_url_to_file(
                "https://www.kilosort.org/static/downloads/kilosort_logo_small.png",
                icon_path, progress=True
                )
        except HTTPError as e:
            print('Unable to download logo')
            print(e)

    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    
    kilosort_application.setWindowIcon(app_icon)

    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")
    pg.setConfigOption("useOpenGL", True)

    kilosort_gui = KiloSortGUI(kilosort_application, filename=filename)
    # kilosort_gui.showMaximized()
    kilosort_gui.show()

    sys.exit(kilosort_application.exec_())

