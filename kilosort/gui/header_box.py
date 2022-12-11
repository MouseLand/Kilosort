from kilosort import __version__
from kilosort.gui.minor_gui_elements import help_popup_text, controls_popup_text
from PyQt5 import QtCore, QtGui, QtWidgets


class HeaderBox(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.layout = QtWidgets.QHBoxLayout()

        # self.kilosort_text = QtWidgets.QLabel()
        # self.kilosort_text.setText(f"Kilosort {__version__[:3]}")
        # self.kilosort_text.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))

        self.controls_button = QtWidgets.QPushButton("Controls")
        self.controls_button.clicked.connect(self.show_controls_popup)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_popup)

        self.reset_gui_button = QtWidgets.QPushButton("Reset GUI")

        # self.layout.addWidget(self.kilosort_text)
        self.layout.addStretch(0)
        self.layout.addWidget(self.controls_button)
        self.layout.addWidget(self.help_button)
        self.layout.addWidget(self.reset_gui_button)

        self.setLayout(self.layout)

    @QtCore.pyqtSlot()
    def show_help_popup(self):
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            help_popup_text,
            QtWidgets.QMessageBox.Ok,
            QtWidgets.QMessageBox.Ok,
        )

    @QtCore.pyqtSlot()
    def show_controls_popup(self):
        QtWidgets.QMessageBox.information(
            self,
            "Controls",
            controls_popup_text,
            QtWidgets.QMessageBox.Ok,
            QtWidgets.QMessageBox.Ok,
        )
