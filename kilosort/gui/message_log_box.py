from pathlib import Path
import pprint

import numpy as np
from kilosort.gui.logger import XStream
from kilosort.utils import probe_as_string
from qtpy import QtCore, QtGui, QtWidgets


class MessageLogBox(QtWidgets.QGroupBox):
    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log Box")
        self.gui = parent
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth) 
        self.log_box.setFont(QtGui.QFont("Monospace"))
        self.layout.addWidget(self.log_box, 0, 0, 1, 3)

        self.popout_button = QtWidgets.QPushButton('Show More')
        self.popout_button.clicked.connect(self.show_log_popout)
        self.layout.addWidget(self.popout_button, 1, 0, 1, 1)
        self.popout_window = ExpandedLog()

        self.print_settings_button = QtWidgets.QPushButton('Print Settings')
        self.print_settings_button.clicked.connect(self.print_settings)
        self.layout.addWidget(self.print_settings_button, 1, 1, 1, 1)

        self.print_probe_button = QtWidgets.QPushButton('Print Probe')
        self.print_probe_button.clicked.connect(self.print_probe)
        self.layout.addWidget(self.print_probe_button, 1, 2, 1, 1)

        log_box_document = self.log_box.document()
        default_font = log_box_document.defaultFont()
        default_font.setPointSize(8)
        log_box_document.setDefaultFont(default_font)

        XStream.stdout().messageWritten.connect(self.update_text)
        XStream.stderr().messageWritten.connect(self.update_text)


    @QtCore.Slot(str)
    def update_text(self, text):
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.appendPlainText(text)
        self.log_box.ensureCursorVisible()
        self.popout_window.update_text(text)

    @QtCore.Slot()
    def show_log_popout(self):
        self.popout_window.show()

    @QtCore.Slot()
    def print_settings(self):
        # For debugging purposes, check for mismatch between displayed parameter
        # values and the values that are actually being used.
        settings_text = "settings = "
        settings = self.gui.settings_box.settings.copy()
        settings['probe'] = '... (use print probe)'
        s = pprint.pformat(settings, indent=4, sort_dicts=False)
        settings_text += s[0] + '\n ' + s[1:-1] + '\n' + s[-1]

        self.update_text(settings_text)

    @QtCore.Slot()
    def print_probe(self):
        probe = self.gui.settings_box.settings['probe']
        probe_text = probe_as_string(probe)
        self.update_text(probe_text)

    def prepare_for_new_context(self):
        self.log_box.clear()

    def get_context(self):
        return self.gui.context

    def reset(self):
        self.log_box.clear()


class ExpandedLog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth) 
        self.log_box.setFont(QtGui.QFont("Monospace"))
        self.layout.addWidget(self.log_box, 0, 0, 1, 1)
        
        center = QtWidgets.QApplication.screens()[0].availableGeometry().center()
        self.setGeometry(0, 0, 800, 500)
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

    def update_text(self, text):
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.appendPlainText(text)
        self.log_box.ensureCursorVisible()
