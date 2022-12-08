from pathlib import Path

from kilosort.gui.logger import XStream
from PyQt5 import QtCore, QtGui, QtWidgets


class MessageLogBox(QtWidgets.QGroupBox):
    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Message Log Box")
        self.gui = parent
        self.layout = QtWidgets.QHBoxLayout()
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_box.setFont(QtGui.QFont("Monospace"))
        self.layout.addWidget(self.log_box)

        log_box_document = self.log_box.document()
        default_font = log_box_document.defaultFont()
        default_font.setPointSize(8)
        log_box_document.setDefaultFont(default_font)

        XStream.stdout().messageWritten.connect(self.update_text)
        XStream.stderr().messageWritten.connect(self.update_text)

        self.setLayout(self.layout)

    @QtCore.pyqtSlot(str)
    def update_text(self, text):
        self.log_box.moveCursor(QtGui.QTextCursor.End)
        self.log_box.appendPlainText(text)
        self.log_box.ensureCursorVisible()

    def prepare_for_new_context(self):
        self.save_log_file()
        self.log_box.clear()

    def save_log_file(self):
        context = self.get_context()
        if context is not None:
            context_path = context.context_path
            log_file_name = context.raw_data.name + ".log"

            log_file_path = Path(context_path) / log_file_name

            with open(log_file_path, "w") as log_file:
                log_file.write(self.log_box.toPlainText())

    def get_context(self):
        return self.gui.context

    def reset(self):
        self.log_box.clear()
