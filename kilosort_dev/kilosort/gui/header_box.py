from kilosort import __version__
from kilosort.gui.minor_gui_elements import help_popup_text, controls_popup_text
from PyQt5 import QtCore, QtGui, QtWidgets


class HeaderBox(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.gui = parent
        self.layout = QtWidgets.QHBoxLayout()

        # self.kilosort_text = QtWidgets.QLabel()
        # self.kilosort_text.setText(f"Kilosort {__version__[:3]}")
        # self.kilosort_text.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))

        self.auto_load_check = QtWidgets.QCheckBox('Auto Load')
        if self.gui.auto_load:
            self.auto_load_check.setCheckState(2)
        else:
            self.auto_load_check.setCheckState(0)
        self.auto_load_check.stateChanged.connect(self.check_auto_load)

        self.show_plots_check = QtWidgets.QCheckBox('Show Plots')
        if self.gui.show_plots:
            self.show_plots_check.setCheckState(2)
        else:
            self.show_plots_check.setCheckState(0)
        self.show_plots_check.clicked.connect(self.check_show_plots)

        # TODO: connect the button clicks to open something in browser instead?
        #       like the paper and/or wiki.
        self.controls_button = QtWidgets.QPushButton("Controls")
        self.controls_button.setToolTip(controls_popup_text)
        #self.controls_button.clicked.connect(self.show_controls_popup)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.setToolTip(help_popup_text)
        #self.help_button.clicked.connect(self.show_help_popup)

        self.reset_gui_button = QtWidgets.QPushButton("Reset GUI")
        
        self.clear_cache_button = QtWidgets.QPushButton("Clear Cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)

        # self.layout.addWidget(self.kilosort_text)
        self.layout.addStretch(0)
        self.layout.addWidget(self.auto_load_check)
        self.layout.addWidget(self.show_plots_check)
        self.layout.addWidget(self.controls_button)
        self.layout.addWidget(self.help_button)
        self.layout.addWidget(self.reset_gui_button)
        self.layout.addWidget(self.clear_cache_button)

        self.setLayout(self.layout)


    @QtCore.pyqtSlot()
    def check_auto_load(self):
        self.gui.auto_load = self.auto_load_check.isChecked()
        self.gui.qt_settings.setValue('auto_load', self.gui.auto_load)

    @QtCore.pyqtSlot()
    def check_show_plots(self):
        self.gui.show_plots = self.show_plots_check.isChecked()
        self.gui.qt_settings.setValue('show_plots', self.gui.show_plots)

    @QtCore.pyqtSlot()
    def clear_cache(self):
        self.gui.qt_settings.clear()
        