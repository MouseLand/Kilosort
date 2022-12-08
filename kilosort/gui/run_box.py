from PyQt5 import QtCore, QtGui, QtWidgets

from kilosort.gui.sorter import KiloSortWorker


class RunBox(QtWidgets.QGroupBox):
    updateContext = QtCore.pyqtSignal(object)
    sortingStepStatusUpdate = QtCore.pyqtSignal(dict)
    disableInput = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.parent = parent

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run All")
        self.preprocess_button = QtWidgets.QPushButton("Preprocess")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        self.export_button = QtWidgets.QPushButton("Export for Phy")

        self.buttons = [
            self.run_all_button,
            self.preprocess_button,
            self.spike_sort_button,
            self.export_button,
        ]

        self.data_path = None
        self.working_directory = None
        self.results_directory = None

        self.sorting_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False,
        }

        self.preprocess_done = False
        self.spikesort_done = False

        self.remote_widgets = None

        self.setup()

    def setup(self):
        self.run_all_button.clicked.connect(self.run_all)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.spike_sort_button.clicked.connect(self.spikesort)
        self.export_button.clicked.connect(self.export)

        self.spike_sort_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.run_all_button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.layout.addWidget(self.run_all_button, 0, 0, 2, 2)
        self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        self.layout.addWidget(self.spike_sort_button, 1, 2, 1, 2)
        self.layout.addWidget(self.export_button, 2, 2, 1, 2)

        self.setLayout(self.layout)

        self.disable_all_buttons()
        self.reenable_buttons()

    def disable_all_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def reenable_buttons(self):
        self.run_all_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)
        if self.sorting_status["preprocess"]:
            self.spike_sort_button.setEnabled(True)
        else:
            self.spike_sort_button.setEnabled(False)
        if self.sorting_status["spikesort"]:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)

    @QtCore.pyqtSlot(bool)
    def disable_all_input(self, value):
        if value:
            self.disable_all_buttons()
        else:
            self.reenable_buttons()

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_working_directory(self, working_directory_path):
        self.working_directory = working_directory_path

    def set_results_directory(self, results_directory_path):
        self.results_directory = results_directory_path

    def get_current_context(self):
        return self.parent.get_context()

    def update_sorting_status(self, step, status):
        self.sorting_status[step] = status
        self.sortingStepStatusUpdate.emit(self.sorting_status)

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict
        self.reenable_buttons()

    @QtCore.pyqtSlot(object)
    def finished_preprocess(self, context):
        self.updateContext.emit(context)
        self.update_sorting_status("preprocess", True)

    @QtCore.pyqtSlot(object)
    def finished_spikesort(self, context):
        self.updateContext.emit(context)
        self.update_sorting_status("spikesort", True)

    @QtCore.pyqtSlot(object)
    def finished_export(self, context):
        self.updateContext.emit(context)
        self.update_sorting_status("export", True)

    @QtCore.pyqtSlot()
    def preprocess(self):
        if self.get_current_context() is not None:
            if self.sorting_status["preprocess"]:
                response = QtWidgets.QMessageBox.warning(
                    self,
                    "Confirmation",
                    "If you redo this step, all intermediate files from the previous "
                    "sorting will be deleted. Are you sure you want to proceed?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )

                if response == QtWidgets.QMessageBox.Yes:
                    self.change_sorting_status(
                        {"preprocess": False,
                         "spikesort": False,
                         "export": False}
                    )
                    self.sortingStepStatusUpdate.emit(self.sorting_status)
                    self.run_steps("preprocess")
            else:
                self.run_steps("preprocess")

    @QtCore.pyqtSlot()
    def spikesort(self):
        if self.get_current_context() is not None:
            self.run_steps("spikesort")

    @QtCore.pyqtSlot()
    def export(self):
        if self.get_current_context() is not None:
            self.run_steps("export")

    @QtCore.pyqtSlot()
    def run_all(self):
        if self.get_current_context() is not None:
            self.run_steps(["preprocess", "spikesort", "export"])

        self.change_sorting_status(
            {"preprocess": True,
             "spikesort": True,
             "export": True}
        )
        self.sortingStepStatusUpdate.emit(self.sorting_status)

    def run_steps(self, steps):
        self.disableInput.emit(True)
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        worker = KiloSortWorker(
            context=self.get_current_context(),
            data_path=self.data_path,
            output_directory=self.results_directory,
            steps=steps,
            plot_widgets=self.remote_widgets,
        )

        worker.finishedPreprocess.connect(self.finished_preprocess)
        worker.finishedSpikesort.connect(self.finished_spikesort)
        worker.finishedAll.connect(self.finished_export)

        QtWidgets.QApplication.restoreOverrideCursor()

        worker.start()
        while worker.isRunning():
            QtWidgets.QApplication.processEvents()
        else:
            self.disableInput.emit(False)

    def prepare_for_new_context(self):
        self.change_sorting_status(
            {"preprocess": False,
             "spikesort": False,
             "export": False}
        )
        self.sortingStepStatusUpdate.emit(self.sorting_status)
