from PyQt5 import QtCore, QtGui, QtWidgets

from kilosort.gui.sorter import KiloSortWorker
from kilosort.gui.sanity_plots import (
    PlotWindow, plot_drift_amount, plot_drift_scatter, plot_diagnostics
    )


class RunBox(QtWidgets.QGroupBox):
    setupContextForRun = QtCore.pyqtSignal()
    updateContext = QtCore.pyqtSignal(object)
    sortingStepStatusUpdate = QtCore.pyqtSignal(dict)
    disableInput = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.parent = parent

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run")
        # self.preprocess_button = QtWidgets.QPushButton("Preprocess")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        
        self.buttons = [
            self.run_all_button,
            # self.preprocess_button,
            # self.spike_sort_button,
            # self.export_button,
        ]

        self.data_path = None
        self.working_directory = None
        self.results_directory = None

        self.sorting_status = {
            # "preprocess": False,
            "spikesort": False,
            "export": False,
        }

        self.preprocess_done = False
        self.spikesort_done = False

        self.remote_widgets = None

        self.progress_bar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progress_bar, 2, 0, 2, 2)
        #self.gui.progress_bar = self.progress_bar 

        self.setup()

    def setup(self):
        # self.run_all_button.clicked.connect(self.run_all)
        self.run_all_button.clicked.connect(self.spikesort)
        # self.preprocess_button.clicked.connect(self.preprocess)
        self.spike_sort_button.clicked.connect(self.spikesort)
        
        self.run_all_button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.layout.addWidget(self.run_all_button, 0, 0, 2, 2)
        # self.layout.addWidget(self.preprocess_button, 0, 2, 1, 2)
        # self.layout.addWidget(self.spike_sort_button, 0, 2, 1, 2)
        
        self.setLayout(self.layout)

        self.disable_all_buttons()
        self.reenable_buttons()

    def disable_all_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def reenable_buttons(self):
        self.run_all_button.setEnabled(True)
        self.spike_sort_button.setEnabled(True)
        # self.preprocess_button.setEnabled(True)
        # if self.sorting_status["preprocess"]:
        #     self.spike_sort_button.setEnabled(True)
        # else:
        #     self.spike_sort_button.setEnabled(False)
        
    @QtCore.pyqtSlot(bool)
    def disable_all_input(self, value):
        if value:
            self.disable_all_buttons()
        else:
            self.reenable_buttons()

    def set_data_path(self, data_path):
        self.data_path = data_path

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

    # @QtCore.pyqtSlot(object)
    # def finished_preprocess(self, context):
    #     self.updateContext.emit(context)
    #     self.update_sorting_status("preprocess", True)

    @QtCore.pyqtSlot(object)
    def finished_spikesort(self, context):
        self.updateContext.emit(context)
        self.current_worker = None
        self.update_sorting_status("spikesort", True)

    # @QtCore.pyqtSlot()
    # def preprocess(self):
    #     if self.get_current_context() is not None:
    #         if self.sorting_status["preprocess"]:
    #             response = QtWidgets.QMessageBox.warning(
    #                 self,
    #                 "Confirmation",
    #                 "If you redo this step, all intermediate files from the previous "
    #                 "sorting will be deleted. Are you sure you want to proceed?",
    #                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
    #                 QtWidgets.QMessageBox.No
    #             )
    #
    #             if response == QtWidgets.QMessageBox.Yes:
    #                 self.change_sorting_status({
    #                     "preprocess": False,
    #                     "spikesort": False,
    #                     "export": False
    #                 })
    #                 self.sortingStepStatusUpdate.emit(self.sorting_status)
    #                 self.run_steps("preprocess")
    #         else:
    #             self.run_steps("preprocess")
    #
    @QtCore.pyqtSlot()
    def spikesort(self):
        if self.get_current_context() is not None:
            self.run_steps("spikesort")

    @QtCore.pyqtSlot()
    def run_all(self):
        if self.get_current_context() is not None:
            self.run_steps([
                # "preprocess",
                "spikesort",
            ])

        self.change_sorting_status({
            # "preprocess": True,
            "spikesort": True,
        })
        self.sortingStepStatusUpdate.emit(self.sorting_status)

    def run_steps(self, steps):
        self.disableInput.emit(True)
        self.setupContextForRun.emit()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        worker = KiloSortWorker(
            context=self.get_current_context(),
            results_directory=self.results_directory,
            steps=steps, device=self.parent.device
        )
        
        worker.progress_bar.connect(self.set_progress_val)
        # worker.finishedPreprocess.connect(self.finished_preprocess)
        worker.finishedSpikesort.connect(self.finished_spikesort)
        worker.finishedAll.connect(self.finished_spikesort)
        # TODO: add option to disable sanity plots?
        worker.plotDataReady.connect(self.add_plot_data)
        self.current_worker = worker
        self.setup_sanity_plots()
        
        QtWidgets.QApplication.restoreOverrideCursor()

        worker.start()
        while worker.isRunning():
            QtWidgets.QApplication.processEvents()
        else:
            self.disableInput.emit(False)

    def set_progress_val(self, val):
        self.progress_bar.setValue(val)

    def prepare_for_new_context(self):
        self.change_sorting_status({
            # "preprocess": False,
            "spikesort": False,
        })
        self.sortingStepStatusUpdate.emit(self.sorting_status)

    def setup_sanity_plots(self):
        self.plots = {
            'drift_amount': PlotWindow(nrows=1, ncols=1, title='Drift Amount'),
            'drift_scatter': PlotWindow(
                nrows=1, ncols=1, title='Drift Scatter', width=10, height=6
                ),
            'diagnostics': PlotWindow(
                nrows=2, ncols=2, width=8, height=8, title='Diagnostics'
                )
        }

    # TODO: have a separate process do the actual plotting, otherwise it
    #       locks up the sorting process. Not a big deal for the sample data,
    #       but on larger datasets the drift scatter could take a while, for example.
    def add_plot_data(self, plot_type):
        settings = self.get_current_context().params
        if plot_type == 'drift':
            # Drift amount over time for each block of probe
            plot_window1 = self.plots['drift_amount']
            plot_window2 = self.plots['drift_scatter']
            dshift = self.current_worker.dshift
            st0 = self.current_worker.st0
            plot_drift_amount(plot_window1, dshift, settings)
            plot_drift_scatter(plot_window2, st0)

        elif plot_type == 'diagnostics':
            plot_window = self.plots['diagnostics']
            Wall3 = self.current_worker.Wall3
            wPCA = self.current_worker.wPCA
            clu0 = self.current_worker.clu0
            plot_diagnostics(plot_window, wPCA, Wall3, clu0, settings)
