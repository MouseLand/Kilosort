from qtpy import QtCore, QtGui, QtWidgets

from kilosort.gui.sorter import KiloSortWorker
from kilosort.gui.sanity_plots import (
    PlotWindow, plot_drift_amount, plot_drift_scatter, plot_diagnostics,
    plot_spike_positions
    )


class RunBox(QtWidgets.QGroupBox):
    setupContextForRun = QtCore.Signal()
    updateContext = QtCore.Signal(object)
    sortingStepStatusUpdate = QtCore.Signal(dict)
    disableInput = QtCore.Signal(bool)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.setTitle("Run")

        self.parent = parent

        self.layout = QtWidgets.QGridLayout()

        self.run_all_button = QtWidgets.QPushButton("Run")
        self.spike_sort_button = QtWidgets.QPushButton("Spikesort")
        self.save_preproc_check = QtWidgets.QCheckBox("Save Preprocessed Copy")
        self.clear_cache_check = QtWidgets.QCheckBox("Clear PyTorch Cache")
        self.do_CAR_check = QtWidgets.QCheckBox("CAR")
        self.invert_sign_check = QtWidgets.QCheckBox("Invert Sign")
        self.verbose_check = QtWidgets.QCheckBox("Verbose Log")

        self.buttons = [
            self.run_all_button
        ]

        self.data_path = None
        self.working_directory = None
        self.results_directory = None

        self.sorting_status = {
            "spikesort": False,
            "export": False,
        }

        self.preprocess_done = False
        self.spikesort_done = False

        self.remote_widgets = None

        self.progress_bar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progress_bar, 5, 0, 3, 6)

        self.setup()

    def setup(self):
        self.run_all_button.clicked.connect(self.spikesort)
        self.spike_sort_button.clicked.connect(self.spikesort)
        self.run_all_button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.save_preproc_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        preproc_text = """
            If enabled, a whitened, filtered, and drift-corrected copy of the
            data will be saved to 'temp_wh.dat' in the results directory. This
            will also reformat the results for Phy so that the preprocessed copy
            is used instead of the raw binary file.
            """
        self.save_preproc_check.setToolTip(preproc_text)

        self.clear_cache_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        cache_text = """
            If enabled, force pytorch to free up memory reserved for its cache in
            between memory-intensive operations.
            Note that setting `clear_cache=True` is NOT recommended unless you
            encounter GPU out-of-memory errors, since this can result in slower
            sorting.
            """
        self.clear_cache_check.setToolTip(cache_text)

        self.do_CAR_check.setCheckState(QtCore.Qt.CheckState.Checked)
        car_text = """
            If enabled, apply common average reference during preprocessing
            (recommended).
            """
        self.do_CAR_check.setToolTip(car_text)

        self.invert_sign_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        invert_sign_text = """
            If enabled, flip positive/negative values in data to conform to
            standard expected by Kilosort4. This is NOT recommended unless you
            know your data is using the opposite sign.
            """
        self.invert_sign_check.setToolTip(invert_sign_text)

        self.verbose_check.setCheckState(QtCore.Qt.CheckState.Unchecked)
        verbose_text = """
            If True, include additional debug-level logging statements for some
            steps. This provides more detail for debugging, but may impact
            performance.
            """
        self.verbose_check.setToolTip(verbose_text)

        self.layout.addWidget(self.run_all_button, 0, 0, 3, 6)
        self.layout.addWidget(self.save_preproc_check, 3, 0, 1, 3)
        self.layout.addWidget(self.clear_cache_check, 3, 3, 1, 3)
        self.layout.addWidget(self.verbose_check, 4, 0, 1, 2)
        self.layout.addWidget(self.invert_sign_check, 4, 2, 1, 2)
        self.layout.addWidget(self.do_CAR_check, 4, 4, 1, 2)
        
        self.setLayout(self.layout)

        self.disable_all_buttons()
        self.reenable_buttons()

    def disable_all_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def reenable_buttons(self):
        self.run_all_button.setEnabled(True)
        self.spike_sort_button.setEnabled(True)
        
    @QtCore.Slot(bool)
    def disable_all_input(self, value):
        if value:
            self.disable_all_buttons()
            # This is done separate from other buttons so that it can be checked
            # on or off without needing to load data.
            self.save_preproc_check.setEnabled(False)
        else:
            self.reenable_buttons()
            self.save_preproc_check.setEnabled(True)

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

    @QtCore.Slot(object)
    def finished_spikesort(self, context):
        self.updateContext.emit(context)
        self.current_worker = None
        self.update_sorting_status("spikesort", True)

    @QtCore.Slot()
    def spikesort(self):
        if self.get_current_context() is not None:
            self.run_steps("spikesort")

    @QtCore.Slot()
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
            steps=steps, device=self.parent.device,
            file_object=self.parent.file_object,
        )
        
        worker.progress_bar.connect(self.set_progress_val)
        # worker.finishedPreprocess.connect(self.finished_preprocess)
        worker.finishedSpikesort.connect(self.finished_spikesort)
        worker.finishedAll.connect(self.finished_spikesort)

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
        if self.parent.show_plots:
            self.plots = {
                'drift_amount': PlotWindow(
                    nrows=1, ncols=1, width=400, height=400, title='Drift Amount'
                    ),
                'drift_scatter': PlotWindow(
                    nrows=1, ncols=1, title='Drift Scatter', width=1500, height=700,
                    background='w'
                    ),
                'diagnostics': PlotWindow(
                    nrows=2, ncols=2, width=800, height=800, title='Diagnostics'
                    ),
                'probe': PlotWindow(
                    nrows=1, ncols=1, width=1500, height=700, title='Spike positions'
                    )
            }
        else:
            self.plots = {}

    # TODO: have a separate process do the actual plotting, otherwise it
    #       locks up the sorting process. Not a big deal for the sample data,
    #       but on larger datasets the drift scatter could take a while, for example.
    def add_plot_data(self, plot_type):
        if not self.parent.show_plots:
            return

        settings = self.get_current_context().params
        if plot_type == 'drift':
            # Drift amount over time for each block of probe
            plot_window1 = self.plots['drift_amount']
            plot_window2 = self.plots['drift_scatter']
            dshift = self.current_worker.dshift
            st0 = self.current_worker.st0
            plot_drift_amount(plot_window1, dshift, settings)
            plot_drift_scatter(plot_window2, st0, settings)

        elif plot_type == 'diagnostics':
            plot_window = self.plots['diagnostics']
            Wall0 = self.current_worker.Wall0
            wPCA = self.current_worker.wPCA
            clu0 = self.current_worker.clu0
            plot_diagnostics(plot_window, wPCA, Wall0, clu0, settings)

        elif plot_type == 'probe':
            plot_window = self.plots['probe']
            clu = self.current_worker.clu
            is_refractory = self.current_worker.is_refractory
            plot_spike_positions(plot_window, clu, is_refractory, settings)
