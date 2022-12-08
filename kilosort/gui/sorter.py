import numpy as np
from PyQt5 import QtCore


class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.pyqtSignal(object)
    finishedSpikesort = QtCore.pyqtSignal(object)
    finishedAll = QtCore.pyqtSignal(object)

    def __init__(
            self,
            context,
            data_path,
            output_directory,
            steps,
            sanity_plots=False,
            plot_widgets=None,
            *args,
            **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = data_path
        self.output_directory = output_directory

        self.sanity_plots = sanity_plots
        self.plot_widgets = plot_widgets

        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]

    def run(self):
        if "preprocess" in self.steps:
            self.context.reset()
            self.context.probe = self.context.raw_probe.copy()
            self.context = run_preprocess(self.context)
            self.finishedPreprocess.emit(self.context)

        if "spikesort" in self.steps:
            self.context = run_spikesort(self.context,
                                         sanity_plots=self.sanity_plots,
                                         plot_widgets=self.plot_widgets)
            self.finishedSpikesort.emit(self.context)

        if "export" in self.steps:
            run_export(self.context, self.data_path, self.output_directory)
            self.finishedAll.emit(self.context)
