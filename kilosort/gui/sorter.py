import os
import numpy as np
import torch
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

from kilosort.gui.logger import setup_logger
from kilosort.run_kilosort import run_kilosort

logger = setup_logger(__name__)


class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.pyqtSignal(object)
    finishedSpikesort = QtCore.pyqtSignal(object)
    finishedAll = QtCore.pyqtSignal(object)
    progress_bar = pyqtSignal(int)

    def __init__(
            self,
            context,
            results_directory,
            steps,
            *args,
            **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = context.data_path
        self.results_directory = results_directory
        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]

    def run(self):
        if "spikesort" in self.steps:
            settings = self.context.params
            probe = self.context.probe
            settings["data_dir"] = self.data_path.parent
            settings["filename"] = self.data_path
            results_directory = self.results_directory
            if not results_directory.exists():
                logger.info(f"Results dir at {results_directory} does not exist. The folder will be created.")
                results_directory.mkdir(parents=True)
            run_kilosort(
                settings=settings,
                probe=probe,
                results_dir=results_directory,
                device=torch.device("cuda"),
                progress_bar=self.progress_bar,
                data_dtype=settings['data_dtype']
            )

            logger.info(f"Spike sorting output saved in\n{results_directory}")
            self.finishedSpikesort.emit(self.context)