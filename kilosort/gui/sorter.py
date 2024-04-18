import time

import numpy as np
import torch
from qtpy import QtCore

from kilosort.gui.logger import setup_logger
from kilosort.run_kilosort import (
    initialize_ops, compute_preprocessing, compute_drift_correction,
    detect_spikes, cluster_spikes, save_sorting
    )

logger = setup_logger(__name__)


class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.Signal(object)
    finishedSpikesort = QtCore.Signal(object)
    finishedAll = QtCore.Signal(object)
    progress_bar = QtCore.Signal(int)
    plotDataReady = QtCore.Signal(str)

    def __init__(self, context, results_directory, steps,
                 device=torch.device('cuda'), file_object=None, *args, **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = context.data_path
        self.results_directory = results_directory
        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]
        self.device = device
        self.file_object = file_object

    def run(self):
        if "spikesort" in self.steps:
            settings = self.context.params
            probe = self.context.probe
            settings["data_dir"] = self.data_path.parent
            settings["filename"] = self.data_path
            results_dir = self.results_directory
            if not results_dir.exists():
                logger.info(f"Results dir at {results_dir} does not exist."
                             "The folder will be created.")
                results_dir.mkdir(parents=True)

            tic0 = time.time()

            # TODO: make these options in GUI
            do_CAR=True
            invert_sign=False
        
            if not do_CAR:
                print("Skipping common average reference.")

            if settings['nt0min'] is None:
                settings['nt0min'] = int(20 * settings['nt']/61)
            data_dtype = settings['data_dtype']
            device = self.device

            ops = initialize_ops(settings, probe, data_dtype, do_CAR,
                                 invert_sign, device)

            # TODO: add support for file object through data conversion
            # Set preprocessing and drift correction parameters
            ops = compute_preprocessing(ops, self.device, tic0=tic0,
                                        file_object=self.file_object)
            np.random.seed(1)
            torch.cuda.manual_seed_all(1)
            torch.random.manual_seed(1)
            ops, bfile, st0 = compute_drift_correction(
                ops, self.device, tic0=tic0, progress_bar=self.progress_bar,
                file_object=self.file_object
                )

            # Will be None if nblocks = 0 (no drift correction)
            if st0 is not None:
                self.dshift = ops['dshift']
                self.st0 = st0
                self.plotDataReady.emit('drift')

            # Sort spikes and save results
            st, tF, Wall0, clu0 = detect_spikes(ops, self.device, bfile, tic0=tic0,
                                                progress_bar=self.progress_bar)

            self.Wall0 = Wall0
            self.wPCA = torch.clone(ops['wPCA'].cpu()).numpy()
            self.clu0 = clu0
            self.plotDataReady.emit('diagnostics')

            clu, Wall = cluster_spikes(st, tF, ops, self.device, bfile, tic0=tic0,
                                       progress_bar=self.progress_bar)
            ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0)

            self.ops = ops
            self.st = st[kept_spikes]
            self.clu = clu[kept_spikes]
            self.tF = tF[kept_spikes]
            self.is_refractory = is_ref
            self.plotDataReady.emit('probe')

            logger.info(f"Spike sorting output saved in\n{results_dir}")
            self.finishedSpikesort.emit(self.context)
