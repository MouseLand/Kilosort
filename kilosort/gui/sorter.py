import time
import pprint
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
from qtpy import QtCore

#from kilosort.gui.logger import setup_logger
import kilosort
from kilosort.run_kilosort import (
    setup_logger, initialize_ops, compute_preprocessing, compute_drift_correction,
    detect_spikes, cluster_spikes, save_sorting
    )
from kilosort.io import save_preprocessing
from kilosort.utils import log_performance, log_cuda_details

#logger = setup_logger(__name__)


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
                results_dir.mkdir(parents=True)
            
            setup_logger(results_dir)

            try:
                logger.info(f"Kilosort version {kilosort.__version__}")
                logger.info(f"Sorting {self.data_path}")
                clear_cache = settings['clear_cache']
                if clear_cache:
                    logger.info('clear_cache=True')
                logger.info('-'*40)

                tic0 = time.time()

                if probe['chanMap'].max() >= settings['n_chan_bin']:
                    raise ValueError(
                        f'Largest value of chanMap exceeds channel count of data, '
                        'make sure chanMap is 0-indexed.'
                    )

                if settings['nt0min'] is None:
                    settings['nt0min'] = int(20 * settings['nt']/61)
                data_dtype = settings['data_dtype']
                device = self.device
                save_preprocessed_copy = settings['save_preprocessed_copy']
                do_CAR = settings['do_CAR']
                invert_sign = settings['invert_sign']
                if not do_CAR:
                    logger.info("Skipping common average reference.")

                ops = initialize_ops(settings, probe, data_dtype, do_CAR,
                                     invert_sign, device, save_preprocessed_copy)
                # Remove some stuff that doesn't need to be printed twice,
                # then pretty-print format for log file.
                ops_copy = ops.copy()
                _ = ops_copy.pop('settings')
                _ = ops_copy.pop('probe')
                print_ops = pprint.pformat(ops_copy, indent=4, sort_dicts=False)
                logger.debug(f"Initial ops:\n{print_ops}\n")

                # TODO: add support for file object through data conversion
                # Set preprocessing and drift correction parameters
                ops = compute_preprocessing(ops, self.device, tic0=tic0,
                                            file_object=self.file_object)
                np.random.seed(1)
                torch.cuda.manual_seed_all(1)
                torch.random.manual_seed(1)
                ops, bfile, st0 = compute_drift_correction(
                    ops, self.device, tic0=tic0, progress_bar=self.progress_bar,
                    file_object=self.file_object, clear_cache=clear_cache
                    )

                # Check scale of data for log file
                b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
                logger.debug(f"First batch min, max: {b1.min(), b1.max()}")

                if save_preprocessed_copy:
                    save_preprocessing(results_dir / 'temp_wh.dat', ops, bfile)

                # Will be None if nblocks = 0 (no drift correction)
                if st0 is not None:
                    self.dshift = ops['dshift']
                    self.st0 = st0
                    self.plotDataReady.emit('drift')

                # Sort spikes and save results
                st, tF, Wall0, clu0 = detect_spikes(
                    ops, self.device, bfile, tic0=tic0,
                    progress_bar=self.progress_bar, clear_cache=clear_cache
                    )

                self.Wall0 = Wall0
                self.wPCA = torch.clone(ops['wPCA'].cpu()).numpy()
                self.clu0 = clu0
                self.plotDataReady.emit('diagnostics')

                clu, Wall = cluster_spikes(
                    st, tF, ops, self.device, bfile, tic0=tic0,
                    progress_bar=self.progress_bar, clear_cache=clear_cache
                    )
                ops, similar_templates, is_ref, est_contam_rate, kept_spikes = \
                    save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0)

            except Exception as e:
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.exception('Out of memory error, printing performance...')
                    log_performance(logger, level='info')
                    log_cuda_details(logger)
                # This makes sure the full traceback is written to log file.
                logger.exception('Encountered error in `run_kilosort`:')
                # Annoyingly, this will print the error message twice for console
                # but I haven't found a good way around that.
                raise

            self.ops = ops
            self.st = st[kept_spikes]
            self.clu = clu[kept_spikes]
            self.tF = tF[kept_spikes]
            self.is_refractory = is_ref
            self.plotDataReady.emit('probe')

            self.finishedSpikesort.emit(self.context)
