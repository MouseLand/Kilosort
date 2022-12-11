import numpy as np
import torch
from PyQt5 import QtCore
from kilosort.gui.logger import setup_logger
from kilosort.run_kilosort import run_kilosort

logger = setup_logger(__name__)


class KiloSortWorker(QtCore.QThread):
    finishedPreprocess = QtCore.pyqtSignal(object)
    finishedSpikesort = QtCore.pyqtSignal(object)
    finishedAll = QtCore.pyqtSignal(object)

    def __init__(
            self,
            context,
            output_directory,
            steps,
            *args,
            **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = context.data_path
        self.output_directory = output_directory

        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]

    def run(self):
        if "spikesort" in self.steps:
            settings = self.context.params
            probe = self.context.probe
            settings["data_folder"] = self.data_path.parent
            ops, st, tF, clu, Wall, is_ref = run_kilosort(
                settings=settings,
                probe=probe,
                device=torch.device("cuda")
            )

            output_folder = self.context.context_path

            ops_arr = np.array(ops)
            np.save(output_folder / "ops.npy", ops_arr)

            np.save(output_folder / "st.npy", st)

            np.save(output_folder / "tF.npy", tF.cpu().numpy())

            np.save(output_folder / "clu.npy", clu)

            np.save(output_folder / "Wall.npy", Wall.cpu().numpy())

            np.save(output_folder / "is_ref.npy", np.array(is_ref))

            self.finishedSpikesort.emit(self.context)

        if "export" in self.steps:
            # run_export(self.context, self.data_path, self.output_directory)
            # self.finishedAll.emit(self.context)
            raise NotImplementedError("Export for Phy has not been implemented yet!")
