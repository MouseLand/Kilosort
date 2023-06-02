from pathlib import Path
import torch
from kilosort import preprocessing
from kilosort.gui import (
    DataViewBox,
    HeaderBox,
    MessageLogBox,
    ProbeViewBox,
    RunBox,
    SettingsBox,
)
from kilosort.gui.logger import setup_logger
from kilosort.io import BinaryFiltered
from kilosort.utils import DOWNLOADS_DIR, download_probes
from PyQt5 import QtCore, QtGui, QtWidgets

logger = setup_logger(__name__)


class KiloSortGUI(QtWidgets.QMainWindow):
    def __init__(self, application, filename=None, **kwargs):
        super(KiloSortGUI, self).__init__(**kwargs)

        self.app = application

        self.data_path = filename
        self.probe_layout = None
        self.params = None
        self.results_directory = None

        # self.probe_files_path = Path(probes.__file__).parent
        # assert self.probe_files_path.exists()

        self.local_config_path = DOWNLOADS_DIR
        self.local_config_path.mkdir(parents=True, exist_ok=True)

        self.new_probe_files_path = self.local_config_path / "probes"
        self.new_probe_files_path.mkdir(exist_ok=True)
        download_probes(self.new_probe_files_path)

        # self.time_range = None
        self.num_channels = None

        self.context = None

        self.content = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout()

        self.header_box = HeaderBox(self)

        self.boxes = QtWidgets.QWidget()
        self.boxes_layout = QtWidgets.QHBoxLayout(self.boxes)
        self.first_boxes_layout = QtWidgets.QVBoxLayout()
        self.second_boxes_layout = QtWidgets.QHBoxLayout()
        self.third_boxes_layout = QtWidgets.QVBoxLayout()
        self.fourth_boxes_layout = QtWidgets.QVBoxLayout()

        self.settings_box = SettingsBox(self)
        self.probe_view_box = ProbeViewBox(self)
        self.data_view_box = DataViewBox(self)
        self.run_box = RunBox(self)
        self.message_log_box = MessageLogBox(self)

        self.setAcceptDrops(True)

        self.setup()

    def keyPressEvent(self, event):
        QtWidgets.QMainWindow.keyPressEvent(self, event)

        if type(event) == QtGui.QKeyEvent:
            modifiers = event.modifiers()

            if event.key() == QtCore.Qt.Key_Up:
                if modifiers:
                    if modifiers == QtCore.Qt.ControlModifier:
                        # logger.debug("Ctrl+Up")
                        self.change_channel_display(1)
                    elif modifiers == QtCore.Qt.AltModifier:
                        # logger.debug("Alt+Up")
                        self.scale_data(1)
                    elif modifiers == QtCore.Qt.ShiftModifier:
                        # logger.debug("Shift+Up")
                        self.zoom_data_view_in_time(1)
                else:
                    # logger.debug("Only Up")
                    self.change_displayed_channel_count(shift=1)
            elif event.key() == QtCore.Qt.Key_Down:
                if modifiers:
                    if modifiers == QtCore.Qt.ControlModifier:
                        # logger.debug("Ctrl+Down")
                        self.change_channel_display(-1)
                    elif modifiers == QtCore.Qt.AltModifier:
                        # logger.debug("Alt+Down")
                        self.scale_data(-1)
                    elif modifiers == QtCore.Qt.ShiftModifier:
                        # logger.debug("Shift+Down")
                        self.zoom_data_view_in_time(-1)
                else:
                    # logger.debug("Only Down")
                    self.change_displayed_channel_count(shift=-1)
            elif event.key() == QtCore.Qt.Key_Left:
                self.shift_data(time_shift=-1)
            elif event.key() == QtCore.Qt.Key_Right:
                self.shift_data(time_shift=1)
            # elif event.key() == QtCore.Qt.Key_C:
            #     self.toggle_view()
            elif event.key() == QtCore.Qt.Key_1:
                self.toggle_mode("raw")
            elif event.key() == QtCore.Qt.Key_2:
                self.toggle_mode("whitened")
            elif event.key() == QtCore.Qt.Key_3:
                self.toggle_mode("prediction")
            elif event.key() == QtCore.Qt.Key_4:
                self.toggle_mode("residual")
            else:
                pass
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        filename = files[0]
        if self.context is None:
            self.settings_box.set_data_file_path_from_drag_and_drop(filename)
            logger.info(f"File at location: {filename} is ready to load!")
        else:
            response = QtWidgets.QMessageBox.warning(
                self,
                "Are you sure?",
                "You are attempting to load a new file while another file "
                "is already loaded. Are you sure you want to proceed?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
                )
                
            if response == QtWidgets.QMessageBox.Yes:
                self.settings_box.set_data_file_path_from_drag_and_drop(filename)


    def setup(self):
        self.setWindowTitle(f"Kilosort4")

        self.third_boxes_layout.addWidget(self.settings_box, 85)
        self.third_boxes_layout.addWidget(self.run_box, 15)

        self.second_boxes_layout.addLayout(self.third_boxes_layout, 50)
        self.second_boxes_layout.addWidget(self.probe_view_box, 50)

        self.first_boxes_layout.addLayout(self.second_boxes_layout, 60)
        self.first_boxes_layout.addWidget(self.message_log_box, 40)

        self.fourth_boxes_layout.addWidget(self.header_box, 3)
        self.fourth_boxes_layout.addWidget(self.data_view_box, 97)

        self.boxes_layout.addLayout(self.first_boxes_layout, 25)
        self.boxes_layout.addLayout(self.fourth_boxes_layout, 75)

        self.boxes.setLayout(self.boxes_layout)
        self.content_layout.addWidget(self.boxes, 90)

        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content.setLayout(self.content_layout)
        self.setCentralWidget(self.content)

        self.header_box.reset_gui_button.clicked.connect(self.reset_gui)

        self.settings_box.settingsUpdated.connect(self.load_data)
        self.settings_box.previewProbe.connect(self.probe_view_box.preview_probe)

        self.data_view_box.channelChanged.connect(self.probe_view_box.update_probe_view)
        self.data_view_box.modeChanged.connect(
            self.probe_view_box.synchronize_data_view_mode
        )

        self.probe_view_box.channelSelected.connect(
            self.data_view_box.change_primary_channel
        )

        self.run_box.updateContext.connect(self.update_context)
        self.run_box.disableInput.connect(self.disable_all_input)
        self.run_box.sortingStepStatusUpdate.connect(self.update_sorting_status)
        self.run_box.setupContextForRun.connect(self.setup_context_for_run)

    def change_channel_display(self, direction):
        if self.context is not None:
            self.data_view_box.shift_primary_channel(direction)

    def shift_data(self, time_shift):
        if self.context is not None:
            self.data_view_box.shift_current_time(time_shift)

    def change_displayed_channel_count(self, shift):
        if self.context is not None:
            self.data_view_box.change_displayed_channel_count(shift)

    def zoom_data_view_in_time(self, direction):
        if self.context is not None:
            self.data_view_box.change_plot_range(direction)

    def scale_data(self, direction):
        if self.context is not None:
            self.data_view_box.change_plot_scaling(direction)

    # def toggle_view(self):
    #     self.data_view_box.toggle_mode()

    def toggle_mode(self, mode):
        if mode == "raw":
            self.data_view_box.raw_button.click()
        elif mode == "whitened":
            self.data_view_box.whitened_button.click()
        # elif mode == "prediction":
        #     self.data_view_box.prediction_button.click()
        # elif mode == "residual":
        #     self.data_view_box.residual_button.click()
        else:
            raise ValueError("Invalid mode requested!")

    @QtCore.pyqtSlot(bool)
    def disable_all_input(self, value):
        self.settings_box.disable_all_input(value)
        self.run_box.disable_all_input(value)

    def load_data(self):
        self.set_parameters()
        self.do_load()

    def set_parameters(self):
        settings = self.settings_box.settings
        # advanced_options = self.settings_box.advanced_options

        self.data_path = settings.pop("data_file_path")
        self.results_directory = settings.pop("results_dir")
        self.probe_layout = settings.pop("probe_layout")
        # self.time_range = settings.pop("time_range")
        self.num_channels = settings["n_chan_bin"]

        # params = KilosortParams()
        # params = params.parse_obj(advanced_options)
        params = settings.copy()

        assert params

        self.params = params

    def do_load(self):
        self.disable_all_input(True)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))

        self.prepare_for_new_context()
        self.setup_context()
        self.load_binary_files()
        self.update_probe_view()
        self.setup_data_view()
        self.update_run_box()

        self.disable_all_input(False)
        QtWidgets.QApplication.restoreOverrideCursor()

    def load_binary_files(self):
        n_channels = self.params["n_chan_bin"]
        sample_rate = self.params["fs"]
        chan_map = self.probe_layout["chanMap"]
        xc = self.probe_layout["xc"]
        yc = self.probe_layout["yc"]
        nskip = self.params["nskip"]
        data_dtype = self.params["data_dtype"]

        binary_file = BinaryFiltered(
            filename=self.data_path,
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            device=torch.device("cpu"),
            dtype=data_dtype,
        )

        self.context.binary_file = binary_file

        self.context.highpass_filter = preprocessing.get_highpass_filter(
            fs=sample_rate,
            device=torch.device("cpu")
        )

        with BinaryFiltered(
            filename=self.data_path,
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            hp_filter=self.context.highpass_filter,
            device=torch.device("cpu"),
            dtype=data_dtype,
        ) as bin_file:
            self.context.whitening_matrix = preprocessing.get_whitening_matrix(
                f=bin_file,
                xc=xc,
                yc=yc,
                nskip=nskip,
            )

        filt_binary_file = BinaryFiltered(
            filename=self.data_path,
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            hp_filter=self.context.highpass_filter,
            whiten_mat=self.context.whitening_matrix,
            device=torch.device("cpu"),
            dtype=data_dtype,
        )

        self.context.filt_binary_file = filt_binary_file

        self.data_view_box.set_whitening_matrix(self.context.whitening_matrix)
        self.data_view_box.set_highpass_filter(self.context.highpass_filter)

    def setup_data_view(self):
        self.data_view_box.setup_seek(self.context)
        # self.data_view_box.create_plot_items()
        self.data_view_box.enable_view_buttons()

    def setup_context(self):
        self.context = Context(
            context_path = self.results_directory,
            probe=self.probe_layout,
            raw_probe=self.probe_layout.copy(),
            params=self.params,
            data_path=self.data_path,
        )

    def setup_context_for_run(self):
        self.set_parameters()
        self.context["params"] = self.params

    @QtCore.pyqtSlot(object)
    def update_context(self, context):
        self.context = context
        self.update_probe_view()
        self.update_data_view()

    @QtCore.pyqtSlot()
    def update_probe_view(self):
        self.probe_view_box.set_layout(self.context)

    def update_data_view(self):
        self.data_view_box.set_whitening_matrix(self.context.whitening_matrix)
        # self.data_view_box.clear_cached_traces()
        self.data_view_box.update_plot(self.context)

    def update_run_box(self):
        self.run_box.set_data_path(self.data_path)
        self.run_box.set_results_directory(self.results_directory)

    @QtCore.pyqtSlot(dict)
    def update_sorting_status(self, status_dict):
        self.run_box.change_sorting_status(status_dict)
        self.data_view_box.change_sorting_status(status_dict)
        self.probe_view_box.change_sorting_status(status_dict)

    def get_context(self):
        return self.context

    def get_probe(self):
        return self.probe_layout

    def get_params(self):
        return self.params

    def get_binary_file(self):
        if self.context is not None:
            return self.context.binary_file
        else:
            return None

    def get_filt_binary_file(self):
        if self.context is not None:
            return self.context.filt_binary_file
        else:
            return None

    def prepare_for_new_context(self):
        self.data_view_box.prepare_for_new_context()
        self.probe_view_box.prepare_for_new_context()
        self.message_log_box.prepare_for_new_context()

        self.close_binary_files()

        self.context = None

    def close_binary_files(self):
        if self.context is not None:
            if self.context.binary_file is not None:
                self.context.binary_file.close()

            if self.context.filt_binary_file is not None:
                self.context.filt_binary_file.close()

    def reset_gui(self):
        # self.time_range = None
        self.num_channels = None
        self.context = None

        self.close_binary_files()

        self.probe_view_box.reset()
        self.data_view_box.reset()
        self.settings_box.reset()
        self.message_log_box.reset()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.message_log_box.save_log_file()
        self.close_binary_files()
        event.accept()


class Context(dict):
    """
    Borrowed from:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/32107024#32107024

    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Context, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Context, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Context, self).__delitem__(key)
        del self.__dict__[key]
