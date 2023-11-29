import os
import pprint
from pathlib import Path

import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from scipy.io.matlab.miobase import MatReadError

from kilosort.gui.logger import setup_logger
from kilosort.gui.minor_gui_elements import ProbeBuilder, create_prb
from kilosort.io import load_probe, BinaryRWFile


logger = setup_logger(__name__)

_DEFAULT_DTYPE = 'int16'
_MAIN_PARAMETERS = [
    # variable name, display name, type, minimum, maximum, exclusions, default
    # min, max are inclusive, so have to specify exclude 0 to get > 0 for example
    ('n_chan_bin', 'number of channels', int, 0, np.inf, [0], 385),
    ('fs', 'sampling frequency', float, 0, np.inf, [0], 30000),
    ('nt', 'nt', int, 1, np.inf, [], 61),
    ('Th', 'Th', float, 0, np.inf, [0], 6),
    ('spkTh', 'spkTh', float, 0, np.inf, [0], 8),
    ('Th_detect', 'Th_detect', float, 0, np.inf, [0], 9),
    ('nwaves', 'nwaves', int, 1, np.inf, [], 6),
    ('nskip', 'nskip', int, 1, np.inf, [], 25),
    ('nt0min', 'nt0min', int, 0, np.inf, [], 20),
    ('NT', 'NT', int, 1, np.inf, [], 60000),
    ('nblocks', 'nblocks', int, 0, np.inf, [], 5),
    ('binning_depth', 'binning depth', float, 0, np.inf, [0], 5),
    ('sig_interp', 'sig_interp', float, 0, np.inf, [0], 20),
    ('artifact_threshold', 'artifact threshold', float, 0, np.inf, [], np.inf)
]

_EXTRA_PARAMETERS = [
    ('whitening_range', 'whitening range', int, 1, np.inf, [], 32),
    ('dmin', 'dmin', float, 0, np.inf, [0], None),
    ('dminx', 'dminx', float, 0, np.inf, [0], None),
    ('acg_threshold', 'acg threshold', float, 0, np.inf, [0], 0.2),
    ('ccg_threshold', 'ccg_threshold', float, 0, np.inf, [0], 0.25),
    ('cluster_downsampling', 'cluster downsampling', int, 1, np.inf, [], 20),
    ('cluster_pcs', 'cluster pcs', int, 1, np.inf, [], 64),
    ('min_template_size', 'min template size', int, 1, np.inf, [], 10),
    ('template_sizes', 'num template sizes', int, 1, np.inf, [], 5),
    ('nearest_chans', 'nearest chans', int, 1, np.inf, [], 10),
    ('nearest_templates', 'nearest templates', int, 1, np.inf, [], 100),
    ('templates_from_data', 'templates from data', bool, None, None, [], False),
    ('n_templates', 'n templates', int, 1, np.inf, [], 6),
    ('n_pcs', 'n pcs', int, 1, np.inf, [], 6),
    ('th_for_wPCA', 'th for wPCA', float, 0, np.inf, [0], 6),
    ('duplicate_spike_bins', 'duplicate spike bins', int, 0, np.inf, [], 15)
]


class SettingsBox(QtWidgets.QGroupBox):
    settingsUpdated = QtCore.pyqtSignal()
    previewProbe = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent
        
        self.select_data_file = QtWidgets.QPushButton("Select Binary File")
        self.data_file_path = Path(self.gui.data_path).resolve()
        self.data_file_path_input = QtWidgets.QLineEdit(os.fspath(self.data_file_path) 
                                                        if self.data_file_path is not None else "")

        if self.data_file_path is not None:
            self.results_directory_path = self.data_file_path.parent.joinpath('kilosort4/')
        self.gui.results_directory = self.results_directory_path 
        self.select_results_directory = QtWidgets.QPushButton(
            "Select Results Dir."
        )
        self.results_directory_input = QtWidgets.QLineEdit(os.fspath(self.results_directory_path) 
                                                            if self.results_directory_path is not None else "")

        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()
        self._probes = []
        self.populate_probe_selector()

        self.dtype_selector_text = QtWidgets.QLabel("Data dtype:")
        self.dtype_selector = QtWidgets.QComboBox()
        self.populate_dtype_selector()

        self.device_selector_text = QtWidgets.QLabel('PyTorch device:')
        self.device_selector = QtWidgets.QComboBox()
        self.populate_device_selector()

        generated_inputs = []
        for i, (var, name, _, _, _, _, default) in enumerate(_MAIN_PARAMETERS):
            setattr(self, f'{var}_text', QtWidgets.QLabel(f'{name}'))
            setattr(self, f'{var}_input', QtWidgets.QLineEdit())
            getattr(self, f'{var}_input').parameter_index = i
            setattr(self, f'{var}', default)
            generated_inputs.append(getattr(self, f'{var}_input'))
        self.data_dtype = _DEFAULT_DTYPE

        self.extra_parameters_window = ExtraParametersWindow(self)
        self.extra_parameters_button = QtWidgets.QPushButton('Extra settings')

        self.load_settings_button = QtWidgets.QPushButton("LOAD")
        self.probe_preview_button = QtWidgets.QPushButton("Preview Probe")
        self.probe_layout = self.gui.probe_layout
        self.probe_name = self.gui.probe_name
  
        self.input_fields = [
            self.data_file_path_input,
            self.results_directory_input,
            self.probe_layout_selector,
            self.dtype_selector
            ]
        self.input_fields.extend(generated_inputs)

        self.buttons = [
            self.load_settings_button,
            self.probe_preview_button,
            self.select_data_file,
            self.select_results_directory,
        ]

        self.settings = {}
        self.setup()

        if self.probe_name is not None:
            self.probe_layout_selector.setCurrentText(self.probe_name)
        if self.probe_layout is not None:
            self.enable_preview_probe()
        if self.data_file_path is not None and self.data_file_path != '':
            if self.check_settings():
                self.enable_load()


    def setup(self):
        self.setTitle("Settings")

        layout = QtWidgets.QGridLayout()
        row_count = 0

        font = self.load_settings_button.font()
        font.setPointSize(18)
        self.load_settings_button.setFont(font)
        self.load_settings_button.setDisabled(True)
        self.load_settings_button.clicked.connect(self.update_settings)
        layout.addWidget(self.load_settings_button, row_count, 0, 1, 5)

        row_count += 1
        layout.addWidget(self.select_data_file, row_count, 0, 1, 3)
        layout.addWidget(self.data_file_path_input, row_count, 3, 1, 2)
        self.select_data_file.clicked.connect(self.on_select_data_file_clicked)
        self.data_file_path_input.textChanged.connect(self.on_data_file_path_changed)
        self.data_file_path_input.editingFinished.connect(
            self.on_data_file_path_changed
        )

        row_count += 1
        layout.addWidget(self.select_results_directory, row_count, 0, 1, 3)
        layout.addWidget(self.results_directory_input, row_count, 3, 1, 2)
        self.select_results_directory.clicked.connect(
            self.on_select_results_dir_clicked
        )
        self.results_directory_input.textChanged.connect(
            self.on_results_directory_changed
        )
        self.results_directory_input.editingFinished.connect(
            self.on_results_directory_changed
        )

        row_count += 1
        layout.addWidget(self.probe_layout_text, row_count, 0, 1, 3)
        layout.addWidget(self.probe_layout_selector, row_count, 3, 1, 2)
        self.probe_layout_selector.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLength
        )
        self.probe_layout_selector.currentTextChanged.connect(
            self.on_probe_layout_selected
        )

        row_count += 1
        self.probe_preview_button.setDisabled(True)
        self.probe_preview_button.clicked.connect(self.show_probe_layout)
        layout.addWidget(
            self.probe_preview_button, row_count, 3, 1, 2)

        row_count += 1
        layout.addWidget(self.dtype_selector_text, row_count, 0, 1, 3)
        layout.addWidget(self.dtype_selector, row_count, 3, 1, 2)
        self.dtype_selector.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self.dtype_selector.currentTextChanged.connect(
            self.on_data_dtype_selected
        )

        row_count += 1
        layout.addWidget(self.device_selector_text, row_count, 0, 1, 3)
        layout.addWidget(self.device_selector, row_count, 3, 1, 2)
        self.device_selector.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        )
        self.device_selector.currentTextChanged.connect(
            self.on_device_selected
        )

        for parameter_info in _MAIN_PARAMETERS:
            row_count += 1
            var = parameter_info[0]
            layout.addWidget(getattr(self, f'{var}_text'), row_count, 0, 1, 3)
            layout.addWidget(getattr(self, f'{var}_input'), row_count, 3, 1, 2)
            getattr(self, f'{var}_input').textChanged.connect(self.update_parameter)

        row_count +=1
        layout.addWidget(self.extra_parameters_button, row_count, 0, 1, 5)
        self.extra_parameters_button.clicked.connect(
            lambda x: self.extra_parameters_window.show()
            )

        self.setLayout(layout)
        self.set_default_field_values()
        self.update_settings()

    def set_default_field_values(self):
        self.dtype_selector.setCurrentText(_DEFAULT_DTYPE)
        for var, _, _,  _, _, _, default in _MAIN_PARAMETERS:
            getattr(self, f'{var}_input').setText(str(default))
        for var, _, _,  _, _, _, default in _EXTRA_PARAMETERS:
            epw = self.extra_parameters_window
            getattr(epw, f'{var}_input').setText(str(default))

    def on_select_data_file_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Choose data file to load...",
            directory=os.path.expanduser("~"),
            options=file_dialog_options,
        )
        if data_file_name:
            self.data_file_path_input.setText(data_file_name)

    def set_data_file_path_from_drag_and_drop(self, filename):
        if Path(filename).suffix in ['.bin', '.dat', '.bat', '.raw']:
            self.data_file_path_input.setText(filename)
            logger.info(f"File at location: {filename} is ready to load!")

        else:
            QtWidgets.QMessageBox.warning(
                self.parent(),
                "Wrong file type!",
                "Drag and drop only works with .bin, .dat, .bat, and .raw files!",
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            logger.warning(
                "Drag and drop only works with .bin, .dat, .bat, and .raw files!"
                )

    def on_select_results_dir_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        results_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(
            parent=self,
            caption="Choose results directory...",
            directory=QtCore.QUrl(os.path.expanduser("~")),
            options=file_dialog_options,
        )
        if results_dir_name:
            self.results_directory_input.setText(results_dir_name.toLocalFile())

    def on_results_directory_changed(self):
        results_directory = Path(self.results_directory_input.text())

        if not results_directory.exists():
            logger.warning(f"The results directory {results_directory} does not exist.")
            logger.warning("It will be (recursively) created upon data load.")

        self.results_directory_path = results_directory
        self.gui.qt_settings.setValue('results_dir', results_directory)

        if self.check_settings():
            self.enable_load()
        else:
            self.disable_load()

    def on_data_file_path_changed(self):
        data_file_path = Path(self.data_file_path_input.text())
        try:
            assert data_file_path.exists()

            parent_folder = data_file_path.parent
            results_folder = parent_folder / "kilosort4"
            self.results_directory_input.setText(results_folder.as_posix())

            self.data_file_path = data_file_path
            self.gui.qt_settings.setValue('data_file_path', data_file_path)

            if self.check_settings():
                self.enable_load()
        except AssertionError:
            logger.exception("Please select a valid file path!")
            self.disable_load()

    def disable_all_input(self, value):
        for button in self.buttons:
            button.setDisabled(value)
        for field in self.input_fields:
            field.setDisabled(value)

    def enable_load(self):
        self.load_settings_button.setEnabled(True)

    def disable_load(self):
        self.load_settings_button.setDisabled(True)

    def enable_preview_probe(self):
        self.probe_preview_button.setEnabled(True)

    def disable_preview_probe(self):
        self.probe_preview_button.setDisabled(True)

    def check_settings(self):
        self.settings = {
            "data_file_path": self.data_file_path,
            "results_dir": self.results_directory_path,
            "probe": self.probe_layout,
            "probe_name": self.probe_name,
            "data_dtype": self.data_dtype,
            }
        for p in _MAIN_PARAMETERS:
            self.settings[p[0]] = getattr(self, p[0])
        for p in _EXTRA_PARAMETERS:
            self.settings[p[0]] = getattr(self.extra_parameters_window, p[0])

        none_allowed = ['dmin', 'dminx']
        for k, v in self.settings.items():
            if v is None and k not in none_allowed:
                return False
        return True
    
    @QtCore.pyqtSlot()
    def update_parameter(self):
        parameter_index = self.sender().parameter_index
        parameter_info = _MAIN_PARAMETERS[parameter_index]
        _check_parameter(self, self, parameter_info)

    @QtCore.pyqtSlot()
    def update_settings(self):
        if self.check_settings():
            if not self.results_directory_path.exists():
                try:
                    os.makedirs(self.results_directory_path)
                    self.settingsUpdated.emit()
                except Exception as e:
                    logger.exception(e)
                    self.disable_load()

            else:
                self.settingsUpdated.emit()

    @QtCore.pyqtSlot()
    def show_probe_layout(self):
        self.previewProbe.emit(self.probe_layout)

    @QtCore.pyqtSlot(str)
    def on_probe_layout_selected(self, name):
        if name not in ["", "[new]", "other..."]:
            probe_path = Path(self.gui.new_probe_files_path).joinpath(name)
            try:
                probe_layout = load_probe(probe_path)
                self.save_probe_selection(probe_layout, probe_path.name)

                total_channels = self.probe_layout["n_chan"]
                total_channels = self.estimate_total_channels(total_channels)
                self.n_chan_bin_input.setText(str(total_channels))

                self.enable_preview_probe()

                if self.check_settings():
                    self.enable_load()
            except MatReadError:
                logger.exception("Invalid probe file!")
                self.disable_load()
                self.disable_preview_probe()

        elif name == "[new]":
            dialog = ProbeBuilder(parent=self)

            if dialog.exec() == QtWidgets.QDialog.Accepted:
                probe_name = dialog.get_map_name()
                probe_layout = dialog.get_probe()

                probe_name = probe_name + ".prb"
                probe_prb = create_prb(probe_layout)
                probe_path = Path(self.gui.new_probe_files_path).joinpath(probe_name)
                with open(probe_path, "w+") as probe_file:
                    str_dict = pprint.pformat(
                        probe_prb, indent=4, compact=False,
                    )
                    str_prb = f"""channel_groups = {str_dict}"""
                    probe_file.write(str_prb)
                assert probe_path.exists()

                self.populate_probe_selector()
                self.probe_name = probe_path.name
                self.probe_layout_selector.setCurrentText(probe_name)
            else:
                self.probe_layout_selector.setCurrentIndex(0)
                self.disable_load()
                self.disable_preview_probe()

        elif name == "other...":
            file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
            probe_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=self,
                caption="Choose probe file...",
                filter="Probe Files (*.mat *.prb *.json)",
                #directory=os.path.expanduser("~"),
                options=file_dialog_options,
            )
            print(probe_path)
            if probe_path:
                try:
                    probe_path = Path(probe_path)
                    assert probe_path.exists()

                    probe_layout = load_probe(probe_path)

                    save_probe_file = QtWidgets.QMessageBox.question(
                        self,
                        "Save probe layout?",
                        "Would you like this probe layout to appear in the list of probe layouts next time?",
                        QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes,
                        QtWidgets.QMessageBox.Yes,
                    )

                    if save_probe_file == QtWidgets.QMessageBox.Yes:
                        probe_prb = create_prb(probe_layout)

                        probe_name = probe_path.with_suffix(".prb").name
                        new_probe_path = (
                            Path(self.gui.new_probe_files_path) / probe_name
                        )

                        if not new_probe_path.exists():
                            with open(new_probe_path, "w+") as probe_file:
                                str_dict = pprint.pformat(
                                    probe_prb, indent=4, compact=False
                                )
                                str_prb = f"""channel_groups = {str_dict}"""
                                probe_file.write(str_prb)

                            self.populate_probe_selector()
                            self.probe_layout_selector.setCurrentText(probe_name)
                            self.probe_name = probe_path.name
                        else:
                            logger.exception("Probe with the same name already exists.")

                    else:
                        self.save_probe_selection(probe_layout, probe_path.name)

                        total_channels = self.probe_layout["n_chan"]
                        total_channels = self.estimate_total_channels(total_channels)
                        self.n_chan_bin_input.setText(str(total_channels))

                        self.enable_preview_probe()

                        if self.check_settings():
                            self.enable_load()

                except AssertionError:
                    logger.exception(
                        "Please select a valid probe file (accepted types: *.prb, *.mat *.json)!"
                    )
                    self.disable_load()
                    self.disable_preview_probe()
            else:
                self.probe_layout_selector.setCurrentIndex(0)
                self.disable_load()
                self.disable_preview_probe()

    def save_probe_selection(self, layout, name):
        self.probe_layout = layout
        self.probe_name = name
        self.gui.qt_settings.setValue('probe_layout', layout)
        self.gui.qt_settings.setValue('probe_name', name)

    def on_data_dtype_selected(self, data_dtype):
        self.data_dtype = data_dtype
        if self.check_settings():
            self.enable_load()

    def on_device_selected(self, device):
        num_gpus = torch.cuda.device_count()
        selector_index = self.device_selector.currentIndex()
        if selector_index >= num_gpus:
            device_id = 'cpu'
        else:
            device_id = f'cuda:{selector_index}'
        self.gui.device = torch.device(device_id)
        if self.check_settings():
            self.enable_load()

    def populate_probe_selector(self):
        self.probe_layout_selector.clear()
        probe_folders = [self.gui.new_probe_files_path]
        
        probes_list = []
        for probe_folder in probe_folders:
            probes = os.listdir(os.fspath(probe_folder))
            probes = [
                probe
                for probe in probes
                if probe.endswith(".mat") or probe.endswith(".prb")
            ]
            probes_list.extend(probes)

        probes_list.sort(key=lambda f: os.path.splitext(f)[1])
        self.probe_layout_selector.addItems([""] + probes_list + ["[new]", "other..."])
        self._probes = probes_list

    def populate_dtype_selector(self):
        self.dtype_selector.clear()
        supported_dtypes = BinaryRWFile.supported_dtypes
        self.dtype_selector.addItems(supported_dtypes)

    def populate_device_selector(self):
        self.device_selector.clear()
        # Add gpus first, so that index in selector matches index in torch's
        # list of gpus.
        gpus = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
            ]
        self.device_selector.addItems(gpus + ['cpu'])

    def estimate_total_channels(self, num_channels):
        if self.data_file_path is not None:
            memmap_data = np.memmap(self.data_file_path, dtype=np.int16)
            data_size = memmap_data.size

            test_n_channels = np.arange(num_channels, num_channels + 31)
            remainders = np.remainder(data_size, test_n_channels)

            possible_results = test_n_channels[np.where(remainders == 0)]

            del memmap_data

            if possible_results.size == 0:
                return num_channels

            else:
                result = possible_results[0]
                logger.info(f"The correct number of channels has been estimated to be {possible_results[0]}.")
                if possible_results.size > 1:
                    logger.info(f"Other possibilities could be {possible_results[1:]}")

                return result

        else:
            return num_channels

    def preapre_for_new_context(self):
        pass

    def reset(self):
        self.data_file_path_input.clear()
        self.results_directory_input.clear()
        self.probe_layout_selector.setCurrentIndex(0)
        self.set_default_field_values()
        self.disable_preview_probe()
        self.disable_load()


class ExtraParametersWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.main_settings = parent
        self.input_fields = []
        for i, (var, name, _, _, _, _, default) in enumerate(_EXTRA_PARAMETERS):
            setattr(self, f'{var}_text', QtWidgets.QLabel(f'{name}'))
            setattr(self, f'{var}_input', QtWidgets.QLineEdit())
            getattr(self, f'{var}_input').parameter_index = i
            setattr(self, f'{var}', default)
            self.input_fields.append(getattr(self, f'{var}_input'))
        
        layout = QtWidgets.QGridLayout()
        row_count = 0
        for parameter_info in _EXTRA_PARAMETERS:
            row_count += 1
            var = parameter_info[0]
            layout.addWidget(getattr(self, f'{var}_text'), row_count, 0, 1, 3)
            layout.addWidget(getattr(self, f'{var}_input'), row_count, 3, 1, 2)
            getattr(self, f'{var}_input').textChanged.connect(self.update_parameter)

        self.setLayout(layout)

        center = QtWidgets.QDesktopWidget().availableGeometry().center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

    @QtCore.pyqtSlot()
    def update_parameter(self):
        parameter_index = self.sender().parameter_index
        parameter_info = _EXTRA_PARAMETERS[parameter_index]
        _check_parameter(self, self.main_settings, parameter_info)


def _check_parameter(sender_obj, main_obj, parameter_info):
    var, name, ptype, pmin, pmax, excl, _ = parameter_info
    try:
        value = getattr(sender_obj, f'{var}_input').text()
        if (value is None):
            v = value
        elif ptype is bool:
            print('inside bool check')
            if value.lower() == 'false':
                v = False
            elif value.lower() == 'true':
                v = True
            else:
                raise TypeError(f'{var} should be True or False.')
            print(f'value is: {value}, and v is: {v}')
        else:
            v = ptype(value)
        if not isinstance(v, bool):
            assert v >= pmin
            assert v <= pmax
            assert v not in excl
        setattr(sender_obj, var, v)

        if main_obj.check_settings():
            main_obj.enable_load()

    except ValueError:
        logger.exception(
            f"Invalid input!\n {name} must be of type: {ptype}."
        )
        main_obj.disable_load()

    except AssertionError:
        logger.exception(
            f"Invalid inputs!\n {name} must be in the range:\n"
            f"{pmin} <= {name} <= {pmax}, {name} != {excl}"
        )
        main_obj.disable_load()
