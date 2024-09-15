import os
import pprint
from pathlib import Path
import json
import ast

import numpy as np
import torch
from qtpy import QtCore, QtWidgets, QtGui
from scipy.io.matlab.miobase import MatReadError

from kilosort.gui.logger import setup_logger
from kilosort.gui.minor_gui_elements import ProbeBuilder, create_prb
from kilosort.io import load_probe, BinaryRWFile
from kilosort.parameters import MAIN_PARAMETERS, EXTRA_PARAMETERS


logger = setup_logger(__name__)

_DEFAULT_DTYPE = 'int16'
_ALLOWED_FILE_TYPES = ['.bin', '.dat', '.bat', '.raw']  # For binary data

class SettingsBox(QtWidgets.QGroupBox):
    settingsUpdated = QtCore.Signal()
    previewProbe = QtCore.Signal(object)
    dataChanged = QtCore.Signal()

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent
        self.load_enabled = False
        self.use_file_object = False
        
        self.select_data_file = QtWidgets.QPushButton("Select Binary File")
        self.data_file_path = self.gui.data_path
        if self.data_file_path is not None:
            self.data_file_path = self.data_file_path.resolve()
        self.data_file_path_input = QtWidgets.QLineEdit(
            self.data_file_path.as_posix()
            if self.data_file_path is not None else None
            )
        
        if not self.gui.qt_settings.contains('last_data_location'):
            self.gui.qt_settings.setValue(
                'last_data_location', Path('~').expanduser().as_posix()
                )

        self.convert_data_button = QtWidgets.QPushButton("Convert to Binary")

        self.results_directory_path = self.gui.results_directory
        if self.data_file_path is not None and self.results_directory_path is None:
            self.results_directory_path = self.data_file_path.parent.joinpath('kilosort4/')

        self.select_results_directory = QtWidgets.QPushButton(
            "Select Results Dir."
        )
        self.results_directory_input = QtWidgets.QLineEdit(
            self.results_directory_path.as_posix()
            if self.results_directory_path is not None else None
            )

        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()
        self._probes = []
        self.populate_probe_selector()

        self.bad_channels_text = QtWidgets.QLabel("Excluded channels:")
        self.bad_channels_input = QtWidgets.QLineEdit()
        if self.gui.qt_settings.contains('bad_channels'):
            bad_channels = self.gui.qt_settings.value('bad_channels')
            if bad_channels is not None:
                # List of ints gets cached as list of strings, so have to convert.
                self.bad_channels = [int(s) for s in bad_channels]
                self.bad_channels_input.setText(str(self.bad_channels))
            else:
                self.bad_channels = []
        else:
            self.bad_channels = []

        self.dtype_selector_text = QtWidgets.QLabel("Data dtype:")
        self.dtype_selector = QtWidgets.QComboBox()
        self.populate_dtype_selector()

        self.device_selector_text = QtWidgets.QLabel('PyTorch device:')
        self.device_selector = QtWidgets.QComboBox()
        self.populate_device_selector()

        generated_inputs = []
        for k, p in MAIN_PARAMETERS.items():
            setattr(self, f'{k}_text', QtWidgets.QLabel(f'{p["gui_name"]}'))
            getattr(self, f'{k}_text').setToolTip(f'{p["description"]}')
            setattr(self, f'{k}_input', QtWidgets.QLineEdit())
            getattr(self, f'{k}_input').var_name = k
            setattr(self, f'{k}', p['default'])
            generated_inputs.append(getattr(self, f'{k}_input'))
        self.data_dtype = _DEFAULT_DTYPE

        self.extra_parameters_window = ExtraParametersWindow(self)
        self.extra_parameters_button = QtWidgets.QPushButton('Extra settings')
        self.import_settings_button = QtWidgets.QPushButton('Import')
        self.export_settings_button = QtWidgets.QPushButton('Export')

        self.load_settings_button = QtWidgets.QPushButton("LOAD")
        self.probe_preview_button = QtWidgets.QPushButton("Preview Probe")
        self.probe_layout = self.gui.probe_layout
        self.probe_name = self.gui.probe_name
  
        self.input_fields = [
            self.data_file_path_input,
            self.results_directory_input,
            self.probe_layout_selector,
            self.dtype_selector,
            self.device_selector
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
        if self.check_valid_binary_path(self.data_file_path):
            if self.check_settings():
                self.enable_load()


    def setup(self):
        self.setTitle("Settings")

        layout = QtWidgets.QGridLayout()
        row_count = 0
        rspan = 3
        col1 = 0
        col2 = 3
        cspan1 = 3
        cspan2 = 3
        dbl = cspan1 + cspan2

        font = self.load_settings_button.font()
        font.setPointSize(18)
        self.load_settings_button.setFont(font)
        self.load_settings_button.setDisabled(True)
        self.load_settings_button.clicked.connect(self.update_settings)
        layout.addWidget(self.load_settings_button, row_count, col1, rspan, dbl)


        ### Data selection / conversion
        row_count += rspan
        layout.addWidget(self.select_data_file, row_count, col1, rspan, cspan1)
        layout.addWidget(self.convert_data_button, row_count, col2, rspan, cspan2)
        self.convert_data_button.clicked.connect(self.open_data_converter)
        row_count += rspan
        layout.addWidget(self.data_file_path_input, row_count, col1, rspan, dbl)
        self.select_data_file.clicked.connect(self.on_select_data_file_clicked)
        self.data_file_path_input.editingFinished.connect(
            self.on_data_file_path_changed
        )


        # Add small vertical space for visual grouping
        row_count += rspan
        layout.addWidget(QtWidgets.QWidget(), row_count, 0, 1, dbl)
        row_count += 1

        ### Results path
        layout.addWidget(
            self.select_results_directory, row_count, col1, rspan, cspan1
            )
        row_count += rspan
        layout.addWidget(
            self.results_directory_input, row_count, col1, rspan, dbl
            )
        self.select_results_directory.clicked.connect(
            self.on_select_results_dir_clicked
        )
        self.results_directory_input.editingFinished.connect(
            self.on_results_directory_changed
        )


        # Add small vertical space for visual grouping
        row_count += rspan
        layout.addWidget(QtWidgets.QWidget(), row_count, 0, 1, dbl)
        row_count += 1

        ### Probe selection
        layout.addWidget(self.probe_layout_text, row_count, col1, rspan, cspan1)
        layout.addWidget(
            self.probe_preview_button, row_count, col2, rspan, cspan2)
        self.probe_preview_button.setDisabled(True)
        self.probe_preview_button.clicked.connect(self.show_probe_layout)

        row_count += rspan
        layout.addWidget(self.probe_layout_selector, row_count, col1, rspan, dbl)
        self.probe_layout_selector.currentTextChanged.connect(
            self.on_probe_layout_selected
        )

        row_count += rspan
        layout.addWidget(self.bad_channels_text, row_count, col1, rspan, cspan1)
        layout.addWidget(self.bad_channels_input, row_count, col2, rspan, cspan2)
        self.bad_channels_input.editingFinished.connect(self.update_bad_channels)
        self.bad_channels_text.setToolTip(
            "A list of channel indices (rows in the binary file) that should "
            "not be included in sorting.\nListing channels here is equivalent to "
            "excluding them from the probe dictionary."
            )


        # Add small vertical space for visual grouping
        row_count += rspan
        layout.addWidget(QtWidgets.QWidget(), row_count, 0, 1, dbl)
        row_count += 1

        ### Settings
        layout.addWidget(self.dtype_selector_text, row_count, col1, rspan, cspan1)
        layout.addWidget(self.dtype_selector, row_count, col2, rspan, cspan2)
        self.dtype_selector.currentTextChanged.connect(
            self.on_data_dtype_selected
        )

        row_count += rspan
        layout.addWidget(self.device_selector_text, row_count, col1, rspan, cspan1)
        layout.addWidget(self.device_selector, row_count, col2, rspan, cspan2)
        self.device_selector.currentTextChanged.connect(
            self.on_device_selected
        )

        for k in list(MAIN_PARAMETERS.keys()):
            row_count += rspan
            layout.addWidget(
                getattr(self, f'{k}_text'), row_count, col1, rspan, cspan1
                )
            layout.addWidget(
                getattr(self, f'{k}_input'), row_count, col2, rspan, cspan2
                )
            inp = getattr(self, f'{k}_input')
            inp.editingFinished.connect(self.update_parameter)

        row_count += rspan
        layout.addWidget(
            self.extra_parameters_button, row_count, col1, rspan, dbl
            )
        self.extra_parameters_button.clicked.connect(
            lambda x: self.extra_parameters_window.show()
            )
        
        row_count += rspan
        layout.addWidget(
            self.import_settings_button, row_count, col1, rspan, cspan1
            )
        self.import_settings_button.clicked.connect(self.import_settings)
        layout.addWidget(
            self.export_settings_button, row_count, col2, rspan, cspan2
            )
        self.export_settings_button.clicked.connect(self.export_settings)


        self.setLayout(layout)
        self.set_cached_field_values()
        self.update_settings()

    def set_default_field_values(self):
        self.dtype_selector.setCurrentText(_DEFAULT_DTYPE)
        epw = self.extra_parameters_window
        for k, p in MAIN_PARAMETERS.items():
            getattr(self, f'{k}_input').setText(str(p['default']))
            getattr(self, f'{k}_input').editingFinished.emit()
        for k, p in EXTRA_PARAMETERS.items():
            getattr(epw, f'{k}_input').setText(str(p['default']))
            getattr(epw, f'{k}_input').editingFinished.emit()

    def set_cached_field_values(self):
        # Only run during setup, so that resetting gui always goes to defaults.
        if self.gui.qt_settings.contains('data_dtype'):
            dtype = self.gui.qt_settings.value('data_dtype')
            self.dtype_selector.setCurrentText(dtype)
        epw = self.extra_parameters_window
        for k, p in MAIN_PARAMETERS.items():
            if self.gui.qt_settings.contains(k):
                # Use cached value
                d = str(self.gui.qt_settings.value(k))
            else:
                # Use default value
                d = str(p['default'])
            getattr(self, f'{k}_input').setText(d)
            getattr(self, f'{k}_input').editingFinished.emit()
        for k, p in EXTRA_PARAMETERS.items():
            if self.gui.qt_settings.contains(k):
                # Use cached value
                v = self.gui.qt_settings.value(k)
                if k == 'drift_smoothing':
                    # List of floats gets cached as list of strings, so
                    # have to convert back.
                    d = str([float(s) for s in v])
                else:
                    d = str(v)
            else:
                # Use default value
                d = str(p['default'])
            getattr(epw, f'{k}_input').setText(d)
            getattr(epw, f'{k}_input').editingFinished.emit()

    @QtCore.Slot()
    def import_settings(self):
        # 1) open file dialog
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        settings_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Choose .json file to load settings from...",
            directory=os.path.expanduser("~"),
            filter='*.json',
            options=file_dialog_options,
        )
        # 2) load to dict through json
        with open(settings_file_name, 'r') as f:
            settings = json.load(f)

        # 3) loop through settings to update values
        epw = self.extra_parameters_window
        for k, v in settings['main'].items():
            getattr(self, f'{k}_input').setText(str(v))
            getattr(self, f'{k}_input').editingFinished.emit()
        for k, v in settings['extra'].items():
            getattr(epw, f'{k}_input').setText(str(v))
            getattr(epw, f'{k}_input').editingFinished.emit()

        self.dtype_selector.setCurrentIndex(settings['misc']['dtype_idx'])
        self.probe_layout_selector.setCurrentIndex(settings['misc']['probe_idx'])
        self.device_selector.setCurrentIndex(settings['misc']['device_idx'])


    @QtCore.Slot()
    def export_settings(self):
        # 1) dump parameters to dict
        #       don't save entire settings dict, other stuff is data-specific
        settings = {'main': {}, 'extra': {}, 'misc': {}}
        for k in list(MAIN_PARAMETERS.keys()):
            settings['main'][k] = getattr(self, k)
        for k in list(EXTRA_PARAMETERS.keys()):
            settings['extra'][k] = getattr(self.extra_parameters_window, k)
        
        # Add dtype, probe, pytorch device
        settings['misc']['dtype_idx'] = self.dtype_selector.currentIndex()
        settings['misc']['probe_idx'] = self.probe_layout_selector.currentIndex()
        settings['misc']['device_idx'] = self.device_selector.currentIndex()

        # 3) open save dialog        
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        settings_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption="Specify a .json file to save settings to...",
            directory=os.path.expanduser("~"),
            filter='*.json',
            options=file_dialog_options,
        )

        # 2) format as .json
        with open(settings_file_name, 'w+') as f:
            f.write(json.dumps(settings))


    def on_select_data_file_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Choose data file to load...",
            directory=self.gui.qt_settings.value('last_data_location'),
            options=file_dialog_options,
        )
        if data_file_name:
            self.data_file_path_input.setText(data_file_name)
            self.data_file_path_input.editingFinished.emit()
            # Cache data folder for the next time a file is selected
            data_folder = Path(data_file_name).parent.as_posix()
            self.gui.qt_settings.setValue('last_data_location', data_folder)

    def set_data_file_path_from_drag_and_drop(self, filename):
        if Path(filename).suffix in ['.bin', '.dat', '.bat', '.raw']:
            self.data_file_path_input.setText(filename)
            self.data_file_path_input.editingFinished.emit()
            logger.info(f"File at location: {filename} is ready to load!")

        else:
            message = (
                "Only .bin, .dat, .bat, and .raw files accepted as binary, "
                "the data conversion tool will be opened instead..."
                )
            QtWidgets.QMessageBox.warning(
                self.parent(),
                "Unrecognized file type",
                message,
                QtWidgets.QMessageBox.StandardButton.Ok,
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            self.gui.converter.filename = filename
            self.gui.converter.filename_input.setText(filename)
            self.gui.converter.show()

    def open_data_converter(self):
        self.gui.converter.show()

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
            self.results_directory_input.editingFinished.emit()

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
            assert self.check_valid_binary_path(data_file_path)

            parent_folder = data_file_path.parent
            results_folder = parent_folder / "kilosort4"
            self.results_directory_input.setText(results_folder.as_posix())
            self.results_directory_input.editingFinished.emit()
            self.data_file_path = data_file_path
            self.gui.qt_settings.setValue('data_file_path', data_file_path)

            if self.check_settings():
                self.enable_load()
                self.dataChanged.emit()

        except AssertionError:
            logger.exception("Please select a valid binary file path.")
            self.disable_load()

    def check_valid_binary_path(self, filename):
        if filename is None:
            print('Binary path is None.')
            return False
        else:
            f = Path(filename)
            if f.exists() and f.is_file():
                if f.suffix in _ALLOWED_FILE_TYPES or self.use_file_object:
                    return True
                else:
                    print(f'Binary file has invalid suffix. Must be {_ALLOWED_FILE_TYPES}')
                    return False
            else:
                print('Binary file does not exist at that path.')
                return False

    def disable_all_input(self, value):
        for button in self.buttons:
            button.setDisabled(value)
        for field in self.input_fields:
            field.setDisabled(value)

    def enable_load(self):
        self.load_settings_button.setEnabled(True)
        self.load_enabled = True

    def disable_load(self):
        self.load_settings_button.setDisabled(True)
        self.load_disabled = True

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
        for k in list(MAIN_PARAMETERS.keys()):
            self.settings[k] = getattr(self, k)
        for k in list(EXTRA_PARAMETERS.keys()):
            self.settings[k] = getattr(self.extra_parameters_window, k)

        if not self.check_valid_binary_path(self.data_file_path):
            return False

        none_allowed = [
            'dmin', 'nt0min', 'max_channel_distance', 'x_centers',
            'shift', 'scale'
            ]
        for k, v in self.settings.items():
            if v is None and k not in none_allowed:
                logger.info(f'`None` not allowed for parameter {k}.')
                return False
        return True
    
    @QtCore.Slot()
    def update_parameter(self):
        parameter_key = self.sender().var_name
        parameter_info = MAIN_PARAMETERS[parameter_key]
        _check_parameter(self, self, parameter_key, parameter_info)

    @QtCore.Slot()
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

    def get_probe_template_args(self):
        epw = self.extra_parameters_window
        template_args = [
            epw.nearest_chans, epw.dmin, epw.dminx, 
            epw.max_channel_distance, epw.x_centers, self.gui.device
            ]
        return template_args

    @QtCore.Slot()
    def show_probe_layout(self):
        if self.check_settings:
            self.previewProbe.emit(self.get_probe_template_args())
        else:
            logger.info("Cannot preview probe layout, invalid settings.")

    @QtCore.Slot(str)
    def on_probe_layout_selected(self, name):
        if name not in ["", "[new]", "other..."]:
            probe_path = Path(self.gui.new_probe_files_path).joinpath(name)
            try:
                probe_layout = load_probe(probe_path)
                self.save_probe_selection(probe_layout, probe_path.name)

                total_channels = self.probe_layout["n_chan"]
                total_channels = self.estimate_total_channels(total_channels)
                self.n_chan_bin_input.setText(str(total_channels))
                self.n_chan_bin_input.editingFinished.emit()

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
                        self.n_chan_bin_input.editingFinished.emit()

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

    def get_bad_channels(self):
        text = self.bad_channels_input.text()
        text = text.replace(']','').replace('[','').replace(' ','')
        if len(text) > 0:
            bad_channels = [int(s) for s in text.split(',')]
        else:
            bad_channels = []
        
        return bad_channels

    def set_bad_channels(self, bad_channels):
        self.bad_channels_input.setText(str(bad_channels))
        self.bad_channels_input.editingFinished.emit()

    @QtCore.Slot()
    def update_bad_channels(self):
        # Remove brackets and white space if present, convert to list of ints.
        self.bad_channels = self.get_bad_channels()
        self.gui.qt_settings.setValue('bad_channels', self.bad_channels)

        # Trigger update so that probe layout in main gets updated, then
        # refresh probe view.
        self.update_settings()
        self.previewProbe.emit(self.get_probe_template_args)


    def on_data_dtype_selected(self, data_dtype):
        self.data_dtype = data_dtype
        self.gui.qt_settings.setValue('data_dtype', data_dtype)
        if self.check_settings():
            self.enable_load()

    def on_device_selected(self, device):
        num_gpus = torch.cuda.device_count()
        selector_index = self.device_selector.currentIndex()
        if (selector_index >= num_gpus) or (selector_index < 0):
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

    def clear_probe_selection(self):
        self.probe_layout_selector.setCurrentIndex(0)
        self.gui.qt_settings.setValue('probe_layout', None)
        self.gui.qt_settings.setValue('probe_name', None)
        self.disable_preview_probe()

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
        if self.check_valid_binary_path(self.data_file_path):
            if self.use_file_object:
                return self.gui.file_object.c
            
            memmap_data = np.memmap(self.data_file_path, dtype=self.data_dtype)
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
        self.data_file_path = None
        self.gui.qt_settings.setValue('data_file_path', None)
        self.results_directory_input.clear()
        self.results_directory_path = None
        self.gui.qt_settings.setValue('results_dir', None)
        self.clear_probe_selection()
        self.set_default_field_values()
        self.disable_load()

    def check_load(self):
        if self.check_settings():
            self.enable_load()
        else:
            self.disable_load()


class ExtraParametersWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.main_settings = parent
        self.input_fields = []
        generated_inputs = []
        for k, p in EXTRA_PARAMETERS.items():
            setattr(self, f'{k}_text', QtWidgets.QLabel(f'{p["gui_name"]}'))
            getattr(self, f'{k}_text').setToolTip(f'{p["description"]}')
            setattr(self, f'{k}_input', QtWidgets.QLineEdit())
            getattr(self, f'{k}_input').var_name = k
            setattr(self, f'{k}', p['default'])
            generated_inputs.append(getattr(self, f'{k}_input'))
        
        layout = QtWidgets.QGridLayout()
        row_count = 0
        col = 0
        heading = None
        self.heading_labels = []
        for k, p in EXTRA_PARAMETERS.items():
            if p['step'] != heading:
                heading = p['step']
                heading_label = QtWidgets.QLabel(heading)
                heading_label.setFont(QtGui.QFont('Arial', 14))
                self.heading_labels.append(heading_label)
                if len(self.heading_labels) % 4 == 0:
                    hgap = QtWidgets.QLabel('         ')
                    layout.addWidget(hgap, row_count, 5, 1, 1)
                    row_count = 0
                    col = 6
                row_count += 1
                gap = QtWidgets.QLabel('')
                gap.setFont(QtGui.QFont('Arial', 4))
                layout.addWidget(gap, row_count, col, 1, 5)
                row_count += 1
                layout.addWidget(self.heading_labels[-1], row_count, col, 1, 5)
            row_count += 1
            layout.addWidget(getattr(self, f'{k}_text'), row_count, col, 1, 3)
            layout.addWidget(getattr(self, f'{k}_input'), row_count, col+3, 1, 2)
            inp = getattr(self, f'{k}_input')
            inp.editingFinished.connect(self.update_parameter)

        self.setLayout(layout)

        center = QtWidgets.QApplication.screens()[0].availableGeometry().center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

    @QtCore.Slot()
    def update_parameter(self):
        parameter_key = self.sender().var_name
        parameter_info = EXTRA_PARAMETERS[parameter_key]
        _check_parameter(self, self.main_settings, parameter_key, parameter_info)


def _check_parameter(sender_obj, main_obj, k, p):
    reset = True
    try:
        value = getattr(sender_obj, f'{k}_input').text()
        if (value is None) or (str(value).lower() == 'none'):
            v = None
        else:
            v = _str_to_type(value, p['type'])
            if isinstance(v, bool) or isinstance(v, list):
                pass
            else:
                assert v >= p['min']
                assert v <= p['max']
                assert v not in p['exclude']
        setattr(sender_obj, k, v)
        main_obj.gui.qt_settings.setValue(k, v)
        reset = False

        if main_obj.check_settings() and not main_obj.load_enabled:
            main_obj.enable_load()

    except ValueError:
        logger.exception(
            f"Invalid input!\n {p['gui_name']} must be of type: {p['type']}."
        )
        main_obj.disable_load()

    except AssertionError:
        logger.exception(
            f"Invalid inputs!\n {p['gui_name']} must be in the range:\n"
            f"{p['min']} <= {p['gui_name']} <= {p['max']},\n"
            f"{p['gui_name']} != {p['exclude']}"
        )
        main_obj.disable_load()

    finally:
        if reset:
            # Invalid input, change back to what it was before.
            v = getattr(sender_obj, k)
            getattr(sender_obj, f'{k}_input').setText(str(v))


def _str_to_type(string, dtype):
    if dtype is bool:
        if string.lower() == 'false':
            v = False
        elif string.lower() == 'true':
            v = True
        else:
            raise TypeError(f'{string} should be True or False for bool.')
    elif dtype is list:
        v = ast.literal_eval(string)
    else:
        v = dtype(string)
    return v
