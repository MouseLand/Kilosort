import os
import pprint
from pathlib import Path

import numpy as np
from kilosort.gui.logger import setup_logger
from kilosort.gui.minor_gui_elements import ProbeBuilder, create_prb
from kilosort.io import load_probe
from PyQt5 import QtCore, QtWidgets
from scipy.io.matlab.miobase import MatReadError

logger = setup_logger(__name__)


DEFAULT_PARAMS = {
    "n_chan_bin"    : 385,
    "fs"            : 30000,
    "nt"            : 61,
    "Th"            : 6,
    "spkTh"         : 8,
    "Th_detect"     : 9,
    "nwaves"        : 6,
    "nskip"         : 25,
    "nt0min"        : 20,
    "NT"            : 60000,
    "nblocks"       : 5,
    "binning_depth" : 5,
    "sig_interp"    : 20,
}


class SettingsBox(QtWidgets.QGroupBox):
    settingsUpdated = QtCore.pyqtSignal()
    previewProbe = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.select_data_file = QtWidgets.QPushButton("Select Data File")
        self.data_file_path_input = QtWidgets.QLineEdit("")

        self.select_working_directory = QtWidgets.QPushButton(
            "Select Working Dir."
        )
        self.working_directory_input = QtWidgets.QLineEdit("")

        self.select_results_directory = QtWidgets.QPushButton(
            "Select Results Dir."
        )
        self.results_directory_input = QtWidgets.QLineEdit("")

        self.probe_layout_text = QtWidgets.QLabel("Select Probe Layout")
        self.probe_layout_selector = QtWidgets.QComboBox()
        self._probes = []
        self.populate_probe_selector()

        self.num_channels_text = QtWidgets.QLabel("number of channels")
        self.num_channels_input = QtWidgets.QLineEdit()

        self.sampling_frequency_text = QtWidgets.QLabel("sampling frequency")
        self.sampling_frequency_input = QtWidgets.QLineEdit()

        self.nt_text = QtWidgets.QLabel("nt")
        self.nt_input = QtWidgets.QLineEdit()

        self.NT_text = QtWidgets.QLabel("NT")
        self.NT_input = QtWidgets.QLineEdit()

        self.Th_text = QtWidgets.QLabel("Th")
        self.Th_input = QtWidgets.QLineEdit()

        self.spkTh_text = QtWidgets.QLabel("spkTh")
        self.spkTh_input = QtWidgets.QLineEdit()

        self.Th_detect_text = QtWidgets.QLabel("Th_detect")
        self.Th_detect_input = QtWidgets.QLineEdit()

        self.nwaves_text = QtWidgets.QLabel("nwaves")
        self.nwaves_input = QtWidgets.QLineEdit()

        self.nskip_text = QtWidgets.QLabel("nskip")
        self.nskip_input = QtWidgets.QLineEdit()

        self.nt0min_text = QtWidgets.QLabel("nt0min")
        self.nt0min_input = QtWidgets.QLineEdit()

        self.nblocks_text = QtWidgets.QLabel("nblocks")
        self.nblocks_input = QtWidgets.QLineEdit()

        self.binning_depth_text = QtWidgets.QLabel("binning depth")
        self.binning_depth_input = QtWidgets.QLineEdit()

        self.sig_interp_text = QtWidgets.QLabel("sig interp")
        self.sig_interp_input = QtWidgets.QLineEdit()

        self.load_settings_button = QtWidgets.QPushButton("LOAD")
        self.probe_preview_button = QtWidgets.QPushButton("Preview Probe")

        self.data_file_path = None
        self.working_directory_path = None
        self.results_directory_path = None
        self.probe_layout = None
        self.num_channels = None
        self.sampling_frequency = None
        self.nt = None
        self.NT = None
        self.Th = None
        self.spkTh = None
        self.Th_detect = None
        self.nwaves = None
        self.nskip = None
        self.nt0min = None
        self.nblocks = None
        self.binning_depth = None
        self.sig_interp = None

        self.input_fields = [
            self.data_file_path_input,
            self.results_directory_input,
            self.working_directory_input,
            self.probe_layout_selector,
            self.num_channels_input,
            self.sampling_frequency_input,
            self.nt_input,
            self.NT_input,
            self.Th_input,
            self.spkTh_input,
            self.Th_detect_input,
            self.nwaves_input,
            self.nskip_input,
            self.nt0min_input,
            self.nblocks_input,
            self.binning_depth_input,
            self.sig_interp_input,
        ]

        self.buttons = [
            self.load_settings_button,
            self.probe_preview_button,
            self.select_data_file,
            self.select_working_directory,
            self.select_results_directory,
        ]

        self.settings = {}

        self.setup()

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
        layout.addWidget(self.select_working_directory, row_count, 0, 1, 3)
        layout.addWidget(self.working_directory_input, row_count, 3, 1, 2)
        self.select_working_directory.clicked.connect(
            self.on_select_working_dir_clicked
        )
        self.working_directory_input.textChanged.connect(
            self.on_working_directory_changed
        )
        self.working_directory_input.editingFinished.connect(
            self.on_working_directory_changed
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
        layout.addWidget(self.num_channels_text, row_count, 0, 1, 3)
        layout.addWidget(self.num_channels_input, row_count, 3, 1, 2)
        self.num_channels_input.textChanged.connect(self.on_number_of_channels_changed)

        row_count += 1
        layout.addWidget(self.sampling_frequency_text, row_count, 0, 1, 3)
        layout.addWidget(self.sampling_frequency_input, row_count, 3, 1, 2)
        self.sampling_frequency_input.textChanged.connect(self.on_sampling_frequency_changed)

        row_count += 1
        layout.addWidget(self.nt_text, row_count, 0, 1, 3)
        layout.addWidget(self.nt_input, row_count, 3, 1, 2)
        self.nt_input.textChanged.connect(self.on_nt_changed)

        row_count += 1
        layout.addWidget(self.NT_text, row_count, 0, 1, 3)
        layout.addWidget(self.NT_input, row_count, 3, 1, 2)
        self.NT_input.textChanged.connect(self.on_NT_changed)

        row_count += 1
        layout.addWidget(self.Th_text, row_count, 0, 1, 3)
        layout.addWidget(self.Th_input, row_count, 3, 1, 2)
        self.Th_input.textChanged.connect(self.on_Th_changed)

        row_count += 1
        layout.addWidget(self.spkTh_text, row_count, 0, 1, 3)
        layout.addWidget(self.spkTh_input, row_count, 3, 1, 2)
        self.spkTh_input.textChanged.connect(self.on_spkTh_changed)

        row_count += 1
        layout.addWidget(self.Th_detect_text, row_count, 0, 1, 3)
        layout.addWidget(self.Th_detect_input, row_count, 3, 1, 2)
        self.Th_detect_input.textChanged.connect(self.on_Th_detect_changed)

        row_count += 1
        layout.addWidget(self.nwaves_text, row_count, 0, 1, 3)
        layout.addWidget(self.nwaves_input, row_count, 3, 1, 2)
        self.nwaves_input.textChanged.connect(self.on_nwaves_changed)

        row_count += 1
        layout.addWidget(self.nskip_text, row_count, 0, 1, 3)
        layout.addWidget(self.nskip_input, row_count, 3, 1, 2)
        self.nskip_input.textChanged.connect(self.on_nskip_changed)

        row_count += 1
        layout.addWidget(self.nt0min_text, row_count, 0, 1, 3)
        layout.addWidget(self.nt0min_input, row_count, 3, 1, 2)
        self.nt0min_input.textChanged.connect(self.on_nt0min_changed)

        row_count += 1
        layout.addWidget(self.nblocks_text, row_count, 0, 1, 3)
        layout.addWidget(self.nblocks_input, row_count, 3, 1, 2)
        self.nblocks_input.textChanged.connect(self.on_nblocks_changed)

        row_count += 1
        layout.addWidget(self.binning_depth_text, row_count, 0, 1, 3)
        layout.addWidget(self.binning_depth_input, row_count, 3, 1, 2)
        self.binning_depth_input.textChanged.connect(self.on_binning_depth_changed)

        row_count += 1
        layout.addWidget(self.sig_interp_text, row_count, 0, 1, 3)
        layout.addWidget(self.sig_interp_input, row_count, 3, 1, 2)
        self.sig_interp_input.textChanged.connect(self.on_sig_interp_changed)

        self.setLayout(layout)

        self.set_default_field_values(DEFAULT_PARAMS)

        self.update_settings()

    def set_default_field_values(self, default_params):
        if default_params is None:
            default_params = DEFAULT_PARAMS

        self.num_channels_input.setText(str(default_params["n_chan_bin"]))
        self.sampling_frequency_input.setText(str(default_params["fs"]))
        self.nt_input.setText(str(default_params["nt"]))
        self.NT_input.setText(str(default_params["NT"]))
        self.Th_input.setText(str(default_params["Th"]))
        self.spkTh_input.setText(str(default_params["spkTh"]))
        self.Th_detect_input.setText(str(default_params["Th_detect"]))
        self.nwaves_input.setText(str(default_params["nwaves"]))
        self.nskip_input.setText(str(default_params["nskip"]))
        self.nt0min_input.setText(str(default_params["nt0min"]))
        self.nblocks_input.setText(str(default_params["nblocks"]))
        self.binning_depth_input.setText(str(default_params["binning_depth"]))
        self.sig_interp_input.setText(str(default_params["sig_interp"]))

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

    def on_select_working_dir_clicked(self):
        file_dialog_options = QtWidgets.QFileDialog.DontUseNativeDialog
        working_dir_name = QtWidgets.QFileDialog.getExistingDirectoryUrl(
            parent=self,
            caption="Choose working directory...",
            directory=QtCore.QUrl(os.path.expanduser("~")),
            options=file_dialog_options,
        )
        if working_dir_name:
            self.working_directory_input.setText(working_dir_name.toLocalFile())

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

    def on_working_directory_changed(self):
        working_directory = Path(self.working_directory_input.text())
        try:
            assert working_directory.exists()

            self.working_directory_path = working_directory
            if self.check_settings():
                self.enable_load()
        except AssertionError:
            logger.exception("Please select an existing working directory!")
            self.disable_load()

    def on_results_directory_changed(self):
        results_directory = Path(self.results_directory_input.text())

        if not results_directory.exists():
            logger.warning(f"The results directory {results_directory} does not exist.")
            logger.warning("It will be (recursively) created upon data load.")

        self.results_directory_path = results_directory

        if self.check_settings():
            self.enable_load()
        else:
            self.disable_load()

    def on_data_file_path_changed(self):
        data_file_path = Path(self.data_file_path_input.text())
        try:
            assert data_file_path.exists()

            parent_folder = data_file_path.parent
            results_folder = parent_folder / "phy_export" / data_file_path.stem
            self.working_directory_input.setText(parent_folder.as_posix())
            self.results_directory_input.setText(results_folder.as_posix())

            self.data_file_path = data_file_path
            self.working_directory_path = parent_folder
            self.results_directory_path = results_folder

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
            "working_directory": self.working_directory_path,
            "results_directory": self.results_directory_path,
            "probe_layout": self.probe_layout,
            "n_chan_bin": self.num_channels,
            "fs": self.sampling_frequency,
            "nt": self.nt,
            "NT": self.NT,
            "spkTh": self.spkTh,
            "Th": self.Th,
            "nwaves": self.nwaves,
            "nskip": self.nskip,
            "nt0min": self.nt0min,
            "nblocks": self.nblocks,
            "binning_depth": self.binning_depth,
            "sig_interp": self.sig_interp,
        }

        return None not in self.settings.values()

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

                self.probe_layout = probe_layout
                total_channels = self.probe_layout["n_chan"]

                total_channels = self.estimate_total_channels(total_channels)

                self.num_channels_input.setText(str(total_channels))

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
                filter="Probe Files (*.mat *.prb)",
                directory=os.path.expanduser("~"),
                options=file_dialog_options,
            )
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

                        else:
                            logger.exception("Probe with the same name already exists.")

                    else:
                        self.probe_layout = probe_layout

                        total_channels = self.probe_layout["n_chan"]
                        total_channels = self.estimate_total_channels(total_channels)
                        self.num_channels_input.setText(str(total_channels))

                        self.enable_preview_probe()

                        if self.check_settings():
                            self.enable_load()

                except AssertionError:
                    logger.exception(
                        "Please select a valid probe file (accepted types: *.prb, *.mat)!"
                    )
                    self.disable_load()
                    self.disable_preview_probe()
            else:
                self.probe_layout_selector.setCurrentIndex(0)
                self.disable_load()
                self.disable_preview_probe()

    def on_number_of_channels_changed(self):
        try:
            number_of_channels = int(self.num_channels_input.text())
            assert number_of_channels > 0

            self.num_channels = number_of_channels

            if self.check_settings():
                self.enable_load()
        except ValueError:
            logger.exception("Invalid input!\nNo. of channels must be an integer!")
            self.disable_load()
        except AssertionError:
            logger.exception("Invalid input!\nNo. of channels must be > 0!")
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_sampling_frequency_changed(self):
        try:
            fs = int(self.sampling_frequency_input.text())
            assert fs > 0.0

            self.sampling_frequency = fs

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "sampling frequency must be an integer!"
            )
            self.disable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that fs > 0.0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_nt_changed(self):
        try:
            nt = int(self.nt_input.text())
            assert nt > 0

            self.nt = nt

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "nt must be an integer!"
            )
            self.disable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that nt > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_NT_changed(self):
        try:
            NT = int(self.NT_input.text())
            assert NT > 0

            self.NT = NT

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "NT must be an integer!"
            )
            self.disable_load()
        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that NT > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_spkTh_changed(self):
        try:
            spkTh = float(self.spkTh_input.text())
            assert spkTh > 0.0

            self.spkTh = spkTh

            if self.check_settings():
                self.enable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that spkTh > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_Th_detect_changed(self):
        try:
            Th_detect = float(self.Th_detect_input.text())
            assert Th_detect > 0.0

            self.Th_detect = Th_detect

            if self.check_settings():
                self.enable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that Th_detect > 0.0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_Th_changed(self):
        try:
            Th = float(self.Th_input.text())
            assert Th > 0.0

            self.Th = Th

            if self.check_settings():
                self.enable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that Th > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_nwaves_changed(self):
        try:
            nwaves = int(self.nwaves_input.text())
            assert nwaves > 0

            self.nwaves = nwaves

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "nwaves must be an integer!"
            )
            self.disable_load()
        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that nwaves > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_nskip_changed(self):
        try:
            nskip = int(self.nskip_input.text())
            assert nskip > 0

            self.nskip = nskip

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "nskip must be an integer!"
            )
            self.disable_load()
        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that nskip > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_nt0min_changed(self):
        try:
            nt0min = int(self.nt0min_input.text())
            assert nt0min > 0

            self.nt0min = nt0min

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "nt0min must be an integer!"
            )
            self.disable_load()
        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that nt0min > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_nblocks_changed(self):
        try:
            nblocks = int(self.nblocks_input.text())
            assert nblocks > 0

            self.nblocks = nblocks

            if self.check_settings():
                self.enable_load()

        except ValueError:
            logger.exception(
                "Invalid input!\n"
                "nblocks must be an integer!"
            )
            self.disable_load()
        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that nblocks > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_binning_depth_changed(self):
        try:
            binning_depth = float(self.binning_depth_input.text())
            assert binning_depth > 0.0

            self.binning_depth = binning_depth

            if self.check_settings():
                self.enable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that binning_depth > 0!"
            )
            self.disable_load()

    @QtCore.pyqtSlot()
    def on_sig_interp_changed(self):
        try:
            sig_interp = float(self.sig_interp_input.text())
            assert sig_interp > 0.0

            self.sig_interp = sig_interp

            if self.check_settings():
                self.enable_load()

        except AssertionError:
            logger.exception(
                "Invalid inputs!\n"
                "Check that sig_interp > 0!"
            )
            self.disable_load()

    def populate_probe_selector(self):
        self.probe_layout_selector.clear()

        probe_folders = [self.gui.new_probe_files_path]

        probes_list = []
        for probe_folder in probe_folders:
            probes = os.listdir(probe_folder)
            probes = [
                probe
                for probe in probes
                if probe.endswith(".mat") or probe.endswith(".prb")
            ]
            probes_list.extend(probes)

        self.probe_layout_selector.addItems([""] + probes_list + ["[new]", "other..."])
        self._probes = probes_list

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
        self.working_directory_input.clear()
        self.results_directory_input.clear()
        self.probe_layout_selector.setCurrentIndex(0)
        self.set_default_field_values(None)
        self.disable_preview_probe()
        self.disable_load()
