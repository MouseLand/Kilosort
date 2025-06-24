import importlib
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets
from kilosort.io import (
    RecordingExtractorAsArray, BinaryRWFile, spikeinterface_to_binary, load_probe
    )
from kilosort.gui.logger import setup_logger

logger = setup_logger(__name__)

# TODO: Add others after testing them. Full list here:
#       https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html
_SPIKEINTERFACE_IMPORTS = {
    'NWB': 'read_nwb_recording',
    'Blackrock': 'read_blackrock',
    'Neuralynx': 'read_neuralynx',
    'Openephys': 'read_openephys',
    'Intan': 'read_intan'
}


class DataConversionBox(QtWidgets.QWidget):
    disableInput = QtCore.Signal(bool)
    fileObjectLoaded = QtCore.Signal(bool)

    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.setWindowTitle('Convert data (REQUIRES SPIKEINTERFACE)')
        self.filename = None
        self.filetype = None
        self.stream_id = None
        self.stream_name = None
        self.data_dtype = None
        self.file_object = None
        self.conversion_thread = None

        if not self.gui.qt_settings.contains('last_data_location'):
            self.gui.qt_settings.setValue(
                'last_data_location', Path('~').expanduser().as_posix()
                )

        layout = QtWidgets.QGridLayout()

        # Top left: filename and filetype selectors
        self.file_button = QtWidgets.QPushButton('Select file')
        self.file_button.clicked.connect(self.select_file)
        self.or_label = QtWidgets.QLabel('OR')
        self.or_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
            )
        self.folder_button = QtWidgets.QPushButton('Select folder')
        self.folder_button.clicked.connect(self.select_folder)
        self.filename_input = QtWidgets.QLineEdit()
        self.filename_input.setReadOnly(True)
        layout.addWidget(self.file_button, 0, 0, 1, 2)
        layout.addWidget(self.or_label, 0, 2, 1, 2)
        layout.addWidget(self.folder_button, 0, 4, 1, 2)
        layout.addWidget(self.filename_input, 1, 0, 1, 6)

        self.filetype_text = QtWidgets.QLabel('File type:')
        self.filetype_selector = QtWidgets.QComboBox()
        self.filetype_selector.addItems([''] + list(_SPIKEINTERFACE_IMPORTS.keys()))
        self.filetype_selector.currentTextChanged.connect(self.select_filetype)
        layout.addWidget(self.filetype_text, 2, 0, 1, 3)
        layout.addWidget(self.filetype_selector, 2, 3, 1, 3)
        if self.gui.qt_settings.contains('filetype'):
            filetype = self.gui.qt_settings.value('filetype')
            if filetype is not None:
                self.filetype_selector.setCurrentText(filetype)

        # Top right: kwargs for spikeinterface loader, dtype
        self.stream_id_text = QtWidgets.QLabel('stream_id (optional)')
        self.stream_id_input = QtWidgets.QLineEdit()
        self.stream_id_input.editingFinished.connect(self.set_stream_id)
        if self.gui.qt_settings.contains('stream_id'):
            stream_id = self.gui.qt_settings.value('stream_id')
            if stream_id is not None:
                self.stream_id_input.setText(stream_id)

        self.stream_name_text = QtWidgets.QLabel('stream_name (optional)')
        self.stream_name_input = QtWidgets.QLineEdit()
        self.stream_name_input.editingFinished.connect(self.set_stream_name)
        if self.gui.qt_settings.contains('stream_name'):
            stream_name = self.gui.qt_settings.value('stream_name')
            if stream_name is not None:
                self.stream_name_input.setText(stream_name)

        self.dtype_text = QtWidgets.QLabel("Data dtype (required):")
        self.dtype_note = QtWidgets.QLabel(
            "NOTE: this is the dtype for the new .bin file.\nIf the existing file"
            " is not the same dtype, data may be altered."
        )
        self.dtype_selector = QtWidgets.QComboBox()
        supported_dtypes = BinaryRWFile.supported_dtypes
        self.dtype_selector.addItems([''] + supported_dtypes)
        self.dtype_selector.currentTextChanged.connect(self.set_dtype)
        if self.gui.qt_settings.contains('converter_dtype'):
            dtype = self.gui.qt_settings.value('converter_dtype')
            if dtype is not None:
                self.dtype_selector.setCurrentText(dtype)
    
        layout.addWidget(self.stream_id_text, 0, 10, 1, 3)
        layout.addWidget(self.stream_id_input, 0, 13, 1, 3)
        layout.addWidget(self.stream_name_text, 1, 10, 1, 3)
        layout.addWidget(self.stream_name_input, 1, 13, 1, 3)
        layout.addWidget(self.dtype_text, 2, 10, 1, 3)
        layout.addWidget(self.dtype_selector, 2, 13, 1, 3)
        layout.addWidget(self.dtype_note, 3, 10, 2, 6)

        # Bottom: convert to binary or load with wrapper
        self.spacer = QtWidgets.QLabel('         ')
        self.wrapper_button = QtWidgets.QPushButton('Load As Wrapper')
        self.wrapper_button.clicked.connect(self.load_as_wrapper)
        self.wrapper_button.setToolTip(
            "'Load as Wrapper' will load the data without converting it, which is much"
            " faster.\n However, sorting will be slower and the results will not "
            "be browsable in Phy.\n We recommend using this option to check that "
            "your data is being loaded correctly."
        )
        self.convert_button = QtWidgets.QPushButton('Convert to Binary')
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_input = QtWidgets.QLineEdit()
        self.convert_input.setPlaceholderText(
            "NOTE: Must include .bin extension in filename."
            )
        self.convert_input.setReadOnly(True)
        self.convert_button.setToolTip(
            "'Convert to Binary' will copy the data to a new .bin file.\n"
            "This process is slow for large recordings, but results will be "
            "browsable in Phy.\n We recommend this option for most use cases.\n"
        )

        layout.addWidget(self.spacer, 5, 0, 2, 6)
        layout.addWidget(self.wrapper_button, 6, 0, 1, 6)
        layout.addWidget(self.convert_button, 7, 0, 1, 6)
        layout.addWidget(self.convert_input, 7, 6, 1, 10)


        # TODO: remove this after bug is fixed
        self.crash_warning = QtWidgets.QLabel(
            "NOTE: we are aware of a bug that causes the GUI to crash after data"
            " is converted.\n If you encounter this issue, the new .bin file"
            " should still have been created,\n so you can open the GUI again "
            "and select it like any other binary file."
        )
        layout.addWidget(self.crash_warning, 8, 0, 3, 16)

        self.setLayout(layout)


    @QtCore.Slot()
    def select_file(self):
        options = QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileNames(
            parent=self,
            caption="Choose file to load data from...",
            directory=self.gui.qt_settings.value('last_data_location'),
            options=options,
        )
        self.filename = filename
        if not isinstance(self.filename, list):
            self.filename = [self.filename]
        self.filename_input.setText(str(filename))
        data_location = Path(filename[0]).parent.as_posix()
        self.gui.qt_settings.setValue('last_data_location', data_location)

    @QtCore.Slot()
    def select_folder(self):
        options = QtWidgets.QFileDialog.DontUseNativeDialog
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self,
            caption="Choose directory to load data from...",
            directory=self.gui.qt_settings.value('last_data_location'),
            options=options,
        )
        self.filename = directory
        self.filename_input.setText(directory)
        data_location = Path(directory).as_posix()
        self.gui.qt_settings.setValue('last_data_location', data_location)

    @QtCore.Slot()
    def select_filetype(self):
        self.filetype = self.filetype_selector.currentText()
        self.gui.qt_settings.setValue('filetype', self.filetype)

    @QtCore.Slot()
    def set_stream_id(self):
        stream_id = self.stream_id_input.text()
        if stream_id is None or stream_id == '':
            stream_id = None
        else:
            stream_id = stream_id
        self.stream_id = stream_id
        self.gui.qt_settings.setValue('stream_id', stream_id)

    @QtCore.Slot()
    def set_stream_name(self):
        stream_name = self.stream_name_input.text()
        if stream_name == '':
            stream_name = None
        self.stream_name = stream_name
        self.gui.qt_settings.setValue('stream_name', stream_name)

    @QtCore.Slot()
    def set_dtype(self):
        self.data_dtype = self.dtype_selector.currentText()
        self.gui.qt_settings.setValue('converter_dtype', self.data_dtype)

    def load_cached_wrapper(self):
        logger.info('Loading cached file with external format as wrapper...')
        filename = self.gui.qt_settings.value('data_file_path')
        self.filename = [str(f) for f in filename]
        self.filetype = self.gui.qt_settings.value('filetype')
        self.data_dtype = self.gui.qt_settings.value('converter_dtype')
        self.stream_id = self.gui.qt_settings.value('stream_id')
        self.stream_name = self.gui.qt_settings.value('stream_name')
        self.load_as_wrapper(keep_probe=True)

    @QtCore.Slot()
    def load_as_wrapper(self, keep_probe=False):
        if None in [self.filename, self.filetype, self.data_dtype]:
            logger.exception(
                'File name, file type, and data dtype must be specified.'
            )
            logger.exception(
                f'File name: {self.filename}\nFile type: {self.filetype}\n'
                f'Dtype: {self.data_dtype}'
            )
            return

        self.hide()
        recording = get_recording_extractor(
            self.filename, self.filetype, self.stream_id, self.stream_name
            )
        f = RecordingExtractorAsArray(recording)
        self.file_object = f
        update_settings_box(self.gui.settings_box, f.c, f.fs, self.data_dtype)
        self.fileObjectLoaded.emit(True)

        # The np.dtype() wrapper is here because multiple strings can represent
        # the same type, e.g. '<f4' for 'float32'
        if np.dtype(f.dtype) != np.dtype(self.data_dtype):
            logger.warn(
                f'NOTE: SpikeInterface recording has dtype: {f.dtype},\n'
                f'but dtype: {self.data_dtype} was selected in GUI.'
                )

        # TODO: let user decide whether probe info should be exported, and
        #       set path for saving it.
        from probeinterface import write_prb
        try:
            pg = recording.get_probegroup()
            probe_filename = Path(self.filename[0]).parent / 'probe.prb'
            write_prb(probe_filename, pg)
            add_probe_to_settings(self.gui.settings_box, probe_filename)
        except ValueError:
            print(
                'SpikeInterface recording contains no probe information,\n'
                'could not write .prb file.'
            )
            if not keep_probe:
                self.gui.settings_box.clear_probe_selection()

        self.gui.settings_box.check_load()
        self.gui.qt_settings.setValue('load_as_wrapper', True)

    @QtCore.Slot()
    def convert_to_binary(self):
        # TODO: add option to specify chunksize for conversion
        if None in [self.filename, self.filetype, self.data_dtype]:
            logger.exception(
                'File name, file type, and data dtype must be specified.'
            )
            logger.exception(
                f'File name: {self.filename[0]}\nFile type: {self.filetype}\n'
                f'Dtype: {self.data_dtype}'
            )
            return

        options = QtWidgets.QFileDialog.DontUseNativeDialog
        bin_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption="Specify a .bin file location to save data to...",
            directory=self.gui.qt_settings.value('last_data_location'),
            filter='*.bin',
            options=options,
        )
        data_location = Path(bin_filename).parent.as_posix()
        self.gui.qt_settings.setValue('last_data_location', data_location)

        self.disableInput.emit(True)
        bin_filename = Path(bin_filename)
        self.hide()
        recording = get_recording_extractor(
            self.filename, self.filetype, self.stream_id, self.stream_name
            )

        self.conversion_thread = ConversionThread(self)
        self.conversion_thread.bin_filename = bin_filename
        self.conversion_thread.recording = recording
        self.conversion_thread.start()
        while self.conversion_thread.isRunning():
            QtWidgets.QApplication.processEvents()
        else:
            self.disableInput.emit(False)

        # Get output of conversion, then delete thread.
        c = self.conversion_thread.c
        fs = self.conversion_thread.fs
        probe_filename = self.conversion_thread.probe_filename
        self.conversion_thread = None

        # Update settings with new data file, probe, etc
        settings = self.gui.settings_box
        update_settings_box(settings, c, fs, self.data_dtype)
        if probe_filename is not None:
            add_probe_to_settings(settings, probe_filename)
        else:
            settings.clear_probe_selection()
        settings.data_file_path_input.setText(bin_filename.as_posix())
        settings.data_file_path_input.editingFinished.emit()
        
        self.gui.settings_box.check_load()


# NOTE: spikeinterface should not be imported anywhere else in this module to
#       avoid an explicit dependency.
def get_recording_extractor(filename, filetype, stream_id=None, stream_name=None):
    '''Load recording by importing an extractor from spikeinterface.'''
    extractor_name = _SPIKEINTERFACE_IMPORTS[filetype]
    import spikeinterface as si
    import spikeinterface.extractors as si_ex
    #extractor_module = importlib.import_module('spikeinterface.extractors')
    if filetype == 'NWB':
        kwargs = {}
    else:
        kwargs = {'stream_id': stream_id, 'stream_name': stream_name}

    if not isinstance(filename, list): filename = [filename]
    extractors = [
        getattr(si_ex, extractor_name)(f, **kwargs)
        for f in filename
        ]
    return si.concatenate_recordings(extractors)


def update_settings_box(settings, n_chans, fs, dtype):
    settings.n_chan_bin_input.setText(str(n_chans))
    settings.n_chan_bin_input.editingFinished.emit()
    settings.fs_input.setText(str(fs))
    settings.fs_input.editingFinished.emit()
    data_idx = settings.dtype_selector.findText(dtype)
    settings.dtype_selector.setCurrentIndex(data_idx)


def add_probe_to_settings(settings, probe_filename):
    settings.probe_layout_selector.setCurrentIndex(0)
    probe = load_probe(probe_filename)
    settings.save_probe_selection(probe, probe_filename.name)
    settings.enable_preview_probe()


class ConversionThread(QtCore.QThread):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

    def run(self):
        _, _, self.c, _, self.fs, self.probe_filename = spikeinterface_to_binary(
            self.recording, self.bin_filename.parent,
            data_name=self.bin_filename.name,
            dtype=self.parent.data_dtype
        )
