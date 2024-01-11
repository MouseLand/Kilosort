import importlib
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from kilosort.io import RecordingExtractorAsArray, spikeinterface_to_binary

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
    def __init__(self):
        super().__init__()
        self.filename = None
        self.filetype = None
        self.stream_id = None
        self.stream_name = None

        layout = QtWidgets.QGridLayout()

        # Top left: filename and filetype selectors
        self.filename_button = QtWidgets.QPushButton('Select file')
        self.filename_button.clicked.connect(self.select_filename)
        self.filename_input = QtWidgets.QLineEdit()
        self.filename_input.setReadOnly(True)
        layout.addWidget(self.filename_button, 0, 0, 1, 3)
        layout.addWidget(self.filename_input, 0, 3, 1, 3)

        self.filetype_text = QtWidgets.QLabel('Select file type')
        self.filetype_selector = QtWidgets.QComboBox()
        self.filetype_selector.addItem('')
        self.filetype_selector.addItems(list(_SPIKEINTERFACE_IMPORTS.keys()))
        self.filetype_selector.currentTextChanged.connect(self.select_filetype)
        layout.addWidget(self.filetype_text, 1, 0, 1, 3)
        layout.addWidget(self.filetype_selector, 1, 3, 1, 3)

        # Top right: kwargs for spikeinterface loader
        self.stream_id_text = QtWidgets.QLabel('stream_id')
        self.stream_id_input = QtWidgets.QLineEdit()
        self.stream_id_input.textChanged.connect(self.set_stream_id)
        self.stream_name_text = QtWidgets.QLabel('stream_name')
        self.stream_name_input = QtWidgets.QLineEdit()
        self.stream_name_input.textChanged.connect(self.set_stream_name)
        layout.addWidget(self.stream_id_text, 0, 10, 1, 3)
        layout.addWidget(self.stream_id_input, 0, 13, 1, 3)
        layout.addWidget(self.stream_name_text, 1, 10, 1, 3)
        layout.addWidget(self.stream_name_input, 1, 13, 1, 3)

        # Bottom: convert to binary or load with wrapper
        self.wrapper_button = QtWidgets.QPushButton('Load As Wrapper')
        self.wrapper_button.clicked.connect(self.load_as_wrapper)
        self.convert_button = QtWidgets.QPushButton('Convert to Binary')
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_input = QtWidgets.QLineEdit()
        self.convert_input.setReadOnly(True)
        layout.addWidget(self.wrapper_button, 3, 2, 1, 6)
        layout.addWidget(self.convert_button, 4, 2, 1, 6)
        layout.addWidget(self.convert_input, 4, 8, 1, 6)

        self.setLayout(layout)


    @QtCore.pyqtSlot()
    def select_filename(self):
        options = QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Choose file to load data from...",
            directory=Path('~').expanduser().as_posix(),
            options=options,
        )
        self.filename = filename
        self.filename_input.setText(filename)

    @QtCore.pyqtSlot()
    def select_filetype(self):
        self.filetype = self.filetype_selector.currentText()

    @QtCore.pyqtSlot()
    def set_stream_id(self):
        stream_id = self.stream_id_input.text()
        if stream_id is None or stream_id == '':
            stream_id = None
        else:
            stream_id = int(stream_id)
        self.stream_id = stream_id

    @QtCore.pyqtSlot()
    def set_stream_name(self):
        stream_name = self.stream_name_input.text()
        if stream_name == '':
            stream_name = None
        self.stream_name = stream_name

    @QtCore.pyqtSlot()
    def load_as_wrapper(self):
        pass

    @QtCore.pyqtSlot()
    def convert_to_binary(self):
        pass




# NOTE: spikeinterface should not be imported anywhere else in this module to
#       avoid an explicit dependency.
def get_recording_extractor(filename, filetype, stream_id=None, stream_name=None):
    '''Load recording by importing an extractor from spikeinterface.'''
    extractor_name = _SPIKEINTERFACE_IMPORTS[filetype]
    extractors = importlib.import_module('spikeinterface.extractors')
    if filetype == 'NWB':
        kwargs = {}
    else:
        kwargs = {'stream_id': stream_id, 'stream_name': stream_name}

    return getattr(extractors, extractor_name)(filename, **kwargs)
