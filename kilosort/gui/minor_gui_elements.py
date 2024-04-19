import numpy as np
from kilosort.gui.logger import setup_logger
from qtpy import QtGui, QtWidgets, QtCore

logger = setup_logger(__name__)


def create_prb(probe):
    chan_map = np.array(probe["chanMap"])
    xc, yc = np.array(probe["xc"]), np.array(probe["yc"])
    probe_prb = {}
    unique_channel_groups = np.unique(np.array(probe["kcoords"]))

    for channel_group in unique_channel_groups:
        probe_prb[channel_group] = {}

        channel_group_pos = np.where(probe["kcoords"] == channel_group)
        group_channels = chan_map[channel_group_pos]
        group_xc = xc[channel_group_pos]
        group_yc = yc[channel_group_pos]

        probe_prb[channel_group]['channels'] = np.asarray(group_channels).tolist()
        geometry = {}

        for c, channel in enumerate(group_channels):
            geometry[channel] = (group_xc[c], group_yc[c])

        probe_prb[channel_group]['geometry'] = geometry
        probe_prb[channel_group]['graph'] = []

    return probe_prb


class ProbeBuilder(QtWidgets.QDialog):
    def __init__(self, parent, *args, **kwargs):
        super(ProbeBuilder, self).__init__(parent=parent, *args, **kwargs)
        self.parent = parent
        self.setWindowFlags(
            self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint
            )

        self.map_name_value = QtWidgets.QLineEdit()
        self.map_name_label = QtWidgets.QLabel("Name for new channel map:")

        self.x_coords_value = QtWidgets.QLineEdit()
        self.x_coords_label = QtWidgets.QLabel("X-coordinates for each site:")

        self.y_coords_value = QtWidgets.QLineEdit()
        self.y_coords_label = QtWidgets.QLabel("Y-coordinates for each site:")

        self.k_coords_value = QtWidgets.QLineEdit()
        self.k_coords_label = QtWidgets.QLabel(
            "Shank index ('kcoords') for each " "site (leave blank for single shank):"
        )

        self.channel_map_value = QtWidgets.QLineEdit()
        self.channel_map_label = QtWidgets.QLabel(
            "Channel map (row in data file for each site, 0-indexed):"
        )

        self.input_list = [
            self.map_name_value,
            self.x_coords_value,
            self.y_coords_value,
            self.k_coords_value,
            self.channel_map_value,
        ]

        self.error_label = QtWidgets.QLabel()
        self.error_label.setText("Invalid inputs!")
        self.error_label.setWordWrap(True)

        self.okay_button = QtWidgets.QPushButton("OK", parent=self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", parent=self)
        self.check_button = QtWidgets.QPushButton("Check", parent=self)

        self.map_name = None
        self.x_coords = None
        self.y_coords = None
        self.k_coords = None
        self.channel_map = None
        self.bad_channels = None

        self.probe = None

        self.values_checked = False

        self.setup()

    def setup(self):
        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(
            "Valid inputs: lists, or numpy expressions (use np for numpy)"
        )

        self.cancel_button.clicked.connect(self.reject)
        self.okay_button.setIcon(
            self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogCancelButton)
        )
        self.okay_button.clicked.connect(self.accept)
        self.okay_button.setIcon(
            self.parent.style().standardIcon(QtWidgets.QStyle.SP_DialogOkButton)
        )
        self.check_button.clicked.connect(self.check_inputs)

        buttons = [self.check_button, self.okay_button, self.cancel_button]

        error_label_size_policy = self.error_label.sizePolicy()
        error_label_size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        error_label_size_policy.setRetainSizeWhenHidden(True)
        self.error_label.setSizePolicy(error_label_size_policy)
        error_label_palette = self.error_label.palette()
        error_label_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("red"))
        self.error_label.setPalette(error_label_palette)
        self.error_label.hide()

        for field in self.input_list:
            field.textChanged.connect(self.set_values_as_unchecked)

        widget_list = [
            self.map_name_label,
            self.map_name_value,
            info_label,
            self.x_coords_label,
            self.x_coords_value,
            self.y_coords_label,
            self.y_coords_value,
            self.k_coords_label,
            self.k_coords_value,
            self.channel_map_label,
            self.channel_map_value,
            self.error_label,
        ]

        button_layout = QtWidgets.QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)

        self.okay_button.setDisabled(True)

        for widget in widget_list:
            layout.addWidget(widget)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_values_as_unchecked(self):
        self.values_checked = False
        self.okay_button.setDisabled(True)

    def set_values_as_checked(self):
        self.values_checked = True
        self.okay_button.setDisabled(False)

    def check_inputs(self):
        try:
            map_name = self.map_name_value.text()
            assert len(map_name.split()) == 1, \
                   'probe name cannot contain spaces'

            x_coords = eval(self.x_coords_value.text())
            y_coords = eval(self.y_coords_value.text())
            x_coords = np.array(x_coords, dtype=np.float64)
            y_coords = np.array(y_coords, dtype=np.float64)
            assert len(x_coords) == len(y_coords), \
                   'x and y positions must have same size'

            k_coords = self.k_coords_value.text()
            if k_coords == "":
                k_coords = np.ones_like(x_coords, dtype=np.float64)
            else:
                k_coords = np.array(eval(k_coords), dtype=np.float64)
                assert x_coords.size == k_coords.size, \
                       'contact positions and shank indices must have same size'

            channel_map = self.channel_map_value.text()
            if channel_map == "":
                channel_map = np.arange(x_coords.size)
            else:
                channel_map = np.array(eval(channel_map), dtype=np.int32)
                assert x_coords.size == channel_map.size, \
                       'contact positions and channel map must be same size'
                assert channel_map.size == np.unique(channel_map).size, \
                       'channel map cannot contain repeats'

        except Exception as e:
            self.error_label.setText(str(e))
            self.error_label.show()

        else:
            self.map_name = map_name
            self.x_coords = x_coords.tolist()
            self.y_coords = y_coords.tolist()
            self.k_coords = k_coords.tolist()
            self.channel_map = channel_map.tolist()

            self.set_values_as_checked()
            self.error_label.hide()

            self.construct_probe()

    # TODO: use `Probe` from pykilosort/params.py
    def construct_probe(self):
        probe = {
            "xc": self.x_coords,
            "yc": self.y_coords,
            "kcoords": self.k_coords,
            "chanMap": self.channel_map,
            "n_chan": len(self.x_coords)
        }

        probe["chanMapBackup"] = probe["chanMap"].copy()

        self.probe = probe

    def get_map_name(self):
        return self.map_name

    def get_probe(self):
        return self.probe


controls_popup_text = """
<font style="font-family:Monospace">
GUI Controls <br>
------------ <br>
<br>
[1 2]        - activate/deactivate raw/filtered views of the dataset <br>
[scroll]         - move forward/backward in time <br>
[ctrl + scroll]  - add/remove channels in colormap mode;
[alt + scroll]   - change data/colormap scaling <br>
[shift + scroll] - zoom in/out in time <br>
[left click]     - move forward/backward in time <br>
<br>
Mouse Controls for Plots <br>
------------------------ <br>
<br>
[left click]     - click and drag to pan.
[right click]    - click to open context menu, drag to rescale axes.
[middle/wheel]   - click and drag to pan, or use wheel to zoom.
</font>
"""

help_popup_text = """
Welcome to Kilosort4!

##### Documentation #####
For documentation of the underlying algorithm or the GUI, please visit https://github.com/MouseLand/Kilosort/
or read the paper at https://www.nature.com/articles/s41592-024-02232-7.

##### Troubleshooting #####
1. Click 'Reset GUI' to clear any GUI problems or strange errors. If the problem persists, try 'Clear Cache' and restarting Kilosort. 
2. If the problem persists, visit https://github.com/MouseLand/Kilosort/issues/ and ask for assistance there with as much detail about the problem as possible.  
"""
