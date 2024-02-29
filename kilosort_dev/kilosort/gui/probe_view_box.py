import numpy as np
import pyqtgraph as pg
from kilosort.gui.logger import setup_logger
from PyQt5 import QtCore, QtGui, QtWidgets

logger = setup_logger(__name__)


class ProbeViewBox(QtWidgets.QGroupBox):

    channelSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        super(ProbeViewBox, self).__init__(parent=parent)
        self.setTitle("Probe View")

        self.gui = parent

        self.probe_view = pg.PlotWidget()

        self.info_message = QtWidgets.QLabel(
            "scroll to zoom, click to view channel,\nright click to disable channel"
        )

        self.setup()

        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.total_channels = None
        self.channel_map = None
        self.channel_map_dict = {}

        self.sorting_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False
        }

        self.configuration = {
            "active_channel": "g",
            "inactive_channel": "b",
            "bad_channel": "r",
        }

        self.active_data_view_mode = "colormap"
        self.primary_channel = 0
        self.active_channels = []

    def setup(self):
        layout = QtWidgets.QVBoxLayout()

        self.probe_view.hideAxis("left")
        self.probe_view.hideAxis("bottom")
        self.probe_view.setMouseEnabled(False, True)

        self.info_message.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Black))
        self.info_message.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.info_message, 5)
        layout.addWidget(self.probe_view, 95)
        self.setLayout(layout)

    def set_active_channels(self, channels_displayed):
        primary_channel = self.primary_channel
        channel_map = np.array(self.channel_map)

        primary_channel_position = int(np.where(channel_map == primary_channel)[0])
        end_channel_position = np.where(
            channel_map == primary_channel + channels_displayed
        )[0]
        # prevent the last displayed channel would be set as the end channel in the case that
        # `primary_channel + displayed_channels` exceeds the total number of channels in the channel map
        if end_channel_position.size == 0:
            end_channel_position = np.argmax(channel_map)
        else:
            end_channel_position = int(end_channel_position)
        self.active_channels = channel_map[
            primary_channel_position:end_channel_position+1
        ].tolist()

    def set_layout(self, context):
        self.probe_view.clear()
        probe = context.raw_probe

        self.set_active_layout(probe)

        self.update_probe_view()

    def set_active_layout(self, probe):
        self.active_layout = probe
        self.kcoords = self.active_layout["kcoords"]
        self.xc, self.yc = self.active_layout["xc"], self.active_layout["yc"]
        self.channel_map_dict = {}
        for ind, (xc, yc) in enumerate(zip(self.xc, self.yc)):
            self.channel_map_dict[(xc, yc)] = ind
        self.total_channels = self.active_layout["n_chan"]
        self.channel_map = self.active_layout["chanMap"]

    def on_points_clicked(self, points):
        selected_point = points.ptsClicked[0]
        x_pos = int(selected_point.pos().x())
        y_pos = int(selected_point.pos().y())

        index = self.channel_map_dict[(x_pos, y_pos)]
        channel = self.channel_map[index]
        self.channelSelected.emit(channel)

    @QtCore.pyqtSlot(str, int)
    def synchronize_data_view_mode(self, mode: str, channels_displayed: int):
        if self.active_data_view_mode != mode:
            self.probe_view.clear()
            self.update_probe_view(channels_displayed=channels_displayed)
            self.active_data_view_mode = mode

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict

    def generate_spots_list(self):
        spots = []
        size = 10
        symbol = "s"

        for ind, (x_pos, y_pos) in enumerate(zip(self.xc, self.yc)):
            pos = (x_pos, y_pos)
            is_active = np.isin(ind, self.active_channels).tolist()
            if is_active:
                color = self.configuration["active_channel"]
            else:
                color = self.configuration["inactive_channel"]
            pen = pg.mkPen(0.5)
            brush = pg.mkBrush(color)
            spots.append(dict(pos=pos, size=size, pen=pen, brush=brush, symbol=symbol))

        return spots

    @QtCore.pyqtSlot(int, int)
    def update_probe_view(self, primary_channel=None, channels_displayed=None):
        if primary_channel is not None:
            self.primary_channel = primary_channel

        if channels_displayed is None:
            channels_displayed = self.total_channels

        self.set_active_channels(channels_displayed)
        self.create_plot()

    @QtCore.pyqtSlot(object)
    def preview_probe(self, probe):
        self.probe_view.clear()
        self.set_active_layout(probe)
        self.create_plot(connect=False)

    def create_plot(self, connect=True):
        spots = self.generate_spots_list()

        scatter_plot = pg.ScatterPlotItem(spots)
        if connect:
            scatter_plot.sigClicked.connect(self.on_points_clicked)
        self.probe_view.addItem(scatter_plot)

    def reset(self):
        self.clear_plot()
        self.reset_current_probe_layout()
        self.reset_active_data_view_mode()
        self.primary_channel = 0

    def reset_active_data_view_mode(self):
        self.active_data_view_mode = "colormap"

    def reset_current_probe_layout(self):
        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.total_channels = None
        self.channel_map = None
        self.channel_map_dict = {}
        self.active_channels = []

    def prepare_for_new_context(self):
        self.clear_plot()
        self.reset_current_probe_layout()

    def clear_plot(self):
        self.probe_view.getPlotItem().clear()
