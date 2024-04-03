from qtpy import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg

from kilosort.spikedetect import template_centers, nearest_chans
from kilosort.gui.logger import setup_logger


logger = setup_logger(__name__)


class ProbeViewBox(QtWidgets.QGroupBox):

    channelSelected = QtCore.Signal(int)

    def __init__(self, parent):
        super(ProbeViewBox, self).__init__(parent=parent)
        self.setTitle("Probe View")
        self.gui = parent
        self.probe_view = pg.PlotWidget()
        self.template_toggle = QtWidgets.QCheckBox('Universal Templates')
        self.spot_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.setup()

        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.xcup = None
        self.ycup = None
        self.total_channels = None
        self.channel_map = None
        self.channel_map_dict = {}
        self.channel_spots = None
        self.template_spots = None

        self.sorting_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False
        }

        self.active_data_view_mode = "colormap"

    def setup(self):
        # This was the previous behavior, zoom will only change vertical scaling
        # and no axes are visible.
        # self.probe_view.hideAxis("left")
        # self.probe_view.hideAxis("bottom")
        # self.probe_view.setMouseEnabled(False, True)
        self.show_templates = False

        self.template_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.template_toggle.stateChanged.connect(self.toggle_templates)

        self.spot_scale.setMinimum(0)
        self.spot_scale.setMaximum(10)
        self.spot_scale.setValue(1)
        self.spot_scale.valueChanged.connect(self.refresh_plot)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.probe_view, 95)
        layout.addWidget(self.template_toggle)
        layout.addWidget(self.spot_scale)
        self.setLayout(layout)

    def set_layout(self, context):
        self.probe_view.clear()
        probe = context.raw_probe
        template_args = self.gui.settings_box.get_probe_template_args()
        self.set_active_layout(probe, template_args)
        self.update_probe_view()

    def set_active_layout(self, probe, template_args):
        self.active_layout = probe
        self.kcoords = self.active_layout["kcoords"]
        self.xc, self.yc = self.active_layout["xc"], self.active_layout["yc"]
        self.xcup, self.ycup = self.get_template_centers(*template_args)
        self.channel_map_dict = {}
        for ind, (xc, yc) in enumerate(zip(self.xc, self.yc)):
            self.channel_map_dict[(xc, yc)] = ind
        self.total_channels = self.active_layout["n_chan"]
        self.channel_map = self.active_layout["chanMap"]

    def get_template_centers(self, nC, dmin, dminx, max_dist, device):
        ops = {
            'yc': self.yc, 'xc': self.xc, 'max_channel_distance': max_dist,
            'settings': {'dmin': dmin, 'dminx': dminx}
            }
        ops = template_centers(ops)
        [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
        ys, xs = ys.flatten(), xs.flatten()
        iC, ds = nearest_chans(ys, self.yc, xs, self.xc, nC, device=device)

        igood = ds[0,:] <= ops['max_channel_distance']**2
        iC = iC[:,igood]
        ds = ds[:,igood]
        ys = ys[igood]
        xs = xs[igood]

        return xs, ys
    
    @QtCore.Slot()
    def toggle_templates(self):
        if self.template_toggle.isChecked():
            self.show_templates = True
        else:
            self.show_templates = False
        # If no layout is loaded, this will just be an empty plot.
        # Otherwise, re-uses current layout to re-generate with or without
        # template centers shown.
        self.refresh_plot()

    @QtCore.Slot()
    def refresh_plot(self):
        template_args = self.gui.settings_box.get_probe_template_args()
        self.preview_probe(self.gui.settings_box.probe_layout, template_args)

    @QtCore.Slot(str, int)
    def synchronize_data_view_mode(self, mode: str):
        if self.active_data_view_mode != mode:
            self.probe_view.clear()
            self.update_probe_view()
            self.active_data_view_mode = mode

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict

    def generate_spots_list(self):
        channel_spots = []
        template_spots = []

        if self.xc is not None:
            size = 10 * self.spot_scale.value()
            symbol = "s"
            color = "g"
            for x_pos, y_pos in zip(self.xc, self.yc):
                pen = pg.mkPen(0.5)
                brush = pg.mkBrush(color)
                channel_spots.append({
                    'pos': (x_pos, y_pos), 'size': size, 'pen': pen, 'brush': brush,
                    'symbol': symbol
                    })
        self.channel_spots = channel_spots

        if self.xcup is not None:
            size = 5 * self.spot_scale.value()
            symbol = "o"
            color = "w"
            for x, y in zip(self.xcup, self.ycup):
                pen = pg.mkPen(0.5)
                brush = pg.mkBrush(color)
                template_spots.append({
                    'pos': (x,y), 'size': size, 'pen': pen, 'brush': brush,
                    'symbol': symbol
                    })
        self.template_spots = template_spots


    @QtCore.Slot(int, int)
    def update_probe_view(self):
        self.create_plot()

    @QtCore.Slot(object, object)
    def preview_probe(self, probe, template_args):
        self.probe_view.clear()
        self.set_active_layout(probe, template_args)
        self.create_plot()

    def create_plot(self):
        self.generate_spots_list()
        spots = self.channel_spots
        if self.show_templates:
            spots += self.template_spots
        scatter_plot = pg.ScatterPlotItem(spots)
        self.probe_view.addItem(scatter_plot)

    def reset(self):
        self.clear_plot()
        self.reset_current_probe_layout()
        self.reset_active_data_view_mode()

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

    def prepare_for_new_context(self):
        self.clear_plot()
        self.reset_current_probe_layout()

    def clear_plot(self):
        self.probe_view.getPlotItem().clear()
