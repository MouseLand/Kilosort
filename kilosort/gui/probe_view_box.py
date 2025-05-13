from qtpy import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg
import torch

from kilosort.spikedetect import template_centers, nearest_chans
from kilosort.clustering_qr import x_centers, y_centers, get_nearest_centers
from kilosort.gui.logger import setup_logger

logger = setup_logger(__name__)


class ProbeViewBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        super(ProbeViewBox, self).__init__(parent=parent)
        self.setTitle("Probe View")
        self.gui = parent
        self.probe_view = pg.PlotWidget()
        self.channel_toggle = QtWidgets.QCheckBox('Show Channels')
        self.template_toggle = QtWidgets.QCheckBox('Universal Templates')
        self.color_toggle = QtWidgets.QCheckBox('Color by Group')
        self.number_toggle = QtWidgets.QCheckBox('Number')
        self.center_toggle = QtWidgets.QCheckBox('Grouping Centers')
        self.aspect_toggle = QtWidgets.QCheckBox('True Aspect Ratio')
        self.spot_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.setup()

        self.reset_spots_variables()

        self.sorting_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False
        }


    def setup(self):
        self.aspect_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.aspect_toggle.stateChanged.connect(self.create_plot)
        self.channel_toggle.setCheckState(QtCore.Qt.CheckState.Checked)
        self.channel_toggle.stateChanged.connect(self.create_plot)
        self.template_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.template_toggle.stateChanged.connect(self.create_plot)
        self.center_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.center_toggle.stateChanged.connect(self.create_plot)
        self.color_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.color_toggle.stateChanged.connect(self.create_plot)
        self.number_toggle.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.number_toggle.stateChanged.connect(self.create_plot)

        self.spot_scale.setMinimum(0)
        # Actually want 0 to 10 scaling, but these are multiplied by 4
        # to get 0.25 increments.
        self.spot_scale.setMaximum(40)
        self.spot_scale.setValue(4)
        self.spot_scale.valueChanged.connect(self.set_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Left click to toggle excluded channels'))
        layout.addWidget(self.probe_view, 95)

        chan_box = QtWidgets.QHBoxLayout()
        chan_box.addWidget(self.aspect_toggle)
        chan_box.addWidget(self.channel_toggle)
        layout.addLayout(chan_box)

        temp_box = QtWidgets.QHBoxLayout()
        temp_box.addWidget(self.template_toggle)
        temp_box.addWidget(self.color_toggle)
        layout.addLayout(temp_box)

        center_box = QtWidgets.QHBoxLayout()
        center_box.addWidget(self.center_toggle)
        center_box.addWidget(self.number_toggle)
        layout.addLayout(center_box)
        layout.addWidget(self.spot_scale)
        
        self.setLayout(layout)

    def reset_spots_variables(self):
        self.active_layout = None
        self.kcoords = None
        self.xc = None
        self.yc = None
        self.xcup = None
        self.ycup = None
        self.center_positions = None
        self.total_channels = None
        self.channel_spots = None
        self.template_spots = None
        self.center_spots = None
        self.channel_map = None
        self.channel_map_dict = {}

    def set_layout(self):
        self.probe_view.clear()
        self.reset_spots_variables()
        probe = self.gui.settings_box.probe_layout  # original, no removed chans
        template_args = self.gui.settings_box.get_probe_template_args()
        self.update_spots_variables(probe, template_args)
        self.create_plot()

    def update_spots_variables(self, probe, template_args):
        self.active_layout = probe
        # Set xc, yc based on revised probe (with bad channels removed) for
        # determining template and center positions, since that's how they would
        # be placed during sorting.
        probe = self.gui.probe_layout
        self.xc, self.yc = probe['xc'], probe['yc']
        self.kcoords = probe['kcoords']
        #if self.template_toggle.isChecked() or self.center_toggle.isChecked():
        self.xcup, self.ycup, self.ops = self.get_template_spots(*template_args)
        self.center_positions, self.skipped_centers, self.nearest_centers = \
            self.get_center_spots()

        # Change xc, yc back to original probe for plotting all channels.
        self.xc, self.yc = self.active_layout["xc"], self.active_layout["yc"]
        self.total_channels = self.active_layout["n_chan"]
        self.channel_map = self.active_layout["chanMap"]
        self.channel_map_dict = {}
        for ind, (xc, yc) in enumerate(zip(self.xc, self.yc)):
            self.channel_map_dict[(xc, yc)] = ind

    def get_template_spots(self, nC, dmin, dminx, max_dist, x_centers):
        ops = {
            'yc': self.yc, 'xc': self.xc, 'max_channel_distance': max_dist,
            'x_centers': x_centers, 'settings': {'dmin': dmin, 'dminx': dminx},
            'kcoords': self.kcoords
            }
        ops = template_centers(ops)
        [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
        ys, xs = ys.flatten(), xs.flatten()
        iC, ds = nearest_chans(
            ys, self.yc, xs, self.xc, nC, device=self.gui.device
            )

        igood = ds[0,:] <= max_dist**2
        iC = iC[:,igood]
        ds = ds[:,igood]
        ys = ys[igood]
        xs = xs[igood]
        ops['xcup'] = xs
        ops['ycup'] = ys

        return xs, ys, ops

    def get_center_spots(self):
        ycent = y_centers(self.ops)
        xcent = x_centers(self.ops)
        xy = np.vstack((self.ops['xcup'], self.ops['ycup']))
        xy = torch.from_numpy(xy)
        nearest_center, x_pos, y_pos = get_nearest_centers(xy, xcent, ycent)
        center_pos = list(zip(x_pos.cpu().numpy(), y_pos.cpu().numpy()))

        skipped_centers = []
        for kk in np.arange(len(ycent)):
            for jj in np.arange(len(xcent)):
                ii = kk + jj*ycent.size
                if ii not in nearest_center:
                    skipped_centers.append(ii)
                else:
                    continue

        return center_pos, skipped_centers, nearest_center

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict

    def generate_spots_list(self):
        self.channel_spots = []
        self.template_spots = []
        self.center_spots = []
        self.number_spots = []
        bad_channels = self.gui.settings_box.get_bad_channels()
        shank_idx = self.gui.settings_box.shank_idx
        chan_map = self.active_layout['chanMap']
        if shank_idx is None:
            shank_channels = chan_map
        else:
            shank_map = (self.active_layout['kcoords'] == shank_idx).nonzero()[0]
            shank_channels = chan_map[shank_map]
        channel_size = 10 * self.spot_scale.value()/4
        template_size = 5 * self.spot_scale.value()/4
        center_size = 20 * self.spot_scale.value()/4
        colors = ["w", "b", "r", "c", "y", "m"]

        if self.channel_toggle.isChecked():
            for x_pos, y_pos in zip(self.xc, self.yc):
                index = self.channel_map_dict[(x_pos, y_pos)]
                channel = self.channel_map[index]
                if (channel in bad_channels) or (channel not in shank_channels):
                    color = "b"
                else:
                    color = "g"
                pen = pg.mkPen(0.5)
                brush = pg.mkBrush(color)
                self.channel_spots.append({
                    'pos': (x_pos, y_pos), 'size': channel_size,
                    'pen': pen, 'brush': brush, 'symbol': "s"
                    })

        if self.template_toggle.isChecked():
            for i, (x, y) in enumerate(zip(self.xcup, self.ycup)):
                pen = pg.mkPen(0.5)
                if self.color_toggle.isChecked():
                    j = self.nearest_centers[i]
                    brush = pg.mkBrush(colors[j % len(colors)])
                else:
                    brush = pg.mkBrush("w")

                self.template_spots.append({
                    'pos': (x,y), 'size': template_size, 'pen': pg.mkPen(0.5),
                    'brush': brush, 'symbol': "o"
                    })
        
        if self.center_toggle.isChecked():
            for j, (x, y) in enumerate(self.center_positions):
                if j in self.skipped_centers:
                    continue
                if self.color_toggle.isChecked():
                    c = colors[j % len(colors)]
                    pen1 = pg.mkPen(color=c)
                else:
                    c = "y"
                    pen1 = pg.mkPen(color=c)
                self.center_spots.append({
                    'pos': (x,y), 'size': center_size,
                    'pen': pen1, 'brush': None, 'symbol': "o"
                })
                
                if self.number_toggle.isChecked():
                    pen2 = pg.mkPen(color=c)
                    symbol, scale = create_label(str(j), center_size)
                    self.number_spots.append({
                        'pos': (x,y), 'size': 0.1/scale,
                        'pen': pen2, 'brush': None, 'symbol': symbol
                    })


    def create_plot(self):
        self.generate_spots_list()
        spots = self.channel_spots
        spots += self.template_spots
        spots += self.center_spots
        spots += self.number_spots
        if self.aspect_toggle.isChecked():
            self.probe_view.setAspectLocked()
        else:
            self.probe_view.setAspectLocked(lock=False)

        self.probe_view.clear()
        scatter_plot = pg.ScatterPlotItem(spots)
        scatter_plot.sigClicked.connect(self.on_points_clicked)
        self.probe_view.addItem(scatter_plot)

    def reset(self):
        self.clear_plot()
        self.reset_current_probe_layout()

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

    @QtCore.Slot(object, object)
    def on_points_clicked(self, points, event):
        selected_point = points.ptsClicked[0]
        x_pos = selected_point.pos().x()
        y_pos = selected_point.pos().y()

        # Get channel number corresponding to clicked position, use that
        # to update the list of bad channels before refreshing plot.
        index = self.channel_map_dict[(x_pos, y_pos)]
        channel = self.channel_map[index]
        bad_channels = self.gui.settings_box.get_bad_channels()
        if channel in bad_channels:
            # Remove it from the list
            bad_channels.remove(channel)
        else:
            bad_channels.append(channel)
        self.gui.settings_box.set_bad_channels(bad_channels)

        self.set_layout()


def create_label(label, size):
    # Adapted from:
    # https://www.geeksforgeeks.org/pyqtgraph-show-text-as-spots-on-scatter-plot-graph/
    
    # Create QPainterPath to use as custom symbol shape
    symbol = QtGui.QPainterPath()
    f = QtGui.QFont()
    f.setPointSize(int(size*5))
    # For pyqtgraph, custom symbol shapes must be centered at (0,0)
    p = QtCore.QPointF(0,0)
    symbol.addText(p, f, label)
    # Also must be scaled to width and height of 1.0
    br = symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)

    return tr.map(symbol), scale
