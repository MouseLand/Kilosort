import pyqtgraph as pg
import scipy
import matplotlib
import numpy as np
import torch
from PyQt5 import QtWidgets

from kilosort.postprocessing import compute_spike_positions
from kilosort.gui.palettes import PROBE_PLOT_COLORS

_COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, *args, title=None, width=500, height=400,
                 background=None, **kwargs):
        super().__init__()
        if title is not None:
            self.setWindowTitle(title)
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.GraphicsLayoutWidget(parent=self)
        if background is not None:
            self.plot_widget.setBackground(background)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.hide()


# TODO: Axis labels don't actually show up anywhere, still debugging

def plot_drift_amount(plot_window, dshift, settings):
    # Drift amount for each block of probe over time
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth shift (um)', 'bottom': 'Time (s)'}
        )
    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs)

    for i in range(dshift.shape[1]):
        color = _COLOR_CODES[i % len(_COLOR_CODES)]
        p1.plot(t, dshift[:,i], pen=color)

    plot_window.show()


def plot_drift_scatter(plot_window, st0, settings):
    # Amplitude of spike over time and depth
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth (um)', 'bottom': 'Time (s)'}
    )
    p1.setTitle('Spike amplitude across time and depth')

    x = st0[:,0]  # spike time in seconds
    y = st0[:,5]  # depth of spike center in microns
    z = st0[:,2]  # spike amplitude (data)
    z[z < 10] = 10
    z[z > 100] = 100

    bin_idx = np.digitize(z, np.logspace(1, 2, 90))
    cm = matplotlib.colormaps['binary']
    brushes = np.empty_like(z, dtype=object)
    pens = np.empty_like(z, dtype=object)
    for i in np.unique(bin_idx):
        # Take mean of all amplitude values within one bin, map to color
        subset = (bin_idx == i)
        a = z[subset].mean()
        r,g,b,a = cm(((a-10)/90))
        brush = pg.mkBrush((r,g,b))
        brushes[subset] = brush
        pen = pg.mkPen((r,g,b))
        pens[subset] = pen

    scatter = pg.ScatterPlotItem(x, y, symbol='o', size=1, pen=None, brush=brushes)
    p1.addItem(scatter)
    p1.getViewBox().setBackgroundColor('w')
    bottom_ax = p1.getAxis('bottom')
    bottom_ax.setPen('k')
    bottom_ax.setTextPen('k')
    bottom_ax.setTickPen('k')
    left_ax = p1.getAxis('left')
    left_ax.setPen('k')
    left_ax.setTextPen('k')
    left_ax.setTickPen('k')

    plot_window.show()


def plot_diagnostics(plot_window, wPCA, Wall0, clu0, settings):
    # Temporal features (top left)
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'bottom': 'Time (s)'}
        )
    p1.setTitle('Temporal Features')
    t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
    for i in range(wPCA.shape[0]):
        color = _COLOR_CODES[i % len(_COLOR_CODES)]
        p1.plot(t, wPCA[i,:], pen=color)

    # Spatial features (top right)
    p2 = plot_window.plot_widget.addPlot(
        row=0, col=1, labels={'bottom': 'Unit Number', 'left': 'Channel Number'}
        )
    p2.setTitle('Spatial Features')
    features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
    img = pg.ImageItem(image=features.T)
    p2.addItem(img)

    # Comput spike counts and mean amplitudes
    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size
    mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()

    # Unit amplitudes (bottom left)
    p3 = plot_window.plot_widget.addPlot(
        row=1, col=0, labels={'bottom': 'Unit Number', 'left': 'Amplitude (a.u.)'}
        )
    p3.setTitle('Unit Amplitudes')
    p3.plot(mean_amp)

    # Amplitude vs Spike Count (bottom right)
    p4 = plot_window.plot_widget.addPlot(
        row=1, col=1,
        labels={'bottom': 'Log(1 + Spike Count)', 'left': 'Amplitude (a.u.)'}
        )
    p4.setTitle('Amplitude vs Spike Count')
    p4.plot(np.log(1 + spike_counts), mean_amp, pen=None, symbol='o')

    # Finished, draw plot
    plot_window.show()


# TODO: need to figure out what to do about scatter w/ color first, this would
#       be done the same as the drift scatter (which is currently not working).
def plot_spike_positions(plot_window, ops, st, clu, tF, is_refractory,
                         device=None):

    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth (um)', 'bottom': 'Lateral (um)'}
    )
    p1.setTitle('Spike position across probe, colored by cluster')

    # 10 colors in palette, last one is gray for non-frefractory
    clu = clu.copy()
    bad_units = np.unique(clu)[is_refractory == 0]
    bad_idx = np.in1d(clu, bad_units)
    clu = np.mod(clu, 9)
    clu[bad_idx] = 9
    cm = pg.ColorMap(pos=np.arange(10), color=PROBE_PLOT_COLORS)
    lookup = cm.getLookupTable(nPts=10)

    # Get x, y positions, scale to 2 micron bins
    xs, ys = compute_spike_positions(st, tF, ops)
    ys = (ys*0.5).astype('int')
    xs = (xs*0.5).astype('int')
    
    # Arange clusters into heatmap
    mat = scipy.sparse.csr_matrix((clu, (ys, xs))).toarray()
    img = pg.ImageItem(image=mat)
    img.setLookupTable(lookup)
    p1.addItem(img)

    # Change tick scaling to match bin sizes
    ax = p1.getAxis('bottom')
    ax.setScale(2)
    ay = p1.getAxis('left')
    ay.setScale(2)

    plot_window.show()
