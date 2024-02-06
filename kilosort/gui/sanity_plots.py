import pyqtgraph as pg
import matplotlib
import scipy
import numpy as np
import torch
from PyQt5 import QtWidgets

_COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, *args, title=None, width=500, height=400, **kwargs):
        super().__init__()
        if title is not None:
            self.setWindowTitle(title)
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.GraphicsLayoutWidget(parent=self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.hide()


# TODO: Axis labels don't actually show up anywhere, still debugging

def plot_drift_amount(plot_window, dshift, settings):
    # Drift amount for each block of probe over time
    p1 = plot_window.plot_widget.addPlot(
        row=0, col=0, labels={'left': 'Depth shift (microns)', 'bottom': 'Time (s)'}
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
        row=0, col=0, labels={'left': 'Depth (microns)', 'bottom': 'Time (s)'}
    )
    p1.setTitle('Spike amplitude across time and depth')

    x = st0[:,0]  # spike time in seconds
    x_ind = (x*10).astype('int')  # 100ms bins
    y = st0[:,5]  # depth of spike center in microns
    y_ind = (y*0.2).astype('int')   # 5 micron bins
    z = st0[:,2]  # spike amplitude (data)
    z[z < 10] = 10
    z[z > 100] = 100

    # TODO: No reason to do this until white background is working, and still
    #       not clear if this is working as intended.
    # Set custom colormap
    # greys = matplotlib.colormaps['Greys_r']
    # pos = np.logspace(1, 2, 90)
    # colors = []
    # for p in pos:
    #     r, g, b, _ = greys(((p-10)/90)*255)  # maps over range 0 to 255
    #     colors.append((r, g, b))
    # cm = pg.ColorMap(pos=pos, color=colors)

    # Can't directly plot csr_matrix with pyqtgraph, so convert to array.
    # Just using this for convenience of building the array format.
    mat = scipy.sparse.csr_matrix((z, (x_ind, y_ind))).toarray()
    img = pg.ImageItem(image=mat)#, colorMap=cm)
    p1.addItem(img)
    p1.getViewBox().invertY(True)


    # Change tick scaling to match bin sizes
    ax = p1.getAxis('bottom')
    ax.setScale(0.1)
    ay = p1.getAxis('left')
    ay.setScale(5)

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
def plot_probe_positions(plot_window):
    pass






# TODO: this "works", but there's no way to color by amplitude that I've
#       been able to find

# p1 = plot_window.plot_widget.addPlot(
#     row=0, col=0, labels={'left': 'Depth (microns)', 'bottom': 'Time (s)'}
# )
# points = p1.plot(st0[:,0], st0[:,5], pen=None, symbol='o')
# points.setSymbolSize(1)
# plot_window.show()


# TODO: Looks like this way is almost working? But it's so slow it's not
#       worth doing, and I still can't get the points to actually show up

# p1 = plot_window.plot_widget.addPlot(
#     row=0, col=0, labels={'left': 'Depth (microns)', 'bottom': 'Time (s)'}
# )
# scatter = pg.ScatterPlotItem()

# # Build list of spot dicts to add to scatter.
# # TODO: Simpler way to do this? Couldn't figure out another way to color
# #       by a 3rd variable.
# x = st0[:,0]  # spike time
# y = st0[:,5]  # depth of spike center
# z = st0[:,2]  # spike amplitude

# # Apply colormap and log normalization to amplitude
# cm = matplotlib.colormaps['Greys']
# LN = matplotlib.colors.LogNorm(vmin=10, vmax=100, clip=True)

# t0 = time.perf_counter()
# z = np.array([cm(LN(a)) for a in z])
# t1 = time.perf_counter()
# print(f'time to map colors: {(t1 - t0):.2f}')

# # Build list of spots for plot
# spots = [
#     {'pos': (x[i], y[i]), 'size': 1, 'pen': {'color': z[i]}, 'brush': z[i]}
#     for i in range(x.size)
# ]
# t2 = time.perf_counter()
# print(f'time to build spot list: {(t2 - t1):.2f}')
# # Add spots to plot area
# scatter.addPoints(spots)
# t3 = time.perf_counter()
# print(f'time to add spots to plot: {(t3-t2):.2f}')

# NOTE: For sample dataset (90s, ~1GB), this broken  down to ~60s to map
#       values to color args, ~nothing to build the spots list, and ~35s to 
#       add the spots to the plot. So, optimizing the color mapping would
#       certainly help, but so long as scatter.addPoints(spots) is the only
#       way to do this with a colormap, that's still way too slow for a full
#       size dataset.
#       By contrast, the matplotlib version takes less than
#       half a second to generate.

# p1.addItem(scatter)
# plot_window.show()

# TODO: third option, try making it into a heatmap

# NOTE: No... this is dumb. Wouldn't be able to load the full thing into
#       memory, would have to do it in batches similar to the raw traces and
#       that's getting way too complicated for a simple scatterplot.
# p1 = plot_window.plot_widget.addPlot(
#     row=0, col=0, labels={'left': 'Depth (microns)', 'bottom': 'Time (s)'}
# )
# x = st0[:,0]  # spike time in seconds
# y = st0[:,5]  # depth of spike center in microns
# z = st0[:,2]  # spike amplitude
# # Convert to heatmap with 5ms time bins, 5 micron depth bins
# nx = int(np.ceil((x.max() - x.min())/0.005))
# _, x_bins = np.histogram(x, bins=nx)
# a = _make_heatmap(x, y, z, x_bin=0.005, y_bin=5)


# TODO: much faster now, and SHOULD be doing the right thing, but instead
#       the plot is still coming out monochrome?


# p1 = plot_window.plot_widget.addPlot(
#     row=0, col=0, labels={'left': 'Depth (microns)', 'bottom': 'Time (s)'}
# )
# p1.setTitle('Spike amplitude across time and depth')
# x = st0[:,0]  # spike time
# y = st0[:,5]  # depth of spike center
# z = st0[:,2]  # spike amplitude
# # Clip values outside [10,100] for better visualization
# # TODO: Will these values work for everyone's data?
# z[z < 10] = 10
# z[z > 100] = 100

# # Apply colormap and log normalization to amplitude
# n_bins = 90
# cm = matplotlib.colormaps['Greys_r']
# bin_idx = np.digitize(z, bins=np.logspace(1, 2, n_bins))
# for i in np.unique(bin_idx):
#     # Take mean of all amplitude values within one bin, map to color
#     subset = (bin_idx == i)
#     a = z[subset].mean()
#     rgba = cm(((a-10)/90))  # map over range 0 to 255
#     brush = pg.mkBrush(rgba)
#     pen = pg.mkPen(rgba)
#     # Add points to plot
#     p1.plot(x[subset], y[subset], pen=None, symbol='o', symbolPen=pen,
#             brush=brush, symbolSize=1)

# plot_window.show()