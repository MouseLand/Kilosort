
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import torch
from PyQt5 import QtWidgets


# Adapted from https://www.pythonguis.com/tutorials/plotting-matplotlib/
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent, nrows, ncols, width=5, height=4, dpi=100):
        self.parent = parent
        fig = Figure(figsize=(width,height), dpi=dpi, layout='constrained')
        self.axes = fig.subplots(nrows, ncols)
        super().__init__(fig)


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, *args, title=None, **kwargs):
        super().__init__()
        if title is not None:
            self.setWindowTitle(title)
        self.canvas = MplCanvas(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.hide()


# TODO: Axis labels don't actually show up anywhere, still debugging

def plot_drift_amount(plot_window, dshift, settings):
    # Drift amount for each block of probe over time
    ax = plot_window.canvas.axes
    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs)

    ax.plot(t, dshift)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth shift (microns)')

    plot_window.canvas.draw_idle()
    plot_window.show()


def plot_drift_scatter(plot_window, st0):
    # Scatter of depth of spike vs time, intensity = spike amplitude
    ax = plot_window.canvas.axes
    s = ax.scatter(st0[:,0], st0[:,5], s=1, c=st0[:,2], cmap='Greys_r')
    plot_window.canvas.figure.colorbar(s, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (microns)')

    plot_window.canvas.draw_idle()
    plot_window.show()


def plot_diagnostics(plot_window, wPCA, Wall3, clu0, settings):
    ax1, ax2, ax3, ax4 = plot_window.canvas.axes.flatten()

    # Temporal features (top left)
    t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
    ax1.plot(t, wPCA.T, linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Temporal Features')

    # Spatial features (top right)
    features = torch.linalg.norm(Wall3, dim=1).cpu().numpy()
    ax2.imshow(features.T)
    ax2.set_xlabel('Unit Number')
    ax2.set_ylabel('Channel Number')
    ax2.set_title('Spatial Features')

    # Comput spike counts and mean amplitudes
    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size
    mean_amp = torch.linalg.norm(Wall3, dim=(1,2)).cpu().numpy()

    # # Unit amplitudes (bottom left)
    ax3.plot(mean_amp)
    ax3.set_xlabel('Unit Number')
    ax3.set_ylabel('Amplitude (a.u.)')
    ax3.set_title('Unit Amplitudes')

    # TODO: Still a mismatch here between unit counts for clu0 vs Wall3
    # # # Amplitude vs Spike Count (bottom right)
    # ax4.scatter(np.log(1 + spike_counts), mean_amp, s=1)
    # ax4.set_xlabel('Spike Count')
    # ax4.set_ylabel('Amplitude (a.u.)')
    # ax4.set_title('Amplitude vs Spike Count')

    plot_window.canvas.draw_idle()
    plot_window.show()
