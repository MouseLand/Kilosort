
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
        fig = Figure(figsize=(width,height), dpi=dpi)
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
    ax.xaxis.set_label('Time (s)')
    ax.yaxis.set_label('Depth shift (microns)')
    plot_window.canvas.figure.tight_layout()

    plot_window.canvas.draw_idle()
    plot_window.show()


def plot_drift_scatter(plot_window, st0):
    # Scatter of depth of spike vs time, intensity = spike amplitude
    ax = plot_window.canvas.axes
    s = ax.scatter(st0[:,0], st0[:,5], s=1, c=st0[:,2], cmap='Greys_r')
    plot_window.canvas.figure.colorbar(s, ax=ax)
    ax.xaxis.set_label('Time (s)')
    ax.yaxis.set_label('Depth (microns)')

    plot_window.canvas.draw_idle()
    plot_window.show()


def plot_features(plot_window, Wall3):
    ax = plot_window.canvas.axes
    features = torch.linalg.norm(Wall3, ord=2, dim=1).cpu().numpy()

    ax.imshow(features.T)
    ax.xaxis.set_label('Unit Number')
    ax.yaxis.set_label('Channel Number')
    ax.set_title('Spatial Features')

    plot_window.canvas.draw_idle()
    plot_window.show()
