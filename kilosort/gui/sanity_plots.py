# Adapted from
# https://github.com/MouseLand/pykilosort/blob/master/pykilosort/gui/sanity_plots.py
# Author: Shashwat Sridhar


import numpy as np
import typing as t
from pyqtgraph import LayoutWidget, ColorMap, RemoteGraphicsView, PlotItem

from kilosort.gui import SANITY_PLOT_COLORS


class SanityPlotWidget(LayoutWidget):
    def __init__(self, num_remote_plots, title):
        super(SanityPlotWidget, self).__init__(parent=None)
        self.num_remote_plots = num_remote_plots
        self.remote_plots = []

        self.setWindowTitle(title)

        self.create_remote_views()
        self.arrange_views()

        #self.hide()

    def create_remote_views(self):
        for _ in range(self.num_remote_plots):
            remote_plot = RemoteGraphicsView(useOpenGL=True)
            remote_plot_item = remote_plot.pg.PlotItem()
            remote_plot_item._setProxyOptions(deferGetattr=True)  # noqa
            remote_plot.setCentralItem(remote_plot_item)

            self.remote_plots.append(
                remote_plot
            )

    def arrange_views(self):
        for i, remote_plot in enumerate(self.remote_plots):
            self.addWidget(remote_plot)

            if (i + 1) % 2 == 0:
                self.nextRow()

    @staticmethod
    def _set_labels_on_plot(plot_item, labels):
        plot_item.setLabels(left=labels.get("left", ""),
                            right=labels.get("right", ""),
                            top=labels.get("top", ""),
                            bottom=labels.get("bottom", ""),
                            title=labels.get("title", ""))

    def add_scatter(self,
                    x_data: np.ndarray,
                    y_data: np.ndarray,
                    plot_pos: int,
                    labels: dict,
                    x_lim: t.Optional[tuple] = None,
                    y_lim: t.Optional[tuple] = None,
                    semi_log_x: t.Optional[bool] = None,
                    semi_log_y: t.Optional[bool] = None,
                    **kwargs: t.Optional[dict],
                    ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa
        remote_view = remote_plot_item.getViewBox()

        remote_plot_item.clear()

        remote_plot_item.setLogMode(x=semi_log_x, y=semi_log_y)

        scatter_plot = remote_plot.pg.ScatterPlotItem(x=x_data, y=y_data, **kwargs)
        remote_plot_item.addItem(scatter_plot)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        if not (semi_log_x or semi_log_y):
            remote_plot_item.setRange(xRange=x_lim, yRange=y_lim)
        else:
            if (x_lim is not None) and (y_lim is not None):
                remote_view.setLimits(
                    xMin=x_lim[0],
                    xMax=x_lim[1],
                    yMin=y_lim[0],
                    yMax=y_lim[1],
                )
            elif (x_lim is not None) and (y_lim is None):
                remote_view.setLimits(
                    xMin=x_lim[0],
                    xMax=x_lim[1],
                )
            elif (y_lim is not None) and (x_lim is None):
                remote_view.setLimits(
                    yMin=y_lim[0],
                    yMax=y_lim[1],
                ),
            else:
                # if both x_lim and y_lim are None, enable autoRange
                remote_plot_item.enableAutoRange()

        return remote_plot_item

    def add_curve(self,
                  x_data: np.ndarray,
                  y_data: np.ndarray,
                  plot_pos: int,
                  labels: dict,
                  x_lim: t.Optional[tuple] = None,
                  y_lim: t.Optional[tuple] = None,
                  semi_log_x: t.Optional[bool] = None,
                  semi_log_y: t.Optional[bool] = None,
                  ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.setLogMode(x=semi_log_x, y=semi_log_y)

        if y_data.ndim == 1:
            y_data = y_data[:, np.newaxis]
        for i in range(y_data.shape[1]):
            remote_plot_item.plot(x=x_data, y=y_data[:, i], clear=True)

        remote_plot_item.setRange(xRange=x_lim, yRange=y_lim)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        return remote_plot_item

    def add_image(self,
                  array: np.ndarray,
                  plot_pos: int,
                  labels: dict,
                  normalize: bool = True,
                  invert_y: bool = True,
                  cmap_style: str = "diverging",
                  **kwargs: t.Optional[dict],
                  ) -> PlotItem:
        remote_plot = self.get_remote_plots()[plot_pos]
        remote_plot_item = remote_plot._view.centralWidget  # noqa

        remote_plot_item.clear()

        if "levels" in kwargs.keys():
            levels = kwargs.pop("levels")
            auto_levels = False
        else:
            levels = None
            auto_levels = True

        if normalize:
            array = self.normalize_array(array)

        if cmap_style == "diagnostic":
            colormap = ColorMap(
                pos=np.linspace(
                    -1, 1, len(SANITY_PLOT_COLORS[cmap_style])
                ),
                color=np.array(SANITY_PLOT_COLORS[cmap_style])
            )

            lut = colormap.getLookupTable(
                start=-1, stop=1, nPts=1024,
            )

        elif cmap_style == "dissimilarity":
            colormap = ColorMap(
                pos=np.linspace(
                    0, 1, len(SANITY_PLOT_COLORS[cmap_style])
                ),
                color=np.array(SANITY_PLOT_COLORS[cmap_style])
            )

            lut = colormap.getLookupTable(
                start=1, stop=0, nPts=1024,
            )
        else:
            raise ValueError("Invalid colormap style requested.")

        if auto_levels:
            image_item = remote_plot.pg.ImageItem(image=array,
                                                  lut=lut)
        else:
            image_item = remote_plot.pg.ImageItem(image=array,
                                                  lut=lut,
                                                  autoLevels=auto_levels,
                                                  levels=levels)

        if not auto_levels:
            image_item.setLevels(levels)

        remote_plot_item.addItem(image_item, **kwargs)

        self._set_labels_on_plot(remote_plot_item, labels)
        remote_plot_item.hideAxis('top')
        remote_plot_item.hideAxis('right')

        remote_plot_item.invertY(invert_y)

        return remote_plot_item

    @staticmethod
    def normalize_array(array):
        return 2. * (array - np.amin(array)) / np.ptp(array) - 1

    def get_remote_plots(self):
        return self.remote_plots

    def close_all_plots(self):
        for plot in self.remote_plots:
            plot.close()


# TODO: Adapt this for KS4
def plot_diagnostics(temporal_comp, spatial_comp, mu, nsp, plot_widget):
    temporal_comp = cp.asnumpy(temporal_comp)
    spatial_comp = cp.asnumpy(spatial_comp)
    mu = cp.asnumpy(mu)
    nsp = cp.asnumpy(nsp)

    plot_widget.add_image(
        array=temporal_comp[:, :, 0].T,
        plot_pos=0,
        labels={"left": "Time (samples)",
                "bottom": "Unit Number",
                "title": "Temporal Components"},
        cmap_style="diagnostic",
        levels=(-0.4, 0.4),
        normalize=False,
    )

    plot_widget.add_image(
        array=spatial_comp[:, :, 0].T,
        plot_pos=1,
        labels={"left": "Channel Number",
                "bottom": "Unit Number",
                "title": "Spatial Components"},
        cmap_style="diagnostic",
        levels=(-0.2, 0.2),
        normalize=False,
    )

    plot_widget.add_curve(
        x_data=np.arange(len(mu)),
        y_data=mu,
        plot_pos=2,
        labels={"left": "Amplitude (arb. units)",
                "bottom": "Unit Number",
                "title": "Unit Amplitudes"},
        y_lim=(0, 100),
    )

    plot_widget.add_scatter(
        x_data=np.log(1+nsp),
        y_data=mu,
        plot_pos=3,
        labels={"left": "Amplitude (arb. units)",
                "bottom": "Spike Count",
                "title": "Amplitude vs. Spike Count"},
        y_lim=(0, 1e2),
        x_lim=(0, np.log(1e5)),
        semi_log_x=True,
        pxMode=True,
        symbol="o",
        size=2,
        pen="w",
    )
    plot_widget.show()


# TODO: looks like I need to ditch the sanity plot widget class and
#       just use pyqtgraph directly, running into errors that some googling indicates
#       are a result of outdated code.

def plot_drift(ops):
    plot = SanityPlotWidget(num_remote_plots=1, title="Drift Correction")
    ymin = np.min(ops['dshift'])
    ymax = np.max(ops['dshift'])
    plot.add_curve(
        x_data=np.arange(ops['dshift'].shape[0]),
        y_data=ops['dshift'],
        plot_pos=0,
        labels={"left": "Depth (microns)",
                "bottom": "Time (units TODO)",
                "title": "Drift amount"},
        y_lim=(ymin, ymax)
    )
