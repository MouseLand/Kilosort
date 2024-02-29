import numpy as np
import pyqtgraph as pg
from kilosort.gui.logger import setup_logger
from kilosort.gui.palettes import COLORMAP_COLORS
from PyQt5 import QtCore, QtWidgets

logger = setup_logger(__name__)


class DataViewBox(QtWidgets.QGroupBox):
    channelChanged = QtCore.pyqtSignal(int, int)
    modeChanged = QtCore.pyqtSignal(str, int)
    updateContext = QtCore.pyqtSignal(object)
    intervalUpdated = QtCore.pyqtSignal()

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)

        self.gui = parent

        self.data_view_widget = KSPlotWidget(useOpenGL=True)
        self.data_x_axis = self.data_view_widget.getAxis("bottom")
        self.plot_item = self.data_view_widget.getPlotItem()
        self.data_view_box = self.data_view_widget.getViewBox()
        self.colormap_image = None

        self.data_seek_widget = pg.PlotWidget(useOpenGL=True)
        self.seek_view_box = self.data_seek_widget.getViewBox()
        self.time_seek = pg.InfiniteLine(
            pen=pg.mkPen((255, 0, 0, 128), width=3), movable=True, name="indicator"
        )
        self.time_label = pg.TextItem(color=(180, 180, 180))

        # self.traces_button = QtWidgets.QPushButton("Traces")
        # self.colormap_button = QtWidgets.QPushButton("Colormap")
        self.raw_button = QtWidgets.QPushButton("Raw")
        self.whitened_button = QtWidgets.QPushButton("Whitened")
        # Set minimum and maximum time in seconds for spike sorting.
        # This should truncate the data view if specified.
        self.tmin_input = self.gui.settings_box.tmin_input
        self.tmin = float(self.tmin_input.text())
        self.tmax_input = self.gui.settings_box.tmax_input
        self.tmax = float(self.tmax_input.text())
        # self.prediction_button = QtWidgets.QPushButton("Prediction")
        # self.residual_button = QtWidgets.QPushButton("Residual")

        self.mode_buttons_group = QtWidgets.QButtonGroup(self)
        self.view_buttons_group = QtWidgets.QButtonGroup(self)

        self.view_buttons = {
            "raw": self.raw_button,
            "whitened": self.whitened_button,
            # "prediction": self.prediction_button,
            # "residual": self.residual_button,
        }

        self.input_fields = [self.tmin_input, self.tmax_input]

        # self.mode_buttons = [self.traces_button, self.colormap_button]

        self._view_button_checked_bg_colors = {
            "raw": "white",
            "whitened": "lightblue",
            # "prediction": "green",
            # "residual": "red",
        }

        self._keys = [
            "raw",
            "whitened",
            # "prediction",
            # "residual",
        ]

        self.view_buttons_state = {key: False for key in self._keys}

        self.primary_channel = 0
        self.current_time = 0
        self.plot_range = 0.1  # seconds

        self.highpass_filter = None

        self.whitening_matrix = None
        self.whitened_traces = None
        self.prediction_traces = None
        self.residual_traces = None

        self.sorting_status = {
            "preprocess": False,
            "spikesort": False,
            "export": False,
        }

        # traces settings
        self.traces_plot_items = {
            "raw": [],
            "whitened": [],
            "prediction": [],
            "residual": [],
        }
        self.good_channel_color = (255, 255, 255)
        self.bad_channel_color = (100, 100, 100)
        self.channels_displayed_traces = 32
        self.channels_displayed_colormap = None
        self.seek_range = (0, 100)
        self.scale_factor = 1.0
        self.traces_scaling_factor = {
            "raw": 1,
            "whitened": 15,
            # "prediction": 15,
            # "residual": 15,
        }
        self.traces_curve_color = {
            "raw": "w",
            "whitened": "c",
            # "prediction": "g",
            # "residual": "r",
        }

        # colormap settings
        self._colors = COLORMAP_COLORS

        self.colormap_min = 0.0
        self.colormap_max = 1.0
        self.lookup_table = self.generate_lookup_table(
            self.colormap_min, self.colormap_max
        )

        self.thread_pool = QtCore.QThreadPool()

        self.setup()

    def setup(self):
        self.setTitle("Data View")

        layout = QtWidgets.QVBoxLayout()

        data_view_layout = QtWidgets.QHBoxLayout()
        data_view_layout.addWidget(self.data_view_widget)

        self.time_label.setParentItem(self.seek_view_box)
        self.time_label.setPos(0, 0)
        self.data_seek_widget.addItem(self.time_seek)

        self.time_seek.sigPositionChanged.connect(self.update_seek_text)
        self.time_seek.sigPositionChanged.connect(self.update_seek_position)
        # self.time_seek.sigPositionChangeFinished.connect(self.update_seek_position)

        self.data_view_widget.setMenuEnabled(False)
        self.data_view_widget.setMouseEnabled(True)
        self.data_view_widget.mouseEnabled = True
        self.data_view_widget.hideAxis("left")
        self.data_view_widget.disableAutoRange()
        self.data_view_widget.sceneObj.sigMouseClicked.connect(self.on_scene_clicked)

        self.data_view_widget.signalChangeChannel.connect(
            self.on_wheel_scroll_plus_control
        )
        self.data_view_widget.signalChangeTimeRange.connect(
            self.on_wheel_scroll_plus_shift
        )
        self.data_view_widget.signalChangeScaling.connect(self.on_wheel_scroll_plus_alt)
        self.data_view_widget.signalChangeTimePoint.connect(self.on_wheel_scroll)

        self.data_seek_widget.setMenuEnabled(False)
        self.data_seek_widget.setMouseEnabled(False, False)
        self.data_seek_widget.hideAxis("left")
        self.data_seek_widget.sceneObj.sigMouseClicked.connect(self.seek_clicked)

        data_controls_layout = QtWidgets.QHBoxLayout()

        # self.traces_button.setCheckable(True)
        # self.colormap_button.setCheckable(True)
        # self.colormap_button.setChecked(True)

        # for mode_button in self.mode_buttons:
        #     self.mode_buttons_group.addButton(mode_button)
        #     mode_button.clicked.connect(self.toggle_mode_from_click)
        # self.mode_buttons_group.setExclusive(True)

        for key in self._keys:
            button = self.view_buttons[key]
            button.setCheckable(True)
            button.setStyleSheet("QPushButton {background-color: black; color: white;}")
            button.clicked.connect(self.on_views_clicked)
            self.view_buttons_group.addButton(button)

        self.raw_button.setChecked(True)
        self.raw_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )

        self.enable_view_buttons()

        self.view_buttons_group.setExclusive(True)

        self.view_buttons_state = [
            self.view_buttons[key].isChecked() for key in self._keys
        ]

        # data_controls_layout.addWidget(self.traces_button)
        # data_controls_layout.addWidget(self.colormap_button)

        # Connect time controls
        self.tmin_input.editingFinished.connect(self.on_tmin_edited)
        self.tmax_input.editingFinished.connect(self.on_tmax_edited)

        data_controls_layout.addStretch(0)
        data_controls_layout.addWidget(self.raw_button)
        data_controls_layout.addWidget(self.whitened_button)
        # data_controls_layout.addWidget(self.prediction_button)
        # data_controls_layout.addWidget(self.residual_button)

        data_seek_layout = QtWidgets.QHBoxLayout()
        data_seek_layout.addWidget(self.data_seek_widget)

        layout.addLayout(data_view_layout, 85)
        layout.addLayout(data_controls_layout, 3)
        layout.addLayout(data_seek_layout, 10)

        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def on_tmin_edited(self):
        try:
            self.tmin = float(self.tmin_input.text())
            self.check_time_interval()
            self.intervalUpdated.emit()
        except ValueError:
            logger.exception('Could not convert tmin to float.')
        except AssertionError:
            logger.exception('Invalid tmin,tmax: must have 0 <= tmin < tmax')

    @QtCore.pyqtSlot()
    def on_tmax_edited(self):
        try:
            self.tmax = float(self.tmax_input.text())
            self.check_time_interval()
            self.intervalUpdated.emit()
        except ValueError:
            logger.exception('Could not convert tmax to float.')
        except AssertionError:
            logger.exception('Invalid tmin,tmax: must have 0 <= tmin < tmax')

    def check_time_interval(self):
        # TODO: any other checks needed?
        assert self.tmin >= 0
        assert self.tmin < self.tmax

    @QtCore.pyqtSlot()
    def on_views_clicked(self):
        current_state = {key: self.view_buttons[key].isChecked() for key in self._keys}

        if current_state != self.view_buttons_state:
            self.view_buttons_state = current_state

            for view, state in self.view_buttons_state.items():
                button = self.view_buttons[view]
                if state:
                    color = self._view_button_checked_bg_colors[view]
                    button.setStyleSheet(
                        f"QPushButton {{background-color: {color}; color: black;}}"
                    )
                else:
                    if button.isEnabled():
                        button.setStyleSheet(
                            "QPushButton {background-color: black; color: white;}"
                        )
                    else:
                        button.setStyleSheet(
                            "QPushButton {background-color: black; color: gray;}"
                        )

            self.update_plot()

    @QtCore.pyqtSlot(float)
    def on_wheel_scroll(self, direction):
        if self.context_set():
            self.shift_current_time(direction)

    @QtCore.pyqtSlot(float)
    def on_wheel_scroll_plus_control(self, direction):
        if self.context_set():
            if self.traces_mode_active():
                self.shift_primary_channel(direction)
            else:
                self.change_displayed_channel_count(direction)

    @QtCore.pyqtSlot(float)
    def on_wheel_scroll_plus_shift(self, direction):
        if self.context_set():
            self.change_plot_range(direction)

    @QtCore.pyqtSlot(float)
    def on_wheel_scroll_plus_alt(self, direction):
        if self.context_set():
            self.change_plot_scaling(direction)

    # @QtCore.pyqtSlot()
    # def toggle_mode_from_click(self):
    #     if self.traces_mode_active():
    #         self.modeChanged.emit("traces", self.get_currently_displayed_channel_count())
    #         self.view_buttons_group.setExclusive(False)
    #         self.update_plot()
    #
    #     if self.colormap_mode_active():
    #         self.modeChanged.emit("colormap", self.get_currently_displayed_channel_count())
    #         self._traces_to_colormap_toggle()
    #         self.update_plot()

    # def toggle_mode(self):
    #     if self.colormap_mode_active():
    #         self.traces_button.toggle()
    #         self.modeChanged.emit("traces", self.get_currently_displayed_channel_count())
    #         self.view_buttons_group.setExclusive(False)
    #         self.update_plot()
    #
    #     elif self.traces_mode_active():
    #         self.colormap_button.toggle()
    #         self.modeChanged.emit("colormap", self.get_currently_displayed_channel_count())
    #         self._traces_to_colormap_toggle()
    #         self.update_plot()
    #
    #     else:
    #         pass
    #
    # def _traces_to_colormap_toggle(self):
    #     if sum([button.isChecked() for button in self.view_buttons.values()]) == 1:
    #         # if exactly one mode is active, that mode persists into colormap mode
    #         self.update_plot()
    #     else:
    #         # if more than one mode is active, raw mode is activated on toggle
    #         for name, button in self.view_buttons.items():
    #             if button.isChecked():
    #                 button.setChecked(False)
    #             if button.isEnabled():
    #                 button.setStyleSheet(
    #                     "QPushButton {background-color: black; color: white;}"
    #                 )
    #             else:
    #                 button.setStyleSheet(
    #                     "QPushButton {background-color: black; color: gray;}"
    #                 )
    #
    #         self.view_buttons["raw"].setChecked(True)
    #         self.view_buttons["raw"].setStyleSheet(
    #             f"QPushButton {{"
    #             f"background-color: {self._view_button_checked_bg_colors['raw']}; "
    #             f"color: black;"
    #             f"}}"
    #         )
    #
    #     self.view_buttons_group.setExclusive(True)

    def traces_mode_active(self):
        return False

    def colormap_mode_active(self):
        return True

    def context_set(self):
        return self.get_context() is not None

    def get_context(self):
        return self.gui.context

    def change_primary_channel(self, channel):
        self.primary_channel = channel
        self.channelChanged.emit(self.primary_channel, self.get_currently_displayed_channel_count())
        self.update_plot()

    def shift_primary_channel(self, shift):
        primary_channel = self.primary_channel
        primary_channel += shift * 5
        total_channels = self.get_total_channels()
        if (0 <= primary_channel < total_channels) and total_channels is not None:
            self.primary_channel = primary_channel
            self.channelChanged.emit(self.primary_channel, self.get_currently_displayed_channel_count())
            self.update_plot()

    def get_currently_displayed_channel_count(self):
        if self.traces_mode_active():
            return self.channels_displayed_traces
        else:
            count = self.channels_displayed_colormap
            if count is None:
                count = self.get_total_channels()
            return count

    def set_currently_displayed_channel_count(self, count):
        if self.traces_mode_active():
            self.channels_displayed_traces = count
        else:
            self.channels_displayed_colormap = count

    def get_total_channels(self):
        return self.gui.probe_view_box.total_channels

    def change_displayed_channel_count(self, direction):
        current_channel = self.primary_channel
        total_channels = self.get_total_channels()
        current_count = self.get_currently_displayed_channel_count()

        new_count = current_count + (direction * 5)
        if (current_channel + new_count) <= total_channels:
            self.set_currently_displayed_channel_count(new_count)
            self.channelChanged.emit(self.primary_channel, new_count)
            self.refresh_plot_on_displayed_channel_count_change()
        elif new_count <= 0 and current_count != 1:
            self.set_currently_displayed_channel_count(1)
            self.channelChanged.emit(self.primary_channel, 1)
            self.refresh_plot_on_displayed_channel_count_change()
        elif (current_channel + new_count) > total_channels:
            self.set_currently_displayed_channel_count(total_channels - current_channel)
            self.channelChanged.emit(self.primary_channel, total_channels)
            self.refresh_plot_on_displayed_channel_count_change()

    def refresh_plot_on_displayed_channel_count_change(self):
        self.plot_item.clear()
        # self.create_plot_items()
        self.update_plot()

    def shift_current_time(self, direction):
        time_shift = direction * self.plot_range / 4  # seconds
        current_time = self.current_time
        new_time = current_time + time_shift
        seek_range_min = self.seek_range[0]
        seek_range_max = self.seek_range[1] - self.plot_range
        if seek_range_min <= new_time <= seek_range_max:
            # if new time is in acceptable limits
            self.time_seek.setPos(new_time)
        elif new_time <= seek_range_min:
            # if new time exceeds lower bound of data
            self.time_seek.setPos(seek_range_min)
        elif new_time >= seek_range_max:
            # if new time exceeds upper bound of data
            self.time_seek.setPos(seek_range_max)

    def change_plot_range(self, direction):
        current_range = self.plot_range
        # neg sign to reverse scrolling behaviour
        new_range = current_range * (1.2 ** -direction)
        if 0.005 < new_range < 1.0:
            diff_range = new_range - current_range
            current_time = self.current_time
            new_time = current_time - diff_range/2.
            seek_range_min = self.seek_range[0]
            seek_range_max = self.seek_range[1]
            if new_time < seek_range_min:
                # if range exceeds lower limit on zooming
                # set lower limit as current time
                self.plot_range = new_range
                self.time_seek.setPos(seek_range_min)
            elif new_time + new_range > seek_range_max:
                # if range exceeds upper limit on zooming
                # set (upper limit - new time) as current range
                alt_range = seek_range_max - new_time
                self.plot_range = alt_range
                self.time_seek.setPos(new_time)
            else:
                # if range is acceptable
                self.plot_range = new_range
                self.time_seek.setPos(new_time)

    def change_plot_scaling(self, direction):
        if self.traces_mode_active():
            scale_factor = self.scale_factor * (1.1 ** direction)
            if 0.1 < scale_factor < 10.0:
                self.scale_factor = scale_factor

                self.update_plot()

        if self.colormap_mode_active():
            colormap_min = self.colormap_min + (direction * 0.05)
            colormap_max = self.colormap_max - (direction * 0.05)
            if 0.0 <= colormap_min < colormap_max <= 1.0:
                self.colormap_min = colormap_min
                self.colormap_max = colormap_max
                self.lookup_table = self.generate_lookup_table(
                    self.colormap_min, self.colormap_max
                )

                self.update_plot()

    def on_scene_clicked(self, ev):
        if self.context_set():
            if ev.button() == QtCore.Qt.LeftButton:
                sample_rate = self.get_context().params["fs"]
                if self.traces_mode_active():
                    x_pos = self.data_view_box.mapSceneToView(ev.pos()).x()
                else:
                    x_pos = self.colormap_image.mapFromScene(ev.pos()).x()
                plot_range = self.plot_range * sample_rate
                fraction = x_pos / plot_range
                if fraction > 0.5:
                    self.shift_current_time(direction=1)
                else:
                    self.shift_current_time(direction=-1)

    def update_context(self, new_context):
        self.updateContext.emit(new_context)

    def seek_clicked(self, ev):
        if self.context_set():
            new_time = self.seek_view_box.mapSceneToView(ev.pos()).x()
            seek_range_min = self.seek_range[0]
            seek_range_max = self.seek_range[1]
            if seek_range_min <= new_time < seek_range_max - self.plot_range:
                self.time_seek.setPos(new_time)
            elif seek_range_max - self.plot_range <= new_time < seek_range_max:
                # make sure the plotted data is always as wide as the plot range
                self.time_seek.setPos(seek_range_max - self.plot_range)

    def setup_seek(self, context):
        binary_file = context.binary_file
        sample_rate = context.params["fs"]

        timepoints = binary_file.shape[0]
        min_time = max(0, self.tmin)
        max_time = (timepoints / sample_rate) + min_time

        self.data_seek_widget.setXRange(
            min=min_time,
            max=max_time,
            padding=0.02
        )
        self.time_seek.setPos(min_time)
        self.time_seek.setBounds((min_time, max_time))
        self.seek_range = (min_time, max_time)

    def update_seek_text(self, seek):
        position = seek.pos()[0]
        self.time_label.setText("t={0:.2f} s".format(position))

    def update_seek_position(self, seek):
        position = seek.pos()[0]
        self.current_time = position - self.tmin
        # self.clear_cached_traces()
        try:
            self.update_plot()
        except ValueError:
            self.time_seek.setPos(self.seek_range[1] - self.plot_range)

    def change_sorting_status(self, status_dict):
        self.sorting_status = status_dict
        self.enable_view_buttons()

    @QtCore.pyqtSlot()
    def enable_view_buttons(self):
        if self.colormap_mode_active():
            # if self.prediction_button.isChecked() or self.residual_button.isChecked():
            self.raw_button.click()
        # else:
        #     if self.prediction_button.isChecked():
        #         self.prediction_button.click()
        #     if self.residual_button.isChecked():
        #         self.residual_button.click()

        if self.whitening_matrix is not None:
            self.whitened_button.setEnabled(True)
            self.whitened_button.setStyleSheet(
                "QPushButton {background-color: black; color: white;}"
            )
        else:
            self.whitened_button.setDisabled(True)
            self.whitened_button.setStyleSheet(
                "QPushButton {background-color: black; color: gray;}"
            )

        # if self.sorting_status["preprocess"] and self.sorting_status["spikesort"]:
        #     self.prediction_button.setEnabled(True)
        #     self.prediction_button.setStyleSheet(
        #         "QPushButton {background-color: black; color: white;}"
        #     )
        #     self.residual_button.setEnabled(True)
        #     self.residual_button.setStyleSheet(
        #         "QPushButton {background-color: black; color: white;}"
        #     )
        # else:
        #     self.prediction_button.setDisabled(True)
        #     self.prediction_button.setStyleSheet(
        #         "QPushButton {background-color: black; color: gray;}"
        #     )
        #     self.residual_button.setDisabled(True)
        #     self.residual_button.setStyleSheet(
        #         "QPushButton {background-color: black; color: gray;}"
        #     )

    def reset(self):
        self.plot_item.clear()
        # self.delete_curve_plot_items()
        # self.clear_cached_traces()
        self.clear_cached_whitening_matrix()

    def prepare_for_new_context(self):
        self.plot_item.clear()
        # self.delete_curve_plot_items()
        # self.clear_cached_traces()
        self.clear_cached_whitening_matrix()

    # @QtCore.pyqtSlot(object)
    # def clear_cached_traces(self, _=None):
    #     self.whitened_traces = None
    #     self.residual_traces = None
    #     self.prediction_traces = None

    def clear_cached_whitening_matrix(self):
        self.whitening_matrix = None

    def generate_lookup_table(self, colormap_min, colormap_max, num_points=8192):
        assert colormap_min >= 0.0 and colormap_max <= 1.0
        positions = np.linspace(colormap_min, colormap_max, len(self._colors))
        color_map = pg.ColorMap(pos=positions, color=self._colors)
        return color_map.getLookupTable(nPts=num_points)

    # def create_plot_items(self):
    #     """
    #     Create curve plot items for each active view.
    #
    #     Loops over all views and creates curve plot items for each view.
    #     Creating plot items beforehand results in quicker plotting and a
    #     smoother GUI experience when scrolling through data in traces
    #     mode.
    #     """
    #
    #     for view in self._keys:
    #         self.traces_plot_items[view] = []
    #         for c, channel in enumerate(
    #                 range(
    #                     self.primary_channel + self.channels_displayed_traces,
    #                     self.primary_channel,
    #                     -1
    #                 )
    #         ):
    #             curve = pg.PlotCurveItem()
    #             curve.setPos(0, 200 * c)
    #             self.plot_item.addItem(curve)
    #
    #             self.traces_plot_items[view].append(curve)
    #
    # def add_traces_to_plot_items(
    #         self,
    #         traces: np.ndarray,
    #         view: str
    # ):
    #     """
    #     Update plot items with traces.
    #
    #     Loops over traces and plots each trace using the setData() method
    #     of pyqtgraph's PlotCurveItem. The color of the trace depends on
    #     the mode requested (raw, whitened, prediction, residual). Each trace
    #     is also scaled by a certain factor defined in self.traces_scaling_factor.
    #
    #     Parameters
    #     ----------
    #     traces : numpy.ndarray
    #         Data to be plotted.
    #     view : str
    #         One of "raw", "whitened", "prediction" and "residual" views
    #     """
    #     for i, curve in enumerate(self.traces_plot_items[view]):
    #         try:
    #             trace = traces[:, i] * self.scale_factor * self.traces_scaling_factor[view]
    #
    #             color = (
    #                 self.traces_curve_color[view]
    #             )
    #             curve.setPen(color=color, width=1)
    #             curve.setData(trace)
    #         except IndexError:
    #             curve.setData()
    #
    # def hide_inactive_traces(self):
    #     """
    #     Use setData() on all PlotCurveItems belonging to inactive views.
    #     """
    #     for view in self._keys:
    #         if not self.view_buttons[view].isChecked():
    #             for curve_item in self.traces_plot_items[view]:
    #                 curve_item.setData()
    #
    # def hide_traces(self):
    #     """
    #     Use setData() on all PlotCurveItems in the plot.
    #
    #     Used when switching from traces mode to colormap mode.
    #     """
    #     for view in self._keys:
    #         for curve_item in self.traces_plot_items[view]:
    #             curve_item.setData()

    # def delete_curve_plot_items(self):
    #     """
    #     Deletes all PlotCurveItems in self.traces_plot_items.
    #     """
    #     for view in self.traces_plot_items.keys():
    #         self.traces_plot_items[view] = []

    def add_image_to_plot(self, raw_traces, level_min, level_max):
        image_item = pg.ImageItem(setPxMode=False)
        image_item.setImage(
            raw_traces,
            autoLevels=False,
            lut=self.lookup_table,
            levels=(level_min, level_max),
            autoDownsample=False,
        )
        self.colormap_image = image_item
        self.plot_item.addItem(image_item)

    @QtCore.pyqtSlot(object)
    def set_whitening_matrix(self, array):
        self.whitening_matrix = array

    @QtCore.pyqtSlot(object)
    def set_highpass_filter(self, filter):
        self.highpass_filter = filter

    # def calculate_approx_whitening_matrix(self, context):
    #     raw_data = context.raw_data
    #     params = context.params
    #     probe = context.probe
    #     # intermediate = context.intermediate
    #
    #     @QtCore.pyqtSlot()
    #     def _call_enable_buttons():
    #         self.enable_view_buttons()
    #
    #     # if "Wrot" in intermediate and self.whitening_matrix is None:
    #     #     self.whitening_matrix = intermediate.Wrot
    #     #     logger.info("Approx. whitening matrix loaded from existing context.")
    #     #     _call_enable_buttons()
    #     #
    #     # elif (self.whitening_matrix is None) and not (self.thread_pool.activeThreadCount() > 0):
    #     if (self.whitening_matrix is None) and not (self.thread_pool.activeThreadCount() > 0):
    #         whitening_worker = WhiteningMatrixCalculator(
    #             raw_data=raw_data,
    #             params=params,
    #             probe=probe
    #         )
    #
    #         whitening_worker.signals.result.connect(self.set_whitening_matrix)
    #         whitening_worker.signals.finished.connect(_call_enable_buttons)
    #
    #         self.thread_pool.start(whitening_worker)

    def update_plot(self, context=None):
        if context is None:
            context = self.gui.context

        if context is not None:  # since context may still be None
            if self.colormap_image is not None:
                self.plot_item.removeItem(self.colormap_image)
                self.colormap_image = None

            params = context.params
            probe = context.probe
            binary_file = context.binary_file
            filt_binary_file = context.filt_binary_file

            sample_rate = params["fs"]

            start_time = int(self.current_time * sample_rate)
            time_range = int(self.plot_range * sample_rate)
            end_time = start_time + time_range

            self.data_view_widget.setXRange(
                min=0,
                max=time_range,
                padding=0.02,
            )
            self.data_view_widget.setLimits(
                xMin=0,
                xMax=time_range
            )

            max_channels = np.size(probe["chanMap"])

            start_channel = self.primary_channel
            active_channels = self.get_currently_displayed_channel_count()
            if active_channels is None:
                active_channels = self.get_total_channels()
            end_channel = start_channel + active_channels

            # good channels after start_channel to display
            to_display = np.arange(start_channel, end_channel, dtype=int)
            to_display = to_display[to_display < max_channels].tolist()

            if self.traces_mode_active():
                self._update_traces(
                    params=params,
                    probe=probe,
                    binary_file=binary_file,
                    filt_binary_file=filt_binary_file,
                    to_display=to_display,
                    start_time=start_time,
                    end_time=end_time,
                )

            if self.colormap_mode_active():
                self._update_colormap(
                    params=params,
                    probe=probe,
                    binary_file=binary_file,
                    filt_binary_file=filt_binary_file,
                    to_display=to_display,
                    start_time=start_time,
                    end_time=end_time,
                )

            min_tick = start_time + (self.tmin * sample_rate)
            self.data_x_axis.setTicks(
                [
                    [
                        (pos, f"{(min_tick + pos) / sample_rate:.3f}")
                        for pos in np.linspace(0, time_range, 20)
                    ]
                ]
            )

            self.data_view_widget.autoRange()

    def _update_traces(
            self,
            params,
            probe,
            binary_file,
            filt_binary_file,
            to_display,
            start_time,
            end_time
    ):
        pass
        # self.hide_inactive_traces()
        #
        # if self.raw_button.isChecked():
        #     raw_traces = binary_file[start_time:end_time].numpy()
        #     self.add_traces_to_plot_items(
        #         traces=raw_traces[to_display, :].T,
        #         view="raw",
        #     )
        #
        # if self.whitened_button.isChecked():
        #     whitened_traces = filt_binary_file[start_time:end_time].numpy()
        #     self.add_traces_to_plot_items(
        #         traces=whitened_traces[to_display, :].T,
        #         view="whitened",
        #     )
        #
        # if self.prediction_button.isChecked():
        #     if self.prediction_traces is None:
        #         prediction_traces = get_predicted_traces(
        #             matrix_U=asnumpy(intermediate.U_s),
        #             matrix_W=asnumpy(intermediate.Wphy),
        #             sorting_result=intermediate.st3,
        #             time_limits=(start_time, end_time),
        #         )
        #         self.prediction_traces = prediction_traces
        #     else:
        #         prediction_traces = self.prediction_traces
        #
        #     processed_traces = np.zeros_like(raw_traces, dtype=np.int16)
        #     processed_traces[:, good_channels] = prediction_traces
        #     self.add_traces_to_plot_items(
        #         traces=processed_traces[:, to_display],
        #         good_channels=good_channels[to_display],
        #         view="prediction",
        #     )
        #
        # if self.residual_button.isChecked():
        #     if self.residual_traces is None:
        #         if self.whitened_traces is None:
        #             whitened_traces = filter_and_whiten(
        #                 raw_traces=raw_traces,
        #                 params=params,
        #                 probe=probe,
        #                 whitening_matrix=self.whitening_matrix,
        #                 good_channels=good_channels,
        #             )
        #
        #             self.whitened_traces = whitened_traces
        #
        #         else:
        #             whitened_traces = self.whitened_traces
        #
        #         if self.prediction_traces is None:
        #             prediction_traces = get_predicted_traces(
        #                 matrix_U=asnumpy(intermediate.U_s),
        #                 matrix_W=asnumpy(intermediate.Wphy),
        #                 sorting_result=intermediate.st3,
        #                 time_limits=(start_time, end_time),
        #             )
        #
        #             self.prediction_traces = prediction_traces
        #
        #         else:
        #             prediction_traces = self.prediction_traces
        #
        #         residual_traces = whitened_traces - prediction_traces
        #
        #         self.residual_traces = residual_traces
        #     else:
        #         residual_traces = self.residual_traces
        #
        #     processed_traces = np.zeros_like(raw_traces, dtype=np.int16)
        #     processed_traces[:, good_channels] = residual_traces
        #     self.add_traces_to_plot_items(
        #         traces=processed_traces[:, to_display],
        #         good_channels=good_channels[to_display],
        #         view="residual",
        #     )

    def _update_colormap(
            self,
            params,
            probe,
            binary_file,
            filt_binary_file,
            to_display,
            start_time,
            end_time
    ):
        # self.hide_traces()

        if self.raw_button.isChecked():
            colormap_min, colormap_max = -32.0, 32.0
            raw_traces = binary_file[start_time:end_time].cpu().numpy()
            self.add_image_to_plot(
                raw_traces[to_display, :].T,
                colormap_min,
                colormap_max,
            )

        elif self.whitened_button.isChecked():
            whitened_traces = filt_binary_file[start_time:end_time].cpu().numpy()

            colormap_min, colormap_max = -4.0, 4.0
            self.add_image_to_plot(
                whitened_traces[to_display, :].T,
                colormap_min,
                colormap_max,
            )

        # elif self.prediction_button.isChecked():
        #     if self.prediction_traces is None:
        #         prediction_traces = get_predicted_traces(
        #             matrix_U=asnumpy(intermediate.U_s),
        #             matrix_W=asnumpy(intermediate.Wphy),
        #             sorting_result=intermediate.st3,
        #             time_limits=(start_time, end_time),
        #         )
        #         self.prediction_traces = prediction_traces
        #     else:
        #         prediction_traces = self.prediction_traces
        #     colormap_min, colormap_max = -4.0, 4.0
        #     processed_traces[:, good_channels] = prediction_traces
        #     self.add_image_to_plot(
        #         processed_traces[:, to_display],
        #         colormap_min,
        #         colormap_max,
        #     )
        #
        # elif self.residual_button.isChecked():
        #     if self.residual_traces is None:
        #         if self.whitened_traces is None:
        #             whitened_traces = filter_and_whiten(
        #                 raw_traces=raw_traces,
        #                 params=params,
        #                 probe=probe,
        #                 whitening_matrix=self.whitening_matrix,
        #                 good_channels=good_channels,
        #             )
        #
        #             self.whitened_traces = whitened_traces
        #
        #         else:
        #             whitened_traces = self.whitened_traces
        #
        #         if self.prediction_traces is None:
        #             prediction_traces = get_predicted_traces(
        #                 matrix_U=asnumpy(intermediate.U_s),
        #                 matrix_W=asnumpy(intermediate.Wphy),
        #                 sorting_result=intermediate.st3,
        #                 time_limits=(start_time, end_time),
        #             )
        #
        #             self.prediction_traces = prediction_traces
        #
        #         else:
        #             prediction_traces = self.prediction_traces
        #
        #         residual_traces = whitened_traces - prediction_traces
        #
        #         self.residual_traces = residual_traces
        #     else:
        #         residual_traces = self.residual_traces
        #     colormap_min, colormap_max = -4.0, 4.0
        #     processed_traces[:, good_channels] = residual_traces
        #     self.add_image_to_plot(
        #         processed_traces[:, to_display],
        #         colormap_min,
        #         colormap_max,
        #     )


class KSPlotWidget(pg.PlotWidget):
    signalChangeTimePoint = QtCore.pyqtSignal(float)
    signalChangeChannel = QtCore.pyqtSignal(float)
    signalChangeTimeRange = QtCore.pyqtSignal(float)
    signalChangeScaling = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(KSPlotWidget, self).__init__(*args, **kwargs)

    def wheelEvent(self, ev):
        # QtWidgets.QGraphicsView.wheelEvent(self, ev)
        # if not self.mouseEnabled:
        #     ev.ignore()
        #     return

        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.angleDelta().x()

        direction = delta / np.abs(delta)
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ControlModifier:
            # control pressed while scrolling
            self.signalChangeChannel.emit(direction)

        elif modifiers == QtCore.Qt.AltModifier:
            # alt pressed while scrolling
            self.signalChangeScaling.emit(direction)

        elif modifiers == QtCore.Qt.ShiftModifier:
            # shift pressed while scrolling
            self.signalChangeTimeRange.emit(direction)

        else:
            # other key / no key pressed while scrolling
            self.signalChangeTimePoint.emit(direction)

        ev.accept()
        return

    def mouseMoveEvent(self, ev):
        pass

# class WhiteningMatrixCalculator(QtCore.QRunnable):
#
#     def __init__(self, raw_data, probe, params):
#         super(WhiteningMatrixCalculator, self).__init__()
#         self.raw_data = raw_data
#         self.params = params
#         self.probe = probe
#
#         self.signals = CalculatorSignals()
#
#     def run(self):
#         try:
#             logger.info("Calculating approx. whitening matrix.")
#             whitening_matrix = get_approx_whitening_matrix(
#                 raw_data=self.raw_data,
#                 params=self.params,
#                 probe=self.probe,
#             )
#         except Exception as e:
#             logger.error(e)
#         else:
#             logger.info("Approx. whitening matrix calculated.")
#             self.signals.result.emit(whitening_matrix)
#         finally:
#             self.signals.finished.emit()
#
#
# class CalculatorSignals(QtCore.QObject):
#     finished = QtCore.pyqtSignal()
#     result = QtCore.pyqtSignal(object)
