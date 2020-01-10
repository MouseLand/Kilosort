
from pathlib import Path
import logging
# import sys

import click
import numpy as np

from phylib.io.model import load_raw_data
from phylib.utils import Bunch, connect
from phylib.utils.geometry import linear_positions

from phy.apps import capture_exceptions
from phy.cluster.views.trace import TraceImageView, select_traces
from phy.cluster.views.probe import ProbeView
from phy.gui import create_app, run_app, GUI, IPythonView
from phy.gui.qt import QSlider, Qt, QAction, QLabel

logger = logging.getLogger(__name__)


class KilosortGUICreator(object):
    def __init__(self, dat_path, **kwargs):
        self.dat_path = Path(dat_path).resolve()
        self.gui_name = 'PythonKilosortGUI'
        self.__dict__.update(kwargs)
        self.load_data()

    def load_data(self):
        dat_path = self.dat_path
        if dat_path.suffix == '.cbin':
            data = load_raw_data(path=dat_path)
            sample_rate = data.sample_rate
            n_channels_dat = data.shape[1]
        else:
            sample_rate = float(self.sample_rate)
            assert sample_rate > 0.

            n_channels_dat = int(self.n_channels_dat)

            dtype = np.dtype(self.dtype)
            offset = int(self.offset or 0)
            order = getattr(self, 'order', None)

            # Memmap the raw data file.
            data = load_raw_data(
                path=dat_path,
                n_channels_dat=n_channels_dat,
                dtype=dtype,
                offset=offset,
                order=order,
            )
        self.data = data
        self.duration = self.data.shape[0] / sample_rate

    def create_params(self):
        # TODO: generate a params.py (with confirmation if overwrite), and ks2_params dictionary
        pass

    def find_dead_channels(self):
        # TODO
        pass

    def preprocess(self):
        # TODO
        pass

    def spike_sort(self):
        # TODO
        pass

    def create_buttons(self, gui):
        action = QAction("Create params.py", gui)
        action.triggered.connect(self.create_params)
        gui._toolbar.addAction(action)

        action = QAction("Find dead channels", gui)
        action.triggered.connect(self.find_dead_channels)
        gui._toolbar.addAction(action)

        action = QAction("Preprocess", gui)
        action.triggered.connect(self.preprocess)
        gui._toolbar.addAction(action)

        action = QAction("Spike sort", gui)
        action.triggered.connect(self.spike_sort)
        gui._toolbar.addAction(action)

    def create_ipython_view(self, gui):
        view = IPythonView()
        view.attach(gui)

        view.inject(gui=gui, creator=self, data=self.data)

        # TODO: redirect KS2 output in it

        return view

    def create_trace_view(self, gui):
        gui._toolbar.addWidget(QLabel("Time selection: "))
        time_slider = QSlider(Qt.Horizontal, gui)
        time_slider.setRange(0, 100)
        time_slider.setTracking(False)
        gui._toolbar.addWidget(time_slider)
        self.time_slider = time_slider

        def _get_traces(interval):
            return Bunch(data=select_traces(self.data, interval, sample_rate=self.sample_rate))

        view = TraceImageView(
            traces=_get_traces,
            n_channels=self.n_channels_dat,
            sample_rate=self.sample_rate,
            duration=self.duration,
            enable_threading=False,
        )
        view.attach(gui)

        self.move_time_slider_to(view.time)

        @time_slider.valueChanged.connect
        def time_slider_changed():
            view.go_to(float(time_slider.sliderPosition()) / 100 * self.duration)

        @connect(sender=view)
        def on_time_range_selected(sender, interval):
            self.move_time_slider_to(.5 * (interval[0] + interval[1]))

        return view

    def move_time_slider_to(self, time):
        self.time_slider.setSliderPosition(int(time / self.duration * 100))

    def create_probe_view(self, gui):
        channel_positions = linear_positions(self.n_channels_dat)
        view = ProbeView(channel_positions)
        view.attach(gui)
        # TODO: update positions dynamically
        return view

    def create_gui(self):
        """Create the spike sorting GUI."""

        gui = GUI(name=self.gui_name, subtitle=self.dat_path.resolve(), enable_threading=False)
        gui.has_save_action = False
        gui.set_default_actions()
        self.create_buttons(gui)
        self.create_ipython_view(gui)
        self.create_trace_view(gui)
        self.create_probe_view(gui)
        # TODO: KeyValueWidget with KS2 params (auto generated from default params)
        # TODO: KeyValueWidget with data params
        # probe coordinates: text box with python code that loads channel_positions.npy, or
        #   xc.npy/kcoords: need to define channel_positions variable
        # channel mapping: load npy, need to define channel_map variable
        # interval_start
        # interval_stop

        return gui


# @phycli.command('kilosort')  # pragma: no cover
@click.command()
@click.argument('dat-path', type=click.Path(exists=True))
@click.option('-s', '--sample-rate', type=float)
@click.option('-d', '--dtype', type=str)
@click.option('-n', '--n-channels', type=int)
@click.option('-h', '--offset', type=int)
@click.option('-f', '--fortran', type=bool, is_flag=True)
# @click.pass_context
def kilosort(dat_path, **kwargs):
    """Launch the trace GUI on a raw data file."""
    create_app()

    with capture_exceptions():
        kwargs['n_channels_dat'] = kwargs.pop('n_channels')
        kwargs['order'] = 'F' if kwargs.pop('fortran', None) else None
        creator = KilosortGUICreator(dat_path, **kwargs)
        gui = creator.create_gui()
        gui.show()
        run_app()
        gui.close()
