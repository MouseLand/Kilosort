
from pathlib import Path
import logging
from textwrap import dedent

import click
import numpy as np

from phylib.io.model import load_raw_data
from phylib.utils import Bunch, connect
from phylib.utils.geometry import linear_positions

from phy.apps import capture_exceptions
from phy.cluster.views.trace import TraceImageView, select_traces
from phy.cluster.views.probe import ProbeView
from phy.gui import create_app, run_app, GUI, IPythonView, KeyValueWidget
from phy.gui.qt import QSlider, Qt, QLabel, QScrollArea, QVBoxLayout, QWidget
from phy.gui import Actions

from .default_params import default_params
from .main import run

logger = logging.getLogger(__name__)


# DEFAULT_PARAMS_PY = '''
# dat_path = {dat_path}
# n_channels_dat = {n_channels_dat}
# dtype = "{dtype}"
# offset = {offset}
# sample_rate = {sample_rate}
# '''


class Parameters(QWidget):
    pass


class KilosortGUICreator(object):
    def __init__(self, dat_path, **kwargs):
        self.dat_path = Path(dat_path).resolve()
        self.gui_name = 'PythonKilosortGUI'
        self.__dict__.update(kwargs)
        self.load_data()

    def load_data(self):
        # TODO: use EphysTraces
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

        # Parameters for the creation of params.py
        self.n_channels_dat = n_channels_dat
        self.offset = offset
        self.dtype = dtype

        self.data = data
        self.duration = self.data.shape[0] / sample_rate

    # def create_params(self):
        # paramspy = DEFAULT_PARAMS_PY.format(
        #     dat_path='["%s"]' % str(self.dat_path),
        #     n_channels_dat=self.n_channels_dat,
        #     offset=self.offset,
        #     dtype=self.dtype,
        #     sample_rate=self.sample_rate,
        # )

    def _run(self, stop_after=None):
        # TODO: test
        run(self.dat_path, self.probe, dir_path=self.dir_path, stop_after=stop_after)

    def find_good_channels(self):
        self._run('good_channels')
        # TODO: update probe view

    def preprocess(self):
        self._run('preprocess')
        # TODO: update trace view

    def spike_sort(self):
        self._run()
        # TODO: create custom logging handler that redirectors to ipython view
        # view.append_stream(...)

    def create_actions(self, gui):
        """Create the actions."""
        self.actions = Actions(gui)
        # self.actions.add(self.create_params, name="Create params.py", toolbar=True)
        self.actions.add(self.find_good_channels, name="Find good channels", toolbar=True)
        self.actions.add(self.preprocess, name="Preprocess", toolbar=True)
        self.actions.add(self.spike_sort, name="Spike sort", toolbar=True)

    def create_ipython_view(self, gui):
        """Add the IPython view."""
        view = IPythonView()
        view.attach(gui)

        view.inject(gui=gui, creator=self, data=self.data)

        return view

    def create_trace_view(self, gui):
        """Add the trace view."""
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
        """Move the time slider."""
        self.time_slider.setSliderPosition(int(time / self.duration * 100))

    def create_probe_view(self, gui):
        """Add the view that shows the probe layout."""
        channel_positions = linear_positions(self.n_channels_dat)
        view = ProbeView(channel_positions)
        view.attach(gui)
        # TODO: update positions dynamically when the probe view changes
        return view

    def create_params_widget(self, gui):
        """Create the widget that allows to enter parameters for KS2."""
        widget = KeyValueWidget(gui)
        for name, default in default_params.items():
            # HACK: None default params in KS2 are floats
            vtype = 'float' if default is None else None
            widget.add_pair(name, default, vtype=vtype)
        # Time interval (TODO: take it into account with EphysTraces).
        widget.add_pair('time interval', [0.0, self.duration])
        widget.add_pair('custom probe', dedent('''
        # Python code that returns a probe variable which is a Bunch instance,
        # with the following variables: NchanTOT, chanMap, xc, yc, kcoords.
        ''').strip(), 'multiline')

        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        # scroll.show()

        widget = Parameters(gui)
        layout = QVBoxLayout(widget)
        layout.addWidget(scroll)

        gui.add_view(widget)
        return widget

    def create_gui(self):
        """Create the spike sorting GUI."""

        gui = GUI(name=self.gui_name, subtitle=self.dat_path.resolve(), enable_threading=False)
        gui.has_save_action = False
        gui.set_default_actions()
        self.create_actions(gui)
        self.create_params_widget(gui)
        self.create_ipython_view(gui)
        self.create_trace_view(gui)
        self.create_probe_view(gui)

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
