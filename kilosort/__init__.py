#__version__ = "4"
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


from .utils import PROBE_DIR, DOWNLOADS_DIR
from .run_kilosort import run_kilosort
from .parameters import DEFAULT_SETTINGS
