import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0,<2.0.0',
    'scipy',
    'scikit-learn',
    'tqdm',
    'torch>=1.6',
    'numba',
    'faiss-cpu',
    'psutil'
]

gui_deps = [
    'pyqtgraph>=0.13.0',
    'qtpy',
    'pyqt6',
    'pyqt6.sip',
    'matplotlib'
]

docs_deps = [
    'sphinx>=3.0',
    'sphinxcontrib-apidoc',
    'nbsphinx',
    'myst_parser',
    'sphinx_rtd_theme',
    'pandoc'
]

### remove torch install if already installed
try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__.split(".")[1])
    if version >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

### remove pyqt6 install if other qt backend installed
try:
    import PyQt5
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
    gui_deps.append("pyqt5")
    gui_deps.append("pyqt5.sip")
except:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kilosort",
    license="BSD",
    author="Marius Pachitariu",
    author_email="pachitarium@janelia.hhmi.org",
    description="spike sorting pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/kilosort",
    setup_requires=[
        'pytest-runner',
        'setuptools-scm',
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires=install_deps,
    tests_require=[
        'pytest'
    ],
    extras_require={
        'docs': docs_deps,
        'gui': gui_deps,
        'all': gui_deps,
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
        ]
    }
)
