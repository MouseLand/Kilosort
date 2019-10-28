# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from pathlib import Path
import re

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


readme = (Path(__file__).parent / 'README.md').read_text()


# Find version number from `__init__.py` without executing it.
with (Path(__file__).parent / 'pykilosort/__init__.py').open('r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='pykilosort',
    version=version,
    license="BSD",
    description='Python port of KiloSort 2',
    long_description=readme,
    author='Cyrille Rossant',
    author_email='cyrille.rossant@gmail.com',
    url='https://github.com/rossant/pykilosort',
    packages=_package_tree('pykilosort'),
    package_dir={'pykilosort': 'pykilosort'},
    package_data={
        'pykilosort': [],
    },
    include_package_data=True,
    keywords='kilosort,spike sorting,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Framework :: IPython",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
