#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

import os
import re
from setuptools import setup, find_packages


def parse_requirements(file):
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        return [line.strip() for line in req_file if '/' not in line]


def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'fd', '__init__.py')) as file:
        return re.findall("__version__ = \'(.*)\'", file.read())[0]


setup(
    name='fd',
    python_requires='>=3.6, <3.8',
    version=get_version(),
    description='EO Research Field Delineation',
    url='https://gitlab.com/nivaeu/uc2_fielddelineation',
    author='Sinergise EO research team',
    author_email='eoresearch@sinergise.com',
    packages=find_packages(),
    package_data={'fd': ['evalscripts/dates_evalscript.js', 'evalscripts/data_evalscript.js']},
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'DEV': parse_requirements('requirements-dev.txt')
    },
    zip_safe=False
)
