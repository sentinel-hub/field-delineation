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
    python_requires='>=3.6',
    version=get_version(),
    description='EO Research Field Delineation',
    url='https://github.com/sentinel-hub/field-delineation',
    author='Sinergise EO research team',
    author_email='eoresearch@sinergise.com',
    packages=find_packages(),
    package_data={'fd': ['evalscripts/data_evalscript.js']},
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'DEV': parse_requirements('requirements-dev.txt')
    },
    zip_safe=False
)
