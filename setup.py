from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).absolute().parent

long_description = (here / Path('README.md')).read_text()

_version = {}
exec((here / Path('gym_datums/_version.py')).read_text(), _version)

setup(
    name='gym_datums',
    version=_version['__version__'],
    description='Gym environments focused around time series.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SwamyDev/gym-datums',
    packages=find_packages(include=['gym_datums', 'gym_datums.*']),
    install_requires=['gym'],
    extras_require={'test': ['pytest', 'pytest-cov']},
)
