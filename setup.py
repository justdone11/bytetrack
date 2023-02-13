from distutils.core import setup
from setuptools import find_packages

setup(
    name='ByteTrackWrapper',
    version='0.1',
    packages=find_packages(include=['bytetrack', 'bytetrack.*']),
    install_requires=[
        'Cython',
        'cython-bbox',
        'numpy>=1.14.5',
        'opencv-python',
        'scipy',
        'lap',
        'six',
        'Pillow'
    ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)