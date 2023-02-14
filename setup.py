import os
from distutils.core import setup
from setuptools import find_packages

# lib_folder = os.path.dirname(os.path.realpath(__file__))
# requirement_path = os.path.join(lib_folder, "requirements.txt")
# with open(requirement_path, "r") as f:
#     install_requires = f.readlines()

setup(
    name='bytetrack',
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