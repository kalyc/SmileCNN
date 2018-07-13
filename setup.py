from setuptools import setup
from setuptools import find_packages

setup(
   name='SmileCNN',
   version='1.0',
   description='Train smile image detection models using Keras-MXNet and infer using MXNet Model Server',
   author='kalyc',
   install_requires=['numpy>=1.9.1',
                     'sklearn'
                     ], #external packages as dependencies
   packages=find_packages()
)
