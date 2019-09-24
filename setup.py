from setuptools import setup, find_packages
import os

version_file = open(os.path.join('./panos_utilities', 'VERSION'))
version = version_file.read().strip()
setup(name="panos_utilities",
      version=version,
      packages=find_packages())
