from setuptools import setup
from distutils.util import convert_path

ver_path = convert_path('panos_utilities/VERSION')
version_file = open(ver_path)
version = version_file.read().strip()
setup(name="panos_utilities",
      version=version,
      packages=['panos_utilities'],
      data_files=[('panos_utilities', ['panos_utilities/VERSION'])],
      install_requires=['numpy', 'scipy'])
