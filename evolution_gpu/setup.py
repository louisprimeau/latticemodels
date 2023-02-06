from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='heatbath_cpp',
      ext_modules=[cpp_extension.CppExtension('heatbath_cpp', ['hetabath.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})