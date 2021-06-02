from distutils.core import setup, Extension 
import os
#name of module 
name  = "osc"
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
#version of module 
version = "1.0"

import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# specify the name of the extension and source files 
# required to compile this 
ext_modules = Extension(name='_osc',sources=["osc.i","osc.c"]) 
  
setup(name=name, 
      version=version, 
      ext_modules=[ext_modules]) 
