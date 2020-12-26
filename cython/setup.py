from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        r'testing',
        [r'cython.pyx']
    ),
]

setup(
    name='testing',
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)