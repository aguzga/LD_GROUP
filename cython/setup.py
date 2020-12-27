from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        r'testing',
        [r'cython2.pyx']
    ),
]

setup(
    name='testing',
    ext_modules=cythonize(ext_modules, annotate = True),
    include_dirs=[numpy.get_include()],
    compiler_directives={'boundscheck': False}
)