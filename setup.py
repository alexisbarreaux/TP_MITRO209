from setuptools import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace

setup(
    name='Densest subgraph',
    ext_modules=cythonize("densest_subgraph.pyx"),
    zip_safe=False,
)