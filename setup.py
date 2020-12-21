from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Densest subgraph',
    ext_modules=cythonize("densest_subgraph.pyx"),
    zip_safe=False,
)