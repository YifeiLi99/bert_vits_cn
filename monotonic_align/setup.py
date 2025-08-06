from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="core",  # ⚠️ 注意不要写成 monotonic_align.core
        sources=["core.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(ext_modules, language_level="3"),
    script_args=['build_ext', '--inplace']
)