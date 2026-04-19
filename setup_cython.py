import os
import glob
from setuptools import setup, Extension
from Cython.Build import cythonize

# Collect all source files except __init__.py
src_files = [f for f in glob.glob('src/**/*.py', recursive=True) if not f.endswith('__init__.py')]

extensions = [
    Extension(
        name=f.replace(os.sep, '.').replace('/', '.').replace('.py', ''),
        sources=[f],
        extra_compile_args=['-O3'] if os.name != 'nt' else ['/O2']
    ) for f in src_files
]

if __name__ == '__main__':
    setup(
        name="AIAvatarLive_C",
        ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    )
