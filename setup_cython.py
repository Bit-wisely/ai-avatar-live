import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import glob

# Find all python files in the src directory
src_files = glob.glob('src/**/*.py', recursive=True)

# Exclude __init__.py files as they usually don't need to be compiled
src_files = [f for f in src_files if not f.endswith('__init__.py')]

# Optionally include run.py, test_mp.py, test_mp_v2.py
# If you want to convert the tests and entry point too:
# src_files.extend(['run.py', 'test_mp.py', 'test_mp_v2.py'])

extensions = []
for file in src_files:
    # Build module name: 'src/camera/webcam_capture.py' -> 'src.camera.webcam_capture'
    module_name = file.replace(os.sep, '.').replace('/', '.').replace('.py', '')
    
    # We compile with optimization flags where possible
    extensions.append(
        Extension(
            name=module_name,
            sources=[file],
            extra_compile_args=['-O3', '-ffast-math'] if os.name != 'nt' else ['/O2']
        )
    )

if __name__ == '__main__':
    setup(
        name="AIAvatarLive_C_Extensions",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,   # Disable array bounds checking for performance
                'wraparound': False,    # Disable negative indexing for performance
                'cdivision': True       # C-style division (no modulo check for 0)
            },
            nthreads=4,          # Parallel conversion
            annotate=True        # Generates HTML files to analyze C conversion
        ),
    )
