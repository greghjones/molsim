project('molsim_accel', 'c',
  version : '0.1.0',
  meson_version: '>=1.1.0',
  default_options : ['warning_level=2','b_lto=true','cpp_std=c++23'],
)

add_languages('fortran', native: false)
add_languages('cpp', native: false)

py = import('python').find_installation(pure: false)

# NumPy >=2.0 provides include dirs via dependency()
np_dep = dependency('numpy')

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check : true
).stdout().strip()

# f2py include dir (for fortranobject.h) is not in dependency('numpy'),
# so add it separately
f2py_dep = declare_dependency(
  include_directories : incdir_f2py,
)

# Generate the f2py wrappers
fort_accel_source = custom_target('fortran_accelmodule.c',
  input : ['molsim/interface.pyf', 'molsim/fort_funcs.f90'],
  output : ['fortran_accelmodule.c', 'fortran_accel-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@']
)

py.extension_module('fortran_accel',
  ['molsim/fort_funcs.f90', fort_accel_source, incdir_f2py / 'fortranobject.c'],
  dependencies : [np_dep, f2py_dep],
  fortran_args : '-march=native',
  subdir: 'molsim',
  install : true,
)

pybind11_dep = dependency('pybind11')

sleef_dep = declare_dependency(
    include_directories : ['molsim/cpp/sleef/include']
)

py.extension_module('molsim_cpp',
  ['molsim/cpp/classes.cxx', 'molsim/cpp/binding.cxx'],
  dependencies : [np_dep, sleef_dep, pybind11_dep],
  cpp_args: ['-fopenmp', '-march=native'],
  subdir: 'molsim',
  install : true,
)

# Install the Python package files
py.install_sources(
  'molsim/__init__.py',
  'molsim/functions.py',
  'molsim/radex/models.py',
  'molsim/radex/interface.py',
  'molsim/radex/__init__.py',
  'molsim/radex/utils.py',
  'molsim/radex/classes.py',
  'molsim/analysis.py',
  'molsim/plotting.py',
  'molsim/constants.py',
  'molsim/__init__.py',
  'molsim/utils.py',
  'molsim/stack.py',
  'molsim/stats.py',
  'molsim/mcmc/models.py',
  'molsim/mcmc/preprocess.py',
  'molsim/mcmc/tests/__init__.py',
  'molsim/mcmc/tests/test_load_data.py',
  'molsim/mcmc/tests/test_mcmc.py',
  'molsim/mcmc/__init__.py',
  'molsim/mcmc/base.py',
  'molsim/mcmc/compute.py',
  'molsim/fitting.py',
  'molsim/file_io.py',
  'molsim/classes.py',
  'molsim/samer/extract_spectra.py',
  'molsim/samer/SAMER_Classes.py',
  'molsim/samer/__init__.py',
  'molsim/samer/SAMER_Functions.py',
  'molsim/file_handling.py',
   pure: false,
   preserve_path: true
 )
