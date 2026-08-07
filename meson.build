project('molsim_accel', 'c',
  version : '0.1.0',
  meson_version: '>=1.1.0',
  default_options : ['warning_level=2','b_lto=true','cpp_std=c++23'],
)

add_languages('fortran', native: false)
add_languages('cpp', native: false)

conf_data = configuration_data()

cppcompiler = meson.get_compiler('cpp')
cppargs = []
cpplinkargs = []
fortargs = []

nativebuild = true

cpufam = host_machine.cpu_family()

if cppcompiler.get_argument_syntax() == 'gcc'

  avx2march = '-march=x86-64-v3'
  avx512march = '-march=x86-64-v4'

  # Apple clang doesn't support OpenMP threading
  if cppcompiler.has_argument('-fopenmp')
    cppargs += '-fopenmp'
    cpplinkargs += '-fopenmp'
  else
    warning('Your compiler (likely Apple clang) does not support OpenMP. Parallelism will be disabled.')
  endif

  # Determine appropriate compiler flags for machine
  # This is very "laptop" focused, and needs some changes for
  # heterogeneous clusters

  if cpufam == 'x86_64'
    if nativebuild
      cppargs  += ['-march=native', '-mtune=native']
      fortargs += ['-march=native', '-mtune=native']
    endif
    conf_data.set('MOLSIM_X86_64', true)
  elif cpufam == 'aarch64'
    if nativebuild
      cppargs += '-mcpu=native'
    endif
    conf_data.set('MOLSIM_ARM', true)
  endif
elif cppcompiler.get_argument_syntax() == 'msvc'
  avx2march = '/arch:AVX2'
  avx512march = '/arch:AVX512'
  if cppcompiler.has_argument('/openmp:experimental')
    cppargs += '/openmp:experimental'
  else
    warning('Your compiler does not support OpenMP. Parallelism will be disabled.')
  endif
endif

# Check for presence of std::print
printcode = '''#include <print>
void f() { std::print("Hello world"); }
'''

printsupport = cppcompiler.compiles(printcode, name: 'C++ compiler supports std::print')

assert(printsupport, 'C++ compiler does not support std::print (too old)')




py = import('python').find_installation(pure: false)

# NumPy >=2.0 provides include dirs via dependency()
np_dep = dependency('numpy', method: 'config-tool')

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check : true
).stdout().strip()

# f2py include dir (for fortranobject.h) is not in dependency('numpy'),
# so add it separately
f2py_dep = declare_dependency(
  include_directories : incdir_f2py,
)

configure_file(output: 'molsim_config.h',
               configuration: conf_data)

# Generate the f2py wrappers
fort_accel_source = custom_target('fortran_accelmodule.c',
  input : ['molsim/interface.pyf'],
  output : ['fortran_accelmodule.c', 'fortran_accel-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@']
)

py.extension_module('fortran_accel',
  ['molsim/fort_funcs.f90', fort_accel_source, incdir_f2py / 'fortranobject.c'],
  dependencies : [np_dep, f2py_dep],
  fortran_args : fortargs,
  subdir: 'molsim',
  install : true,
)

pybind11_dep = dependency('pybind11')

sleef_dep = declare_dependency(
    include_directories : ['molsim/cpp/sleef/include']
)

cpp_sources = ['molsim/cpp/classes.cxx', 'molsim/cpp/functional.cxx', 'molsim/cpp/functional_purec.cxx',
               'molsim/cpp/detect_cpu.cxx', 'molsim/cpp/binding.cxx']
    
cpp_deps = [np_dep, sleef_dep, pybind11_dep]

if cpufam == 'aarch64'
    cpp_sources += 'molsim/cpp/functional_neon.cxx'
elif cpufam == 'x86_64'
    molsim_avx2 = static_library('molsim_avx2', ['molsim/cpp/functional_avx2.cxx'],
                                 cpp_args: [avx2march]+cppargs) # prepend, so native build march will supersede
    cpp_deps += molsim_avx2
endif


py.extension_module('molsim_cpp',
  cpp_sources,
  dependencies : cpp_deps,
  cpp_args: cppargs,
  subdir: 'molsim',
  install : true,
  link_args: cpplinkargs
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
