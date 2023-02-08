.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _getting_started-label:

*********************************************
Getting Started With RAJA Performance Suite
*********************************************

This section should help get you up and running the RAJA Performannce Suite
quickly.

============
Requirements
============

The primary requirement for using the RAJA Performance Suite is a C++14 
standard compliant compiler. Different kernel variants use different 
programming models like CUDA or HIP and must be supported by the compiler 
you chose to build and run them. For the most part, available configuration 
options and how to enable or disable them are similar to those in RAJA,
which are described in `RAJA Build Options <https://raja.readthedocs.io/en/develop/sphinx/user_guide/config_options.html#configopt-label>`_. Later in this
section, we describe a few options that are specific to the RAJA Performance
Suite.

To build the RAJA Performance Suite and run basic kernel variants, you will 
need:

- C++ compiler with C++14 support
- `CMake <https://cmake.org/>`_ version 3.23 or greater when building the HIP back-end, and version 3.20 or greater otherwise.

==================
Get the Code
==================

The RAJA Performance Suite project is hosted on GitHub:
`GitHub RAJA Performance Suite project <https://github.com/LLNL/RAJAPerf>`_. 
To get the code, clone the repository into a local working space using the 
command::

   $ git clone --recursive https://github.com/LLNL/RAJAPerf.git

The ``--recursive`` option above is used to pull RAJA Performance Suite 
Git *submodules*, on which it depends, into your local copy of the repository.

After running the clone command, a copy of the RAJA Performance Suite
repository will reside in the ``RAJAPerf`` subdirectory where you ran the 
clone command. You will be on the ``develop`` branch, which is the default 
RAJA Performance Suite branch.

If you do not pass the ``--recursive`` argument to the ``git clone``
command, you can also type the following commands after cloning::

  $ cd RAJAPerf
  $ git submodule update --init --recursive

Either way, the end result is the same and you should be good to configure the
code and build it.

.. note:: * If you switch branches in a RAJA Performance Suite repo (e.g., 
            you are on a branch, with everything up-to-date, and you run the 
            command ``git checkout <different branch name>``, you may need to 
            run the command ``git submodule update`` to set the Git submodule
            versions to what is used by the new branch.
          * If the set of submodules in a new branch is different than the
            previous branch you were on, you may need to run the command
            ``git submodule update --init --recursive`` to pull in the
            correct set of submodule and versions.

.. _getting_started_depend-label:

==================
Dependencies
==================

The RAJA Performance Suite has several dependencies that are required based 
on how you want to build and use it. The RAJA Performance Suite Git repository 
has submodules that contain these dependencies. We recommend using the 
submodules which are pinned to specific versions of those libraries for 
most usage.

The most important dependencies are:

- `RAJA <https://github.com/LLNL/RAJA>`_
- `BLT build system <https://github.com/LLNL/blt>`_
- `CMake <https://cmake.org/>`_ version 3.23 or greater when building the HIP back-end, and version 3.20 or greater otherwise.
- A C++ 14 standard compliant compiler

Please see `RAJA Dependencies <https://raja.readthedocs.io/en/develop/sphinx/user_guide/getting_started.html#dependencies>`_ for more information.

.. _getting_started_build-label:

==================
Build and Install
==================

The process to build and install the RAJA Performance Suite is similar to
the process for RAJA. Please see `RAJA Build and Install <https://raja.readthedocs.io/en/develop/sphinx/user_guide/getting_started.html#build-and-install>`_
for more information.

When building the RAJA Performance Suite,
RAJA and the RAJA Performance Suite are built together using the same CMake
configuration. For convenience, we include scripts in the ``scripts``
directory that invoke corresponding configuration files (CMake cache files)
in the RAJA submodule. For example, the ``scripts/lc-builds`` directory
contains scripts that show how we build code for testing on platforms in
the Lawrence Livermore Computing Center. Each build script creates a
descriptively-named build space directory in the top-level RAJA Performance 
Suite directory and runs CMake with a configuration appropriate for the 
platform and compilers used. After CMake completes, enter the build directory 
and type `make` (or `make -j <N>` for a parallel build using N processor 
cores). If you omit the number of cores, the code will build in parallel 
using all available cores on the node you are running on to compile the code. 
For example::

  $ ./scripts/blueos_nvcc_clang.sh 10.2.89 70 10.0.1
  $ cd build_blueos_nvcc10.2.89-sm_70-clang10.0.1
  $ make -j 

will build the code for CPU-GPU execution using the clang 10.0.1 compiler for
the CPU and CUDA 10.2.89 for the GPU. The GPU executable code will target
the CUDA compute architecture ``sm_70``.

.. note:: The scripts in the ``scripts/lc-builds`` directory contain
          helpful examples of running CMake to generate a variety of 
          build configurations.

The provided configurations will only build the Performance Suite code by
default; i.e., it will not build the RAJA Performance Suite test codes. If you 
want to build the tests, for example, to verify your build is working properly,
just pass the following options to CMake ``-DENABLE_TESTS=On`` and
``-DRAJA_PERFSUITE_ENABLE_TESTS=On``, either on the command line if you run 
CMake directly or edit the script you are running to do this. Then, when the 
build completes, you can type `make test` to run the RAJA Performance Suite 
tests.

You can also create your own build directory and run CMake with your own
options from there. For example::

  & mkdir my-build
  & cd my-build
  & cmake <cmake args> ../
  & make -j 

Building with MPI
-----------------

Some of the provided configurations will build the Performance Suite with
MPI support enabled. For example::

  $ ./scripts/blueos_spectrum_nvcc_clang.sh rolling-release 10.2.89 70 10.0.1
  $ cd build_lc_blueos-spectrumrolling-release-nvcc10.2.89-sm_70-clang10.0.1
  $  make -j

In general MPI support can be enabled by passing the `-DENABLE_MPI=On` option
to CMake and providing a mpi compiler wrapper via the
``-DMPI_CXX_COMPILER=/path/to/mpic++`` option to CMake in addition to other 
CMake options. For example::

  $ mkdir my-mpi-build
  $ cd my-mpi-build
  $ cmake -DENABLE_MPI=On -DMPI_CXX_COMPILER=/path/to/mpic++ <cmake args> ../
  $ make -j

When MPI is enabled, you can run the RAJA Performance Suite in a way that
mimicks how a real application would run, such as by fully utilizing a GPU
or all CPU cores. MPI-enabled execution is supported to generate realistic
performance data with the Suite.

Building with specific GPU thread-block size tunings
-----------------------------------------------------

