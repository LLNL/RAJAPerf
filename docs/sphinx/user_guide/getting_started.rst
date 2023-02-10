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

This section should help get you building the RAJA Performance Suite code
quickly.

.. _getting_started_reqs-label:

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

.. _getting_started_getcode-label:

==================
Getting the Code
==================

The RAJA Performance Suite project is hosted on GitHub:
`GitHub RAJA Performance Suite project <https://github.com/LLNL/RAJAPerf>`_. 
To get the code, clone the repository into a local working space using the 
command::

   $ git clone --recursive https://github.com/LLNL/RAJAPerf.git

The ``--recursive`` option is used to pull all RAJA Performance Suite 
Git *submodules*, on which it depends, into your local copy of the repository.

After running the ``git clone`` command, a copy of the RAJA Performance Suite
repository will reside in the ``RAJAPerf`` subdirectory where you ran the 
clone command. You will be on the ``develop`` branch, which is the default 
RAJA Performance Suite branch. For example::

  $ cd RAJAPerf
  $ git branch | grep \*
  * develop

If you do not pass the ``--recursive`` argument to the ``git clone``
command, you can also type the following command in the ``RAJAPerf`` 
directory after cloning::

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

The RAJA Performance Suite has several dependencies that are required, and
which are contained in the Suite submodules. For most usage, we recommend 
using the submodules which are pinned to specific versions of those libraries 
for each branch/release.

The most important dependencies are:

- `RAJA <https://github.com/LLNL/RAJA>`_
- `BLT build system <https://github.com/LLNL/blt>`_
- `CMake <https://cmake.org/>`_ version 3.23 or greater when building the HIP back-end, and version 3.20 or greater otherwise.
- A C++ 14 standard compliant compiler

Please see `RAJA Dependencies <https://raja.readthedocs.io/en/develop/sphinx/user_guide/getting_started.html#dependencies>`_ for more information about
RAJA dependencies.

.. _getting_started_build-label:

==================
Build and Install
==================

The build and install process for the RAJA Performance Suite is similar to
the process for RAJA. Please see `RAJA Build and Install <https://raja.readthedocs.io/en/develop/sphinx/user_guide/getting_started.html#build-and-install>`_
for more information.

When building the RAJA Performance Suite,
RAJA and the RAJA Performance Suite are built together using the same CMake
configuration. For convenience, we include scripts in the ``scripts``
directory that invoke associated configuration files (CMake cache files)
in the RAJA submodule. For example, the ``scripts/lc-builds`` directory
contains scripts that show how we build code for testing on platforms in
the Computing Center at Lawrence Livermore National Laboratory. Each build 
script creates a
descriptively-named build space directory in the top-level RAJA Performance 
Suite directory and runs CMake with a configuration appropriate for the 
platform and specified compiler(s). After CMake completes, enter the build 
directory and type ``make`` (or ``make -j <N>`` or ``make -j`` for a parallel 
build using N processor cores, or all available processor cores on a node,
respectively). For example::

  $ ./scripts/blueos_nvcc_clang.sh 10.2.89 70 10.0.1
  $ cd build_blueos_nvcc10.2.89-sm_70-clang10.0.1
  $ make -j 

will build the code for CPU-GPU execution using the clang 10.0.1 compiler for
the CPU and CUDA 10.2.89 for the GPU. The GPU executable code will target
the CUDA compute architecture ``sm_70``.

.. note:: The scripts in the ``scripts/lc-builds`` directory contain
          helpful examples of running CMake to generate a variety of 
          build configurations.

You can also create your own build directory and run CMake with your own
options from there. For example::

  & mkdir my-build
  & cd my-build
  & cmake <cmake args> ../
  & make -j 

When no CMake test options are provided, only the RAJA Performance Suite code 
will be built. If you want to build both the Suite tests and RAJA tests (to
verify that everything is built properly), pass the following options to 
CMake: ``-DENABLE_TESTS=On`` and ``-DRAJA_PERFSUITE_ENABLE_TESTS=On``. This 
can be done on the command line if you run CMake directly or by editing the 
build script you are using. If you want to build the Suite tests, but not 
RAJA tests, pass the two CMake options above plus the option 
``-DRAJA_ENABLE_TESTS=Off``. 

In any case, after the build completes, you can type `make test` to run the 
tests you have chosen to build and see the results.

.. note:: Which kernel variants that can be run depend on which programming
          model features have been enabled for a build. By default, only
          *sequential* CPU RAJA and baseline variants will be built. To
          additionally enable OpenMP variants, for example, you must pass the 
          ``DENABLE_OPENMP=On`` option to CMake. Similarly, for CUDA, HIP,
          and other programming model variants.

.. important:: For GPU-enabled builds, only one GPU back-end can be enabled
               in a single executable. However, CPU and GPU enabled execution
               can be enabled in a single executable. For example, one can
               enable CPU sequential, OpenMP, and CUDA GPU variants in a build.
               Similarly for HIP GPU variants. 

Building with MPI
-----------------

Some provided configurations will build the Performance Suite with
MPI support enabled. For example::

  $ ./scripts/blueos_spectrum_nvcc_clang.sh rolling-release 10.2.89 70 10.0.1
  $ cd build_lc_blueos-spectrumrolling-release-nvcc10.2.89-sm_70-clang10.0.1
  $  make -j

In general MPI support can be enabled by passing the `-DENABLE_MPI=On` option
to CMake and providing a MPI compiler wrapper via the
``-DMPI_CXX_COMPILER=/path/to/mpic++`` option to CMake in addition to other 
CMake options. For example::

  $ mkdir my-mpi-build
  $ cd my-mpi-build
  $ cmake <cmake args \
    -DENABLE_MPI=On -DMPI_CXX_COMPILER=/path/to/mpic++ \
    ..
  $ make -j

When MPI is enabled, you can run the RAJA Performance Suite in a way that
mimics how a real application would run, such as by fully utilizing a GPU
or all CPU cores. MPI-enabled execution is supported to generate realistic
performance data with the Suite.

Building with specific GPU thread-block size tunings
-----------------------------------------------------

If desired, you can build a version of the RAJA Performance Suite code with 
multiple GPU kernel versions that will run with different GPU thread-block 
sizes. The CMake option for this is 
``-DRAJA_PERFSUITE_GPU_BLOCKSIZES=<list,of,block,sizes>``. For example::

  $ mkdir my-gpu-build
  $ cd my-gpu-build
  $ cmake <cmake args> \
    -DRAJA_PERFSUITE_GPU_BLOCKSIZES=64,128,256,512,1024 \
    ..
  $ make -j

