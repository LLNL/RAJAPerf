.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _build-label:

*********************************************
Building the RAJA Performance Suite
*********************************************

This section will help you build the RAJA Performance Suite code so you can
start running it. 

.. _build_reqs-label:

============
Requirements
============

The primary requirement for building the RAJA Performance Suite are:

- C++ compiler with C++14 support
- `CMake <https://cmake.org/>`_ version 3.23 or greater when building the HIP back-end, and version 3.20 or greater otherwise.

For the most part, available configuration options and how to enable or 
disable them are similar to `RAJA build options <https://raja.readthedocs.io/en/develop/sphinx/user_guide/config_options.html#configopt-label>`_. 

Later in this section, we discuss options that are specific to the 
RAJA Performance Suite.

.. _build_getcode-label:

==================
Getting the Code
==================

The RAJA Performance Suite code is hosted on the 
`GitHub RAJA Performance Suite project <https://github.com/LLNL/RAJAPerf>`_. 
To get the code, clone the repository into a local working space using the 
command::

   $ git clone --recursive https://github.com/LLNL/RAJAPerf.git

The ``--recursive`` option is required to pull all RAJA Performance Suite 
Git *submodules* into your local copy of the repository.

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

Either way, the result is the same and you should be good to configure the
code and build it.

.. note:: * If you are in your local copy of the RAJA Performance Suite repo
            and you switch to a different branch (e.g., you run the 
            command ``git checkout <different branch name>``), you may need to 
            run the command ``git submodule update`` to set the Git *submodule
            versions* to what is used by the new branch. To see if this is 
            required, the ``git status`` command will indicate whether the
            submodules are at the proper versions. 
          * If the *set of submodules* in a new branch is different than the
            previous branch you were on, you may need to run the command
            ``git submodule update --init --recursive`` (described above) to 
            pull in the correct set of submodule and versions.

.. _build_depend-label:

==================
Dependencies
==================

The RAJA Performance Suite has several required dependencies. These are
contained in the Suite Git submodules. So for most usage, we recommend 
using the submodules which are pinned to specific versions of those libraries 
in each branch or release.

The most important dependencies are:

- `RAJA <https://github.com/LLNL/RAJA>`_
- `BLT build system <https://github.com/LLNL/blt>`_

RAJA also contains dependencies, which are discussed in 
`RAJA Dependencies <https://raja.readthedocs.io/en/develop/sphinx/user_guide/build.html#dependencies>`_.

.. _build_build-label:

==================
Build and Install
==================

The build and install process for the RAJA Performance Suite is similar to
the process for RAJA, which is described in `RAJA Build and Install <https://raja.readthedocs.io/en/develop/sphinx/user_guide/build.html#build-and-install>`_.

When building the RAJA Performance Suite, RAJA and the RAJA Performance Suite 
are built together using the same CMake configuration. The basic process for 
generating a build space and configuration is to create a build directory and 
run CMake in it. For example::

  $ pwd
  path/to/RAJAPerf
  $ mkdir my-build
  $ cd my-build
  $ cmake <cmake args> ..
  $ make -j

For convenience and informational purposes, we maintain scripts in the 
``scripts`` directory for various build configurations. These scripts invoke 
associated *host-config* files (CMake cache files) in the RAJA submodule. For 
example, the ``scripts/lc-builds`` directory contains scripts that we use 
during development to generate build configurations for machines
in the Livermore Computing Center at Lawrence Livermore National Laboratory. 
These scripts are designed to be run in the top-level RAJAPerf directory. Each 
script creates a descriptively-named build space directory and runs CMake with 
a configuration appropriate for the platform and specified compiler(s). To 
compile the code after CMake completes, enter the build directory and type 
``make`` (or ``make -j <N>`` or ``make -j`` for a parallel build using N 
processor cores, or all available processor cores on a node, respectively). 
For example::

  $ ./scripts/lc-builds/blueos_nvcc_clang.sh 10.2.89 70 10.0.1
  $ cd build_blueos_nvcc10.2.89-70-clang10.0.1
  $ make -j 

will build the code for CPU-GPU execution using the clang 10.0.1 compiler for
the CPU and CUDA 10.2.89 for the GPU. The GPU executable code will target
the CUDA compute architecture ``sm_70``.

.. note:: The scripts in the ``scripts/lc-builds`` directory contain
          helpful examples of running CMake to generate a variety of 
          build configurations.

When no CMake test options are provided, only the RAJA Performance Suite code 
will be built. If you want to build both the Suite tests and RAJA tests (to
verify that everything is built properly), pass the following options to 
CMake: ``-DENABLE_TESTS=On`` and ``-DRAJA_PERFSUITE_ENABLE_TESTS=On``. This 
can be done on the command line if you run CMake directly or by editing the 
build script you are using. If you want to build the Suite tests, but not 
RAJA tests, pass the two CMake options above plus the option 
``-DRAJA_ENABLE_TESTS=Off``. In any case, after the build completes, you can 
type ``make test`` to run the tests you have built and see the results.

.. note:: The kernel variants that can be run depends on which programming
          model features have been enabled in a build configuration. By 
          default, only *sequential* CPU RAJA and baseline variants will be 
          built. To additionally enable OpenMP variants, for example, you must 
          pass the ``DENABLE_OPENMP=On`` option to CMake. Similar options will
          enable other variants for CUDA, HIP, and other programming models.

.. note:: For GPU-enabled builds, only one GPU back-end can be enabled in a 
          single executable. However, CPU and GPU execution can be 
          enabled in a single executable. For example, one can enable CPU 
          sequential, OpenMP, and CUDA GPU variants in a build. Similarly 
          for HIP GPU variants. 

Building with MPI
-----------------

Earlier, we mentioned that the Suite can be built with MPI enabled and
described why this is useful. Some configuration scripts we provide will 
configure a build with MPI support enabled. For example::

  $ ./scripts/lc-builds/lc-blueos_spectrum_nvcc_clang.sh rolling-release 10.2.89 70 10.0.1
  $ cd build_lc_blueos-spectrumrolling-release-nvcc10.2.89-70-clang10.0.1
  $  make -j

This will configure a build to use the *rolling release* of the Spectrum MPI
implementation for an appropriate Livermore Computing system.

In general, MPI support can be enabled by passing the `-DENABLE_MPI=On` option
to CMake and providing a MPI compiler wrapper via the
``-DMPI_CXX_COMPILER=/path/to/mpic++`` option to CMake, in addition to other 
necessary CMake options. For example::

  $ mkdir my-mpi-build
  $ cd my-mpi-build
  $ cmake <cmake args> \
    -DENABLE_MPI=On -DMPI_CXX_COMPILER=/path/to/mpic++ \
    ..
  $ make -j

Building with specific GPU thread-block size tunings
-----------------------------------------------------

If desired, you can build a version of the RAJA Performance Suite code with 
multiple versions of GPU kernels that will run with different GPU thread-block 
sizes. The CMake option for this is 
``-DRAJA_PERFSUITE_GPU_BLOCKSIZES=<list,of,block,sizes>``. For example::

  $ mkdir my-gnu-build
  $ cd my-gpu-build
  $ cmake <cmake args> \
    -DRAJA_PERFSUITE_GPU_BLOCKSIZES=64,128,256,512,1024 \
    ..
  $ make -j

will build versions of GPU kernels that use 64, 128, 256, 512, and 1024 threads
per GPU thread-block.

Building with Caliper
---------------------

RAJAPerf Suite may also use Caliper instrumentation, with per variant & tuning output into .cali files. While Caliper is low-overhead
it is not zero, so it will add a small amount of timing skew in its data as 
compared to the original. Caliper output enables usage of performance analysis tools like Hatchet and Thicket.
For much more on Caliper, Hatchet and Thicket, read their documentation here:

| - `Caliper Documentation <http://software.llnl.gov/Caliper/>`_ 
| - `Hatchet User Guide <https://llnl-hatchet.readthedocs.io/en/latest/user_guide.html>`_ 
| - `Thicket User Guide <https://thicket.readthedocs.io/en/latest/>`_ 


Caliper *annotation* is in the following tree structure::

  RAJAPerf
    Group
      Kernel

| Build against these Caliper versions
|
|   **caliper@2.9.0** (preferred target)
|   **caliper@master** (if using older Spack version)

1: Use one of the caliper build scripts in `scripts/lc-builds/*_caliper.sh`

2: Add the build options manually to an existing build::

  In Cmake scripts add
    **-DRAJA_PERFSUITE_USE_CALIPER=On**

  Add to **-DCMAKE_PREFIX_PATH**
    ;${CALIPER_PREFIX}/share/cmake/caliper;${ADIAK_PREFIX}/lib/cmake/adiak

  or use
    -Dcaliper_DIR -Dadiak_DIR package prefixes

For Spack : raja_perf +caliper ^caliper@2.9.0

For Uberenv: python3 scripts/uberenv/uberenv.py --spec +caliper ^caliper@2.9.0

If you intend on passing nvtx or roctx annotation to Nvidia or AMD profiling tools, 
build Caliper with +cuda cuda_arch=XX or +rocm respectively. Then you can specify
an additional Caliper service for nvtx or roctx like so: roctx example:

CALI_SERVICES_ENABLE=roctx rocprof --roctx-trace --hip-trace raja-perf.exe 
