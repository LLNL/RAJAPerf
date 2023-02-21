.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _run-label:

*********************************************
Running the RAJA Performance Suite
*********************************************

This section describes how to run the Suite and which execution options are 
available.

After the Suite code is compiled, following the instructions provided in
:ref:`build-label`, the executable will reside in the ``bin`` subdirectory 
of the build space. The executable will be able to run all kernels and 
variants that have been built depending on which CMake options were specified
to configure the build.

To run the Suite in its default mode, type the executable name with no 
command-line arguments::

  $ ./bin/raja-perf.exe

This will run all kernels and variants that have been built in their default
configurations. Information describing how the Suite will run along with
some information about each kernel will appear on the screen. More information
about kernel and execution details will also appear in a run report files 
generated in the run directory after Suite execution completes. 

.. note:: * You can pass the ``--dryrun`` command-line option to the executable
            to see a summary of how the Suite will execute without actually 
            running it.
          * You can choose the directory for output file names as well as
            output file names using command line options.

The Suite can be run in a variety of ways determined by the command-line 
options passed to the executable. For example, you can run or exclude subsets 
of kernels, variants, or groups. You can also pass options to set problem 
sizes, number of times each kernel is run (sampled), and many other run 
parameters.The goal is to build the code once and use scripts or other means 
to run the Suite in different ways for analyses you want to perform.

Each option appears in a *long form* with a double hyphen prefix (i.e., '--').
Commonly used options are also available in a one or two character *short form*
with a single hyphen prefix (i.e., '-') for convenience. To see available 
options along with a brief description of each, pass the `--help` or `-h` 
option to the executable::

  $ ./bin/raja-perf.exe --help

or

  $ ./bin/raja-perf.exe -h

.. note:: To see all available Suite execution options, pass the `--help` or 
          `-h` option to the executable.

Lastly, the program will report specific errors if given incorrect input, such
as an option that requires a value and no value is provided. It will also emit 
a summary of command-line arguments it was given if the input contains 
something that the code does not know how to parse. 

.. note: The Suite executable will attempt to provide helpful information
         if it is given incorrect input, such as command-line arguments that 
         it does not know how to parse. Ill-formed input will be noted in
         screen output, hopefully making it easy for users to correct erroneous 
         usage, such as mis-spelled option names.

.. _run_mpi-label:

==================
Running with MPI
==================

Running the Suite with MPI is just like running any other MPI application.
For example::

  $ srun -n 2 ./bin/raja-perf.exe

will run the entire Suite (all kernels and variants) in their default 
configurations on each of 2 MPI ranks. 

The kernel information output shows how each kernel is run on each rank. 
Timing is reported on rank 0 and is gathered by invoking an MPI barrier, 
starting a timer, running the kernel, invoking an MPI barrier, and then 
stopping the timer. Total problem size across all MPI ranks can be 
calculated, if desired, by multiplying the number of MPI ranks by the problem 
size reported in the kernel information. 

.. _run_omptarget-label:

======================
OpenMP target offload
======================

OpenMP target offload variants of the kernels in the Suite are 
considered a work-in-progress since the RAJA OpenMP target offload back-end 
is a work-in-progress. If you configure them to build, they can be run with
the executable `./bin/raja-perf-omptarget.exe` which is distinct from the one 
described above. When the OpenMP target offload variants were developed, it 
was not possible for them to co-exist in the same executable as CUDA 
variants, for example. In the future, the build system may be reworked so 
that the OpenMP target variants can be run from the same executable as the 
other variants.
