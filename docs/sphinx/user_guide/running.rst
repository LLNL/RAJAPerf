.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _running-label:

*********************************************
Running the RAJA Performance Suite
*********************************************

This section describes how to run the Suite and which execution options are 
available.

After the Suite is compiled, the executable will be located in the ``bin``
subdirectory of the build space directory. The executable will be able to run
all kernels and variants that have been built depending on which programming
model back-ends have been enabled via CMake options.

To execute the Suite in its default mode, run the executable with no 
command-line arguments::

  $ ./bin/raja-perf.exe

This will run all kernels and variants that have been built in their default
configurations. Some information describing how the Suite will run along with
some information about each kernel will appear on the screen. More information
about kernel details will also appear in a run report file generated in your 
run directory after Suite execution completes. 

.. note:: You can pass the ``--dryrun`` command-line option to the executable
          to see a summary of how the Suite will execute without actually 
          running it.

The Suite can be run in a variety of ways that are determined by the options 
passed to the executable. For example, you can run or exclude subsets of 
kernels, variants, or groups. You can also pass other options to set problem 
sizes, number of times each kernel is run (sampled), etc. The idea is to build 
the code once and use scripts or other means to run the Suite in different 
ways for analyses you want to perform.

All options appear in a *long form* with a double hyphen prefix (i.e., '--').
Commonly used options are also available in a one or two character *short form*
with a single hyphen prefix (i.e., '-') for convenience. To see available 
options along with a brief description of each, pass the `--help` or `-h` 
option to the executable::

  $ ./bin/raja-perf.exe --help

or

  $ ./bin/raja-perf.exe -h

.. note:: To see all available Suite options, pass the `--help` or `-h` 
          option to the executable.

Lastly, the program will emit a summary of command-line arguments it was given
if the input contains something that the code does not know how to parse. 
The ill-formed input will be noted in the summary. Hopefully, this makes
it easy for users to correct erroneous usage, such as mis-spelled option names.

==================
Running with MPI
==================

Running the Suite with MPI is just like running any other MPI application.
For example::

  $ srun -n 2 ./bin/raja-perf.exe

will run the entire Suite (all kernels and variants) in their default 
configurations on each of 2 MPI ranks. 

The kernel information output shows how
each kernel is run on each rank. The total problem size across all MPI ranks
can be calculated by multiplying the number of MPI ranks by the problem
size in the kernel information. Timing is reported on rank 0 and is gathered
by doing an MPI barrier, starting the timer, running the kernel repetitions,
doing an MPI barrier, and then stopping the timer.

======================
OpenMP target offload
======================

The OpenMP target offload variants of the kernels in the Suite are 
considered a work-in-progress since the RAJA OpenMP target offload back-end 
is a work-in-progress. If you configure them to build, they can be run with
the executable `./bin/raja-perf-omptarget.exe` which is distinct from the one 
described above. At the time the OpenMP target offload variants were 
developed, it was not possible for them to co-exist in the same executable
as the CUDA variants, for example. In the future, the build system may
be reworked so that the OpenMP target variants can be run from the same
executable as the other variants.
