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

This section describes how to run the Suite, after the Suite code is compiled 
following the instructions provided in :ref:`build-label`. 

.. _run_test-label:

==================
Running a test
==================

After compilation, a test executable will reside in the ``test`` subdirectory
of the build space. We use this test for our continuous integration testing
to make sure everything works when changes are made to the code. 
To run the test, type the test executable name::

  $ ./test/test-raja-perf-suite.exe

This will run a few iterations of each kernel and variant that was built 
based on the CMake options specified to configure the build. 

You can also run an individual kernel by setting an environment variable
to the name of the kernel you want to run. For example, 
if you use a csh/tcsh shell::

  $ setenv RAJA_PERFSUITE_UNIT_TEST DAXPY
  $ ./test/test-raja-perf-suite.exe 

or, if you use a bash shell::

  $ RAJA_PERFSUITE_UNIT_TEST=DAXPY ./test/test-raja-perf-suite.exe

In either case, the test will run all compiled variants of the 'DAXPY' 
kernel.

.. _run_suite-label:

==================
Running the Suite
==================

After compilation, the main executable will reside in the ``bin`` subdirectory 
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

.. note:: You can pass the ``--dryrun`` command-line option to the executable to see a summary of how the Suite will execute without actually running it.

The Suite can be run in a variety of ways determined by the command-line 
options passed to the executable. For example, you can run or exclude subsets 
of kernels, variants, or groups. You can also pass options to set problem 
sizes, number of times each kernel is run (sampled), and many other run 
parameters. The goal is to build the code once and use scripts or other means 
to run the Suite in different ways for the analyses you want to perform.

Each option appears in a *long form* with a double hyphen prefix (i.e., '--').
Commonly used options are also available in a one or two character *short form*
with a single hyphen prefix (i.e., '-') for convenience. To see available 
options along with a brief description of each, pass the `--help` or `-h` 
option to the executable::

  $ ./bin/raja-perf.exe --help

or::

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

============================
Additional Caliper Use Cases
============================

If you specified building with Caliper (``-DRAJA_PERFSUITE_USE_CALIPER=On``),
the generation of Caliper .cali files are automated for the most part.

However, there are a couple of other supported use cases.

Collecting PAPI topdown statistics on Intel Architectures
---------------------------------------------------------

On Intel systems, you can collect topdown PAPI counter statistics by using
command line arguments

``--add-to-spot-config, -atsc <string> [Default is none]``

This appends additional parameters to the built-in Caliper spot config.

To include some PAPI counters (Intel arch), add the following to the command 
line

``-atsc topdown.all``

Caliper's topdown service generates derived metrics from raw PAPI counters; 
a hierarchy of metrics to identify bottlenecks in out-of-order processors. 
This is based on an an approach described in Ahmad Yasin's paper 
*A Top-Down Method for Performance Analysis and Counters Architecture*. The 
top level of the hierarchy has a reliable set of four derived metrics or 
starting weights (sum to 1.0) which include:

#. **Frontend Bound.** Stalls attributed to the front end which is responsible for fetching and decoding program code.    
#. **Bad Speculation.** Fraction of the workload that is affected by incorrect execution paths, i.e. branch misprediction penalties
#. **Retiring.** Increases in this category reflects overall Instructions Per Cycle (IPC) fraction which is good in general. However, a large retiring fraction for non-vectorized code could also be a hint to the user to vectorize their code (see Yasin's paper) 
#. **Backend Bound.** Memory Bound where execution stalls are related to the memory subsystem, or Core Bound where execution unit occupancy is sub-optimal lowering IPC (more compiler dependent)

.. note:: Backend Bound = 1 - (Frontend Bound + Bad Speculation + Retiring)

.. note:: Caveats: 

          #. When collecting PAPI data in this way you'll be limited to running              only one variant, since Caliper maintains only one PAPI context.
          #. Small kernels should be run at large problem sizes to minimize 
             anomalous readings.
          #. Measured values are only relevant for the innermost level of the 
             Caliper tree hierarchy, i.e. Kernel.Tuning under investigation.
          #. Some lower level derived quantities may appear anomalous 
             with negative values. Collecting raw counters can help identify 
             the discrepancy.

``-atsc topdown-counters.all``

.. note:: Other caveats: Raw counter values are often noisy and require a lot 
          of accommodation to collect accurate data including: 
 
            * Turning off Hyperthreading
            * Turning off Prefetch as is done in Intel's Memory Latency 
              Checker (requires root access) 
            * Adding LFENCE instruction to serialize and bracket code under 
              test 
            * Disabling preemption and hard interrupts 

          See Andreas Abel's dissertation `Automatic Generation of Models of 
          Microarchitectures` for more info on this and for a comprehensive 
          look at the nanobench machinery.

Some helpful references:

`Yasin's Paper <https://www.researchgate.net/publication/269302126_A_Top-Down_method_for_performance_analysis_and_counters_architecture>`_

`Vtune-cookbook topdown method <https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html>`_

`Automatic Generation of Models of Microarchitectures <https://uops.info/dissertation.pdf>`_

Generating trace events (time-series) for viewing in chrome://tracing or Perfetto
---------------------------------------------------------------------------------

`Perfetto <https://ui.perfetto.dev/>`_

Use Caliper's event trace service to collect timestamp info, where kernel 
timing can be viewed using browser trace profile views. For example,

``CALI_CONFIG=event-trace,event.timestamps ./raja-perf.exe -ek PI_ATOMIC INDEXLIST  -sp``

This will produce a separate .cali file with date prefix which looks something 
like ``221108-100718_724_ZKrHC68b77Yd.cali``

Then, we need to convert this .cali file to JSON records. But first, we need 
to make sure Caliper's python reader is available in the ``PYTHONPATH`` 
environment variable 

``export PYTHONPATH=caliper-source-dir/python/caliper-reader``

then run ``cali2traceevent.py``. For example,

``python3 ~/workspace/Caliper/python/cali2traceevent.py 221108-102406_956_9WkZo6xvetnu.cali RAJAPerf.trace.json``

You can then load the resulting JSON file either in Chrome by going to 
``chrome://tracing`` or in ``Perfetto``.

For CUDA, assuming you built Caliper with CUDA support, you can collect and 
combine trace information for memcpy, kernel launch, synchronization, and 
kernels. For example,

``CALI_CONFIG="event-trace(event.timestamps,trace.cuda=true,cuda.activities)" ./raja-perf.exe -v RAJA_CUDA Base_CUDA -k Algorithm_REDUCE_SUM -sp``

.. warning::
  When you run cali2traceevent.py you need to add --sort option before the filenames.
  This is needed because the trace.cuda event records need to be sorted before processing.
  Failing to do so may result in a Python traceback.
  New versions of the Caliper Python package have this option built in by default to avoid this issue.

``~/workspace/Caliper/python/cali2traceevent.py --sort file.cali file.json``

For HIP, substitute ``rocm.activities`` for ``cuda.activities``.

.. note:: Currently there is no analog ``trace.rocm``.
