.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors. 
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##


########################
RAJA Performance Suite
########################

=============================
Motivation and Background
=============================

The RAJA Performance Suite is designed to explore performance of loop-based
computational kernels found in HPC applications. Specifically, it is
used to assess and monitor runtime performance of kernels implemented using
`RAJA C++ performance portability abstractions <https://github.com/LLNL/RAJA>`_.
The Suite contains a variety of kernels implemented using common parallel 
programming models, such as OpenMP and CUDA. Some important 
terminology used in the Suite implementation and discussion includes:

  * **Kernel** is a distinct loop-based computation that appears in the Suite in
    multiple variants (or implementations), each of which performs the same
    computation.
  * **Group** is a subset of kernels in the Suite that originated from the 
    same source, such as a specific benchmark suite.
  * **Variant** refers to implementations of Suite kernels that share the same 
    implementation approach and programming model, such as *baseline OpenMP*, 
    *RAJA OpenMP*, etc.
  * **Tuning** refers to an implementation of kernels with a particular 
    execution parameterization, such as GPU thread-block size 128, GPU 
    thread-block size 256, etc.

The kernels in the Suite originate from various HPC benchmark suites and 
applications. For example, the "Stream" group contains kernels from the Babel 
Stream benchmark, the "Apps" group contains kernels extracted from
real scientific computing applications, and so forth. Each kernel in the Suite 
appears in multiple RAJA and non-RAJA (i.e., *baseline*) variants that use 
parallel programming models supported by RAJA. Some kernels have multiple 
tunings of a variant to explore the performance implications of options that 
a programming model supports.

.. note:: Available variants for a kernel do not need to include all possible
          variants in the Suite. In some cases, a kernel appears only in the 
          subset of variants that makes sense for the particular kernel.

The Suite can be run as a single process or with multiple processes when
configured with MPI support. When running with multiple MPI processes, the same 
code is executed on each rank. Ranks are synchronized before and after each
kernel executes to gather timing data to rank zero. Running with multiple 
MPI processes helps the Suite generate performance data that is more 
realistic for HPC applications than performance data generated running with 
a single process. For example, running sequentially with one MPI 
rank per core vs running sequentially with a single process yields different 
performance results on most multi-core CPUs because bandwidth resources are
exercised differently. Similarly, for GPU systems where multiple MPI ranks
are necessary to fully utilize GPU compute resources.

More information about running the Suite for different types of performance 
studies is provided in the 
:doc:`RAJA Performance Suite User Guide <sphinx/user_guide/index>`

=================================
Git Repository and Issue Tracking
=================================

The main interaction hub for the RAJA Performance Suite is its
`GitHub project <https://github.com/LLNL/RAJAPerf>`_. There you will find 
the Git source code repository, issue tracker, release history, and other 
information about the project.

================================
Communicating with the RAJA Team
================================

If you have questions, find a bug, have ideas about expanding the
functionality or applicability, or wish to contribute to RAJA Performance Suite
development, please do not hesitate to contact us. We are always
interested in improving the Suite and exploring new ways to use it. 

The best way to communicate with us is via our email list: ``raja-dev@llnl.gov``

=========================================================
RAJA Performance Suite User and Developer Documentation
=========================================================

  * :doc:`RAJA Performance Suite User Guide <sphinx/user_guide/index>`

  * :doc:`RAJA Performance Suite Developer Guide <sphinx/dev_guide/index>`

======================================================
RAJA Copyright and License Information
======================================================

Please see :ref:`rajaperf-copyright`.

.. toctree::
   :hidden: 
   :caption: User Documentation

   sphinx/user_guide/index

.. toctree::
   :hidden: 
   :caption: Developer Documentation

   sphinx/dev_guide/index
   sphinx/rajaperf_license
