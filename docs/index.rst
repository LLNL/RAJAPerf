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
computational kernels found in HPC applications. Specifically, it can be
used to assess and monitor runtime performance of kernels implemented using
`RAJA <https://github.com/LLNL/RAJA>`_ C++ performance portability 
abstractions. The Suite contains a variety of kernels implemented using
common parallel programming models, such as OpenMP and CUDA. Some important 
terminology used in the Suite includes:

  * `Kernel` is a distinct loop-based computation that appears in the Suite in
    multiple variants (or implementations), each of which performs the same
    computation.
  * `Group` is a collection of kernels in the Suite that are grouped together
    because they originate from the same source, such as a specific benchmark
    suite.
  * `Variant` refers to implementations of Suite kernels that share the same 
    approach/abstraction and programming model, such as baseline OpenMP, RAJA 
    OpenMP, etc.
  * `Tuning` is a particular implementation of a variant of a kernel in the
    Suite, such as GPU thread-block size 128, GPU thread-block size 256, etc.

Each kernel in the Suite appears in multiple RAJA and non-RAJA (i.e., baseline)
variants using parallel programming models that RAJA supports. Some kernels have
multiple tunings of a variant to explore some of the parametrization that the
programming model supports. The kernels originate from various HPC benchmark
suites and applications. For example, the "Stream" group contains kernels from
the Babel Stream benchmark, the "Apps" group contains kernels extracted from
real scientific computing applications, and so forth.

The Suite can be run as a single process or with multiple processes when
configured with MPI support. When running with multiple MPI ranks, the same 
code is executed on all ranks. Ranks are synchronized before and after each
kernel executes to gather timing data to rank zero. Running with MPI in the 
same configuration used by an HPC app allows the Suite to generate performance 
data that is more relevant for that HPC app than performance data generated 
running with a single process.  For example, running sequentially with one MPI 
rank per core vs running sequentially with a single process yields different 
performance results on most multi-core CPUs because bandwidth resources are
exercised differently.

More information about running the Suite for different types of performance 
studies is provided in the 
:doc:`RAJA Performance Suite User Guide <sphinx/user_guide/index>`

=================================
Git Repository and Issue Tracking
=================================

The main interaction hub for the RAJA Performance Suite is
`GitHub <https://github.com/LLNL/RAJAPerf>`_ There you will find the Git 
source code repository, issue tracker, release history, and other information 
about the project.

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
