.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci-label:

************************************************************
RAJA Performance Suite Continuous Integration (CI) Testing
************************************************************

The RAJA Performance Suite uses the same continuous integration tools as RAJA.

One important difference to note is that the RAJA Performance Suite inherits 
most of its support for GitLab CI testing from its RAJA submodule. As a result,
RAJA submodules that support RAJA GitLab CI, such as 
`Uberenv <https://github.com/LLNL/uberenv>`_ and
`RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_,
do not appear in the RAJA Performance Suite repository. However, the
RAJA Performance Suite does include files that are specific to the project
and play the same roles and follow the structure as the similarly named
files in the RAJA repository.

Files that support Azure Pipelines testing for the RAJA Performance Suite
are also maintained in the project repository.

Please see `RAJA Continuous Integration Testing <https://raja.readthedocs.io/en/develop/sphinx/dev_guide/ci.html>`_ for more information.

.. _ci_tasks-label:

******************************************************
Continuous Integration (CI) Testing Maintenance Tasks
******************************************************

Tasks for maintaining continuous integration in the RAJA Performance Suite
are similar to those for RAJA. Please see `RAJA Continuous Integration Testing 
Maintenance Tasks <https://raja.readthedocs.io/en/develop/sphinx/dev_guide/ci_tasks.html>`_ for more information.

