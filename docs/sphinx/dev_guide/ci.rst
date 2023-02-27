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

The RAJA Performance Suite project uses the same continuous integration tools 
as the RAJA project, namely Azure Pipelines and GitLab CI in the Livermore
Computing Collaboration Zone (LC CZ). Please see `RAJA Continuous Integration Testing <https://raja.readthedocs.io/en/develop/sphinx/dev_guide/ci.html>`_ for more information.

The RAJA Performance Suite project includes files that support GitLab and
Azure Pipelines CI testing that are specific to the project. These file are
similar to those in the RAJA project and play the same roles and follow the
same structure as in the RAJA project. Such files are described in `RAJA Continuous Integration Testing <https://raja.readthedocs.io/en/develop/sphinx/dev_guide/ci.html>`_.

An important difference to note between RAJA Performance Suite GitLab CI and 
RAJA GitLab CI is that the RAJA Performance Suite inherits most of its support 
from its RAJA submodule. As a result, submodules that support RAJA GitLab CI, 
such as `Uberenv <https://github.com/LLNL/uberenv>`_ and
`RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_,
do not appear in the RAJA Performance Suite repository. Instead, the 
``RAJAPerf/scripts/gitlab/build_and_test.sh`` script invokes the 
``uberenv.py`` script in the RAJA submodule to drive Spack and the 
``RAJAPerf/.uberenv_config.json`` file points to the RADIUSS Spack Configs
content in the RAJA submodule:

.. literalinclude:: ../../../.uberenv_config.json
   :language: json

Like RAJA, the RAJA Performance Suite Spack package is maintained in the 
RADIUSS Spack Configs project.

.. _ci_tasks-label:

******************************************************
Continuous Integration (CI) Testing Maintenance Tasks
******************************************************

Tasks for maintaining continuous integration in the RAJA Performance Suite
project are similar to those for RAJA. Please see 
`RAJA Continuous Integration Testing Maintenance Tasks <https://raja.readthedocs.io/en/develop/sphinx/dev_guide/ci_tasks.html>`_ for details.

