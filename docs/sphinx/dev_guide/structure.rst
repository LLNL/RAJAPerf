.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _structure-label:

************************************************************************
RAJA Performance Suite Structure: Adding Kernels, Variants, and Tunings 
************************************************************************

This section describes how to add new Kernels, Variants, and Tunings to the 
Suite. The discussion should make clear the organization of the code and 
how it works, which is useful to understand when making a contribution.

It is important to note that Group and Feature modifications are not required 
unless a new Group or exercised RAJA Feature is added when a new Kernel is 
introduced.

It is also essential that the appropriate targets are updated in the 
appropriate ``CMakeLists.txt`` files when files are added to the Suite so 
that they will be compiled.
