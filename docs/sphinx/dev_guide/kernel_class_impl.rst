.. ##
.. ## Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
.. ## and RAJA Performance Suite project contributors.
.. ## See the RAJAPerf/LICENSE file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _kernel_class_impl-label:

=============================
Kernel Class Implementation
=============================

Each kernel in the Suite follows a similar source file organization and 
implementation pattern for consistency and ease of analysis and understanding.
Here, we describe important and conventions applies in each kernel class
implementation that must be followed to ensure that all kernels integrate into
the RAJA Performance Suite in the same way.

.. _kernel_class_impl_gen-label:

----------------------
General class methods
----------------------

Class methods that do not execute kernel variants and which are not specific to
any kernel variant implementation are defined in one implementation file. For
the **ADD** kernel that we are describing this is the source file ``ADD.cpp``,
which in its entirety is:

.. literalinclude:: ../../../src/stream/ADD.cpp
   :start-after: _add_impl_gen_start
   :end-before: _add_impl_gen_end
   :language: C++
