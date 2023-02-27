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
the **ADD** kernel that we are describing, this is the source file ``ADD.cpp``,
which in its entirety is:

.. literalinclude:: ../../../src/stream/ADD.cpp
   :language: C++

The methods in the source file are:

  * **Class constructor**, which calls the ``KernelBase`` class constructor
    passing the ``KernelID`` and the ``RunParams`` object, which are used
    to initialize the base class. The constructor calls other base class 
    methods to set information about the kernel, which is specific to the
    kernel. Such information includes:

      * Default problem size and number of kernel repetitions to generate 
        execution run time.
      * The actual problem size that will be run, which is a function of
        the default size and command-line input.
      * The number of *loop iterations* that are performed and the number of 
        loop kernels that run each time the kernel is executed. Note that the 
        **ADD** kernel is based on a simple, single for-loop. However, other 
        kernels in the Suite execute multiple loop kernels.
      * The number of bytes read and written and the number of FLOPS performed 
        for each kernel execution.
      * Which RAJA features the kernel exercises.
      * Which Suite variants are defined, or implemented for the kernel. Each
        variant requires a call to the ``setVariantDefined`` method. Note 
        that not every kernel implements every variant. So this is a mechanism
        to account for what is being run for analysis proposes.

  * **Class destructor**, which must be provided to deallocate kernel state 
    that is allocated in the constructor and which persists throughout the
    execution of the Suite. Note that in the case of the **ADD** kernel, the
    destructor is empty since no state is dynamically allocated in the
    constructor.

  * ``setUp`` method, which allocates and initializes data required for the
    kernel to execute and produce results.

  * ``tearDown`` method, which deallocates and resets any data that will be 
    re-allocated and/or initialized in subsequent kernel executions.
  
  * ``updateChecksum`` method, which computes a checksum from the results of
    an execution of the kernel and adds it to the checksum value for the 
    variant and tuning index that was run.

.. important:: There will only be one instance of each kernel class created 
               by the program. Thus, each kernel class constructor and 
               destructor must only perform operations that are not specific 
               to any kernel variant.

The ``setUp``, ``tearDown``, and ``updateChecksum`` methods will be 
called **each time a kernel variant is run**. We allocate and deallocate
data arrays in the ``setUp`` and ``tearDown`` methods to prevent any 
performance timing bias that may be introduced by artificially reusing data
in cache for example, when doing performance experiments.

Also, note that the ``setUp`` and ``tearDown`` methods pass a ``VariantID``
value to data allocation and initialization, and deallocation methods so
this data management can be done in a variant-specific manner as needed.

To simplify these operations and help ensure consistency, there exist utility 
methods to allocate, initialize, deallocate, and copy data, and compute 
checksums defined in the various *data utils* files in the ``common``
directory.
 
    
