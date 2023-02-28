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
 
.. _kernel_class_impl_exec-label:

-------------------------
Kernel execution methods
-------------------------

In the discussion of the **ADD** :ref:`kernel_class-label`, we noted 
that the class implementation involves multiple files containing variants for
each execution back-end. In particular, these files contain implementations of
the *run* methods declared in the **ADD** :ref:`kernel_class_header-label`
to execute the variants.

Each method takes a variant ID argument that identifies the variant to run and 
a tuning index that identifies the tuning of the variant to run. Note that the 
tuning index can be ignored when there is only one tuning. Each method is 
responsible for multiple tasks which involve a combination of kernel and 
variant specific operations and calling kernel base class methods, such as:

  * Setting up and initializing data needed by a kernel variant before it is run
  * Starting an execution timer before a kernel is run
  * Running the proper number of kernel executions
  * Stopping the time after the kernel is run
  * Putting the class member data in an appropriate state to update a checksum 

For example, here is the method to run sequential CPU variants of the **ADD**
kernel:

.. literalinclude:: ../../../src/stream/ADD.hpp
   :start-after: _add_run_seq_start
   :end-before: _add_run_seq_end
   :language: C++

All kernel source files follow a similar organization and implementation 
pattern for each set of back-end exeuction variants.

.. important:: Following the established implementation patterns for kernels
               in the Suite help to ensure that the code is consistent, 
               understandable, easily maintained, and needs minimal 
               documentation.

A few items are worth noting:

  * Thee tuning index argument is ignored because there is only one tuning for 
    the sequential kernel variants.
  * Execution parameters, such as kernel loop length and number of execution
    repetitions, are set by calling base class methods which return values
    based on kernel defaults and input parameters. This ensures that the
    execution will be consistent across run variants and results will be 
    what is expected.
  * Simple switch-case statement logic is used to execute the proper variant
    based on the ``VariantID`` argument.
  * Macros defined in the ``ADD.hpp`` header file are used to reduce the amount
    of redundant code, such as for data initialization (``ADD_DATA_SETUP``) 
    and the kernel body (``ADD_BODY``).

