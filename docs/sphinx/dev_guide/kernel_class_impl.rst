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
  
    .. note:: The ``tearDown`` method frees and/or resets all kernel
              data that is allocated and/or initialized in the ``setUp``
              method.

  * ``updateChecksum`` method, which computes a checksum from the results of
    an execution of the kernel and adds it to the checksum value, which is a
    member of the ``KernelBase`` class, for the variant and tuning index that
    was run.

    .. note:: The checksum must be computed in the same way for each
              variant of a  kernel so that checksums for different
              variants can be compared to help identify differences, and
              potential errors in implementations, compiler optimizations,
              programming model execution, etc.

The ``setUp``, ``tearDown``, and ``updateChecksum`` methods are
called **each time a kernel variant is run**. We allocate and deallocate
data arrays in the ``setUp`` and ``tearDown`` methods to prevent any 
performance timing bias that may be introduced by artificially reusing data 
in cache, for example, when doing performance experiments. Also, note that 
the ``setUp`` and ``tearDown`` methods take a ``VariantID`` argument and pass
it to data allocation, initialization, and deallocation methods so
this data management can be done in a variant-specific manner as needed.

To simplify these operations and help ensure consistency, there exist utility 
methods to allocate, initialize, deallocate, and copy data, and compute 
checksums defined in the various *data utils* files in the ``common``
directory.

---------------------------
Kernel object construction 
---------------------------

It is important to note that there will only be one instance of each kernel 
class created by the program. Thus, each kernel class constructor and 
destructor must only perform operations that are not specific to any kernel 
variant.

The ``Executor`` class in the ``common`` directory creates kernel objects,
one for each kernel that will be run based on command-line input options. To
ensure a new kernel object will be created properly, add a call to its class 
constructor based on its ``KernelID`` in the ``getKernelObject()`` method in 
the ``RAJAPerfSuite.cpp`` file. For example::

  KernelBase* getKernelObject(KernelID kid,
                              const RunParams& run_params)
  {
    KernelBase* kernel = 0;

    switch ( kid ) {

      ...

      case Stream_ADD : {
        kernel = new stream::ADD(run_params);
        break;
      }

      ...

    } // end switch on kernel id

    return kernel;
  }

  }

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
kernel in the ``ADD-Seq.cpp`` file:

.. literalinclude:: ../../../src/stream/ADD-Seq.cpp
   :start-after: _add_run_seq_start
   :end-before: _add_run_seq_end
   :language: C++

A few details are worth noting:

  * Thee tuning index argument is ignored because there is only one tuning for 
    the sequential kernel variants.
  * Execution parameters, such as kernel loop length and number of execution
    repetitions, are set by calling base class methods which return values
    based on kernel defaults and input parameters. This ensures that the
    execution will be consistent across run variants and results will be 
    what is expected.
  * Simple switch-case statement logic is used to execute the proper variant
    based on the ``VariantID`` argument.
  * We guard sequential variants apart from the ``Base_Seq`` variant with 
    the ``RUN_RAJA_SEQ`` macro. This ensures that the base sequential variant
    will always run to be used as a reference variant for execution timing.
    By default, we turn off the other sequential variants when we build an
    executable with OpenMP target offload enabled.
  * Macros defined in the ``ADD.hpp`` header file are used to reduce the amount
    of redundant code, such as for data initialization (``ADD_DATA_SETUP``) 
    and the kernel body (``ADD_BODY``).

All kernel source files follow a similar organization and implementation 
pattern for each set of back-end execution variants. However, there are some
important differences to note that we describe next in the discussion of
the CUDA variant execution file.

The key contents related to execution of CUDA GPU variants of the **ADD** 
kernel in the ``ADD-Cuda.cpp`` file are:

.. literalinclude:: ../../../src/stream/ADD-Cuda.cpp
   :start-after: _add_run_cuda_start
   :end-before: _add_run_cuda_end
   :language: C++

Notable differences with the sequential variant file are:

  * Most of the file is guarded using the ``RAJA_ENABLE_CUDA`` macro.

    .. note:: The contents of all non-sequential variant implementation files
              are guarded using the ``RAJA_ENABLE_<backend>`` macros.

  * In addition to using the ``ADD_DATA_SETUP`` macro, which is also used
    in the sequential variant implementation file discussed above, we
    define two other macros, ``ADD_DATA_SETUP_CUDA`` and 
    ``ADD_DATA_TEARDOWN_CUDA``. The first macro allocates GPU device data needed
    to run a kernel and initialize the data by copying host CPU data to it. 
    After a kernel executes, the second macro copies data needed to compute a
    checksum to the host and then deallocates the device data.
  * A CUDA GPU kernel ``add`` is implemented for the ``Base_CUDA`` variant.
  * The method to exjcute the CUDA kernel variants ``ADD::runCudaVariantImpl``
    is templated on a ``block_size`` parameter, which represents the 
    *tuning parameter*, and is passes to the kernel lauch methods.
  * The ``RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE`` macro is
    used (outside the method implementation, to generate different kernel 
    tuning implementations at compile-time to run the GPU ``block_size``
    versions specified via command-line input mentioned in 
    :ref:`build_build-label`.

.. important:: Following the established implementation patterns for kernels
               in the Suite help to ensure that the code is consistent, 
               understandable, easily maintained, and needs minimal 
               documentation.

