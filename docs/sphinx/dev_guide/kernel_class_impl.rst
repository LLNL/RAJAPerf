.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA Performance Suite.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _kernel_class_impl-label:

=============================
Kernel Class Implementation
=============================

Each kernel in the Suite follows a similar source file organization and 
implementation pattern for consistency and ease of analysis and understanding.
Here, we describe important conventions that apply in each kernel class
implementation that **must be followed** to ensure that all kernels look the
same and integrate into the RAJA Performance Suite in the same way.

.. _kernel_class_impl_gen-label:

----------------------
General class methods
----------------------

Class methods that do not execute kernel variants and which are not specific to
any kernel variant implementation are defined in one implementation file, for
example the source file ``ADD.cpp`` for the **ADD** kernel that we are
describing. The file contents in their entirety are:

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
        the default size and command-line input for target problem size.
      * The number of *loop iterations* that are executed and the number of 
        loop kernels that run each time the kernel is executed. Note that the 
        **ADD** kernel contains a single for-loop so the number of iterations 
        is the problem size and the number of loop kernels is one. Other kernels
        in the Suite may execute multiple loop kernels with different sizes,
        so these methods are used to describe this.
      * The number of bytes allocated for each kernel execution.
      * The number of bytes read for each kernel execution.
      * The number of bytes written for each kernel execution.
      * The number of bytes read, modified, and written for each kernel execution.
      * The number of bytes atomically read, modified, and written for each
        kernel execution.
      * The number of floating point operations (FLOPS) performed for each
        kernel execution.
      * The consistency of the checksums of the kernel. If the kernel
        always produces the same checksum value for all variant tunings then the
        checksums are ``Consistent``. Most kernels get a different but consistent
        checksum for each variant tuning so the checksums are
        ``ConsistentPerVariantTuning``. On the other hand, some kernels have
        variant tunings that get different checksums on each run of that variant
        tuning, for example due to the ordering of floating-point atomic add
        operations, so the checksums are ``Inconsistent``.
      * The tolerance of the checksums of the kernel. A number of predefined
        values are available in the ``KernelBase\:\:ChecksumTolerance`` class. If
        the kernel consistently produces the same checksums then ``zero`` tolerance
        is used. Most kernels use the ``normal`` tolerance. Some kernels are very
        simple, for example they have a single floating-point operation per
        iteration, so they use the ``tight`` tolerance.
      * The scale factor to use with the checksums of the kernel. This is an
        arbitrary multiplier on the checksum values used to scale the checksums
        to a desired range. Mostly used for kernels with floating-point
        operation complexity that does not scale linearly with problem size.
      * The operational complexity of the kernel, where N is the *problem size*
        of the kernel.
      * The number of levels in the largest perfectly nested loop. This only counts
        parallelized dimensions and ignores inner or outer sequential loops. For
        example the GEMM kernel has 2 perfectly nested loop levels as the inner
        loop is implemented sequentially to perform a reduction.
      * The dimensionality of the problem domain, regardless of physical data
        layout. For example, the LTIMES kernel has a problem dimensionality of 3,
        because phi (g, m, and z) and psi (g, d, and z) are indexed over 3
        dimensions.
      * Which RAJA features the kernel exercises.
      * Adding Suite variants and tunings via ``addVariantTunings``. This calls
        the various ``define*VariantTunings`` methods that are defined in the
        source file where the variants and tunings are implemented. Note that
        not every kernel implements every variant, so ``KernelBase`` provides a
        "default" implementation that defines no variants or tunings.

    ..note:: The byte counters are intended to count traffic to and from main
             memory like DRAM or HBM under idealized conditions with perfect
             caching. They are not intended to count the total number of bytes
             requested by load and store instructions. So, even if a memory
             address is read in multiple different iterations of a loop with a
             stencil access pattern it is only counted once in bytes read.
             However caching is not assumed between loops/kernel launches so an
             address is counted once for each separate loop or kernel launch.

    ..note:: To simplify counting each address accessed should only be counted
             in one of the byte counter attributes. For example an address
             that is read and written is counted in the "read, modified, and
             written" counter, but not in the "read" or "written" counters. The
             final output however does add the "read" and "read, modified, and
             written" counters when showing the bytes read.

    ..note:: Available variant tunings for each kernel are specified using a
             ``...BOILERPLATE...`` macro invocation in each kernel variant
             source file. This is discussed in :ref:`kernel_class_impl_exec-label`.

  * **Class destructor**, which must be provided to deallocate kernel state 
    that is allocated in the constructor and which persists throughout the
    execution of the Suite. Note that in the case of the **ADD** kernel, the
    destructor is empty since no state is dynamically allocated in the
    constructor.

  * ``setUp`` method, which allocates and initializes data required for the
    kernel to execute and produce results. This method is called before
    each kernel variant is run.

  * ``tearDown`` method, which deallocates and resets any data that will be
    re-allocated and/or initialized in subsequent kernel executions.
  
    .. note:: The ``tearDown`` method frees and/or resets all kernel
              data that is allocated and/or initialized in the ``setUp``
              method.

  * ``updateChecksum`` method, which computes a checksum from the results of
    an execution of the kernel and adds it to the checksum value, which is a
    member of the ``KernelBase`` class, for the variant and tuning index that
    was run.

    .. note:: Kernel checksum values are used to determine whether kernels 
              variants are producing the same results. Thus, the checksum
              **must** be computed in the same way for each
              variant of a kernel so that checksums for different
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

Data Utility Methods
--------------------

To simplify these operations and help ensure consistency, there exist utility 
methods to allocate, initialize, deallocate, and copy data, and compute 
checksums defined in the various *data utils* files in the ``common``
directory.

When calculating checksums use the ``addToChecksum`` methods. Individual numbers
are added directly to the overall checksum. Arrays of numbers are checksummed to
an intermediate checksum value via a function in the *data utils* file discussed
below and then the intermediate checksum value is added into the overall
checksum. Checksums are calculated via a Kahan sum to improve accuracy.

This function transforms each number in the array before adding the number to
its checksum. The function converts the number into the checksum type, takes the
absolute value of the result, and multiplies the result by a value that differs
for each member of the array and depends on the sign of the number. This
procedure creates different checksums for permutations of the same numbers and
numbers with opposite signs. See the implementation below for details.

.. literalinclude:: ../../../src/common/DataUtils.cpp
   :start-after: _calc_checksum_impl_start
   :end-before: _calc_checksum_impl_end
   :language: C++

---------------------------
Kernel object construction 
---------------------------

It is important to note that there will only be one instance of each kernel 
class created by the program. Thus, each kernel class constructor and 
destructor must only perform operations that are not specific to any kernel 
variant.

The ``Executor`` class in the ``common`` directory creates kernel objects,
one for each kernel that will be run based on command-line input options. To
ensure a new kernel object will be created properly, make sure to add a call
to its class constructor in the switch case section for its ``KernelID`` in the
``getKernelObject()`` method in the ``RAJAPerfSuite.cpp`` file. For example::

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

By convention each of the *run* methods takes a variant ID argument that
identifies the variant to run. Some kernels have multiple *run* methods for
different tunings of some variants. Each method is responsible for multiple
tasks which involve a combination of kernel and variant specific operations and
calling kernel base class methods, such as:

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
  * The ``RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE`` macro appears in the file
    outside of the class implementation to define the variants and "default"
    tunings for the ``ADD`` kernel for the ``Seq`` back-end. The macro defines
    the ``defineSeqVariantTunings`` method for this kernel and assumes the
    existence of ``runSeqVariant``. Note that it adds a "default" tuning for
    each of the variants listed in the final arguments to the macro.
  * Each kernel variant execution contains a call to ``startTimer()`` and
    ``endTimer()`` methods which are used to gather execution timing
    information. Between these method calls is a loop over
    *kernel repetitions*. Each kernel defines a default number of run
    repetitions to record a sufficient amount of execution time to reduce
    execution timing noise. The number of repetitions can be controlled via
    command-line arguments.

    .. note:: The kernel repetition loop counter increment is done using the
    macro ``RP_REPCOUNTINC`` for consistency and to reduce code verbosity.
    The loop counter variable ``irep`` is declared ``volatile`` to prevent
    compilers from optimizing certain loop execution operations. The syntax
    inside the macro definition quiets compiler warnings when the code is 
    compiled using the C++20 standard.
  * Every timed kernel body should be marked with
    ``RP_CALI_SUBKERNEL_BEGIN("<name>")`` and
    ``RP_CALI_SUBKERNEL_END("<name>")``. These macros add the Caliper
    ``subkernel`` attribute when Caliper support and
    ``RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL`` are enabled, and compile away
    otherwise. Configure with ``-DRAJA_PERFSUITE_USE_CALIPER_SUBKERNEL=Off`` to
    exclude subkernel regions from Caliper output.
  * Name fixed-count subkernels by appending a one-based numeric suffix to the
    kernel name, for example ``DAXPY_1``, ``POLYBENCH_JACOBI_1D_1``, and
    ``POLYBENCH_JACOBI_1D_2``. Single-body kernels use the ``_1`` suffix. For a
    variable number of launches, append ``_k`` instead, for example
    ``HALO_PACKING_pack_k`` or ``HALO_EXCHANGE_unpack_k``.

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
              are guarded using the ``RAJA_ENABLE_<back-end>`` macros.

  * A CUDA GPU kernel ``add`` is implemented for the ``Base_CUDA`` variant.
  * The method to execute the CUDA kernel variants ``ADD::runCudaVariantImpl``
    is templated on a ``block_size`` parameter, which represents the 
    *tuning parameter*, and is passed to the kernel launch methods. The
    ``setBlockSize`` function is called to provide this block_size to caliper.
  * The ``RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE`` macro is
    used outside the method implementation, to define "block_size" tunings for
    the ``ADD`` kernel for the ``Cuda`` back-end. The macro defines the
    ``defineCudaVariantTunings`` method for this kernel and assumes the
    existence of ``runCudaVariantImpl<block_size>`` and ``gpu_block_sizes_type``.
    Note that this sets the kernel up to run with the GPU ``block_size``
    information specified via command-line input mentioned in :ref:`build_build-label`.
  * Finally, the method ``RPLaunchCudaKernel`` method is used to launch the
    non-RAJA variants so that they use the same launch mechanics for GPU
    kernels used inside RAJA for a fair timing comparison. A similar method
    ``RPLaunchHipKernel`` is used for HIP back-end kernel variants. 

.. important:: Following the established implementation patterns for kernels
               in the Suite help to ensure that the code is consistent,
               understandable, easily maintained, and needs minimal specific
               documentation for any individual kernel or variant.

--------------------------------
Kernel tuning definition methods
--------------------------------

Following on from the previous section, the **ADD** :ref:`kernel_class-label`
back-end files also contain *define* methods which add tuning definitions
for the variants of the kernel in the same back-end file.

By convention each of the *define* methods takes no arguments. Each kernel must
provide such a *define* method for each of the back-ends that it implements.
This method defines the tunings for the variants of the back-end by calling
the ``addVariantTuning`` kernel base class method for each variant and tuning.

For example, here is the method to define CUDA GPU variants of the **MEMSET**
kernel in the ``MEMSET-Cuda.cpp`` file:

.. literalinclude:: ../../../src/algorithm/MEMSET-Cuda.cpp
   :start-after: _memset_define_cuda_start
   :end-before: _memset_define_cuda_end
   :language: C++

  * The template parameter is a pointer to the member function that implements
    the tuning; e.g., ``&MEMSET::runCudaVariantLibrary``
  * The first argument is the variant id
  * The second argument is the name of the tuning given as a string

A few details are worth noting:

  * The loop over variants and variant conditionals helps define all of the
    relevant tunings for each of the variants.
  * The ``seq_for`` function over ``gpu_block_sizes_type{}`` passes a compile
    time integer constant type into ``block_size`` so it may be used as a
    template argument.
  * The ``addVariantTuning`` method is called with a unique name for each
    tuning. Note that the same tuning may appear for multiple variants.
