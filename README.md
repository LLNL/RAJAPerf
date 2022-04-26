[comment]: # (#################################################################)
[comment]: # (Copyright 2017-2022, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA Performance Suite project contributors.)
[comment]: # (See the RAJAPerf/LICENSE file for details.)
[comment]: #
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)


RAJA Performance Suite
======================

[![Build Status](https://travis-ci.org/LLNL/RAJAPerf.svg?branch=develop)](https://travis-ci.org/LLNL/RAJAPerf)

The RAJA Performance Suite is designed to explore performance of loop-based
computational kernels found in HPC applications. Specifically, it can be
used to assess and monitor runtime performance of kernels implemented using
[RAJA] C++ performance portability abstractions and compare those to variants
implemented using common parallel programming models, such as OpenMP and CUDA,
directly. Some important terminology used in the Suite includes:

  * `Kernel` is a distinct loop-based computation that appears in the Suite in
    multiple variants (or implementations), each of which performs the same
    computation.
  * `Variant` is an implementation or set of implementations of a kernel in the
    Suite that share the same approach/abstraction and programming model,
    such as baseline OpenMP, RAJA OpenMP, etc.
  * `Tuning` is a particular implementation of a variant of a kernel in the
    Suite, such as gpu block size 128, gpu block size 256, etc.
  * `Group` is a collection of kernels in the Suite that are grouped together
    because they originate from the same source, such as a specific benchmark
    suite.

Each kernel in the Suite appears in multiple RAJA and non-RAJA (i.e., baseline)
variants using parallel programming models that RAJA supports. Some kernels have
multiple tunings of a variant to explore some of the parametrization that the
programming model supports. The kernels originate from various HPC benchmark
suites and applications. For example, the "Stream" group contains kernels from
the Babel Stream benchmark, the "Apps" group contains kernels extracted from
real scientific computing applications, and so forth.

The suite can be run as a single process or with multiple processes when
configured with MPI support. Running with MPI in the same configuration used
by an hpc app allows the suite to gather performance data that is more relevant
for that hpc app than performance data gathered running single process. For
example running sequentially with one MPI rank per core vs running sequentially
with a single process yields different performance results on most multi-core
CPUs.

* * *

Table of Contents
=================

1. [Building the Suite](#building-the-suite)
2. [Running the Suite](#running-the-suite)
3. [Generated output](#generated-output)
4. [Adding kernels and variants](#adding-kernels-and-variants)
4. [Continuous Integration](#continuous-integration)
5. [Contributions](#contributions)
6. [Authors](#authors)
7. [Copyright and Release](#copyright-and-release)

* * *

# Building the Suite

To build the Suite, you must first obtain a copy of the source code by cloning
the GitHub repository. For example,

```
> mkdir RAJA-PERFSUITE
> cd RAJA-PERFSUITE
> git clone --recursive https://github.com/llnl/RAJAPerf.git
```

The repository will reside in a `RAJAPerf` sub-directory in the directory into
which is was cloned.

The Performance Suite has two Git submodules, [RAJA] and the CMake-based [BLT]
build system. The `--recursive` option tells Git to clone the submodules
as well as any submodules that they use. If you switch to a different branch
in your working copy of the repository, you should update the submodules to
make sure you have the right versions of them for the branch. For example,

```
> cd RAJAPerf
> git checkout <some branch name>
> git submodule update --recursive
```

Note that the `--recursive` option will update submodules within submodules,
similar to usage with the `git clone` as described above.

RAJA and the Performance Suite are built together using the same CMake
configuration. For convenience, we include scripts in the `scripts`
directory that invoke corresponding configuration files (CMake cache files)
in the RAJA submodule. For example, the `scripts/lc-builds` directory
contains scripts that show how we build code for testing on platforms in
the Lawrence Livermore Computing Center. Each build script creates a
descriptively-named build space directory in the top-level Performance Suite
directory and runs CMake with a configuration appropriate for the platform and
compilers used. After CMake completes, enter the build directory and type
`make` (or `make -j <N>` for a parallel build using N processor cores; if you
omit the number of cores, the code will build in parallel using all available
cores on the node you are running on) to compile the code. For example,

```
> ./scripts/blueos_nvcc_clang.sh 10.2.89 sm_70 10.0.1
> cd build_blueos_nvcc10.2.89-cm_70-clang10.0.1
> make -j
```

The build scripts and associated CMake `host-config` files in RAJA are
useful sources of information for building the Suite on various platforms.
For example, they show how to enable specific back-end kernel variants and
compiler options we use for testing.

You can also create your own build directory and run CMake with your own
options from there. For example, :

```
> mkdir my-build
> cd my-build
> cmake <cmake args> ../
> make -j
```

The provided configurations will only build the Performance Suite code by
default; i.e., it will not build any RAJA test or example codes. If you
want to build the RAJA tests, for example, to verify your build of RAJA is
working properly, just pass the `-DENABLE_TESTS=On` option to CMake, either
on the command line if you run CMake directly or edit the script you are
running to do this. Then, when the build completes, you can type `make test`
to run the RAJA tests.

## Building with MPI

Some of the provided configurations will build the Performance Suite with
MPI support enabled. For example,

```
> ./scripts/blueos_spectrum_nvcc_clang.sh rolling-release 10.2.89 sm_70 10.0.1
> cd build_lc_blueos-spectrumrolling-release-nvcc10.2.89-sm_70-clang10.0.1
> make -j
```

In general MPI support can be enabled by passing the `-DENABLE_MPI=On` option
to CMake and providing a mpi compiler wrapper via the
`-DMPI_CXX_COMPILER=/path/to/mpic++` option to CMake in addition to other CMake
options. For example,

```
> mkdir my-mpi-build
> cd my-mpi-build
> cmake -DENABLE_MPI=On -DMPI_CXX_COMPILER=/path/to/mpic++ <cmake args> ../
> make -j
```

## Building with specific GPU block size tunings

Some of the provided configurations will build the Performance Suite with
GPU support enabled. This will build with the default GPU block size tuning for
each kernel. For example,

```
> ./scripts/blueos_nvcc_clang.sh 10.2.89 sm_70 10.0.1
> cd build_lc_blueos-nvcc10.2.89-sm_70-clang10.0.1
> make -j
```

Using a specific set of GPU block sizes is done by by passing the
`-DRAJA_PERFSUITE_GPU_BLOCKSIZES=<list,of,block,sizes>` option to CMake via the
`-DMPI_CXX_COMPILER=/path/to/mpic++` option to CMake in addition to other CMake
options. For example,

```
> mkdir my-gpu-build
> cd my-gpu-build
> cmake -DRAJA_PERFSUITE_GPU_BLOCKSIZES=64,128,256,512,1024 <cmake args> ../
> make -j
```

* * *

# Running the Suite

The Suite is run by invoking the executable in the `bin` sub-directory in the
build space directory. For example, if you provide no command line options,

```
> ./bin/raja-perf.exe
```

the entire Suite (all kernels and variants) will execute in their default
configurations. How the Suite will run and some details about each kernel
will appear on the screen before it is run. Kernel detail information will
also appear in a run report file generated in your run directory
after the Suite executes. You can pass the ''--dryrun'' option along with
any other runtime options to see a summary of how the Suite will execute
without actually running it.

The Suite can be run in a variety of ways via options passed to the executable.
For example, you can run subsets of kernels and variants by specifying
variants, groups, or individual kernels explicitly. Other configuration
options to set problem sizes, number of times each kernel is run, etc. can
also be specified. You build the code once and use scripts or other mechanisms
to run the Suite in different ways for analyses you want to perform.

All options appear in a 'long form' with a double hyphen prefix (i.e., '--').
Some options are available in a one or two character 'short form' with a
single hyphen prefix (i.e., '-') for convenience. To see available options
along with a brief description of each, pass the `--help` or `-h` option:

```
> ./bin/raja-perf.exe --help
```

or

```
> ./bin/raja-perf.exe -h
```

Lastly, the program will generate a summary of provided input if it is given
input that the code does not know how to parse. Ill-formed input will be noted
in the summary output. Hopefully, this will make it easy for users to correct
erroneous usage, such as mis-spelled option names.

## Running with MPI

Running the Suite with MPI is as simple as running any other MPI application.
For example,

```
> srun -n 2 ./bin/raja-perf.exe
```
the entire Suite (all kernels and variants) will execute in their default
configurations on each of the 2 ranks. The kernel information output shows how
each kernel is run on each rank. The total problem size across all MPI ranks
can be calculated by multiplying the number of MPI ranks by the problem
size in the kernel information. Timing is reported on rank 0 and is gathered
by doing an MPI barrier, starting the timer, running the kernel repetitions,
doing an MPI barrier, and then stopping the timer.

## Important note

 * The OpenMP target offload variants of the kernels in the Suite are a
   work-in-progress since the RAJA OpenMP target offload back-end is also
   a work-in-progress. If you configure them to build, they can be run with
   the executable `./bin/raja-perf-omptarget.exe` which is distinct from
   the one described above. At the time the OpenMP target offload variants were
   developed, it was not possible for them to co-exist in the same executable
   as the CUDA variants, for example. In the future, the build system may
   be reworked so that the OpenMP target variants can be run from the same
   executable as the other variants.

* * *

# Generated output

When the Suite is run, several output files are generated that contain
data describing the run. The file names start with the file prefix
provided via a command line option in the output directory, also specified
on the command line. If no such options are provided, files will be located
in the current run directory and be named `RAJAPerf-*`, where '*' is a string
indicating the contents of the file.

Currently, there are five files generated:

1. Timing -- execution time (sec.) of each loop kernel and variant run
2. Checksum -- checksum values for each loop kernel and variant run to ensure they are producing the same results (typically, checksum differences of ~1e-10 or less indicate that all kernel variants ran correctly).
3. Speedup -- runtime speedup of each loop kernel and variant with respect to a reference variant. The reference variant can be set with a command line option. If not specified, the first variant run will be used as the reference. The reference variant used will be noted in the file.
4. Figure of Merit (FOM) -- basic statistics about speedup of RAJA variant vs. baseline for each programming model run. Also, when a RAJA variant timing differs from the corresponding baseline variant timing by more than some tolerance, this will be noted in the file with `OVER_TOL`. By default the tolerance is 10%. This can be changed via a command line option.
5. Kernel -- Basic information about each kernel that is run, which is the same
for each variant of the kernel that is run. See description of output information below.

All output files are text files. Other than the checksum file, all are in
'csv' format for easy processing by common tools and generating plots.

## Kernel information definitions

Information about kernels that are run is located in the ''RAJAPerf-kernels.csv'' file. This information is for each process individually, so when running with MPI the total problem size aggregated across all ranks is the number of ranks times the problem size shown in the kernel information. Kernel information includes the following:

1. Kernel name -- Format is group name followed by kernel name, separated by an underscore.
2. Feature -- RAJA feature(s) exercised in RAJA variants of kernel.
3. Problem size -- Size of the problem represented by a kernel. Please see notes below for more information.
4. Reps -- Number of times a kernel runs in a single pass through the Suite.
5. Iterations/rep -- Sum of sizes of all parallel iteration spaces for all loops run in a single kernel execution (see 'problem size' above).
6. Kernels/rep -- total number of loop structures run (or GPU kernels launched) in each kernel repetition.
7. Bytes/rep -- Total number of bytes read from and written to memory for each repetition of kernel.
8. FLOPs/rep -- Total number of floating point operations executed for each repetition of kernel. Currently, we count arithmetic operations (+, -, *, /) and functions, such as exp, sin, code, etc. as on FLOP. We do not currently count operations like abs and comparisons (<, >, etc.) in the FLOP count. So these numbers are rough estimates. For actual FLOP counts, a performancce analysis tool should be used.

### Notes about 'problem size'

 * Problem size is always output per process/MPI rank. To get the total problem
   size across all ranks when running with MPI multiply the problem size by
   the number of MPI ranks.
 * The Suite uses three notions of problem size for each kernel: 'default',
   'target', and 'actual'. Default is the 'default' problem size defined for a
   kernel and the size that will be run if no runtime options are
   provided to run a different size. Target is the desired problem size to run
   based on default settings and alterations to that if input is provided to
   change the default. Actual is the problem size that is run based on how
   each kernel calculates this.
 * The concept of problem size is subjective and can be interpreted differently
   depending on the kernel structure and what one is trying to measure. For
   example, problem size could refer to the amount of data needed to be stored
   in memory to run the problem, or it could refer to the amount of parallel
   work that is possible, etc.
 * We employ the following, admittedly loose definition, which depends on the
   particular kernel structure. Of all the 'loop structures' (e.g., single
   loop, nested loops, etc.) that are run for a kernel (note that some kernels
   run multiple loops, possibly with different sizes or loop structures),
   problem size refers to the size of the data set required to generate the
   kernel result. The interpretation of this and the definition of problem
   size for each kernel in the suite is determined by the kernel developer
   and team discussion.

Here are a few examples to give a better sense of how we determine problem
size for various kernels in the Suite.

Vector addition.
```cpp
for (int i = 0; i < 0; i < N; ++i) {
  c[i] = a[i] + b[i];
}
```
The problem size for this kernel is 'N', the loop length. Note that this
happens to match the size of the vectors a, b, c and the total amount of
parallel work in the kernel. This is common for simple, data parallel kernels.

Matrix-vector multiplication.
```cpp
for (int r = 0; r < N_r; ++r) {
  b[r] = 0;
  for (int c = 0; c < N_c; ++c) {
    b[r] += A[r][c] + x[c];
  }
}
```
The problem size if N_r * N_c, the size of the matrix. Note that this matches
the total size of the problem iteration space, but the total amount of parallel
work is N_r, the number of rows in the matrix and the length of the vector b.

Matrix-matrix multiplication.
```cpp
for (int i = 0; i < N_i; ++i) {
  for (int j = 0; j < N_j; ++j) {
    A[i][j] = 0;
    for (int k = 0; k < N_k; ++k) {
      A[i][j] += B[i][k] * C[k][j];
    }
  }
}
```
Here, we are multiplying matrix B (N_i x N_k) and matrix C (N_k x N_j) and
storing the result in matrix A (N_i X N_j). Problem size could be chosen to
be the maximum number of entries in matrix B or C. We choose the size of
matrix A (N_i * N_j), which is more closely aligned with the number of
independent operations (i.e., the amount of parallel work) in the kernels.


* * *

# Adding kernels, variants, and Tunings

This section describes how to add new kernels, variants and/or tunings to the
Suite. *Group* and *feature* modifications are not required unless a new group
or exercised RAJA feature is added when a new kernel is introduced. The
information in this section also provides insight into how the performance
Suite operates.

It is essential that the appropriate targets are updated in the appropriate
`CMakeLists.txt` files when files are added to the Suite so that they will
be compiled.

## Adding a kernel

Adding a new kernel to the Suite involves three main steps:

1. Add a unique kernel ID and a unique kernel name to the Suite.
2. If the kernel is part of a new kernel group or exercises a new RAJA feature, also add a unique group ID and name for the group. Similarly, if a new RAJA
feature is exercised by a new kernel.
3. Implement a kernel class that contains all operations needed to run it, with source files organized as described below.

These steps are described in the following sections.

### Add the kernel ID and name

Two key pieces of information identify a kernel: the group in which it
resides and the name of the kernel itself. For concreteness, we describe
how to add a kernel "FOO" that lives in the kernel group "Basic". The files
`RAJAPerfSuite.hpp` and `RAJAPerfSuite.cpp` in the `src/common` directory
define enumeration values and arrays of string names for the kernels,
respectively.

First, add an enumeration value identifier for the kernel, that is unique
among all kernels, in the enum 'KernelID' in the header file `RAJAPerfSuite.hpp`:

```cpp
enum KernelID {
..
  Basic_FOO,
..
};
```

Note: the enumeration value for the kernel is the group name followed
by the kernel name, separated by an underscore. It is important to follow
this convention so that the kernel works properly with the Performance
Suite machinery.

Second, add the kernel name to the array of strings `KernelNames` in the file
`RAJAPerfSuite.cpp`:

```cpp
static const std::string KernelNames [] =
{
..
  std::string("Basic_FOO"),
..
};
```

Note: the kernel string name is just a string version of the kernel ID.
This convention must be followed so that the kernel works properly with the
Performance Suite machinery. Also, the values in the KernelID enum and the
strings in the `KernelNames` array must be kept consistent (i.e., same order
and matching one-to-one). Within each kernel group, we keep the kernel names
and IDs in alphabetical order to make the organization clear.


### Add new group if needed

If a kernel is added as part of a new group of kernels in the Suite, a
new value must be added to the `GroupID` enum in the header file
`RAJAPerfSuite.hpp` and an associated group string name must be added to
the `GroupNames` array of strings in the file `RAJAPerfSuite.cpp`. Again,
the enumeration values and items in the string array must be kept
consistent (same order and matching one-to-one).

Adding a new RAJA feature is similar.


### Add the kernel class

Each kernel in the Suite is implemented in a class whose header and
implementation files live in the directory named for the group
in which the kernel lives. The kernel class is responsible for implementing
all operations needed to manage data, execute, and record execution timing and
checksum information for each variant and tuning of the kernel. To properly plug
in to the Performance Suite framework, the kernel class must be a subclass of
the `KernelBase` base class that defines the interface for kernels in the Suite.

Continuing with our example, we add a 'FOO' class header file `FOO.hpp`,
and multiple implementation files described in the following sections:

  * `FOO.cpp` contains the methods to setup and teardown the memory for the
    'FOO' kernel, and compute and record a checksum on the result after it
    executes. It also specifies kernel information in the kernel class
    constructor.
  * `FOO-Seq.cpp` contains sequential CPU variants and tunings of the kernel.
  * `FOO-OMP.cpp` contains OpenMP CPU multithreading variants and tunings of the kernel.
  * `FOO-OMPTarget.cpp` contains OpenMP target offload variants and tunings of the kernel.
  * `FOO-Cuda.cpp` contains CUDA GPU variants and tunings of the kernel.
  * `FOO-Hip.cpp` contains HIP GPU variants and tunings of the kernel.

All kernels in the Suite follow the same implementation pattern. Inspect the
files for any kernel to understand the overall organization.

Note: if a new execution back-end variant is added that is not listed here,
that variant should go in the file `FOO-<backend-name>.cpp`. Keeping the
back-end variants in separate files helps to understand compiler optimizations
when looking at generated assembly code, for example.

#### Kernel class header

Here is what a header file for the FOO kernel object should look like:

```cpp
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Foo kernel reference implementation:
///
/// Describe it here...
///

#ifndef RAJAPerf_Basic_FOO_HPP
#define RAJAPerf_Basic_FOO_HPP

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams; // Forward declaration for ctor arg.

namespace basic
{

class FOO : public KernelBase
{
public:

  FOO(const RunParams& params);

  ~FOO();

  void setUp(VariantID vid, size_t tuning_idx);
  void updateChecksum(VariantID vid, size_t tuning_idx);
  void tearDown(VariantID vid, size_t tuning_idx);

  void runSeqVariant(VariantID vid, size_t tuning_idx);
  void runOpenMPVariant(VariantID vid, size_t tuning_idx);
  void runCudaVariant(VariantID vid, size_t tuning_idx);
  void runHipVariant(VariantID vid, size_t tuning_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tuning_idx);

private:
  // Kernel-specific data (pointers, scalars, etc.) as needed...
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
```

The kernel object header has a uniquely-named header file include guard and
the class is nested within the `rajaperf` and `basic` namespaces. The
constructor takes a reference to a `RunParams` object, which contains the
input parameters for running the Suite -- we'll say more about this later.
The methods that take a variant ID and tuning index arguments must be provided
as they are pure virtual in the KernelBase class. Their names are descriptive of
what they do and we'll provide more details about them when we describe the
class implementation next.

#### Kernel class implementation

Each kernel in the Suite follows a similar implementation pattern for
consistency and ease of analysis and understanding. Here, we describe several
key steps and conventions that must be followed to ensure that all kernels
interact with the performance Suite machinery in the same way:

1. Initialize the `KernelBase` class object with `KernelID` and `RunParams` object passed to the FOO class constructor.
2. In the class constructor, define kernel information. This includes: default problem size, default run repetition count, iterations per rep, kernels per rep, bytes per rep, FLOPs per rep, the RAJA features used by the kernel, and kernel variants defined (i.e., implemented) by calling the appropriate members in the `KernelBase` class. See the *.cpp file for any existing kernel in the suite for examples of how this is done. Note that tuning names are added in step 6.
3. Implement data allocation and initialization operations for each kernel variant in the `setUp` method.
4. Compute the checksum for each variant in the `updateChecksum` method.
5. Deallocate and reset any data that will be allocated and/or initialized in subsequent kernel executions in the `tearDown` method.
6. Implement kernel execution for the associated variants and tunings in the `run*Variant` methods in the proper source files. Add tuning names for the tunings of each variant by overriding the `KernelBase` methods `set*TuningDefinitions`. Note that this is not necessary if there is only one tuning.

##### Constructor and destructor

It is important to note that there will only be one instance of each kernel
class created by the program. Thus, each kernel class constructor and
destructor must only perform operations that are not specific to any kernel
variant.

The constructor must pass the kernel ID and RunParams object to the base
class `KernelBase` constructor. The body of the constructor must also call
base class methods to set kernel information described above. Note that
the arguments passed to each method are specific to each kernel, in general.
This code snippets shows a typical way this looks for a simple single for-loop
data parallel kernel.

```cpp
FOO::FOO(const RunParams& params)
  : KernelBase(rajaperf::Basic_Foo, params),
{
  setDefaultProblemSize(1000000);  // length of the for-loop
  setDefaultReps(1000);            // number of times the kernel will execute
                                   // to generate an execution run time value

  setActualProblemSize( getTargetProblemSize() );  // actual problem size may
                                                   // be different than the
                                                   // default size based on
                                                   // user-provided run time
                                                   // options

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( ... );  // value set based on data read and written when
                          // kernel executes
  setFLOPsPerRep( ... );  // value set based on floating-point operations
                          // performed when kernel executes

  setUsesFeature(Forall); // the kernel uses the RAJA::forall construct and
                          // no other RAJA features.

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  // etc.
}
```

The class destructor doesn't have any requirements beyond freeing memory
owned by the class object as needed. Often, it is empty.

##### setUp() method

The `setUp()` method is responsible for allocating and initializing data
necessary to run the kernel for the variant specified by its variant ID
argument. For example, a baseline variant may have aligned data allocation
to help enable SIMD optimizations, an OpenMP variant may initialize arrays
following a pattern of "first touch" based on how memory and threads are
mapped to CPU cores, a CUDA variant may initialize data in host memory,
which will be copied to device memory when a CUDA variant executes, etc.

It is important to use the same data allocation and initialization operations
for all kernel variants so that checksums can be compared at the end of a run.

Note: to simplify these operations and help ensure consistency, there exist
utility methods to allocate, initialize, deallocate, and copy data, and compute
checksums defined in the `DataUtils.hpp` `CudaDataUtils.hpp`,
`OpenMPTargetDataUtils.hpp`, etc. header files in the `common` directory.

##### run methods

Which files contain which 'run' methods and associated variant and tuning
implementations is described above. Each method takes a variant ID argument
which identifies the variant to be run and a tuning index which identifies
the tuning of the variant to run. Note that the tuning index can be ignored
when there is only one tuning. Each method is also responsible for calling base
class methods to start and stop execution timers when a loop variant is run.
A typical kernel execution code section may look like:

```cpp
void Foo::runSeqVariant(VariantID vid, size_t /*tuning_idx*/)
{
  const Index_type run_reps = getRunReps();
  // ...

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         // Implementation of Base_Seq kernel variant...

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        // Implementation of Lambda_Seq kernel variant...

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        // Implementation of RAJA_Seq kernel variant...

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  <kernel-name> : Unknown variant id = " << vid << std::endl;
    }

  }
}
```

All kernel implementation files are organized in this way. So following this
pattern will ensure all new additions are consistent.

Important notes:

  * As mentioned earlier, there are multiple source files for each kernel.
    The reason for this is that it makes it easier to apply unique compiler
    flags to different variants and to manage compilation and linking issues
    that arise when some kernel variants are combined in the same translation
    unit.

  * For convenience, we make heavy use of macros to define data declarations
    and kernel bodies in the Suite. While seemingly cryptic, this significantly
    reduces the amount of redundant code required to implement multiple variants
    for each kernel and make sure things are the same as much as possible. The
    kernel class implementation files in the Suite provide many examples of
    the basic pattern we use.

  * We also use macros to define some methods used with GPU block size tunings.
    While seemingly cryptic, this significantly reduces the amount of redundant
    code required to implement calling and naming each of the multiple tunings
    for each kernel and make sure things are the same as much as possible.

##### updateChecksum() method

The `updateChecksum()` method is responsible for adding the checksum
for the current kernel (based on the data the kernel computes) to the
checksum value for the variant and tuning of the kernel just executed, which is
held in the KernelBase base class object.

It is important that the checksum be computed in the same way for
each variant of the kernel so that checksums for different variants can be
compared to help identify differences, and potential errors in
implementations, compiler optimizations, programming model execution, etc.

Note: to simplify checksum computations and help ensure consistency, there
are methods to compute checksums, a weighted sum of array values for example,
are defined in the `DataUtils.hpp` header file in the `common` directory.

##### tearDown() method

The `tearDown()` method frees and/or resets all kernel data that is
allocated and/or initialized in the `setUp()` method execution to prepare for
other kernel variants run subsequently.


### Add object construction operation

The `Executor` class in the `common` directory is responsible for creating
kernel objects for the kernels to be run based on the Suite input options.
To ensure a new kernel object will be created properly, add a call to its
class constructor based on its `KernelID` in the `getKernelObject()`
method in the `RAJAPerfSuite.cpp` file.


## Adding a variant

Each variant in the RAJA Performance Suite is identified by an enumeration
value and a string name. Adding a new variant requires adding these two
items similarly to adding those for a kernel as described above.

### Add the variant ID and name

First, add an enumeration value identifier for the variant, that is unique
among all variants, in the enum 'VariantID' in the header file
`RAJAPerfSuite.hpp`:

```cpp
enum VariantID {
..
  NewVariant,
..
};
```

Second, add the variant name to the array of strings `VariantNames` in the file
`RAJAPerfSuite.cpp`:

```cpp
static const std::string VariantNames [] =
{
..
  std::string("NewVariant"),
..
};
```

Note that the variant string name is just a string version of the variant ID.
This convention must be followed so that the variant works properly with the
Performance Suite machinery. Also, the values in the VariantID enum and the
strings in the `VariantNames` array must be kept consistent (i.e., same order
and matching one-to-one).

### Add kernel variant implementations

In the classes containing kernels to which the new variant applies,
add implementations for the variant in the setup, kernel execution,
checksum computation, and teardown methods as needed. Also, make sure to
define the variant for those kernels in the kernel class constructors by
calling `setVariantDefined(NewVariant)` so that the variant can be run.
These operations are described in earlier sections for adding a new kernel
above.

### Add kernel tuning implementations

In the classes containing kernels to which the new tuning applies,
add implementations for the tuning in the kernel execution and tuning naming
methods as needed. Note that the tuning indices are determined by the order that
the tuning names are added in the `set*TuningDefinitions` method. Therefore
the `run*Variant` methods should have similar logic in order to run the correct
tuning based on the index.

* * *

# Continuous Integration

RAJAPerf Suite uses continuous integration to ensure that changes added to the repository are well integrated and tested for compatibility with the rest of the existing code base. Our CI tests include a variety of vetted configurations that run on different LC machines.

RAJAPerf Suite shares its Gitlab CI workflow with other projects. The documentation is therefore [shared](https://radiuss-ci.readthedocs.io/en/latest).

* * *

# Contributions

The RAJA Performance Suite is a work-in-progress, with new kernels and variants
added as new features and back-end support are developed in RAJA. We encourage
interested parties to contribute to it so that C++ compiler optimizations and
support for programming models like RAJA continue to improve.

The Suite developers follow the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. Folks wishing to contribute to the Suite,
should include their work in a feature branch created from the Performance
Suite `develop` branch. Then, create a pull request with the `develop` branch
as the destination when it is ready to be reviewed. The `develop` branch
contains the latest work in RAJA Performance Suite. Periodically, we merge the
develop branch into the `main` branch and tag a new release.

If you would like to contribute to the RAJA Performance Suite, or have
questions about doing so, please contact the maintainer of the Suite listed
below.

* * *

# Authors

The primary developer/maintainer of the RAJA Performance Suite:

  * Rich Hornung (hornung1@llnl.gov)

Please see the {RAJA Performance Suite Contributors Page](https://github.com/LLNL/RAJAPerf/graphs/contributors), to see the full list of contributors to the
project.

* * *

# LICENSE

The RAJA Performance Suite is licensed under the BSD 3-Clause license,
(BSD-3-Clause or https://opensource.org/licenses/BSD-3-Clause).

Copyrights and patents in the RAJAPerf project are retained by contributors.
No copyright assignment is required to contribute to RAJAPerf.

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-738930`  `OCEC-17-159`

For release details and restrictions, please see the information in the
following:
- [RELEASE](./RELEASE)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)

* * *

# SPDX Usage

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

* * *

# External Packages

The RAJA Performance Suite has some external dependencies, which are included
as Git submodules. These packages are covered by various permissive licenses.
A summary listing follows. See the license included with each package for
full details.

PackageName: BLT
PackageHomePage: https://github.com/LLNL/blt/
PackageLicenseDeclared: BSD-3-Clause

PackageName: RAJA
PackageHomePage: http://github.com/LLNL/RAJA/
PackageLicenseDeclared: BSD-3-Clause

* * *

[RAJA]: https://github.com/LLNL/RAJA
[BLT]: https://github.com/LLNL/blt

