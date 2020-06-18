[comment]: # (#################################################################)
[comment]: # (Copyright 2017-20, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA Performance Suite project contributors.)
[comment]: # (See the RAJA/COPYRIGHT file for details.)
[comment]: #
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)


RAJA Performance Suite
======================

[![Build Status](https://travis-ci.org/LLNL/RAJAPerf.svg?branch=develop)](https://travis-ci.org/LLNL/RAJAPerf)

The RAJA performance suite is developed to explore performance of loop-based 
computational kernels found in HPC applications. Specifically, it
is used to assess, monitor, and compare runtime performance of kernels 
implemented using RAJA and variants implemented using standard or 
vendor-supported parallel programming models directly. Each kernel in the 
suite appears in multiple RAJA and non-RAJA (i.e., baseline) variants using 
parallel programming models such as OpenMP and CUDA.

The kernels originate from various HPC benchmark suites and applications. 
Kernels are partitioned into "groups" --  each group
indicates the origin of its kernels, the algorithm patterns they represent, 
etc. For example, the "Apps" group contains a collection of kernels extracted 
from real scientific computing applications, the "Basic" group contains 
kernels that are small and simple, but exhibit challenges for compiler 
optimization, and so forth.

* * *

Table of Contents
=================

1. [Building the suite](#building-the-suite)
2. [Running the suite](#running-the-suite)
3. [Generated output](#generated-output)
4. [Adding kernels and variants](#adding-kernels-and-variants)
5. [Contributions](#contributions)
6. [Authors](#authors)
7. [Copyright and Release](#copyright-and-release)

* * *

# Building the suite

To build the suite, you must first obtain a copy of the code by cloning the
source repository. For example,

```
> mkdir RAJA-PERFSUITE
> cd RAJA-PERFSUITE
> git clone --recursive https://github.com/llnl/RAJAPerf.git
> ls 
RAJAPerf
```

The Performance Suite has [RAJA] and the CMake-based [BLT] build system
as Git submodules. The `--recursive` argument will clone the appropriate
version of the submodules into the Performance Suite source code. So
if you switch to a different branch, you will have to update the 
submodules. For example,

```
> cd RAJAPerf
> git checkout <some branch name>
> git submodule init
> git submodule update --recursive
```

Note that the `--recursive` will update submodules within submodules, similar
to usage with the `git clone` as described above.

RAJA and the Performance Suite are built together using the same CMake
configuration. For convenience, we include scripts in the `scripts`
directory that invoke corresponding configuration files (CMake cache files) 
in the RAJA submodule. For example, the `scripts/lc-builds` directory 
contains scripts that show how we build code for testing on platforms in
the Lawrence Livermore Computing Center. Each build script creates a 
descriptively-named build space directory in the top-level Performance Suite 
directory and runs CMake with a configuration appropriate for the platform and 
compilers used. After CMake completes, enter the build directory and type 
`make` (or `make -j <N>` for a parallel build) to compile the code. For example,

```
> ./scripts/blueos_nvcc8.0_clang-coral.sh
> cd build_blueos_nvcc8.0_clang-coral
> make -j
```

You can also create your own build directory and run CMake with your own
arguments from there; e.g., :

```
> mkdir my-build
> cd my-build
> cmake <cmake args> ../
> make -j
```

The provided configurations will only build the Performance Suite code by
default; i.e., it will not build any RAJA test or example codes. If you
want to build the RAJA tests for example and verify your build of RAJA is 
working properly, just add the `-DENABLE_TESTS=On` option to CMake, either
on the command line if you run CMake directly or edit the script you are 
running to do this. Then, when the build completes, you can type `make test`
to run the tests.


* * *

# Running the suite

The suite is run by invoking the executable in the `bin` directory in the 
build space. For example, giving it no options:

```
> ./bin/raja-perf.exe
```

will run the entire suite (all kernels and variants) in their default 
configurations.

The suite can be run in a variety of ways by passing options to the executable.
For example, you can run subsets of kernels by specifying variants, groups, or
listing individual kernels explicitly. Other configuration options to set 
problem sizes, number of kernel repetitions, etc. can also be specified. The 
goal is to build the code once and use scripts or other mechanisms to run the 
suite in different ways.

Note: most options appear in a long or short form for ease of use.

To see available options along with a brief description of each, pass the 
`--help` or `-h` option:

```
> ./bin/raja-perf.exe --help
```

or

```
> ./bin/raja-perf.exe -h
```

Lastly, the program will emit a summary of provided input if it is given 
input that the code does not know how to parse. Hopefully, this will make it 
easy for users to correct erroneous usage.

# Important notes

 * The OpenMP target offload variants of the kernels in the Suite are a 
   work-in-progress. Depending on the compiler you have available, you 
   may not be able to compile them. If you can build them, they will 
   appear in an executable named `./bin/raja-perf-omptarget.exe` which is
   distinct from the one named above. The build system will be reworked in 
   the future so that the OpenMP target variants can be run from the same 
   executable as the other variants.

* * *

# Generated output

Running the suite will generate several output files whose name starts with
the specified file prefix in the specified  out put directory. If no such
preferences are provided, files will be located in the current directory
and be named `RAJAPerf*`.

Currently, there are up to four files generated:

1. Timing -- execution time (sec.) of each loop kernel and variant
2. Checksum -- checksum value from results of each loop kernel and variant
3. Speedup -- runtime speedup of each loop kernel and variant with respect to reference variant. Reference variant can be set with command line option.
4. Figure of Merit (FOM) -- basic statistics about speedup of RAJA variant vs. baseline for each programming model run. PASS/FAIL tolerance can be set with command line option.

The name of each file is indicative of its contents. All files are text files. 
Other than the checksum file, all are in 'csv' format for easy processing 
by various tools.

* * *

# Adding kernels and variants

This section describes how to add new kernels and/or kernel variants to the
RAJA Performance Suite. Group modifications are not required unless a new
group is added. The information in this section also provides insight into 
how the performance suite operates.

It is essential that the appropriate targets are updated in the appropriate
`CMakeLists.txt` files when files are added to the suite.

## Adding a kernel

Adding a new kernel to the suite involves three main steps:

1. Add unique kernel ID and unique name to the suite. 
2. If the kernel is part of a new kernel group, also add a unique group ID and name for the group.
3. Implement a kernel class that contains all operations needed to run it, with source files organized as described below.

These steps are described in the following sections.

### Add the kernel ID and name

Two key pieces of information identify a kernel: the group in which it 
resides and the name of the kernel itself. For concreteness, we describe
how to add a kernel "Foo" that lives in the kernel group "Bar". The files 
`RAJAPerfSuite.hpp` and `RAJAPerfSuite.cpp` define enumeration 
values and arrays of string names for the kernels, respectively. 

First, add an enumeration value identifier for the kernel, that is unique 
among all kernels, in the enum 'KerneID' in the header file `RAJAPerfSuite.hpp`:

```cpp
enum KernelID {
..
  Bar_Foo,
..
};
```

Note: the enumeration value for the kernel is the group name followed
by the kernel name, separated by an underscore. It is important to follow
this convention so that the kernel works with existing performance
suite machinery. 

Second, add the kernel name to the array of strings 'KernelNames' in the file
`RAJAPerfSuite.cpp`:

```cpp
static const std::string KernelNames [] =
{
..
  std::string("Bar_Foo"),
..
};
```

Note: the kernel string name is just a string version of the kernel ID.
This convention must be followed so that the kernel works with existing 
performance suite machinery. Also, the values in the KernelID enum and the
strings in the KernelNames array must be kept consistent (i.e., same order
and matching one-to-one).


### Add new group if needed

If a kernel is added as part of a new group of kernels in the suite, a
new value must be added to the 'GroupID' enum in the header file 
`RAJAPerfSuite.hpp` and an associated group string name must be added to
the 'GroupNames' array of strings in the file `RAJAPerfSuite.cpp`. Again,
the enumeration values and items in the string array must be kept
consistent.


### Add the kernel class

Each kernel in the suite is implemented in a class whose header and 
implementation files live in the directory named for the group
in which the kernel lives. The kernel class is responsible for implementing
all operations needed to manage data, execute and record execution timing and 
result checksum information for each variant of the kernel. 
To properly plug in to the Perf Suite framework, the kernel class must 
inherit from the `KernelBase` base class that defines the interface for 
a kernel in the suite.

Continuing with our example, we add a 'Foo' class header file 'Foo.hpp', 
and multiple implementation files described in the following sections: 
  * 'Foo.cpp' contains the methods to setup and teardown the memory for the
    'Foo kernel, and compute and record a checksum on the result after it 
    executes;
  * 'Foo-Seq.cpp' contains CPU variants of the kernel;
  * 'Foo-OMP.cpp' contains OpenMP CPU multithreading variants of the kernel;
  * 'Foo-Cuda.cpp' contains CUDA GPU variants of the kernel; and
  * 'Foo-OMPTarget.cpp' contains OpenMP target offload variants of the kernel.


#### Kernel class header

Here is what the header file for the Foo kernel object may look like:

```cpp
#ifndef RAJAPerf_Bar_Foo_HXX
#define RAJAPerf_Bar_Foo_HXX


///
/// Foo kernel reference implementation:
///
/// Describe it here...
///


#include "common/KernelBase.hpp"

namespace rajaperf  
{
class RunParams; // Forward declaration for ctor arg.

namespace bar   
{

class Foo : public KernelBase
{
public:

  Foo(const RunParams& params);

  ~Foo();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid); 

private:
  // Kernel-specific data (pointers, scalars, etc.) used in kernel...
};

} // end namespace bar
} // end namespace rajaperf

#endif // closing endif for header file include guard
```

The kernel object header has a uniquely-named header file include guard and
the class is nested within the 'rajaperf' and 'bar' namespaces. The 
constructor takes a reference to a 'RunParams' object, which contains the
input parameters for running the suite -- we'll say more about this later. 
The seven methods that take a variant ID argument must be provided as they are
pure virtual in the KernelBase class. Their names are descriptive of what they
do and we'll provide more details when we describe the class implementation
next.

#### Kernel class implementation

All kernels in the suite follow a similar implementation pattern for 
consistency and ease of analysis and understanding. Here, we describe several 
steps and conventions that must be followed to ensure that all kernels 
interact with the performance suite machinery in the same way:

1. Initialize the 'KernelBase' class object with KernelID, default size, and default repetition count in the `class constructor`.
2. Implement data allocation and initialization operations for each kernel variant in the `setUp` method.
3. Implement kernel execution for the associated variants in the `run` methods.
4. Compute the checksum for each variant in the `updateChecksum` method.
5. Deallocate and reset any data that will be allocated and/or initialized in subsequent kernel executions in the `tearDown` method.


##### Constructor and destructor

It is important to note that there will only be one instance of each kernel
class created by the program. Thus, each kernel class constructor and 
destructor must only perform operations that are non-specific to any kernel 
variant.

The constructor must pass the kernel ID and RunParams object to the base
class 'KernelBase' constructor. The body of the constructor must also call
base class methods to set the default size for the iteration space of the 
kernel (e.g., typically the number of loop iterations, but can be 
kernel-dependent) and the number of times to repeat (i.e., execute) the kernel 
with each pass through the suite to generate adequate timing information. 
These values will be modified based on input parameters to define the actual 
size and number of reps applied when the suite is run. Here is how this 
typically looks:

```cpp
Foo::Foo(const RunParams& params)
  : Foo(rajaperf::Bar_Foo, params),
    // default initialization of class members
{
   setDefaultSize(100000);
   setDefaultReps(1000);
}
```

The class destructor doesn't have any requirements beyond freeing memory
owned by the class object as needed.

##### setUp() method

The 'setUp()' method is responsible for allocating and initializing data 
necessary to run the kernel for the variant specified by its variant ID 
argument. For example, a baseline variant may have aligned data allocation
to help enable SIMD optimizations, an OpenMP variant may initialize arrays
following a pattern of "first touch" based on how memory and threads are 
mapped to CPU cores, a CUDA variant may initialize data in host memory and 
copy it into device memory, etc.

It is important to use the same data allocation and initialization operations
for RAJA and non-RAJA variants that are related. Also, the state of all 
input data for the kernel should be the same for all variants so that 
checksums can be compared at the end of a run.

Note: to simplify these operations and help ensure consistency, there exist 
utility methods to allocate, initialize, deallocate, and copy data, and compute
checksums defined in the `DataUtils.hpp` `CudaDataUtils.hpp`, and 
`OpenMPTargetDataUtils.hpp` header files in the 'common' directory.

##### run methods

Which files contain which 'run' methods and associated variant implementations 
is described above. Each method take a variant ID argument which identifies
the variant to be run for each programming model back-end. Each method is also 
responsible for calling base class methods to start and stop execution timers 
when a loop variant is run. A typical kernel execution code section may look 
like:

```cpp
void Foo::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  // ...

  switch ( vid ) {

    case Base_Seq : {

      // Declare data for baseline sequential variant of kernel...

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
         // Implementation of Base_Seq kernel variant...
      }
      stopTimer();

      // ...

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
      std::cout << "\n  <kernel-name> : Unknown variant id = " << vid << std::endl;
    }

  }
}
```

All kernel implementation files are organized in this way. So following this
pattern will keep all new additions consistent. 

Note: As described earlier, there are five source files for each kernel.
The reason for this is that it makes it easier to apply unique compiler flags 
to different variants and to manage compilation and linking issues that arise 
when some kernel variants are combined in the same translation unit.

Note: for convenience, we make heavy use of macros to define data 
declarations and kernel bodies in the suite. This significantly reduces
the amount of redundant code required to implement multiple variants
of each kernel and make sure things are the same as much as possible. 
The kernel class implementation files in the suite provide many examples of 
the basic pattern we use.

##### updateChecksum() method

The 'updateChecksum()' method is responsible for adding the checksum
for the current kernel (based on the data the kernel computes) to the 
checksum value for the variant of the kernel just executed, which is held 
in the KernelBase base class object. 

It is important that the checksum be computed in the same way for
each variant of the kernel so that checksums for different variants can be 
compared to help identify differences, and potentially errors, in 
implementations, compiler optimizations, programming model execution, etc.

Note: to simplify checksum computations and help ensure consistency, there 
are methods to compute checksums defined in the `DataUtils.hpp` header file 
in the 'common' directory.

##### tearDown() method

The 'tearDown()' method free and/or reset all kernel data that is
allocated and/or initialized in the 'setUp' method execution to prepare for 
other kernel variants run subsequently.


### Add object construction operation

The 'Executor' class object is responsible for creating kernel objects 
for the kernels to be run based on the suite input options. To ensure a new
kernel object will be created properly, add a call to its class constructor 
based on its 'KernelID' in the 'getKernelObject()' method in the 
`RAJAPerfSuite.cpp` file.

  
## Adding a variant

Each variant in the RAJA Performance Suite is identified by an enumeration
value and a string name. Adding a new variant requires adding these two
items similar to adding a kernel as described above. 

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

Second, add the variant name to the array of strings 'VariantNames' in the file
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
This convention must be followed so that the variant works with existing
performance suite machinery. Also, the values in the VariantID enum and the
strings in the VariantNames array must be kept consistent (i.e., same order
and matching one-to-one).

### Add kernel variant implementations

In the classes containing kernels to which the new variant applies, 
add implementations for the variant in the setup, kernel execution, 
checksum computation, and teardown methods. These operations are described
in earlier sections for adding a new kernel above.

* * *

# Contributions

The RAJA Performance Suite is intended to be a work-in-progress, with new
kernels and variants added over time. We encourage interested parties to 
contribute to it so that C++ compiler optimizations and support for programming
models like RAJA continue to improve. 

The Suite developers follow the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. Folks wishing to contribute to the Suite, should include their work in a feature branch created from the RAJA `develop` 
branch. Then, create a pull request with the `develop` branch as the 
destination when it is ready to be reviewed. The `develop` branch contains the 
latest work in RAJA Performance Suite. Periodically, we will merge the
develop branch into the `main` branch and tag a new release.

If you would like to contribute to the RAJA Performance Suitea, or have 
questions about doing so, please contact one of its developers. See below.

* * *

# Authors

The primary developer of the RAJA Performance Suite:

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

