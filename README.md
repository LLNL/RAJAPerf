RAJA Performance Suite
======================

The RAJA performance suite is used to explore performance of loop-based 
computational kernels of the sort found in HPC applications. Each kernel 
appears in multiple variants that enable experiments to evaluate and 
compare runtime performance achieved by RAJA and non-RAJA variants, using 
different parallel programming models (e.g., OpenMP, CUDA, etc.), compiler 
optimizations (e.g., SIMD vectorization), etc.

The kernels are borrowed from a variety of sources, such as benchmark 
suites and applications. Kernels are partitioned into "groups", which indicates 
the origin of the kernels, algorithm patterns they represent, etc. For 
example, the "Apps" group contains a collection of kernels extracted from 
real scientific computing applications, kernels in the "Basic" group are 
small and simple, but exhibit challenges for compiler optimizations, etc.

* * *

Table of Contents
=================

1. [Building the suite](#building-the-suite)
2. [Running the suite](#running-the-suite)
3. [Generated output](#generated-output)
4. [Adding kernels and variants](#adding-kernels-and-variants)

* * *

# Building the suite

Before building the suite, you must get a copy of the code by cloning the
necessary source repositories.

First, clone the RAJA repo into a directory of your choice; e.g.
```
> mkdir RAJA-stuff
> cd RAJA-stuff
> git clone https://github.com/LLNL/RAJA.git
> ls 
RAJA
```

Next, clone the RAJA Performance Suite repo into a specific location in 
the RAJA source tree. Starting after the last step above:
```
> cd RAJA
> git clone ssh://git@cz-bitbucket.llnl.gov:7999/raja/raja-perfsuite.git ./extra/performance
```

The process of cloning the performance suite repo into the RAJA code in this 
way is temporary. It will change when the performance suite is moved to a 
project on GitHub.

Finally, use [CMake] to build the code. The simplest way to build the code is 
to create a build directory in the top-level RAJA directory (in-source builds 
are not allowed!) and run CMake from there; i.e., :
```
> mkdir build
> cd build
> cmake -DRAJA_ENABLE_PERFSUITE=On ../
> make raja-perf.exe
```

RAJA and the Performance Suite are built together using the CMake 
configuration. Please see the RAJA Quickstart Guide (add link) for details 
on building RAJA, configuration options, etc.

* * *

# Running the suite

The suite is executed by running the executable in the top-level performance 
suite 'src' directory in the build space. For example, giving it no options:
```
> ./extra/performance/src/raja-perf.exe
```
will run the entire suite (all kernels and variants) in their default 
configurations.

The suite can be run in a variety of ways (e.g., specified subsets of kernels, 
variants, groups) by passing appropriate options to the executable. To see 
available options along with a brief description, pass the '--help' or '-h'
option:
```
> ./extra/performance/src/raja-perf.exe --help
```

* * *

# Generated output

Fill this in when we have a basic implementation... 

* * *

# Adding loop kernels and variants

The following describes how to add new kernels and/or kernel variants to the
RAJA Performance Suite. Group modifications are implicit and do not require
any significant additional steps. The information in this section also
provides insight into how the performance suite operates.

It is essential that the appropriate targets are updated in the appropriate
'CMakeLists.txt' files when loop kernels are added.

## Adding a kernel

Two key pieces of information identify a kernel: the group in which it 
resides and the name of the kernel itself. For concreteness, we describe
how to add a kernel "Foo" that lives in the kernel group "Bar".

### Add the kernel ID and name

The files `RAJAPerfSuite.hxx` and `RAJAPerfSuite.cxx` define enumeration 
values and arrays of string names for the kernels, respectively. 

First, add an enumeration value identifier for the kernel, that is unique 
among all kernels, in the enum 'KerneID' in the header file `RAJAPerfSuite.hxx`:

```cpp
enum KernelID {
..
  Bar_Foo,
..
};
```

Note that the enumeration value for the kernel is the group name followed
by the kernel name, separated by an underscore. It is important to follow
this convention so that the kernel works with existing performance
suite machinery. 

Second, add the kernel name to the array of strings 'KernelNames' in the file
`RAJAPerfSuite.cxx`:

```cpp
static const std::string KernelNames [] =
{
..
  std::string("Bar_Foo"),
..
};
```

Note that the kernel string name is just a string version of the kernel ID.
This convention must be followed so that the kernel works with existing 
performance suite machinery. Also, the values in the KernelID enum and the
strings in the KernelNames array must be kept consistent (i.e., same order
and matching one-to-one).


### Add new group if needed

If a kernel is added as part of a new group of kernels in the suite, a
new value must be added to the 'GroupID' enum in the header file 
`RAJAPerfSuite.hxx` and an associated group string name must be added to
the array of strings 'GroupNames' in the file `RAJAPerfSuite.cxx`. Again,
the enumeration values and items in the string array must be kept
consistent.


### Add the kernel class

Each kernel in the RAJA Performance Suite is implemented in a class whose
header and implementation files live in the directory named for the group
in which the kernel lives. The kernel class is responsible for implementing
all operations needed to execute and record execution timing and result 
checksum information for each variant of the kernel. 

Continuing with our example, we add 'Foo' class header and implementation 
files 'Foo.hxx' and 'Foo.cxx' to the 'src/bar' directory. The class must 
inherit from the 'KernelBase' base class that defined the interface for 
kernels in the suite. 

#### Kernel class header

Here is what the header file for the Foo kernel object may look like:

```cpp
#ifndef RAJAPerf_Bar_Foo_HXX
#define RAJAPerf_Bar_Foo_HXX

#include "common/KernelBase.hxx"
#include "RAJA/RAJA.hxx"

namespace rajaperf  // RAJA Performance Suite namespeace
{
namespace bar   // Kernel group namespace
{

class Foo : public KernelBase
{
public:

  Foo(const RunParams& params);

  ~Foo();

  void setUp(VariantID vid);
  void runKernel(VariantID vid);
  void computeChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  // Kernel-specific data (pointers, scalares, etc.) used in kernel...
};

} // end namespace bar
} // end namespace rajaperf

#endif // closing endif for header file include guard
```

The kernel object header has a uniquely-named header file include guard and
the class is nested within the 'rajaperf' and 'bar' namespaces. The 
constructor takes a reference to a 'RunParams' object, which contains the
input parameters for running the suite -- we'll say more about this later. 
The four methods that take a variant ID argument must be provided as they are
pure virtual in the KernelBase class. Their names are descriptive of what they
do and we'll provide more details when we describe the class implementation
next.

#### Kernel class implementation

Providing the Foo class implementation is straightforward. However, several
conventions that we describe here must be followed.

##### Constructor and destructor

The constructor must pass the kernel ID and RunParams object to the base
class 'KernelBase' constructor. The body of the constructor must also call
base class methods to set the default size for the iteration space of the 
kernel (e.g., the number of loop iterations) and the number of samples to 
execute the kernel with each pass through the suite. These values will be 
modified based on input parameters to define the size and number of samples 
actually executed when the suite is run. Here is how this might look:

```cpp
Foo::Foo(const RunParams& params)
  : Foo(rajaperf::Bar_Foo, params),
    // default initialization of class members
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}
```

The class destructor doesn't have any requirements beyond freeing memory
owned by the class object.

##### setUp() method

The 'setUp()' method is responsible for allocating and initializing data 
necessary to run the kernel for the variant defined by the method's variant ID 
argument. For example, a baseline variant may have aligned data allocation
to help enable SIMD optimizations, an OpenMP variant may initialize arrays
following a pattern of "first touch" based on how memory and threads are 
mapped to CPU cores, a CUDA variant may initialize data on a host and copy
it into device memory, etc.

It is important to use the same data allocation and initialization operations
for RAJA and non-RAJA variants that are related. Also, the state of all 
input data for the kernel should be the same for all variants so that 
checksums can be compared at the end of a run.

Note: there are some utilities to data allocation and initialization
in the 'common' directory that can be shared by different kernels.

##### runKernel() method

The 'runKernel()' method executes the kernel for the variant defined by the 
method's variant ID argument. The method is also responsible for calling
base class methods to start and stop execution timers for the loop variant.
A typical kernel execution code section may look like:

```cpp
void Foo::runKernel(VariantID vid)
{
  // ...

  // Execute vid variant of kernel
  startTimer();
  for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
     // Kernel implementation for vid variant
  }
  stopTimer();

  // ...
}
```

##### computeChecksum() method

The 'computeChecksum()' method is responsible for updating the checksum 
value for the variant of the kernel just executed, which is held in the
KernelBase base class. The checksum is based on data that the kernel
computes; e.g., if the kernel computes an array of values, a reasonable
checksum could be a weighted sum of the elements of the array.

It is important that the checksum be computed in the same way for
each variant of the kernel so that checksums for different variants can be 
compared to help identify differences, and potentially errors, in 
implementations, compiler optimizations, programming model execution, etc.

Note: there are some utilities to update checksums in the 'common' directory 
that can be shared by different kernels.

##### tearDown() method

The 'tearDown()' method must free and/or reset all kernel data that is
allocated and/or initialized in the 'setUp' method execution for other
kernel variants run subsequently.


### Add object construction operation

The 'Executor' class creates objects representing kernels to be run
based on the suite run parameters. To make sure the object for a new kernel
is created properly, add a call to its class constructor in the 
'getKernelObject()' method in the 'RAJAPerfSuite.cxx' file.

  
## Adding a variant

Each variant in the RAJA Performance Suite is identified by an enumeration
value and a string name. Adding a new variant requires adding these two
items similar to adding a kernel as described above. 

### Add the variant ID and name

First, add an enumeration value identifier for the variant, that is unique 
among all variants, in the enum 'VariantID' in the header file 
`RAJAPerfSuite.hxx`:

```cpp
enum VariantID {
..
  NewVariant,
..
};
```

Second, add the variant name to the array of strings 'VariantNames' in the file
`RAJAPerfSuite.cxx`:

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
in the sections for adding a new kernel above.


* * *

[RAJA]: https://github/LLNL/RAJA
[CMake]: http://www.cmake.org
