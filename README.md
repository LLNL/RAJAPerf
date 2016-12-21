RAJA Performance Suite
======================

The RAJA performance suite is used to explore performance of different 
variants of loop-based computational kernels. Each kernel appears in multiple 
variants that enable experiments to be run to evaluate and compare runtime 
performance achieved by RAJA and non-RAJA variants, using different parallel 
programming models (e.g., OpenMP, CUDA, etc.), compiler optimizations, etc.

The kernels are borrowed from a variety of sources, including other benchmark 
suites and applications. The kernels in the suite are partitioned into 
"sub-suites", that we refer to simply as "suites", indicating their origin. 
For example, the "Apps" suite contains a collection of kernels extracted from
scientific computing applications.

* * *

Table of Contents
=================

1. [Building the suite](#building-the-suite)
2. [Running the suite](#running-the-suite)
3. [Generated output](#generated-output)
4. [Adding a new suite](#adding-a-new-suite)
5. [Adding kernels and variants](#adding-kernels-and-variants)

## Building the suite

Before building the suite, you must get a copy of the code by cloning the
necessary source repositories.

First, clone the RAJA repo into a directory of your choice; e.g.
```
> mkdir RAJA-stuff
> cd RAJA-stuff
> git clone https://github.com/LLNL/RAJA.git
```

Next, clone the performance suite repo into a specific location in the RAJA 
source tree. Starting after the last step above:
```
> cd RAJA
> git clone ssh://git@cz-bitbucket.llnl.gov:7999/raja/raja-perfsuite.git ./extra/performance
```

Note that the process of cloning the performance suite repo into the RAJA code
in this way is temporary. It will change when the performance suite is moved
to a project on GitHub.

Finally, use [CMake] to build the code. The simplest way to build the code is 
to create a build directory in the top-level RAJA directory (in-source builds 
are not allowed!) and run CMake from there; i.e., :
```
> mkdir build
> cd build
> cmake -DRAJA_ENABLE_PERFSUITE=On ../
> make raja-perf.exe
```

Note that the directory `host-config` in the RAJA repo contains `host config` 
files that can be passed to CMake with the "-C" option to seed the CMake
cache. These files are often a useful guide for using different compilers 
and configurations.


## Running the suite

After the suite is compiled, the suite is run via the executable in the 
performance suite directory. For example, giving it no options:
```
> ./extra/performance/src/raja-perf.exe
```
will run the entire suite (all kernels and variants) in their default 
configurations.

The suite can be run in a variety of ways (e.g., subsets of kernels, variants,
suites) with different configurations by passing appropriate command line
options to the executable. To see the available options and a brief description
of them, run:
```
> ./extra/performance/src/raja-perf.exe --help
```

## Generated output

Fill this in when we have a basic implementation... 

## Adding a new suite


## Adding kernels and variants


* * * 

[RAJA]: https://github/LLNL/RAJA
[CMake]: http://www.cmake.org
