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

This section describes how to add Kernels, Variants, Groups, and Tunings to the
Suite. The discussion aims to make clear the organization of the code and 
how it works, which is useful to understand when making a contribution.

All files containing RAJA Performance Suite infrastructure and kernels reside 
in the ``src`` directory of the project. If you list the contents of that 
directory, you will see the following::

  $ ls -c1 -F
  lcals/
  stream/
  rajaperf_config.hpp.in
  polybench/
  common/
  basic/
  apps/
  algorithm/
  RAJAPerfSuiteDriver.cpp
  CMakeLists.txt

Each directory contains files for kernels in the Group associated with the 
directory name. For example, the ``lcals`` directory contains kernels from
the LCALS benchmark suite, the ``stream`` directory contains kernels from a
stream benchmark suite, and so on. The one exception is the ``common`` 
directory, which contains files that implement the Suite infrastructure and 
utilities, such as data management routines, used throughout the Suite.

The following discussion describes how to modify and add files with new 
content to the Suite, such as new kernels, variants, etc.

.. _structure_addkernel-label:

================
Adding a Kernel
================

Adding a kernel to the Suite involves five main steps:

#. Add a unique kernel ID and a unique kernel name to the Suite.
#. If the kernel is part of a new kernel group or exercises a new RAJA feature,
   add a unique group ID and group name. 
#. If the kernel exercises a RAJA feature that is not currently used in the 
   Suite, add a unique feature ID and feature name.
#. Implement a kernel class that defines all operations needed to integrate
   the kernel into the Suite. This includes adding the kernel class header
   file and source files that contain kernel variant implementations.
#. Add appropriate targets to ``CMakeLists.txt`` files, where needed, so
   that the new kernel code will be compiled when the Suite is built.

We describe the steps in the following sections.

.. _structure_addkernel_name-label:

Adding a kernel ID and name
----------------------------

Two key pieces of information are used to identify each kernel in the Suite: 
the group in which it resides and the name of the kernel. Kernel IDs and
names are maintained in the files ``RAJAPerfSuite.hpp`` and 
``RAJAPerfSuite.cpp``, respectively, which reside in the ``src/common`` 
directory.

For concreteness, we describe how one would add the **ADD** kernel, which 
already exists in the Suite in the **Stream** kernel group.

First, add an enumeration value identifier for the kernel, that is unique 
among all Suite kernels, in the enum ``KernelID`` in the
``src/common/RAJAPerfSuite.hpp`` header file::

  enum KernelID {

    ...

    //
    // Stream kernels...
    //
    Stream_ADD,
    ...

  };

Second, add the kernel name to the array of strings ``KernelNames`` in the
``src/common/RAJAPerfSuite.cpp`` source file::

  static const std::string KernelNames [] =
  {

    ...

    //
    // Stream kernels...
    //
    std::string("Stream_ADD"),
    ...

  };

Several conventions are important to note for a kernel ID and name. Following 
them will ensure that the kernel integrates properly into the RAJA Performance 
Suite machinery.

.. note:: * The enumeration value label for a kernel is the **group name followed by the kernel name separated by an underscore**.
          * Kernel ID enumeration values for kernels in the same group must
            appear consecutively in the enumeration.
          * Kernel ID enumeration labels must in alphabetical order, with 
            respect to the base kernel name in each group.
          * The kernel string name is just a string version of the kernel ID.
          * The values in the ``KernelID`` enum must match the strings in the
            ``KernelNames`` array one-to-one and in the same order.

Typically, adding a new Group or Feature is not needed when adding a Kernel.
One or both of these needs to be added only if the Kernel is not part of an
existing Group of kernels, or exercises a RAJA Feature that is not used in an
existing Kernel. For completeness, we describe the addition of a new group and
feature in case either is needed.

.. _structure_addkernel_group-label:

Adding a group 
----------------------------

If a kernel is added as part of a new group of kernels in the Suite, a new 
value must be added to the ``GroupID`` enum in the ``RAJAPerfSuite.hpp`` 
header file and an associated group string name must be added to the 
``GroupNames`` string array in the ``RAJAPerfSuite.cpp`` source file. The
process is similar to adding a new kernel ID and name described above.

.. note:: Enumeration values and string array entries for Groups must be kept 
          consistent, in the same order and matching one-to-one.

.. _structure_addkernel_feature-label:

Adding a feature
----------------------------

If a kernel is added that exercises a RAJA Feature that is not used in an
existing kernel, a new value must be added to the ``FeatureID`` enum in the
``RAJAPerfSuite.hpp`` header file and an associated feature string name must 
be added to the ``FeatureNames`` string array in the ``RAJAPerfSuite.cpp`` 
source file. The process is similar to adding a new kernel ID and name 
described above.

.. note:: Enumeration values and string array entries for Features must be kept 
          consistent, in the same order and matching one-to-one.

.. _structure_addvariant-label:

================
Adding a Variant
================

Similar to a Kernel, each Variant in the Suite is is identified by an 
enumeration value and a string name. Adding a new variant requires adding 
these two items in a similar fashion to adding a kernel name and ID as 
described above.

Adding a variant to the Suite involves four main steps:

#. Add a unique variant ID and a unique variant name to the Suite.
#. Add the pure virtual method to execute the variant to the ``KernelBase``
   class header file. For example::

     virtual void run<variant-name>Variant(VariantID vid, size_t tune_idx) = 0;

#. For the kernel(s) to which the variant applies, provide kernel variant
   implementations in associated ``<kernel-name>-<variant-name>.cpp`` files.
#. Add appropriate targets to ``CMakeLists.txt`` files, where needed, so
   that the new kernel variant code will be compiled when the Suite is built.

We describe the steps in the following sections.

.. _structure_addvariant_name-label:

Adding a variant ID and name
----------------------------

Variant IDs and names are maintained in the files ``RAJAPerfSuite.hpp`` and
``RAJAPerfSuite.cpp``, respectively, which reside in the ``src/common``
directory. Adding a new variant ID and name is essentially the same as
adding a kernel ID and name, which is described in 
:ref:`structure_addkernel_name-label`.

.. note:: A variant string name is just a string version of the variant ID.               enum value label. This convention must be followed so that each
          variant works properly within the RAJA Performance Suite 
          machinery. Also, the values in the VariantID enum and the 
          strings in the VariantNames array must be kept consistent 
          (i.e., same order and matching one-to-one).

.. _structure_addvariant_impl-label:

Adding kernel variant implementations
-------------------------------------

In the classes containing kernels to which a new variant applies, add 
implementations for the variant in kernel execution methods in files named
``<kernen-name>-<variant-name>.cpp``. This is described in detail in 
:ref:`kernel_class_impl_exec-label`. 

.. note:: Make sure to enable the variant for those kernels in the kernel 
          class constructors by calling the ``KernelBase`` class  method
          ``setVariantDefined(VariantID vid))`` so that the variant can be 
          run. 

.. _structure_addtuning-label:

================
Adding a Tuning
================

For kernels to which a new tuning applies, add implementations for the tuning 
in the kernel execution and tuning naming methods as needed. Note that the 
tuning indices are determined by the order that the tuning names are added 
in the ``set<backend-name>TuningDefinitions(VariantID vid)`` method which is
virtual in the ``KernelBase`` class. 

.. note:: ``run<backend-name>Variant(VariantID vid, size_t tune_idx) methods 
          should have similar logic to the 
          ``set<backend-name>TuningDefinitions(VariantID vid)`` method so that
          the correct tuning will be run based on the index.
