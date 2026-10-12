[comment]: # (#################################################################)
[comment]: # (Copyright Lawrence Livermore National Security, LLC and other)
[comment]: # (RAJA Project Developers. See top-level LICENSE and COPYRIGHT)
[comment]: # (files for dates and other details. No copyright assignment is)
[comment]: # (required to contribute to RAJA Performance Suite.)
[comment]: #
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# <img src="/share/raja/logo/RAJA_LOGO_Color.png?raw=true" width="128" valign="middle" alt="RAJA"/>

RAJA Performance Suite
======================

[![Build Status](https://github.com/llnl/RAJAPerf/actions/workflows/build.yml/badge.svg)](https://github.com/llnl/RAJAPerf/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/rajaperf/badge/?version=develop)](https://raja.readthedocs.io/en/develop/?badge=develop)
[![GitHub release](https://img.shields.io/github/release/llnl/RAJAPerf.svg)](https://github.com/llnl/RAJAPerf/releases/latest)

The RAJA Performance Suite is a companion project to the [RAJA] C++ performance
portability abstraction library. The Performance Suite designed to explore
performance of loop-based computational kernels found in HPC applications.
Specifically, it is used to assess and monitor runtime performance of kernels 
implemented using [RAJA] compare those to variants implemented using common 
parallel programming models, such as OpenMP and CUDA, directly.

User Documentation
-------------------

The RAJA Performance Suite User Guide is the best place to start learning 
about it -- how to build it, how to run it, etc. 

The RAJA Performance Suite Developer Guide contains information about 
how the source code is structured, how to contribute to it, etc.

The most recent version of these documents (develop branch) is available here: https://rajaperf.readthedocs.io

To access docs for other branches or version versions: https://readthedocs.org/projects/rajaperf/

A performance benchmark dashboard tracking changes on `develop` is available here:
https://software.llnl.gov/RAJAPerf/dev/bench/

Please see the [RAJA] project for more information about RAJA.

To cite RAJA Performance Suite, please use the following references:

* RAJA Performance Suite. https://github.com/LLNL/RAJAPerf

* Olga Pearce, Jason Burmark, Rich Hornung, Befikir Bogale, Ian Lumsden, Michael McKinsey, Dewi Yokelson, David Boehme, Stephanie Brink, Michela Taufer, Tom Scogland, "RAJA Performance Suite: Performance Portability Analysis with Caliper and Thicket", in 2024 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC) at the International Conference on High Performance Computing, Network, Storage, and Analysis (SC-W 2024). [Download here](https://dl.acm.org/doi/pdf/10.1109/SCW63240.2024.00162)


Communicate with Us
-------------------

The most effective way to communicate with the RAJA development team
is via our mailing list: **raja-dev@llnl.gov** 

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of the RAJA Performance Suite and are 
interested in contributing to its development, please do not hesitate to 
contact us. We are very interested in improving the Suite and exploring new 
ways to use it.

Authors
-----------

Please see the [RAJA Performance Suite Contributors Page](https://github.com/LLNL/RAJAPerf/graphs/contributors), to see the full list of contributors to the project.

License
--------

The RAJA Performance Suite is licensed under the BSD 3-Clause license,
(BSD-3-Clause or https://opensource.org/licenses/BSD-3-Clause).

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-738930`  `OCEC-17-159`

Copyrights and patents in the RAJAPerf project are retained by contributors.
No copyright assignment is required to contribute to RAJAPerf.

For release details and restrictions, please see the information in the
following:
- [COPYRIGHT](./COPYRIGHT)
- [LICENSE](./LICENSE)

SPDX Usage
-----------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

For example, files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
------------------

The RAJA Performance Suite has some external dependencies, which are included
as Git submodules. These packages are covered by various permissive licenses.
A summary listing follows. See the license included with each package for
full details.

PackageName: RAJA  
PackageHomePage: http://github.com/LLNL/RAJA/   
PackageLicenseDeclared: BSD-3-Clause

PackageName: BLT  
PackageHomePage: https://github.com/LLNL/blt/  
PackageLicenseDeclared: BSD-3-Clause

* * *

[RAJA]: https://github.com/LLNL/RAJA
[BLT]: https://github.com/LLNL/blt

