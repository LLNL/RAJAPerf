[comment]: # (#################################################################)
[comment]: # (Copyright 2017-23, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA Performance Suite project contributors.)
[comment]: # (See the RAJAPerf/LICENSE file for details.)
[comment]: #
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# <img src="/tpl/RAJA/share/raja/logo/RAJA_LOGO_Color.png?raw=true" width="128" valign="middle" alt="RAJA"/>

RAJA Performance Suite
======================

[![Azure Piepline Build Status](https://dev.azure.com/llnl/RAJAPerf/_apis/build/status/LLNL.RAJAPerf?branchName=develop)](https://dev.azure.com/llnl/RAJAPerf/_build/latest?definitionId=1&branchName=develop)
[![Documentation Status](https://readthedocs.org/projects/rajaperf/badge/?version=develop)](https://raja.readthedocs.io/en/develop/?badge=develop)

The RAJA Performance Suite is a companion project to the [RAJA] C++ performance
portability abstraction library. The Performance Suite designed to eplore
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

Please see the [RAJA] project for more information about RAJA.

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

Copyrights and patents in the RAJAPerf project are retained by contributors.
No copyright assignment is required to contribute to RAJAPerf.

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-738930`  `OCEC-17-159`

For release details and restrictions, please see the information in the
following:
- [RELEASE](./RELEASE)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)

SPDX Usage
-----------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
------------------

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

