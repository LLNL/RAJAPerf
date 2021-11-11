###############################################################################
# Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################
set(RAJAPERF_COMMON_SRC
        Executor.cpp
        KernelBase.cpp
        OutputUtils.cpp
        RunParams.cpp
        RAJAPerfSuite.cpp
        )

if(NOT INFRASTRUCTURE_ONLY)
  #MESSAGE(FATAL_ERROR "TODO ERROR ${INFRASTRUCTURE_ONLY}")
  LIST(APPEND RAJAPERF_COMMON_SRC
          DataUtils.cpp
          PerfsuiteKernelDefinitions.cpp
          )
  endif()
blt_add_library(
  NAME common
  SOURCES ${RAJAPERF_COMMON_SRC}
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
  )


