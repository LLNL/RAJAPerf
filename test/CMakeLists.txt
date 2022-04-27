###############################################################################
# Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS
    common
    apps
    basic
    lcals
    polybench
    stream
    algorithm)
list(APPEND RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS})
 
raja_add_test(
  NAME test-raja-perf-suite
  SOURCES test-raja-perf-suite.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS}
  )

target_include_directories(test-raja-perf-suite.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
