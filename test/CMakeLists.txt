###############################################################################
# Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
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
    algorithm
    comm)
list(APPEND RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS})

if (RAJA_PERFSUITE_ENABLE_MPI)
  raja_add_test(
    NAME test-raja-perf-suite
    SOURCES test-raja-perf-suite.cpp
    DEPENDS_ON ${RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS}
    NUM_MPI_TASKS ${RAJA_PERFSUITE_NUM_MPI_TASKS}
    )
else()
  raja_add_test(
    NAME test-raja-perf-suite
    SOURCES test-raja-perf-suite.cpp
    DEPENDS_ON ${RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS}
    )
endif()

target_include_directories(test-raja-perf-suite.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
