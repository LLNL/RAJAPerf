###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS
    common
    apps
    apps-kokkos
    basic
    basic-kokkos
    lcals
    lcals-kokkos
    polybench
    polybench-kokkos
    stream
    stream-kokkos
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
