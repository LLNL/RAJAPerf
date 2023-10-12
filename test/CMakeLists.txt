###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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

list(APPEND RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS} gtest ${CMAKE_THREAD_LIBS_INIT})

set(test_name test-raja-perf-suite)

raja_add_executable(
  NAME ${test_name}.exe
  SOURCES ${test_name}.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_TEST_EXECUTABLE_DEPENDS}
  TEST On)

blt_add_test(
  NAME ${test_name}
  NUM_MPI_TASKS 2
  COMMAND ${TEST_DRIVER} ${test_name})

raja_set_failtest(${test_name})

target_include_directories(test-raja-perf-suite.exe PRIVATE ${PROJECT_SOURCE_DIR}/src)
