###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-xxxxxx
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For details about use and distribution, please read raja-perfsuite/LICENSE.
#
###############################################################################

include_directories(.)

add_subdirectory(common)
add_subdirectory(apps)
add_subdirectory(basic)
add_subdirectory(lcals)
add_subdirectory(polybench)
add_subdirectory(stream)

set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    common
    apps
    basic
    lcals
    polybench
    stream)
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )
