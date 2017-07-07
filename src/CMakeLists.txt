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
# For more information, please see the file LICENSE in the top-level directory.
#
###############################################################################

include_directories(.)

add_subdirectory(common)
add_subdirectory(apps)
add_subdirectory(basic)
#add_subdirectory(livloops)
#add_subdirectory(polybench)
add_subdirectory(stream)

set(rajaperf_executable_depends
    common
    apps
    basic
    stream
    RAJA)
if(ENABLE_OPENMP)
  list(APPEND rajaperf_executable_depends openmp) 
endif()

blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${rajaperf_executable_depends}
  )

#blt_add_executable(
#  NAME raja-perf.exe
#  SOURCES RAJAPerfSuiteDriver.cpp
#  DEPENDS_ON 
#  common
#  apps
#  basic
###  livloops
###  polybench
#  stream
#  RAJA
#  )
