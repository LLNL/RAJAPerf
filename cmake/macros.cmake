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

macro(raja_add_benchmark name)
  add_test(
    NAME ${name}-test
    COMMAND ${name} --benchmark_format=json --benchmark_output=${name}-benchmark-results.json)
endmacro(raja_add_benchmark)
