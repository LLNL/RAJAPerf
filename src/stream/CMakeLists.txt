###############################################################################
# Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME stream
  SOURCES ADD.cpp
          ADD-Seq.cpp 
          ADD-Hip.cpp
          ADD-Cuda.cpp
          ADD-OMP.cpp
          ADD-OMPTarget.cpp
	  ADD-Sycl.cpp
          COPY.cpp 
          COPY-Seq.cpp 
          COPY-Hip.cpp
          COPY-Cuda.cpp
          COPY-OMP.cpp
          COPY-OMPTarget.cpp
	  COPY-Sycl.cpp
          DOT.cpp 
          DOT-Seq.cpp 
          DOT-Hip.cpp 
          DOT-Cuda.cpp 
          DOT-OMP.cpp 
          DOT-OMPTarget.cpp 
	  DOT-Sycl.cpp 
          MUL.cpp 
          MUL-Seq.cpp 
          MUL-Hip.cpp 
          MUL-Cuda.cpp 
          MUL-OMP.cpp 
          MUL-OMPTarget.cpp 
	  MUL-Sycl.cpp 
          TRIAD.cpp 
          TRIAD-Seq.cpp 
          TRIAD-Hip.cpp 
          TRIAD-Cuda.cpp 
          TRIAD-OMPTarget.cpp 
          TRIAD-OMP.cpp 
	  TRIAD-Sycl.cpp 
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
