###############################################################################
# Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_PERFSUITE_ENABLE_DOCUMENTATION)
  if (SPHINX_FOUND)
    blt_add_sphinx_target( raja_docs )
  endif()

  if ( NOT SPHINX_FOUND )
    message(WARNING "RAJA_PERFSUITE_ENABLE_DOCUMENTATION=On, but Sphinx not found. \
      Documentation won't be built.")
  endif ()
endif()
