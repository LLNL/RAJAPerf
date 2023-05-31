//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MPI_HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#include <cmath>

namespace rajaperf
{
namespace apps
{

MPI_HALOEXCHANGE::MPI_HALOEXCHANGE(const RunParams& params)
  : HALOEXCHANGE_base(rajaperf::Apps_MPI_HALOEXCHANGE, params)
{
  setUsesFeature(Forall);
  setUsesFeature(MPI);

  setVariantDefined( Base_Seq );
  // setVariantDefined( Lambda_Seq );
  // setVariantDefined( RAJA_Seq );

  // setVariantDefined( Base_OpenMP );
  // setVariantDefined( Lambda_OpenMP );
  // setVariantDefined( RAJA_OpenMP );

  // setVariantDefined( Base_OpenMPTarget );
  // setVariantDefined( RAJA_OpenMPTarget );

  // setVariantDefined( Base_CUDA );
  // setVariantDefined( RAJA_CUDA );

  // setVariantDefined( Base_HIP );
  // setVariantDefined( RAJA_HIP );
}

MPI_HALOEXCHANGE::~MPI_HALOEXCHANGE()
{
}

void MPI_HALOEXCHANGE::setUp(VariantID vid, size_t tune_idx)
{
  HALOEXCHANGE_base::setUp(vid, tune_idx);

  m_mpi_ranks.resize(s_num_neighbors, -1);

  MPI_Comm_rank(MPI_COMM_WORLD, &m_my_mpi_rank);

  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    m_mpi_ranks[l] = m_my_mpi_rank; // send and recv to own rank
  }
}

void MPI_HALOEXCHANGE::tearDown(VariantID vid, size_t tune_idx)
{
  m_mpi_ranks.clear();

  HALOEXCHANGE_base::tearDown(vid, tune_idx);
}

} // end namespace apps
} // end namespace rajaperf

#endif
