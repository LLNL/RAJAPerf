//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#include <cmath>

namespace rajaperf
{
namespace apps
{

HALOEXCHANGE_FUSED::HALOEXCHANGE_FUSED(const RunParams& params)
  : HALOEXCHANGE_base(rajaperf::Apps_HALOEXCHANGE_FUSED, params)
{
  setUsesFeature(Workgroup);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

HALOEXCHANGE_FUSED::~HALOEXCHANGE_FUSED()
{
}

void HALOEXCHANGE_FUSED::setUp(VariantID vid, size_t tune_idx)
{
  int my_mpi_rank = 0;
  const int mpi_dims[3] = {1,1,1};
  setUp_base(my_mpi_rank, mpi_dims, vid, tune_idx);

  m_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[l];
    allocAndInitData(m_buffers[l], buffer_len, vid);
  }
}

void HALOEXCHANGE_FUSED::tearDown(VariantID vid, size_t tune_idx)
{
  for (int l = 0; l < s_num_neighbors; ++l) {
    deallocData(m_buffers[l], vid);
  }
  m_buffers.clear();

  tearDown_base(vid, tune_idx);
}

} // end namespace apps
} // end namespace rajaperf
