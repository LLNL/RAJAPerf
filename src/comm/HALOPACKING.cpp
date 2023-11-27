//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOPACKING.hpp"

#include "RAJA/RAJA.hpp"

#include <cmath>

namespace rajaperf
{
namespace comm
{

HALOPACKING::HALOPACKING(const RunParams& params)
  : HALO_base(rajaperf::Comm_HALOPACKING, params)
{
  setDefaultReps(200);

  m_num_vars = s_num_vars_default;
  m_var_size = m_grid_plus_halo_size ;

  setItsPerRep( m_num_vars * (m_var_size - getActualProblemSize()) );
  setKernelsPerRep( 2 * s_num_neighbors * m_num_vars );
  setBytesPerRep( (0*sizeof(Int_type)  + 1*sizeof(Int_type) ) * getItsPerRep() +  // pack
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getItsPerRep() +  // pack
                  (0*sizeof(Int_type)  + 1*sizeof(Int_type) ) * getItsPerRep() +  // unpack
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getItsPerRep() ); // unpack
  setFLOPsPerRep(0);

  setUsesFeature(Forall);

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

HALOPACKING::~HALOPACKING()
{
}

void HALOPACKING::setUp(VariantID vid, size_t tune_idx)
{
  int my_mpi_rank = 0;
  const int mpi_dims[3] = {1,1,1};
  setUp_base(my_mpi_rank, mpi_dims, vid, tune_idx);

  m_vars.resize(m_num_vars, nullptr);
  for (Index_type v = 0; v < m_num_vars; ++v) {
    allocAndInitData(m_vars[v], m_var_size, vid);
    auto reset_var = scopedMoveData(m_vars[v], m_var_size, vid);

    Real_ptr var = m_vars[v];

    for (Index_type i = 0; i < m_var_size; i++) {
      var[i] = i + v;
    }
  }

  m_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[l];
    allocAndInitData(m_buffers[l], buffer_len, vid);
  }
}

void HALOPACKING::updateChecksum(VariantID vid, size_t tune_idx)
{
  for (Real_ptr var : m_vars) {
    checksum[vid][tune_idx] += calcChecksum(var, m_var_size, vid);
  }
}

void HALOPACKING::tearDown(VariantID vid, size_t tune_idx)
{
  for (int l = 0; l < s_num_neighbors; ++l) {
    deallocData(m_buffers[l], vid);
  }
  m_buffers.clear();

  for (int v = 0; v < m_num_vars; ++v) {
    deallocData(m_vars[v], vid);
  }
  m_vars.clear();

  tearDown_base(vid, tune_idx);
}

} // end namespace comm
} // end namespace rajaperf
