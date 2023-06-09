//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MPI_HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#include <cmath>

namespace rajaperf
{
namespace apps
{

MPI_HALOEXCHANGE_FUSED::MPI_HALOEXCHANGE_FUSED(const RunParams& params)
  : HALOEXCHANGE_base(rajaperf::Apps_MPI_HALOEXCHANGE_FUSED, params)
{
  m_mpi_size = params.getMPISize();
  m_my_mpi_rank = params.getMPIRank();
  m_mpi_dims = params.getMPI3DDivision();

  setUsesFeature(Workgroup);
  setUsesFeature(MPI);

  if (params.validMPI3DDivision()) {
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
}

MPI_HALOEXCHANGE_FUSED::~MPI_HALOEXCHANGE_FUSED()
{
}

void MPI_HALOEXCHANGE_FUSED::setUp(VariantID vid, size_t tune_idx)
{
  setUp_base(m_my_mpi_rank, m_mpi_dims.data(), vid, tune_idx);

  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy);

  m_pack_buffers.resize(s_num_neighbors, nullptr);
  m_send_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[l];
    if (separate_buffers) {
      allocAndInitData(getDataSpace(vid), m_pack_buffers[l], buffer_len);
      allocAndInitData(DataSpace::Host, m_send_buffers[l], buffer_len);
    } else {
      allocAndInitData(getMPIDataSpace(vid), m_pack_buffers[l], buffer_len);
      m_send_buffers[l] = m_pack_buffers[l];
    }
  }

  m_unpack_buffers.resize(s_num_neighbors, nullptr);
  m_recv_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_unpack_index_list_lengths[l];
    if (separate_buffers) {
      allocAndInitData(getDataSpace(vid), m_unpack_buffers[l], buffer_len);
      allocAndInitData(DataSpace::Host, m_recv_buffers[l], buffer_len);
    } else {
      allocAndInitData(getMPIDataSpace(vid), m_unpack_buffers[l], buffer_len);
      m_recv_buffers[l] = m_unpack_buffers[l];
    }
  }
}

void MPI_HALOEXCHANGE_FUSED::tearDown(VariantID vid, size_t tune_idx)
{
  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy);

  for (int l = 0; l < s_num_neighbors; ++l) {
    if (separate_buffers) {
      deallocData(DataSpace::Host, m_recv_buffers[l]);
      deallocData(getDataSpace(vid), m_unpack_buffers[l]);
    } else {
      deallocData(getMPIDataSpace(vid), m_unpack_buffers[l]);
    }
  }
  m_recv_buffers.clear();
  m_unpack_buffers.clear();

  for (int l = 0; l < s_num_neighbors; ++l) {
    if (separate_buffers) {
      deallocData(DataSpace::Host, m_send_buffers[l]);
      deallocData(getDataSpace(vid), m_pack_buffers[l]);
    } else {
      deallocData(getMPIDataSpace(vid), m_pack_buffers[l]);
    }
  }
  m_send_buffers.clear();
  m_pack_buffers.clear();

  tearDown_base(vid, tune_idx);
}

} // end namespace apps
} // end namespace rajaperf

#endif
