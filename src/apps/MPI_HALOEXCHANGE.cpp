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
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

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

  MPI_Comm_rank(MPI_COMM_WORLD, &m_my_mpi_rank);

  m_mpi_ranks.resize(s_num_neighbors, -1);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    m_mpi_ranks[l] = m_my_mpi_rank; // send and recv to own rank
  }

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

void MPI_HALOEXCHANGE::tearDown(VariantID vid, size_t tune_idx)
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

  m_mpi_ranks.clear();

  HALOEXCHANGE_base::tearDown(vid, tune_idx);
}

} // end namespace apps
} // end namespace rajaperf

#endif
