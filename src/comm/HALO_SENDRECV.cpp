//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_SENDRECV.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

namespace rajaperf
{
namespace comm
{

HALO_SENDRECV::HALO_SENDRECV(const RunParams& params)
  : HALO_base(rajaperf::Comm_HALO_SENDRECV, params)
{
  m_mpi_size = params.getMPISize();
  m_my_mpi_rank = params.getMPIRank();
  m_mpi_dims = params.getMPI3DDivision();

  setDefaultReps(200);

  m_num_vars = params.getHaloNumVars();
  m_var_size = m_grid_plus_halo_size ;

  setItsPerRep( m_num_vars * (m_var_size - getActualProblemSize()) );
  setKernelsPerRep( 0 );
  setBytesReadPerRep( 1*sizeof(Real_type) * getItsPerRep() ); // send
  setBytesWrittenPerRep( 1*sizeof(Real_type) * getItsPerRep() ); // recv
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(0);

  setComplexity(Complexity::N_to_the_two_thirds);

  setUsesFeature(Forall);
  setUsesFeature(MPI);

  if (params.validMPI3DDivision()) {
    setVariantDefined( Base_Seq );

    setVariantDefined( Base_OpenMP );

    setVariantDefined( Base_OpenMPTarget );

    setVariantDefined( Base_CUDA );

    setVariantDefined( Base_HIP );
  }
}

HALO_SENDRECV::~HALO_SENDRECV()
{
}

void HALO_SENDRECV::setUp(VariantID vid, size_t tune_idx)
{
  setUp_base(m_my_mpi_rank, m_mpi_dims.data(), vid, tune_idx);

  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy);

  m_send_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[l];
    if (separate_buffers) {
      allocAndInitData(DataSpace::Host, m_send_buffers[l], buffer_len);
    } else {
      allocAndInitData(getMPIDataSpace(vid), m_send_buffers[l], buffer_len);
    }
  }

  m_recv_buffers.resize(s_num_neighbors, nullptr);
  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_unpack_index_list_lengths[l];
    if (separate_buffers) {
      allocAndInitData(DataSpace::Host, m_recv_buffers[l], buffer_len);
    } else {
      allocAndInitData(getMPIDataSpace(vid), m_recv_buffers[l], buffer_len);
    }
  }
}

void HALO_SENDRECV::updateChecksum(VariantID vid, size_t tune_idx)
{
  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy);

  for (Index_type l = 0; l < s_num_neighbors; ++l) {
    Index_type buffer_len = m_num_vars * m_unpack_index_list_lengths[l];
    if (separate_buffers) {
      checksum[vid][tune_idx] += calcChecksum(DataSpace::Host, m_recv_buffers[l], buffer_len, vid);
    } else {
      checksum[vid][tune_idx] += calcChecksum(getMPIDataSpace(vid), m_recv_buffers[l], buffer_len, vid);
    }
  }
}

void HALO_SENDRECV::tearDown(VariantID vid, size_t tune_idx)
{
  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy);

  for (int l = 0; l < s_num_neighbors; ++l) {
    if (separate_buffers) {
      deallocData(DataSpace::Host, m_recv_buffers[l]);
    } else {
      deallocData(getMPIDataSpace(vid), m_recv_buffers[l]);
    }
  }
  m_recv_buffers.clear();

  for (int l = 0; l < s_num_neighbors; ++l) {
    if (separate_buffers) {
      deallocData(DataSpace::Host, m_send_buffers[l]);
    } else {
      deallocData(getMPIDataSpace(vid), m_send_buffers[l]);
    }
  }
  m_send_buffers.clear();

  tearDown_base(vid, tune_idx);
}

} // end namespace comm
} // end namespace rajaperf

#endif
