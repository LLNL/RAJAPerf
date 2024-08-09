//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_EXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

namespace rajaperf
{
namespace comm
{

HALO_EXCHANGE::HALO_EXCHANGE(const RunParams& params)
  : HALO_base(rajaperf::Comm_HALO_EXCHANGE, params)
{
  m_mpi_size = params.getMPISize();
  m_my_mpi_rank = params.getMPIRank();
  m_mpi_dims = params.getMPI3DDivision();

  setDefaultReps(200);

  m_num_vars = params.getHaloNumVars();
  m_var_size = m_grid_plus_halo_size ;

  setItsPerRep( m_num_vars * (m_var_size - getActualProblemSize()) );
  setKernelsPerRep( 2 * s_num_neighbors * m_num_vars );
  setBytesReadPerRep( 1*sizeof(Int_type) * getItsPerRep() +   // pack
                      1*sizeof(Real_type) * getItsPerRep() +  // pack

                      1*sizeof(Real_type) * getItsPerRep() +  // send

                      1*sizeof(Int_type) * getItsPerRep() +   // unpack
                      1*sizeof(Real_type) * getItsPerRep() ); // unpack
  setBytesWrittenPerRep( 1*sizeof(Real_type) * getItsPerRep() +  // pack

                         1*sizeof(Real_type) * getItsPerRep() +  // recv

                         1*sizeof(Real_type) * getItsPerRep() ); // unpack
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(0);

  setComplexity(Complexity::N_to_the_two_thirds);

  setUsesFeature(Forall);
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

HALO_EXCHANGE::~HALO_EXCHANGE()
{
}

void HALO_EXCHANGE::setUp(VariantID vid, size_t tune_idx)
{
  setUp_base(m_my_mpi_rank, m_mpi_dims.data(), vid, tune_idx);

  m_vars.resize(m_num_vars, nullptr);
  for (Index_type v = 0; v < m_num_vars; ++v) {
    allocAndInitData(m_vars[v], m_var_size, vid);
    auto reset_var = scopedMoveData(m_vars[v], m_var_size, vid);

    Real_ptr var = m_vars[v];

    for (Index_type i = 0; i < m_var_size; i++) {
      var[i] = i + v;
    }
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

void HALO_EXCHANGE::updateChecksum(VariantID vid, size_t tune_idx)
{
  for (Real_ptr var : m_vars) {
    checksum[vid][tune_idx] += calcChecksum(var, m_var_size, vid);
  }
}

void HALO_EXCHANGE::tearDown(VariantID vid, size_t tune_idx)
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

  for (int v = 0; v < m_num_vars; ++v) {
    deallocData(m_vars[v], vid);
  }
  m_vars.clear();

  tearDown_base(vid, tune_idx);
}

} // end namespace comm
} // end namespace rajaperf

#endif
