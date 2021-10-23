//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


TRIDIAG_ELIM::TRIDIAG_ELIM(const RunParams& params)
  : KernelBase(rajaperf::Lcals_TRIDIAG_ELIM, params)
{
  setDefaultGPUBlockSize( gpu_block_size::get_first(gpu_block_sizes_type()) );
  setActualGPUBlockSize( (params.getGPUBlockSize() > 0) ? params.getGPUBlockSize()
                                                        : getDefaultGPUBlockSize() );

  setDefaultProblemSize(1000000);
  setDefaultReps(1000);

  setActualProblemSize( getTargetProblemSize() );

  m_N = getActualProblemSize();

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type ) + 3*sizeof(Real_type )) * (m_N-1) );
  setFLOPsPerRep(2 * (getActualProblemSize()-1));

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

TRIDIAG_ELIM::~TRIDIAG_ELIM()
{
}

void TRIDIAG_ELIM::setUp(VariantID vid)
{
  allocAndInitDataConst(m_xout, m_N, 0.0, vid);
  allocAndInitData(m_xin, m_N, vid);
  allocAndInitData(m_y, m_N, vid);
  allocAndInitData(m_z, m_N, vid);
}

void TRIDIAG_ELIM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_xout, getActualProblemSize());
}

void TRIDIAG_ELIM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_xout);
  deallocData(m_xin);
  deallocData(m_y);
  deallocData(m_z);
}

bool TRIDIAG_ELIM::isGPUBlockSizeSupported() const
{
  return gpu_block_size::invoke_or(
      gpu_block_size::Equals(getActualGPUBlockSize()), gpu_block_sizes_type());
}

} // end namespace lcals
} // end namespace rajaperf
