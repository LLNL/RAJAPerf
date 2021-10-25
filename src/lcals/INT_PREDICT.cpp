//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
  setDefaultGPUBlockSize( gpu_block_size::get_default_or_first(default_gpu_block_size, gpu_block_sizes_type()) );
  setActualGPUBlockSize( (params.getGPUBlockSize() > 0) ? params.getGPUBlockSize()
                                                        : getDefaultGPUBlockSize() );

  setDefaultProblemSize(1000000);
  setDefaultReps(400);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type ) + 10*sizeof(Real_type )) * getActualProblemSize() );
  setFLOPsPerRep(17 * getActualProblemSize());

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

INT_PREDICT::~INT_PREDICT()
{
}

void INT_PREDICT::setUp(VariantID vid)
{
  m_array_length = getActualProblemSize() * 13;
  m_offset = getActualProblemSize();

  m_px_initval = 1.0;
  allocAndInitDataConst(m_px, m_array_length, m_px_initval, vid);

  initData(m_dm22);
  initData(m_dm23);
  initData(m_dm24);
  initData(m_dm25);
  initData(m_dm26);
  initData(m_dm27);
  initData(m_dm28);
  initData(m_c0);
}

void INT_PREDICT::updateChecksum(VariantID vid)
{
  for (Index_type i = 0; i < getActualProblemSize(); ++i) {
    m_px[i] -= m_px_initval;
  }

  checksum[vid] += calcChecksum(m_px, getActualProblemSize());
}

void INT_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
}

bool INT_PREDICT::isGPUBlockSizeSupported() const
{
  return gpu_block_size::invoke_or(
      gpu_block_size::Equals(getActualGPUBlockSize()), gpu_block_sizes_type());
}

} // end namespace lcals
} // end namespace rajaperf
