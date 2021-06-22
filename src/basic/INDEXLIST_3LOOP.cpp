//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST_3LOOP.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


INDEXLIST_3LOOP::INDEXLIST_3LOOP(const RunParams& params)
  : KernelBase(rajaperf::Basic_INDEXLIST_3LOOP, params)
{
  setDefaultSize(1000000);
  setDefaultReps(100);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

#if defined(_OPENMP) && _OPENMP >= 201811
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
#endif
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

INDEXLIST_3LOOP::~INDEXLIST_3LOOP()
{
}

void INDEXLIST_3LOOP::setUp(VariantID vid)
{
  allocAndInitDataRandSign(m_x, getRunSize(), vid);
  allocAndInitData(m_list, getRunSize(), vid);
  m_len = -1;
}

void INDEXLIST_3LOOP::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_list, getRunSize());
  checksum[vid] += Checksum_type(m_len);
}

void INDEXLIST_3LOOP::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_list);
}

} // end namespace basic
} // end namespace rajaperf
