//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


INDEXLIST::INDEXLIST(const RunParams& params)
  : KernelBase(rajaperf::Basic_INDEXLIST, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(100);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Index_type) + 1*sizeof(Index_type)) +
                  (1*sizeof(Int_type) + 0*sizeof(Int_type)) * getActualProblemSize() / 2 + // about 50% output
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(0);

  setUsesFeature(Forall);
  setUsesFeature(Scan);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

#if defined(_OPENMP) && _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP_SCAN)
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
#endif

  setVariantDefined( Base_CUDA );
}

INDEXLIST::~INDEXLIST()
{
}

void INDEXLIST::setUp(VariantID vid)
{
  allocAndInitDataRandSign(m_x, getActualProblemSize(), vid);
  allocAndInitData(m_list, getActualProblemSize(), vid);
  m_len = -1;
}

void INDEXLIST::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_list, getActualProblemSize());
  checksum[vid] += Checksum_type(m_len);
}

void INDEXLIST::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_list);
}

} // end namespace basic
} // end namespace rajaperf