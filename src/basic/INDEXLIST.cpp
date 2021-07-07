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
  setDefaultSize(1000000);
  setDefaultReps(100);

  setProblemSize( getRunSize() );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Index_type) + 1*sizeof(Index_type)) +
                  (1*sizeof(Int_type) + 0*sizeof(Int_type)) * getRunSize() / 2 + // about 50% output
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunSize() );
  setFLOPsPerRep(0);

  setUsesFeature(Forall);
  setUsesFeature(Scan);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

#if defined(_OPENMP) && _OPENMP >= 201811
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
#endif
}

INDEXLIST::~INDEXLIST()
{
}

void INDEXLIST::setUp(VariantID vid)
{
  allocAndInitDataRandSign(m_x, getRunSize(), vid);
  allocAndInitData(m_list, getRunSize(), vid);
  m_len = -1;
}

void INDEXLIST::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_list, getRunSize());
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
