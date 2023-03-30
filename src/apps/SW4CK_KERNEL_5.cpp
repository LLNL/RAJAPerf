//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SW4CK_KERNEL_5.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>


namespace rajaperf
{
namespace apps
{


SW4CK_KERNEL_5::SW4CK_KERNEL_5(const RunParams& params)
  : KernelBase(rajaperf::Apps_SW4CK_KERNEL_5, params)
{
  setDefaultProblemSize(100*100*100);  // See rzmax in ADomain struct
  setDefaultReps(100);

  Index_type rzmax = std::cbrt(getTargetProblemSize())+1;
  //m_domain = new ADomain(rzmax, /* ndims = */ 3);

  //m_array_length = m_domain->nnalls;

  //setActualProblemSize( m_domain->lpz+1 - m_domain->fpz );

  //setItsPerRep( m_domain->lpz+1 - m_domain->fpz );
  setKernelsPerRep(1);
  // touched data size, not actual number of stores and loads
  //  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) * getItsPerRep() +
  //(0*sizeof(Real_type) + 3*sizeof(Real_type)) * (getItsPerRep() + 1+m_domain->jp+m_domain->kp) );
  
  //setFLOPsPerRep(72 * (m_domain->lpz+1 - m_domain->fpz));

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Teams);

  //Goal is to get the following three variants right first
  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  /*
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
  */
}

SW4CK_KERNEL_5::~SW4CK_KERNEL_5()
{
  //  delete m_domain;
}

void SW4CK_KERNEL_5::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

}

void SW4CK_KERNEL_5::updateChecksum(VariantID vid, size_t tune_idx)
{
  //checksum[vid][tune_idx] += calcChecksum(m_vol, m_array_length, checksum_scale_factor );
}

void SW4CK_KERNEL_5::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

}

} // end namespace apps
} // end namespace rajaperf
