//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NESTED_INIT kernel reference implementation:
///
/// for (Index_type k = 0; k < nk; ++k ) {
///   for (Index_type j = 0; j < nj; ++j ) {
///     for (Index_type i = 0; i < ni; ++i ) {
///       array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
///     }
///   }
/// }
///

#ifndef RAJAPerf_Basic_NESTED_INIT_HPP
#define RAJAPerf_Basic_NESTED_INIT_HPP


#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class NESTED_INIT : public KernelBase
{
public:

  NESTED_INIT(const RunParams& params);

  ~NESTED_INIT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_array_length;

  Real_ptr m_array;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nk_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
