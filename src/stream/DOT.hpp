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
/// DOT kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   dot += a[i] * b[i];
/// }
///

#ifndef RAJAPerf_Stream_DOT_HPP
#define RAJAPerf_Stream_DOT_HPP


#define DOT_BODY  \
  dot += a[i] * b[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class DOT : public KernelBase
{
public:

  DOT(const RunParams& params);

  ~DOT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_type m_dot;
  Real_type m_dot_init;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
