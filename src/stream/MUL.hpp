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
/// MUL kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   b[i] = alpha * c[i] ;
/// }
///

#ifndef RAJAPerf_Stream_MUL_HPP
#define RAJAPerf_Stream_MUL_HPP


#define MUL_BODY  \
  b[i] = alpha * c[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class MUL : public KernelBase
{
public:

  MUL(const RunParams& params);

  ~MUL();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
