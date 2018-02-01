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
/// ADD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   c[i] = a[i] + b[i];
/// }
///

#ifndef RAJAPerf_Stream_ADD_HPP
#define RAJAPerf_Stream_ADD_HPP


#define ADD_BODY  \
  c[i] = a[i] + b[i]; 


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class ADD : public KernelBase
{
public:

  ADD(const RunParams& params);

  ~ADD();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;

};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
