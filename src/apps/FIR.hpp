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
/// FIR kernel reference implementation:
///
/// #define FIR_COEFFLEN (16)
///
/// Real_type coeff[FIR_COEFFLEN] = { 3.0, -1.0, -1.0, -1.0,
///                                  -1.0, 3.0, -1.0, -1.0,
///                                  -1.0, -1.0, 3.0, -1.0,
///                                  -1.0, -1.0, -1.0, 3.0 };
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type sum = 0.0;
///   for (Index_type j = 0; j < coefflen; ++j ) {
///     sum += coeff[j]*in[i+j];
///   }
///   out[i] = sum;
/// }
///

#ifndef RAJAPerf_Apps_FIR_HPP
#define RAJAPerf_Apps_FIR_HPP


#define FIR_COEFFLEN (16)

#define FIR_COEFF \
  Real_type coeff_array[FIR_COEFFLEN] = { 3.0, -1.0, -1.0, -1.0, \
                                         -1.0, 3.0, -1.0, -1.0, \
                                         -1.0, -1.0, 3.0, -1.0, \
                                         -1.0, -1.0, -1.0, 3.0 };

#define FIR_BODY \
  Real_type sum = 0.0; \
\
  for (Index_type j = 0; j < coefflen; ++j ) { \
    sum += coeff[j]*in[i+j]; \
  } \
  out[i] = sum;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class FIR : public KernelBase
{
public:

  FIR(const RunParams& params);

  ~FIR();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_in;
  Real_ptr m_out;

  Index_type m_coefflen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
