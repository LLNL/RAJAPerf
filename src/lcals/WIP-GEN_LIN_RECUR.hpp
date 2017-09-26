//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_GEN_LIN_RECUR_HPP
#define RAJAPerf_Basic_GEN_LIN_RECUR_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class GEN_LIN_RECUR : public KernelBase
{
public:

  GEN_LIN_RECUR(const RunParams& params);

  ~GEN_LIN_RECUR();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_b5;
  Real_ptr m_sa;
  Real_ptr m_sb;

  Real_type m_stb5;

  Index_type m_kb5i;
  Index_type m_len;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
