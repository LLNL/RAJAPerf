//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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


#ifndef RAJAPerf_Apps_VOL3D_HPP
#define RAJAPerf_Apps_VOL3D_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{
struct ADomain;

class VOL3D : public KernelBase
{
public:

  VOL3D(const RunParams& params);

  ~VOL3D();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_ptr m_vol;

  Real_type m_vnormq;

  ADomain* m_domain;
  Index_type m_array_length; 
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
