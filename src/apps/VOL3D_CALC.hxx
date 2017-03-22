/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel VOL3D_CALC.
 *
 ******************************************************************************
 */

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
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Apps_VOL3D_CALC_HXX
#define RAJAPerf_Apps_VOL3D_CALC_HXX

#include "common/KernelBase.hxx"

#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class VOL3D_CALC : public KernelBase
{
public:

  VOL3D_CALC(const RunParams& params);

  ~VOL3D_CALC();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  RAJA::Real_ptr m_x;
  RAJA::Real_ptr m_y;
  RAJA::Real_ptr m_z;
  RAJA::Real_ptr m_vol;

  RAJA::Real_type m_vnormq;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
