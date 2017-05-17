/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel VOL3D.
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
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
