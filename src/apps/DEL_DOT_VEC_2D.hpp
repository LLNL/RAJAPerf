/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel DEL_DOT_VEC_2D.
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


#ifndef RAJAPerf_Apps_DEL_DOT_VEC_2D_HPP
#define RAJAPerf_Apps_DEL_DOT_VEC_2D_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{
struct ADomain;

class DEL_DOT_VEC_2D : public KernelBase
{
public:

  DEL_DOT_VEC_2D(const RunParams& params);

  ~DEL_DOT_VEC_2D();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_xdot;
  Real_ptr m_ydot;
  Real_ptr m_div;

  Real_type m_ptiny;
  Real_type m_half;

  ADomain* m_domain;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
