//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D Mass matrix via partial assembly
///
/// TODO add description
///
///

#ifndef RAJAPerf_Apps_MASS3DPA_HPP
#define RAJAPerf_Apps_MASS3DPA_HPP

#define MASS3DPA_DATA_SETUP \
Real_ptr B = m_B; \
Real_ptr Bt = m_Bt; \
Real_ptr D = m_D; \
Real_ptr X = m_X; \
Real_ptr Y = m_Y; \
Index_type NE = m_NE; 

#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

  using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
#if defined(RAJA_ENABLE_CUDA)
                                                 ,
                                                 RAJA::expt::cuda_launch_t<true>
#endif
#if defined(RAJA_ENABLE_HIP)
                                                 ,
                                                 RAJA::expt::hip_launch_t<true>
#endif
                                                 >;

#if defined(RAJA_ENABLE_OPENMP)
using omp_launch_policy =
  RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t
#if defined(RAJA_ENABLE_CUDA)
                           ,
                           RAJA::expt::cuda_launch_t<true>
#endif
#if defined(RAJA_ENABLE_HIP)
                           ,
                           RAJA::expt::hip_launch_t<true>
#endif
                           >;
#endif

  using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_CUDA)
  using gpu_block_x_policy = RAJA::cuda_block_x_loop;
  using gpu_thread_x_policy = RAJA::cuda_thread_x_loop;
  using gpu_thread_y_policy = RAJA::cuda_thread_y_loop;
#endif

#if defined(RAJA_ENABLE_HIP)
  using gpu_block_x_policy = RAJA::hip_block_x_loop;
  using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
  using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
#endif

  using teams_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                         ,
                                       gpu_block_x_policy
#endif
                                         >;

  using threads_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                         gpu_thread_x_policy
#endif
                                           >;

  using threads_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                         gpu_thread_y_policy
#endif
                                           >;
#if defined(RAJA_ENABLE_OPENMP)
  using omp_teams = RAJA::expt::LoopPolicy<RAJA::omp_for_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                       gpu_block_x_policy
#endif
                                           >;
#endif

namespace rajaperf 
{
class RunParams;

namespace apps
{

//
// These index value types cannot be defined in function scope for
// RAJA CUDA variant to work.
//

class MASS3DPA : public KernelBase
{
public:

  MASS3DPA(const RunParams& params);

  ~MASS3DPA();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  //void runOpenMPTargetVariant(VariantID vid);

private:

  int m_Q1D = 5; 
  int m_D1D = 4;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_NE;
  Index_type m_NE_default;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
