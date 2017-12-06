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

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/DataUtils.hpp"

#include <iostream>

//#define ADD_USE_MAP
#define ADD_USE_TARGETALLOC
//#define ADD_USE_UM

#if defined(ADD_USE_UM)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace rajaperf
{
namespace stream
{

#define NUMTEAMS 128

void ADD::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_OpenMPTarget ) {

#if defined(ADD_USE_MAP)
    ADD_DATA;

    int n = getRunSize();
    #pragma omp target enter data map(to:a[0:n],b[0:n],c[0:n])

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ADD_BODY;
      }

    }
    stopTimer();

    #pragma omp target exit data map(from:c[0:n]) map(delete:a[0:n],b[0:n])

#elif defined(ADD_USE_TARGETALLOC)

    int h = omp_get_initial_device();
    int d = omp_get_default_device();

    Real_ptr a;
    Real_ptr b;
    Real_ptr c;

    a = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );
    b = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );
    c = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );

    omp_target_memcpy( a, m_a, iend * sizeof(Real_type), 0, 0, d, h );
    omp_target_memcpy( b, m_b, iend * sizeof(Real_type), 0, 0, d, h );
    omp_target_memcpy( c, m_c, iend * sizeof(Real_type), 0, 0, d, h );

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(a, b, c) device( d )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ADD_BODY;
      }

    }
    stopTimer();

    omp_target_memcpy( m_c, c, iend * sizeof(Real_type), 0, 0, h, d );

    omp_target_free( a, d );
    omp_target_free( b, d );
    omp_target_free( c, d );

#elif defined(ADD_USE_UM)
    Real_ptr a;
    Real_ptr b;
    Real_ptr c;

    cudaMallocManaged( (void**)&a, iend*sizeof(Real_type), cudaMemAttachGlobal);
    cudaMallocManaged( (void**)&b, iend*sizeof(Real_type), cudaMemAttachGlobal);
    cudaMallocManaged( (void**)&c, iend*sizeof(Real_type), cudaMemAttachGlobal);
    cudaDeviceSynchronize();

    for (int i = 0; i < iend; ++i) {
      a[i] = m_a[i]; 
      b[i] = m_b[i]; 
      c[i] = m_c[i]; 
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//    #pragma omp target is_device_ptr(a, b, c) device( d )
      #pragma omp teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ADD_BODY;
      }

    }
    stopTimer();

    cudaDeviceSynchronize();
    for (int i = 0; i < iend; ++i) {
      m_c[i] = c[i];
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

#else
#error ADD -- NO memory model defined!
#endif 

  } else if ( vid == RAJA_OpenMPTarget ) {

#if defined(ADD_USE_MAP)
    ADD_DATA;

    int n = getRunSize();
    #pragma omp target enter data map(to:a[0:n],b[0:n],c[0:n])

    startTimer();
    #pragma omp target data use_device_ptr(a,b,c)
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

    #pragma omp target exit data map(from:c[0:n]) map(delete:a[0:n],b[0:n])

#elif defined(ADD_USE_TARGETALLOC)

    int h = omp_get_initial_device();
    int d = omp_get_default_device();

    Real_ptr a;
    Real_ptr b;
    Real_ptr c;

    a = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );
    b = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );
    c = static_cast<Real_ptr>( omp_target_alloc(iend*sizeof(Real_type), d) );

    omp_target_memcpy( a, m_a, iend * sizeof(Real_type), 0, 0, d, h );
    omp_target_memcpy( b, m_b, iend * sizeof(Real_type), 0, 0, d, h );
    omp_target_memcpy( c, m_c, iend * sizeof(Real_type), 0, 0, d, h );

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//      #pragma omp target data use_device_ptr(a,b,c)
      RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

    omp_target_memcpy( m_c, c, iend * sizeof(Real_type), 0, 0, h, d );

    omp_target_free( a, d );
    omp_target_free( b, d );
    omp_target_free( c, d );

#elif defined(ADD_USE_UM)

#else
#error ADD -- NO memory model defined!
#endif 

  } else {
     std::cout << "\n  ADD : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

