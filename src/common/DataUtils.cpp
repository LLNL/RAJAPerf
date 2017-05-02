/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file containing routines for data management.
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

#include "DataUtils.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"


namespace rajaperf
{

static int data_init_count = 0;

/*
 * Reset counter for data initialization.
 */
void resetDataInitCount()
{
  data_init_count = 0;
}


/*
 * Allocate and initialize aligned data arrays.
 */
void allocAndInitData(Real_ptr& ptr, int len, VariantID vid)
{
  ptr = 
    RAJA::allocate_aligned_type<Real_type>(RAJA::DATA_ALIGN, 
                                           len*sizeof(Real_type));
  initData(ptr, len, vid);
}

void allocAndInitData(Complex_ptr& ptr, int len, VariantID vid)
{
  // Should we do this differently for alignment??
  ptr = new Complex_type[len];
  initData(ptr, len, vid);
}

#if defined(RAJA_ENABLE_CUDA)
/*
 * Allocate and initialize CUDA device data arrays.
 */
void allocAndInitCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len) 
{
  cudaErrchk( cudaMalloc( (void**)&dptr, len * sizeof(Real_type) ) );

  cudaErrchk( cudaMemcpy( dptr, hptr, len * sizeof(Real_type), 
              cudaMemcpyHostToDevice ) );

  data_init_count++;
}
#endif

#if defined(RAJA_ENABLE_CUDA)
/*
 * Copy CUDA device data arrays back to host.
 */
void getCudaDeviceData(Real_ptr& hptr, const Real_ptr dptr, int len)
{
  cudaErrchk( cudaMemcpy( hptr, dptr, len * sizeof(Real_type), 
                          cudaMemcpyDeviceToHost ) );
}
#endif


/*
 * Free data arrays.
 */
void deallocData(Real_ptr& ptr)
{ 
  if (ptr) {
    RAJA::free_aligned(ptr);
    ptr = 0;
  }
}

void deallocData(Complex_ptr& ptr)
{
  if (ptr) { 
    delete [] ptr;
    ptr = 0;
  }
}

#if defined(RAJA_ENABLE_CUDA)
/*
 * Free CUDA device data arrays.
 */
void deallocCudaDeviceData(Real_ptr& dptr) 
{
  cudaErrchk( cudaFree( dptr ) );
  dptr = 0; 
}
#endif

/*
 * Initialize data arrays.
 */
void initData(Real_ptr& ptr, int len, VariantID vid) 
{
  (void) vid;

  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

  if ( vid == Baseline_OpenMP || 
       vid == RAJALike_OpenMP || 
       vid == RAJA_OpenMP ) {
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int i = 0; i < len; ++i) { 
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    };
  } else {
    for (int i = 0; i < len; ++i) {
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    } 
  }

  data_init_count++;
}

void initData(Complex_ptr& ptr, int len, VariantID vid)
{
  (void) vid;

  Complex_type factor = ( data_init_count % 2 ?  Complex_type(0.1,0.2) :
                                                 Complex_type(0.2,0.3) );

  if ( vid == Baseline_OpenMP ||
       vid == RAJALike_OpenMP ||
       vid == RAJA_OpenMP ) {
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int i = 0; i < len; ++i) { 
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    };
  } else {
    for (int i = 0; i < len; ++i) {
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    }
  }

  data_init_count++;
}



/*
 * Initialize scalar data.
 */
void initData(Real_type& d, VariantID vid)
{
  (void) vid;

  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );
  d = factor*1.1/1.12345;

  data_init_count++;
}


/*
 * Calculate and return checksum for data arrays.
 */
long double calcChecksum(const Real_ptr ptr, int len, 
                         Real_type scale_factor)
{
  long double tchk = 0.0;
  for (Index_type j = 0; j < len; ++j) {
    tchk += (j+1)*ptr[j]*scale_factor;
#if 0 // RDH DEBUG
    if ( (j % 100) == 0 ) {
      std::cout << "j : tchk = " << j << " : " << tchk << std::endl;
    }
#endif
  }
  return tchk;
}

long double calcChecksum(const Complex_ptr ptr, int len,
                         Real_type scale_factor)
{
  long double tchk = 0.0;
  for (Index_type j = 0; j < len; ++j) {
    tchk += (j+1)*(real(ptr[j])+imag(ptr[j]))*scale_factor;
#if 0 // RDH DEBUG
    if ( (j % 100) == 0 ) {
      std::cout << "j : tchk = " << j << " : " << tchk << std::endl;
    }
#endif
  }
  return tchk;
}


}  // closing brace for rajaperf namespace
