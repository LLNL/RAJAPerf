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

#include "DataUtils.hxx"

#include "RAJA/RAJA.hxx"

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
void allocAndInit(Real_ptr& ptr, int len, VariantID vid)
{
  ptr = 
    RAJA::allocate_aligned_type<RAJA::Real_type>(RAJA::DATA_ALIGN, 
                                                 len*sizeof(Real_type));
  initData(ptr, len, vid);
}

void allocAndInit(Complex_ptr& ptr, int len, VariantID vid)
{
  // Should we do this differently for alignment??
  ptr = new Complex_type[len];
  initData(ptr, len, vid);
}


/*
 * Free data arrays.
 */
void dealloc(Real_ptr& ptr)
{
  RAJA::free_aligned(ptr);
  ptr = 0;
}

void dealloc(Complex_ptr& ptr)
{
  delete [] ptr;
  ptr = 0;
}


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
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, len, [=](Index_type i) {
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    });
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
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, len, [=](Index_type i) {
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    });
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
