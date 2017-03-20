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

#include <iostream>

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
 * Allocate and initialize aligned data array.
 */
void allocAndInitAligned(RAJA::Real_ptr& ptr, int len, VariantID vid)
{
  ptr = 
    RAJA::allocate_aligned_type<RAJA::Real_type>(RAJA::DATA_ALIGN, 
                                                 len*sizeof(RAJA::Real_type));
  initData(ptr, len, vid);
}

/*
 * Free aligned data array.
 */
void freeAligned(RAJA::Real_ptr& ptr)
{
  RAJA::free_aligned(ptr);
  ptr = 0;
}


/*
 * Initialize data array.
 */
void initData(RAJA::Real_ptr& ptr, int len, VariantID vid) 
{
  (void) vid;

  RAJA::Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

  if ( vid == Baseline_OpenMP || 
       vid == RAJALike_OpenMP || 
       vid == RAJA_OpenMP ) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, len, [=](RAJA::Index_type i) {
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
void initData(RAJA::Real_type& d, VariantID vid)
{
  (void) vid;

  RAJA::Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );
  d = factor*1.1/1.12345;

  data_init_count++;
}


/*
 * Calculate and return checksum for data array.
 */
long double calcChecksum(const RAJA::Real_ptr ptr, int len, 
                         RAJA::Real_type scale_factor)
{
  long double tchk = 0.0;
  for (RAJA::Index_type j = 0; j < len; ++j) {
    tchk += (j+1)*ptr[j]*scale_factor;
#if 0 // RDH DEBUG
    if ( (j % 100) == 0 ) {
      std::cout << "j : tchk = " << j << " : " << tchk << std::endl;
    }
#endif
  }
  return tchk;
}


}  // closing brace for rajaperf namespace
