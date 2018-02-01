//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
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


#include "DataUtils.hpp"


#include "RAJA/internal/MemUtils_CPU.hpp"

#include <cstdlib>

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
 * Increment counter for data initialization.
 */
void incDataInitCount()
{
  data_init_count++;
}


/*
 * Allocate and initialize aligned integer data arrays.
 */
void allocAndInitData(Int_ptr& ptr, int len, VariantID vid)
{
  // Should we do this differently for alignment?? If so, change dealloc()
  ptr = new Int_type[len];
  initData(ptr, len, vid);
}

/*
 * Allocate and initialize aligned data arrays.
 */
void allocAndInitData(Real_ptr& ptr, int len, VariantID vid )
{
  ptr = 
    RAJA::allocate_aligned_type<Real_type>(RAJA::DATA_ALIGN, 
                                           len*sizeof(Real_type));
  initData(ptr, len, vid);
}

void allocAndInitDataConst(Real_ptr& ptr, int len, Real_type val,
                           VariantID vid)
{
  (void) vid;

  ptr = 
    RAJA::allocate_aligned_type<Real_type>(RAJA::DATA_ALIGN, 
                                           len*sizeof(Real_type));
  initDataConst(ptr, len, val, vid);
}

void allocAndInitDataRandSign(Real_ptr& ptr, int len, VariantID vid)
{
  ptr =
    RAJA::allocate_aligned_type<Real_type>(RAJA::DATA_ALIGN,
                                           len*sizeof(Real_type));
  initDataRandSign(ptr, len, vid);
}

void allocAndInitData(Complex_ptr& ptr, int len, VariantID vid)
{
  // Should we do this differently for alignment?? If so, change dealloc()
  ptr = new Complex_type[len];
  initData(ptr, len, vid);
}


/*
 * Free data arrays of given type.
 */
void deallocData(Int_ptr& ptr)
{ 
  if (ptr) {
    delete [] ptr;
    ptr = 0;
  }
}

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


/*
 * \brief Initialize Int_type data array to 
 * randomly signed positive and negative values.
 */
void initData(Int_ptr& ptr, int len, VariantID vid)
{
  (void) vid;

// First touch...
#if defined(RAJA_ENABLE_OPENMP)
  if ( vid == Base_OpenMP ||
       vid == RAJA_OpenMP ) {
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) {
      ptr[i] = 0;
    };
  } 
#endif

  srand(4793);

  Real_type signfact = 0.0;

  for (int i = 0; i < len; ++i) {
    signfact = Real_type(rand())/RAND_MAX;
    ptr[i] = ( signfact < 0.5 ? -1 : 1 );
  };

  signfact = Real_type(rand())/RAND_MAX; 
  Int_type ilo = len * signfact;
  ptr[ilo] = -58;

  signfact = Real_type(rand())/RAND_MAX; 
  Int_type ihi = len * signfact;
  ptr[ihi] = 19;

  incDataInitCount();
}

/*
 * Initialize Real_type data array to non-random 
 * positive values (0.0, 1.0) based on their array position 
 * (index) and the order in which this method is called.
 */
void initData(Real_ptr& ptr, int len, VariantID vid) 
{
  (void) vid;

  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

// first touch...
#if defined(RAJA_ENABLE_OPENMP)
  if ( vid == Base_OpenMP || 
       vid == RAJA_OpenMP ) {
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) { 
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    };
  } 
#endif

  for (int i = 0; i < len; ++i) {
    ptr[i] = factor*(i + 1.1)/(i + 1.12345);
  } 

  incDataInitCount();
}

/*
 * Initialize Real_type data array to constant values.
 */
void initDataConst(Real_ptr& ptr, int len, Real_type val,
                   VariantID vid) 
{

// first touch...
#if defined(RAJA_ENABLE_OPENMP)
  if ( vid == Base_OpenMP ||
       vid == RAJA_OpenMP ) {
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) {
      ptr[i] = 0;
    };
  }
#endif

  for (int i = 0; i < len; ++i) {
    ptr[i] = val;
  };

  incDataInitCount();
}

/*
 * Initialize Real_type data array with random sign.
 */
void initDataRandSign(Real_ptr& ptr, int len, VariantID vid)
{
  (void) vid;

// First touch...
#if defined(RAJA_ENABLE_OPENMP)
  if ( vid == Base_OpenMP ||
       vid == RAJA_OpenMP ) {
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) {
      ptr[i] = 0.0;
    };
  }
#endif

  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

  srand(4793);

  for (int i = 0; i < len; ++i) {
    Real_type signfact = Real_type(rand())/RAND_MAX;
    signfact = ( signfact < 0.5 ? -1.0 : 1.0 );
    ptr[i] = signfact*factor*(i + 1.1)/(i + 1.12345);
  };

  incDataInitCount();
}

/*
 * Initialize Complex_type data array.
 */
void initData(Complex_ptr& ptr, int len, VariantID vid)
{
  (void) vid;

  Complex_type factor = ( data_init_count % 2 ?  Complex_type(0.1,0.2) :
                                                 Complex_type(0.2,0.3) );

#if defined(RAJA_ENABLE_OPENMP)
  if ( vid == Base_OpenMP ||
       vid == RAJA_OpenMP ) {
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) { 
      ptr[i] = factor*(i + 1.1)/(i + 1.12345);
    };
  }
#endif

  for (int i = 0; i < len; ++i) {
    ptr[i] = factor*(i + 1.1)/(i + 1.12345);
  }

  incDataInitCount();
}

/*
 * Initialize scalar data.
 */
void initData(Real_type& d, VariantID vid)
{
  (void) vid;

  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );
  d = factor*1.1/1.12345;

  incDataInitCount();
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
