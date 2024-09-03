//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DataUtils.hpp"
#include "CudaDataUtils.hpp"
#include "HipDataUtils.hpp"
#include "OpenMPTargetDataUtils.hpp"
#include "SyclDataUtils.hpp"

#include "KernelBase.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include<direct.h>
#else
#include <unistd.h>
#endif

namespace rajaperf
{

namespace detail
{

/*!
 * \brief Get if the data space is a host DataSpace.
 */
bool isHostDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::Host:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a omp DataSpace.
 */
bool isOpenMPDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::Omp:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a omp target DataSpace.
 */
bool isOpenMPTargetDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::OmpTarget:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a cuda DataSpace.
 */
bool isCudaDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::CudaPinned:
    case DataSpace::CudaManaged:
    case DataSpace::CudaManagedHostPreferred:
    case DataSpace::CudaManagedDevicePreferred:
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    case DataSpace::CudaDevice:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a hip DataSpace.
 */
bool isHipDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::HipHostAdviseFine:
    case DataSpace::HipHostAdviseCoarse:
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
    case DataSpace::HipManagedAdviseCoarse:
    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a sycl DataSpace.
 */
bool isSyclDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::SyclPinned:
    case DataSpace::SyclManaged:
    case DataSpace::SyclDevice:
      return true;
    default:
      return false;
  }
}


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
 * Copy memory len bytes from src to dst.
 */
void copyHostData(void* dst_ptr, const void* src_ptr, Size_type len)
{
  std::memcpy(dst_ptr, src_ptr, len);
}


/*
 * Allocate data arrays of given type.
 */
void* allocHostData(Size_type len, Size_type align)
{
  return RAJA::allocate_aligned_type<Int_type>(
      align, len);
}


/*
 * Free data arrays of given type.
 */
void deallocHostData(void* ptr)
{
  if (ptr) {
    RAJA::free_aligned(ptr);
  }
}


/*
 * Allocate data arrays of given dataSpace.
 */
void* allocData(DataSpace dataSpace, Size_type nbytes, Size_type align)
{
  void* ptr = nullptr;

  switch (dataSpace) {
    case DataSpace::Host:
    {
      ptr = detail::allocHostData(nbytes, align);
    } break;

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case DataSpace::Omp:
    {
      ptr = detail::allocHostData(nbytes, align);
    } break;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case DataSpace::OmpTarget:
    {
      ptr = detail::allocOpenMPDeviceData(nbytes);
    } break;
#endif

#if defined(RAJA_ENABLE_CUDA)
    case DataSpace::CudaPinned:
    {
      ptr = detail::allocCudaPinnedData(nbytes);
    } break;
    case DataSpace::CudaManaged:
    {
      ptr = detail::allocCudaManagedData(nbytes);
    } break;
    case DataSpace::CudaManagedHostPreferred:
    {
      ptr = detail::allocCudaManagedHostPreferredData(nbytes);
    } break;
    case DataSpace::CudaManagedDevicePreferred:
    {
      ptr = detail::allocCudaManagedDevicePreferredData(nbytes);
    } break;
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    {
      ptr = detail::allocCudaManagedHostPreferredDeviceAccessedData(nbytes);
    } break;
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    {
      ptr = detail::allocCudaManagedDevicePreferredHostAccessedData(nbytes);
    } break;
    case DataSpace::CudaDevice:
    {
      ptr = detail::allocCudaDeviceData(nbytes);
    } break;
#endif

#if defined(RAJA_ENABLE_HIP)
    case DataSpace::HipHostAdviseFine:
    {
      ptr = detail::allocHostData(nbytes, align);
      detail::adviseHipFineData(ptr, nbytes);
    } break;
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipHostAdviseCoarse:
    {
      ptr = detail::allocHostData(nbytes, align);
      detail::adviseHipCoarseData(ptr, nbytes);
    } break;
#endif
    case DataSpace::HipPinned:
    {
      ptr = detail::allocHipPinnedData(nbytes);
    } break;
    case DataSpace::HipPinnedFine:
    {
      ptr = detail::allocHipPinnedFineData(nbytes);
    } break;
    case DataSpace::HipPinnedCoarse:
    {
      ptr = detail::allocHipPinnedCoarseData(nbytes);
    } break;
    case DataSpace::HipManaged:
    {
      ptr = detail::allocHipManagedData(nbytes);
    } break;
    case DataSpace::HipManagedAdviseFine:
    {
      ptr = detail::allocHipManagedData(nbytes);
      detail::adviseHipFineData(ptr, nbytes);
    } break;
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipManagedAdviseCoarse:
    {
      ptr = detail::allocHipManagedData(nbytes);
      detail::adviseHipCoarseData(ptr, nbytes);
    } break;
#endif
    case DataSpace::HipDevice:
    {
      ptr = detail::allocHipDeviceData(nbytes);
    } break;
    case DataSpace::HipDeviceFine:
    {
      ptr = detail::allocHipDeviceFineData(nbytes);
    } break;
#endif

#if defined(RAJA_ENABLE_SYCL)
    case DataSpace::SyclPinned:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      ptr = detail::allocSyclPinnedData(nbytes, qu);
    } break;
    case DataSpace::SyclManaged:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      ptr = detail::allocSyclManagedData(nbytes, qu);
    } break;
    case DataSpace::SyclDevice:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      ptr = detail::allocSyclDeviceData(nbytes, qu);
    } break;
#endif


    default:
    {
      throw std::invalid_argument("allocData : Unknown data space");
    } break;
  }

  return ptr;
}

/*!
 * \brief Copy data from one dataSpace to another.
 */
void copyData(DataSpace dst_dataSpace, void* dst_ptr,
              DataSpace src_dataSpace, const void* src_ptr,
              Size_type nbytes)
{
  if (hostCopyDataSpace(dst_dataSpace) == dst_dataSpace &&
      hostCopyDataSpace(src_dataSpace) == src_dataSpace) {
    detail::copyHostData(dst_ptr, src_ptr, nbytes);
  }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  else if (isOpenMPTargetDataSpace(dst_dataSpace) ||
           isOpenMPTargetDataSpace(src_dataSpace)) {
    auto dst_did = isOpenMPTargetDataSpace(dst_dataSpace) ? getOpenMPTargetDevice()
                                                          : getOpenMPTargetHost();
    auto src_did = isOpenMPTargetDataSpace(src_dataSpace) ? getOpenMPTargetDevice()
                                                          : getOpenMPTargetHost();
    detail::copyOpenMPTargetData(dst_ptr, src_ptr, nbytes,
        dst_did, src_did);
  }
#endif

#if defined(RAJA_ENABLE_CUDA)
  else if (isCudaDataSpace(dst_dataSpace) ||
           isCudaDataSpace(src_dataSpace)) {
    detail::copyCudaData(dst_ptr, src_ptr, nbytes);
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  else if (isHipDataSpace(dst_dataSpace) ||
           isHipDataSpace(src_dataSpace)) {
    detail::copyHipData(dst_ptr, src_ptr, nbytes);
  }
#endif

#if defined(RAJA_ENABLE_SYCL)
  else if (isSyclDataSpace(dst_dataSpace) ||
           isSyclDataSpace(src_dataSpace)) {
    auto qu = camp::resources::Sycl::get_default().get_queue();
    detail::copySyclData(dst_ptr, src_ptr, nbytes, qu);
  }
#endif

  else {
    throw std::invalid_argument("copyData : Unknown data space");
  }
}

/*!
 * \brief Deallocate data array (ptr).
 */
void deallocData(DataSpace dataSpace, void* ptr)
{
  switch (dataSpace) {
    case DataSpace::Host:
    case DataSpace::Omp:
#if defined(RAJA_ENABLE_HIP)
    case DataSpace::HipHostAdviseFine:
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipHostAdviseCoarse:
#endif
#endif
    {
      detail::deallocHostData(ptr);
    } break;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case DataSpace::OmpTarget:
    {
      detail::deallocOpenMPDeviceData(ptr);
    } break;
#endif

#if defined(RAJA_ENABLE_CUDA)
    case DataSpace::CudaPinned:
    {
      detail::deallocCudaPinnedData(ptr);
    } break;
    case DataSpace::CudaManaged:
    {
      detail::deallocCudaManagedData(ptr);
    } break;
    case DataSpace::CudaManagedHostPreferred:
    {
      detail::deallocCudaManagedHostPreferredData(ptr);
    } break;
    case DataSpace::CudaManagedDevicePreferred:
    {
      detail::deallocCudaManagedDevicePreferredData(ptr);
    } break;
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    {
      detail::deallocCudaManagedHostPreferredDeviceAccessedData(ptr);
    } break;
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    {
      detail::deallocCudaManagedDevicePreferredHostAccessedData(ptr);
    } break;
    case DataSpace::CudaDevice:
    {
      detail::deallocCudaDeviceData(ptr);
    } break;
#endif

#if defined(RAJA_ENABLE_HIP)
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
    {
      detail::deallocHipPinnedData(ptr);
    } break;
    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipManagedAdviseCoarse:
#endif
    {
      detail::deallocHipManagedData(ptr);
    } break;
    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
    {
      detail::deallocHipDeviceData(ptr);
    } break;
#endif

#if defined(RAJA_ENABLE_SYCL)
    case DataSpace::SyclPinned:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      detail::deallocSyclPinnedData(ptr, qu);
    } break;
    case DataSpace::SyclManaged:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      detail::deallocSyclManagedData(ptr, qu);
    } break;
    case DataSpace::SyclDevice:
    {
      auto qu = camp::resources::Sycl::get_default().get_queue();
      detail::deallocSyclDeviceData(ptr, qu);
    } break;
#endif



    default:
    {
      throw std::invalid_argument("deallocData : Unknown data space");
    } break;
  }
}


/*
 * \brief Initialize Int_type data array to
 * randomly signed positive and negative values.
 */
void initData(Int_ptr& ptr, Size_type len)
{
  srand(4793);

  Real_type signfact = 0.0;

  for (Size_type i = 0; i < len; ++i) {
    signfact = Real_type(rand())/RAND_MAX;
    ptr[i] = ( signfact < 0.5 ? -1 : 1 );
  };

  signfact = Real_type(rand())/RAND_MAX;
  Size_type ilo = len * signfact;
  ptr[ilo] = -58;

  signfact = Real_type(rand())/RAND_MAX;
  Size_type ihi = len * signfact;
  ptr[ihi] = 19;

  incDataInitCount();
}

/*
 * Initialize Real_type data array to non-random
 * positive values (0.0, 1.0) based on their array position
 * (index) and the order in which this method is called.
 */
void initData(Real_ptr& ptr, Size_type len)
{
  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

  for (Size_type i = 0; i < len; ++i) {
    ptr[i] = factor*(i + 1.1)/(i + 1.12345);
  }

  incDataInitCount();
}

/*
 * Initialize Real_type data array to constant values.
 */
void initDataConst(Real_ptr& ptr, Size_type len, Real_type val)
{
  for (Size_type i = 0; i < len; ++i) {
    ptr[i] = val;
  };

  incDataInitCount();
}

/*
 * Initialize Index_type data array to constant values.
 */
void initDataConst(Index_type*& ptr, Size_type len, Index_type val)
{
  for (Size_type i = 0; i < len; ++i) {
    ptr[i] = val;
  };

  incDataInitCount();
}

/*
 * Initialize Real_type data array with random sign.
 */
void initDataRandSign(Real_ptr& ptr, Size_type len)
{
  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );

  srand(4793);

  for (Size_type i = 0; i < len; ++i) {
    Real_type signfact = Real_type(rand())/RAND_MAX;
    signfact = ( signfact < 0.5 ? -1.0 : 1.0 );
    ptr[i] = signfact*factor*(i + 1.1)/(i + 1.12345);
  };

  incDataInitCount();
}

/*
 * Initialize Real_type data array with random values.
 */
void initDataRandValue(Real_ptr& ptr, Size_type len)
{
  srand(4793);

  for (Size_type i = 0; i < len; ++i) {
    ptr[i] = Real_type(rand())/RAND_MAX;
  };

  incDataInitCount();
}

/*
 * Initialize Complex_type data array.
 */
void initData(Complex_ptr& ptr, Size_type len)
{
  Complex_type factor = ( data_init_count % 2 ?  Complex_type(0.1,0.2) :
                                                 Complex_type(0.2,0.3) );

  for (Size_type i = 0; i < len; ++i) {
    ptr[i] = factor*(i + 1.1)/(i + 1.12345);
  }

  incDataInitCount();
}

/*
 * Initialize scalar data.
 */
void initData(Real_type& d)
{
  Real_type factor = ( data_init_count % 2 ? 0.1 : 0.2 );
  d = factor*1.1/1.12345;

  incDataInitCount();
}

/*
 * Calculate and return checksum for data arrays.
 */
template < typename Data_getter >
long double calcChecksumImpl(Data_getter data, Size_type len,
                             Real_type scale_factor)
{
  long double tchk = 0.0;
  long double ckahan = 0.0;
  for (Size_type j = 0; j < len; ++j) {
    long double x = (std::abs(std::sin(j+1.0))+0.5) * data(j);
    long double y = x - ckahan;
    volatile long double t = tchk + y;
    volatile long double z = t - tchk;
    ckahan = z - y;
    tchk = t;
#if 0 // RDH DEBUG
    if ( (j % 100) == 0 ) {
      getCout() << "j : tchk = " << j << " : " << tchk << std::endl;
    }
#endif
  }
  tchk *= scale_factor;
  return tchk;
}

long double calcChecksum(Int_ptr ptr, Size_type len,
                         Real_type scale_factor)
{
  return calcChecksumImpl([=](Size_type j) {
    return static_cast<long double>(ptr[j]);
  }, len, scale_factor);
}

long double calcChecksum(unsigned long long* ptr, Size_type len,
                         Real_type scale_factor)
{
  return calcChecksumImpl([=](Size_type j) {
    return static_cast<long double>(ptr[j]);
  }, len, scale_factor);
}

long double calcChecksum(Real_ptr ptr, Size_type len,
                         Real_type scale_factor)
{
  return calcChecksumImpl([=](Size_type j) {
    return static_cast<long double>(ptr[j]);
  }, len, scale_factor);
}

long double calcChecksum(Complex_ptr ptr, Size_type len,
                         Real_type scale_factor)
{
  return calcChecksumImpl([=](Size_type j) {
    return static_cast<long double>(real(ptr[j])+imag(ptr[j]));
  }, len, scale_factor);
}

}  // closing brace for detail namespace


/*!
 * \brief Get a host data space to use when making a host copy of data in the given
 *        dataSpace.
 *
 * The returned host data space should reside in memory attached to the host.
 *
 * The intention is to get a data space with high performance on the host.
 * Return the given data space if its already performant and fall back on a
 * host data space that performs well in explicit copy operations with the
 * given space.
 */
DataSpace hostCopyDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::Host:
    case DataSpace::Omp:
    case DataSpace::CudaPinned:
    case DataSpace::CudaManagedHostPreferred:
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    case DataSpace::HipHostAdviseFine:
    case DataSpace::HipHostAdviseCoarse:
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
    case DataSpace::HipManagedAdviseCoarse:
    case DataSpace::SyclPinned:
      return dataSpace;

    case DataSpace::OmpTarget:
      return DataSpace::Host;

    case DataSpace::CudaManaged:
    case DataSpace::CudaManagedDevicePreferred:
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    case DataSpace::CudaDevice:
      return DataSpace::CudaPinned;

    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
      return DataSpace::HipPinned;

    case DataSpace::SyclManaged:
    case DataSpace::SyclDevice:
      return DataSpace::SyclPinned;

    default:
    {
      throw std::invalid_argument("hostCopyDataSpace : Unknown data space");
    } break;
  }
}

/*!
 * \brief Get a data space accessible to the host for the given dataSpace.
 *
 * The returned host data space may reside in memory attached to another device.
 *
 * The intention is to get a data space accessible on the host even if it is not
 * performant. Return the given data space if its already accessible and fall
 * back on a space that is host accessible and performs well in explicit copy
 * operations with the given space.
 */
DataSpace hostAccessibleDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::Host:
    case DataSpace::Omp:
    case DataSpace::CudaPinned:
    case DataSpace::CudaManaged:
    case DataSpace::CudaManagedHostPreferred:
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    case DataSpace::CudaManagedDevicePreferred:
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    case DataSpace::HipHostAdviseFine:
    case DataSpace::HipHostAdviseCoarse:
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
    case DataSpace::HipManagedAdviseCoarse:
    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
    case DataSpace::SyclPinned:
    case DataSpace::SyclManaged:
      return dataSpace;

    case DataSpace::OmpTarget:
      return DataSpace::Host;

    case DataSpace::CudaDevice:
      return DataSpace::CudaPinned;

    case DataSpace::SyclDevice:
      return DataSpace::SyclPinned;

    default:
    {
      throw std::invalid_argument("hostAccessibleDataSpace : Unknown data space");
    } break;
  }
}

}  // closing brace for rajaperf namespace
