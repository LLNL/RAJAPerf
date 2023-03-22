//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_DataUtils_HPP
#define RAJAPerf_DataUtils_HPP

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"
#include "common/OpenMPTargetDataUtils.hpp"
#include "common/CudaDataUtils.hpp"
#include "common/HipDataUtils.hpp"

#include <limits>
#include <new>
#include <type_traits>

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#endif

namespace rajaperf
{

namespace detail
{

/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();

/*!
 * Increment counter for data initialization.
 */
void incDataInitCount();

void copyHostData(void* dst_ptr, const void* src_ptr, size_t len);

/*!
 * \brief Allocate data arrays.
 */
void allocHostData(Int_ptr& ptr, int len, int align);
///
void allocHostData(Index_type*& ptr, int len, int align);
///
void allocHostData(Real_ptr& ptr, int len, int align);
///
void allocHostData(Complex_ptr& ptr, int len, int align);

/*!
 * \brief Free data arrays.
 */
void deallocHostData(Int_ptr& ptr);
///
void deallocHostData(Index_type*& ptr);
///
void deallocHostData(Real_ptr& ptr);
///
void deallocHostData(Complex_ptr& ptr);


/*!
 * \brief Touch Int_type data array with omp threads.
 */
void touchOmpData(Int_ptr& ptr, int len);

/*!
 * \brief Touch Real_type data array with omp threads.
 */
void touchOmpData(Real_ptr& ptr, int len);

/*!
 * \brief Touch Complex_type data array with omp threads.
 */
void touchOmpData(Complex_ptr& ptr, int len);

/*!
 * \brief Initialize Int_type data array.
 *
 * Array entries are randomly initialized to +/-1.
 * Then, two randomly-chosen entries are reset, one to
 * a value > 1, one to a value < -1.
 */
void initData(Int_ptr& ptr, int len);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set (non-randomly) to positive values
 * in the interval (0.0, 1.0) based on their array position (index)
 * and the order in which this method is called.
 */
void initData(Real_ptr& ptr, int len);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set to given constant value.
 */
void initDataConst(Real_ptr& ptr, int len, Real_type val);

/*!
 * \brief Initialize Real_type data array with random sign.
 *
 * Array entries are initialized in the same way as the method
 * initData(Real_ptr& ptr...) above, but with random sign.
 */
void initDataRandSign(Real_ptr& ptr, int len);

/*!
 * \brief Initialize Real_type data array with random values.
 *
 * Array entries are initialized with random values in the interval [0.0, 1.0].
 */
void initDataRandValue(Real_ptr& ptr, int len);

/*!
 * \brief Initialize Complex_type data array.
 *
 * Real and imaginary array entries are initialized in the same way as the
 * method allocAndInitData(Real_ptr& ptr...) above.
 */
void initData(Complex_ptr& ptr, int len);

/*!
 * \brief Initialize Real_type scalar data.
 *
 * Data is set similarly to an array enttry in the method
 * initData(Real_ptr& ptr...) above.
 */
void initData(Real_type& d);


/*!
 * \brief Calculate and return checksum for data arrays.
 *
 * Checksums are computed as a weighted sum of array entries,
 * where weight is a simple function of elemtn index.
 *
 * Checksumn is multiplied by given scale factor.
 */
long double calcChecksum(Int_ptr d, int len,
                         Real_type scale_factor);
///
long double calcChecksum(Real_ptr d, int len,
                         Real_type scale_factor);
///
long double calcChecksum(Complex_ptr d, int len,
                         Real_type scale_factor);

}  // closing brace for detail namespace


/*!
 * \brief Get an host accessible data space for this dataSpace.
 *
 * Intended to be a space that is quick to copy to from the given space if
 * the given space is not accessible on the Host.
 */
inline DataSpace hostAccessibleDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::Host:
    case DataSpace::Omp:
    case DataSpace::CudaPinned:
    case DataSpace::HipHostAdviseFine:
    case DataSpace::HipHostAdviseCoarse:
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
      return dataSpace;

    case DataSpace::OmpTarget:
      return DataSpace::Host;

    case DataSpace::CudaManaged:
    case DataSpace::CudaDevice:
      return DataSpace::CudaPinned;

    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
    case DataSpace::HipManagedAdviseCoarse:
      return dataSpace;

    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
      return DataSpace::HipPinned;

    default:
    {
      throw std::invalid_argument("hostAccessibleDataSpace : Unknown data space");
    } break;
  }
}

/*!
 * \brief Get if the data space is a host DataSpace.
 */
inline bool isHostDataSpace(DataSpace dataSpace)
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
inline bool isOpenMPDataSpace(DataSpace dataSpace)
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
inline bool isOpenMPTargetDataSpace(DataSpace dataSpace)
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
inline bool isCudaDataSpace(DataSpace dataSpace)
{
  switch (dataSpace) {
    case DataSpace::CudaPinned:
    case DataSpace::CudaManaged:
    case DataSpace::CudaDevice:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Get if the data space is a hip DataSpace.
 */
inline bool isHipDataSpace(DataSpace dataSpace)
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
 * \brief Allocate data array (ptr).
 */
template <typename T>
inline void allocData(DataSpace dataSpace, T& ptr, int len, int align)
{
  switch (dataSpace) {
    case DataSpace::Host:
    {
      detail::allocHostData(ptr, len, align);
    } break;

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case DataSpace::Omp:
    {
      detail::allocHostData(ptr, len, align);
      detail::touchOmpData(ptr, len);
    } break;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case DataSpace::OmpTarget:
    {
      detail::allocOpenMPDeviceData(ptr, len);
    } break;
#endif

#if defined(RAJA_ENABLE_CUDA)
    case DataSpace::CudaPinned:
    {
      detail::allocCudaPinnedData(ptr, len);
    } break;
    case DataSpace::CudaManaged:
    {
      detail::allocCudaManagedData(ptr, len);
    } break;
    case DataSpace::CudaDevice:
    {
      detail::allocCudaDeviceData(ptr, len);
    } break;
#endif

#if defined(RAJA_ENABLE_HIP)
    case DataSpace::HipHostAdviseFine:
    {
      detail::allocHostData(ptr, len, align);
      detail::adviseHipFineData(ptr, len);
    } break;
    case DataSpace::HipHostAdviseCoarse:
    {
      detail::allocHostData(ptr, len, align);
      detail::adviseHipCoarseData(ptr, len);
    } break;
    case DataSpace::HipPinned:
    {
      detail::allocHipPinnedData(ptr, len);
    } break;
    case DataSpace::HipPinnedFine:
    {
      detail::allocHipPinnedFineData(ptr, len);
    } break;
    case DataSpace::HipPinnedCoarse:
    {
      detail::allocHipPinnedCoarseData(ptr, len);
    } break;
    case DataSpace::HipManaged:
    {
      detail::allocHipManagedData(ptr, len);
    } break;
    case DataSpace::HipManagedAdviseFine:
    {
      detail::allocHipManagedData(ptr, len);
      detail::adviseHipFineData(ptr, len);
    } break;
    case DataSpace::HipManagedAdviseCoarse:
    {
      detail::allocHipManagedData(ptr, len);
      detail::adviseHipCoarseData(ptr, len);
    } break;
    case DataSpace::HipDevice:
    {
      detail::allocHipDeviceData(ptr, len);
    } break;
    case DataSpace::HipDeviceFine:
    {
      detail::allocHipDeviceFineData(ptr, len);
    } break;
#endif

    default:
    {
      throw std::invalid_argument("allocData : Unknown data space");
    } break;
  }
}

/*!
 * \brief Deallocate data array (ptr).
 */
template <typename T>
inline void deallocData(DataSpace dataSpace, T& ptr)
{
  switch (dataSpace) {
    case DataSpace::Host:
    case DataSpace::Omp:
    case DataSpace::HipHostAdviseFine:
    case DataSpace::HipHostAdviseCoarse:
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
    case DataSpace::HipManagedAdviseCoarse:
    {
      detail::deallocHipManagedData(ptr);
    } break;
    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine:
    {
      detail::deallocHipDeviceData(ptr);
    } break;
#endif

    default:
    {
      throw std::invalid_argument("deallocData : Unknown data space");
    } break;
  }
}

/*!
 * \brief Copy data from one array to another.
 */
template <typename T>
inline void copyData(DataSpace dst_dataSpace, T* dst_ptr,
                     DataSpace src_dataSpace, const T* src_ptr,
                     int len)
{
  if (hostAccessibleDataSpace(dst_dataSpace) == dst_dataSpace &&
      hostAccessibleDataSpace(src_dataSpace) == src_dataSpace) {
    detail::copyHostData(dst_ptr, src_ptr, sizeof(T)*len);
  }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  else if (isOpenMPTargetDataSpace(dst_dataSpace) ||
           isOpenMPTargetDataSpace(src_dataSpace)) {
    auto dst_did = isOpenMPTargetDataSpace(dst_dataSpace) ? omp_get_default_device()
                                                          : omp_get_initial_device();
    auto src_did = isOpenMPTargetDataSpace(src_dataSpace) ? omp_get_default_device()
                                                          : omp_get_initial_device();
    detail::copyOpenMPTargetData(dst_ptr, src_ptr, sizeof(T)*len,
        dst_did, src_did);
  }
#endif

#if defined(RAJA_ENABLE_CUDA)
  else if (isCudaDataSpace(dst_dataSpace) ||
           isCudaDataSpace(src_dataSpace)) {
    detail::copyCudaData(dst_ptr, src_ptr, sizeof(T)*len);
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  else if (isHipDataSpace(dst_dataSpace) ||
           isHipDataSpace(src_dataSpace)) {
    detail::copyHipData(dst_ptr, src_ptr, sizeof(T)*len);
  }
#endif

  else {
    throw std::invalid_argument("copyData : Unknown data space");
  }
}

/*!
 * \brief Move data array into new dataSpace.
 */
template <typename T>
inline void moveData(DataSpace new_dataSpace, DataSpace old_dataSpace,
                     T*& ptr, int len, int align)
{
  if (new_dataSpace != old_dataSpace) {

    T* new_ptr = nullptr;

    allocData(new_dataSpace, new_ptr, len, align);

    copyData(new_dataSpace, new_ptr, old_dataSpace, ptr, len);

    deallocData(old_dataSpace, ptr);

    ptr = new_ptr;
  }
}

/*!
 * \brief Allocate and initialize data array.
 */
template <typename T>
inline void allocAndInitData(DataSpace dataSpace, T*& ptr, int len, int align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initData(ptr, len);

  if (init_dataSpace != dataSpace) {
    moveData(dataSpace, init_dataSpace, ptr, len, align);
  }
}


/*!
 * \brief Allocate and initialize aligned Real_type data array.
 *
 * Array entries are initialized using the method
 * initDataConst(Real_ptr& ptr...) below.
 */
template <typename T>
inline void allocAndInitDataConst(DataSpace dataSpace, T*& ptr, int len, int align,
                                  Real_type val)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataConst(ptr, len, val);

  if (init_dataSpace != dataSpace) {
    moveData(dataSpace, init_dataSpace, ptr, len, align);
  }
}

/*!
 * \brief Allocate and initialize aligned Real_type data array with random sign.
 *
 * Array is initialized using method initDataRandSign(Real_ptr& ptr...) below.
 */
template <typename T>
inline void allocAndInitDataRandSign(DataSpace dataSpace, T*& ptr, int len, int align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataRandSign(ptr, len);

  if (init_dataSpace != dataSpace) {
    moveData(dataSpace, init_dataSpace, ptr, len, align);
  }
}

/*!
 * \brief Allocate and initialize aligned Real_type data array with random
 *        values.
 *
 * Array is initialized using method initDataRandValue(Real_ptr& ptr...) below.
 */
template <typename T>
inline void allocAndInitDataRandValue(DataSpace dataSpace, T*& ptr, int len, int align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataRandValue(ptr, len);

  if (init_dataSpace != dataSpace) {
    moveData(dataSpace, init_dataSpace, ptr, len, align);
  }
}

/*
 * Calculate and return checksum for arrays.
 */
template <typename T>
inline long double calcChecksum(DataSpace dataSpace, T* ptr, int len, int align,
                                Real_type scale_factor)
{
  T* check_ptr = ptr;
  T* copied_ptr = nullptr;

  DataSpace check_dataSpace = hostAccessibleDataSpace(dataSpace);
  if (check_dataSpace != dataSpace) {
    allocData(check_dataSpace, copied_ptr, len, align);

    copyData(check_dataSpace, copied_ptr, dataSpace, ptr, len);

    check_ptr = copied_ptr;
  }

  auto val = detail::calcChecksum(check_ptr, len, scale_factor);

  if (check_dataSpace != dataSpace) {
    deallocData(check_dataSpace, copied_ptr);
  }

  return val;
}


/*!
 * \brief Holds a RajaPool object and provides access to it via a
 *        std allocator compliant type.
 */
template < typename RajaPool >
struct RAJAPoolAllocatorHolder
{

  /*!
   * \brief Std allocator compliant type that uses the pool owned by
   *        the RAJAPoolAllocatorHolder that it came from.
   *
   * Note that this must not outlive the RAJAPoolAllocatorHolder
   * used to create it.
   */
  template < typename T >
  struct Allocator
  {
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    Allocator() = default;

    Allocator(Allocator const&) = default;
    Allocator(Allocator &&) = default;

    Allocator& operator=(Allocator const&) = default;
    Allocator& operator=(Allocator &&) = default;

    template < typename U >
    constexpr Allocator(Allocator<U> const& other) noexcept
      : m_pool_ptr(other.get_impl())
    { }

    Allocator select_on_container_copy_construction()
    {
      return *this;
    }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type *ptr = nullptr;
      if (m_pool_ptr != nullptr) {
        ptr = m_pool_ptr->template malloc<value_type>(num);
      }

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t) noexcept
    {
      if (m_pool_ptr != nullptr) {
        m_pool_ptr->free(ptr);
      }
    }

    RajaPool* const& get_impl() const
    {
      return m_pool_ptr;
    }

    template <typename U>
    friend inline bool operator==(Allocator const& lhs, Allocator<U> const& rhs)
    {
      return lhs.get_impl() == rhs.get_impl();
    }

    template <typename U>
    friend inline bool operator!=(Allocator const& lhs, Allocator<U> const& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    friend RAJAPoolAllocatorHolder;

    RajaPool* m_pool_ptr = nullptr;

    constexpr Allocator(RajaPool* pool_ptr) noexcept
      : m_pool_ptr(pool_ptr)
    { }
  };

  template < typename T >
  Allocator<T> getAllocator()
  {
    return Allocator<T>(&m_pool);
  }

  ~RAJAPoolAllocatorHolder()
  {
    // manually free memory arenas, as this is not done automatically
    m_pool.free_chunks();
  }

private:
  RajaPool m_pool;
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
