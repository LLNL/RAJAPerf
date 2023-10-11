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

void copyHostData(void* dst_ptr, const void* src_ptr, Size_type len);

/*!
 * \brief Allocate data arrays.
 */
void* allocHostData(Size_type len, Size_type align);

/*!
 * \brief Free data arrays.
 */
void deallocHostData(void* ptr);


/*!
 * \brief Allocate data array in dataSpace.
 */
void* allocData(DataSpace dataSpace, Size_type nbytes, Size_type align);

/*!
 * \brief Copy data from one dataSpace to another.
 */
void copyData(DataSpace dst_dataSpace, void* dst_ptr,
              DataSpace src_dataSpace, const void* src_ptr,
              Size_type nbytes);

/*!
 * \brief Free data arrays in dataSpace.
 */
void deallocData(DataSpace dataSpace, void* ptr);


/*!
 * \brief Initialize Int_type data array.
 *
 * Array entries are randomly initialized to +/-1.
 * Then, two randomly-chosen entries are reset, one to
 * a value > 1, one to a value < -1.
 */
void initData(Int_ptr& ptr, Size_type len);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set (non-randomly) to positive values
 * in the interval (0.0, 1.0) based on their array position (index)
 * and the order in which this method is called.
 */
void initData(Real_ptr& ptr, Size_type len);

/*!
 * \brief Initialize Real_type data array.
 *
 * Array entries are set to given constant value.
 */
void initDataConst(Real_ptr& ptr, Size_type len, Real_type val);

/*!
 * \brief Initialize Index_type data array.
 *
 * Array entries are set to given constant value.
 */
void initDataConst(Index_type*& ptr, Size_type len, Index_type val);

/*!
 * \brief Initialize Real_type data array with random sign.
 *
 * Array entries are initialized in the same way as the method
 * initData(Real_ptr& ptr...) above, but with random sign.
 */
void initDataRandSign(Real_ptr& ptr, Size_type len);

/*!
 * \brief Initialize Real_type data array with random values.
 *
 * Array entries are initialized with random values in the interval [0.0, 1.0].
 */
void initDataRandValue(Real_ptr& ptr, Size_type len);

/*!
 * \brief Initialize Complex_type data array.
 *
 * Real and imaginary array entries are initialized in the same way as the
 * method allocAndInitData(Real_ptr& ptr...) above.
 */
void initData(Complex_ptr& ptr, Size_type len);

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
long double calcChecksum(Int_ptr d, Size_type len,
                         Real_type scale_factor);
///
long double calcChecksum(Real_ptr d, Size_type len,
                         Real_type scale_factor);
///
long double calcChecksum(Complex_ptr d, Size_type len,
                         Real_type scale_factor);

}  // closing brace for detail namespace


/*!
 * \brief Get an host accessible data space for this dataSpace.
 *
 * Intended to be a space that is quick to copy to from the given space if
 * the given space is not accessible on the Host.
 */
DataSpace hostAccessibleDataSpace(DataSpace dataSpace);

/*!
 * \brief Allocate data array (ptr).
 */
template <typename T>
inline void allocData(DataSpace dataSpace, T*& ptr_ref, Size_type len, Size_type align)
{
  Size_type nbytes = len*sizeof(T);
  T* ptr = static_cast<T*>(detail::allocData(dataSpace, nbytes, align));

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  if (dataSpace == DataSpace::Omp) {
    // perform first touch on Omp Data
    #pragma omp parallel for
    for (Size_type i = 0; i < len; ++i) {
      ptr[i] = T{};
    };
  }
#endif

  ptr_ref = ptr;
}

/*!
 * \brief Deallocate data array (ptr).
 */
template <typename T>
inline void deallocData(DataSpace dataSpace, T*& ptr)
{
  detail::deallocData(dataSpace, ptr);
  ptr = nullptr;
}

/*!
 * \brief Copy data from one array to another.
 */
template <typename T>
inline void copyData(DataSpace dst_dataSpace, T* dst_ptr,
                     DataSpace src_dataSpace, const T* src_ptr,
                     Size_type len)
{
  Size_type nbytes = len*sizeof(T);
  detail::copyData(dst_dataSpace, dst_ptr, src_dataSpace, src_ptr, nbytes);
}

/*!
 * \brief Move data array into new dataSpace.
 */
template <typename T>
inline void moveData(DataSpace new_dataSpace, DataSpace old_dataSpace,
                     T*& ptr, Size_type len, Size_type align)
{
  if (new_dataSpace != old_dataSpace) {

    T* new_ptr = nullptr;

    allocData(new_dataSpace, new_ptr, len, align);

    copyData(new_dataSpace, new_ptr, old_dataSpace, ptr, len);

    deallocData(old_dataSpace, ptr);

    ptr = new_ptr;
  }
}


template <typename T>
struct AutoDataMover
{
  AutoDataMover(DataSpace new_dataSpace, DataSpace old_dataSpace,
                T*& ptr, Size_type len, Size_type align)
    : m_ptr(&ptr)
    , m_new_dataSpace(new_dataSpace)
    , m_old_dataSpace(old_dataSpace)
    , m_len(len)
    , m_align(align)
  { }

  AutoDataMover(AutoDataMover const&) = delete;
  AutoDataMover& operator=(AutoDataMover const&) = delete;

  AutoDataMover(AutoDataMover&& rhs)
    : m_ptr(std::exchange(rhs.m_ptr, nullptr))
    , m_new_dataSpace(rhs.m_new_dataSpace)
    , m_old_dataSpace(rhs.m_old_dataSpace)
    , m_len(rhs.m_len)
    , m_align(rhs.m_align)
  { }
  AutoDataMover& operator=(AutoDataMover&& rhs)
  {
    finalize();
    m_ptr = std::exchange(rhs.m_ptr, nullptr);
    m_new_dataSpace = rhs.m_new_dataSpace;
    m_old_dataSpace = rhs.m_old_dataSpace;
    m_len = rhs.m_len;
    m_align = rhs.m_align;
    return *this;
  }

  void finalize()
  {
    if (m_ptr) {
      moveData(m_new_dataSpace, m_old_dataSpace,
          *m_ptr, m_len, m_align);
      m_ptr = nullptr;
    }
  }

  ~AutoDataMover()
  {
    finalize();
  }

private:
  T** m_ptr;
  DataSpace m_new_dataSpace;
  DataSpace m_old_dataSpace;
  Size_type m_len;
  Size_type m_align;
};

/*!
 * \brief Allocate and initialize data array.
 */
template <typename T>
inline void allocAndInitData(DataSpace dataSpace, T*& ptr, Size_type len, Size_type align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initData(ptr, len);

  moveData(dataSpace, init_dataSpace, ptr, len, align);
}


/*!
 * \brief Allocate and initialize aligned Real_type data array.
 *
 * Array entries are initialized using the method initDataConst.
 */
template <typename T>
inline void allocAndInitDataConst(DataSpace dataSpace, T*& ptr, Size_type len, Size_type align,
                                  T val)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataConst(ptr, len, val);

  moveData(dataSpace, init_dataSpace, ptr, len, align);
}

/*!
 * \brief Allocate and initialize aligned Real_type data array with random sign.
 *
 * Array is initialized using method initDataRandSign.
 */
template <typename T>
inline void allocAndInitDataRandSign(DataSpace dataSpace, T*& ptr, Size_type len, Size_type align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataRandSign(ptr, len);

  moveData(dataSpace, init_dataSpace, ptr, len, align);
}

/*!
 * \brief Allocate and initialize aligned Real_type data array with random
 *        values.
 *
 * Array is initialized using method initDataRandValue.
 */
template <typename T>
inline void allocAndInitDataRandValue(DataSpace dataSpace, T*& ptr, Size_type len, Size_type align)
{
  DataSpace init_dataSpace = hostAccessibleDataSpace(dataSpace);

  allocData(init_dataSpace, ptr, len, align);

  detail::initDataRandValue(ptr, len);

  moveData(dataSpace, init_dataSpace, ptr, len, align);
}

/*
 * Calculate and return checksum for arrays.
 */
template <typename T>
inline long double calcChecksum(DataSpace dataSpace, T* ptr, Size_type len, Size_type align,
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
    value_type* allocate(Size_type num)
    {
      if (num > std::numeric_limits<Size_type>::max() / sizeof(value_type)) {
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
