//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
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

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define RAJA_PERFSUITE_HOST_DEVICE __host__ __device__
#else
#define RAJA_PERFSUITE_HOST_DEVICE
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
 * \brief Initialize Index_type data array.
 *
 * Array entries are randomly initialized to +/-1.
 * Then, two randomly-chosen entries are reset, one to
 * a value > 1, one to a value < -1.
 */
void initData(Index_ptr& ptr, Size_type len);

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
long double calcChecksum(Index_ptr d, Size_type len,
                         Real_type scale_factor);
///
long double calcChecksum(unsigned long long* d, Size_type len,
                         Real_type scale_factor);
///
long double calcChecksum(Real_ptr d, Size_type len,
                         Real_type scale_factor);
///
long double calcChecksum(Complex_ptr d, Size_type len,
                         Real_type scale_factor);

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
DataSpace hostCopyDataSpace(DataSpace dataSpace);

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
  DataSpace init_dataSpace = hostCopyDataSpace(dataSpace);

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
  DataSpace init_dataSpace = hostCopyDataSpace(dataSpace);

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
  DataSpace init_dataSpace = hostCopyDataSpace(dataSpace);

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
  DataSpace init_dataSpace = hostCopyDataSpace(dataSpace);

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

  DataSpace check_dataSpace = hostCopyDataSpace(dataSpace);
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


namespace detail {

constexpr size_t next_pow2(size_t v) noexcept
{
  static_assert(sizeof(size_t) == 8, "");
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v++;
  return v;
}

constexpr size_t log2(size_t v) noexcept
{
  size_t result = 0;
  while(v >>= 1) { result++; }
  return result;
}


template < typename ... Ts >
struct PackNumBytes
{
  static constexpr size_t value = (0 + ... + sizeof(Ts));
};

/*!
 * \brief Pack 2-bit integer into least significant 2 bits of double precision
 * floating point number's significand.
 */
struct DoubleChar2Bits
{
  static_assert(sizeof(double) == 8, "");
  static_assert(sizeof(unsigned long long) == 8, "");

  static constexpr bool packable = true;
  using type = unsigned long long;

  // little endian bit layout of double
  struct BitLayout
  {
    unsigned long long char2 : 2; // least-significant 2 bits of significand
    unsigned long long significand : 50;
    unsigned long long exponent : 11;
    unsigned long long sign : 1;
  };

  RAJA_PERFSUITE_HOST_DEVICE
  static inline type pack(double const& d, char const& c)
  {
    BitLayout bit_val;
    memcpy(&bit_val, &d, sizeof(double));
    bit_val.char2 = c;
    type packed_val;
    memcpy(&packed_val, &bit_val, sizeof(double));
    return packed_val;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  static inline void unpack(type const& packed_val, double& d, char& c)
  {
    BitLayout bit_val;
    memcpy(&bit_val, &packed_val, sizeof(double));
    c = bit_val.char2;
    bit_val.char2 = 0;
    memcpy(&d, &bit_val, sizeof(double));
  }
};

/*!
 * \brief Pack 2-bit integer into most significant 2 bits of 64-bit integer.
 */
template < typename Integral >
struct IntegralChar2bits
{
  static_assert(sizeof(Integral) == 8, "");
  static_assert(sizeof(unsigned long long) == 8, "");

  static constexpr bool packable = true;
  using type = unsigned long long;

  RAJA_PERFSUITE_HOST_DEVICE
  static inline type pack(Integral const& i, char const& c)
  {
    Integral packed_tmp = (i << 2) | (c & 0x3);
    type packed_val;
    memcpy(&packed_val, &packed_tmp, sizeof(type));
    return packed_val;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  static inline void unpack(type const& packed_val, Integral& i, char& c)
  {
    Integral packed_tmp;
    memcpy(&packed_tmp, &packed_val, sizeof(type));
    c = static_cast<char>(packed_tmp & 0x3);
    i = packed_tmp >> 2;
  }
};


/*!
 * \brief Atomic Data Packer, packs data into a single type that can be used
 * with atomics.
 */
template < typename, typename ... Ts >
struct AtomicDataPackerImpl;

template < typename ... Ts >
struct AtomicDataPackerImpl<std::enable_if_t<(PackNumBytes<Ts...>::value > 8)>, Ts...>
{
  static constexpr bool packable = false;
  using type = void;
};
///
template < typename ... Ts >
struct AtomicDataPackerImpl<std::enable_if_t<(PackNumBytes<Ts...>::value <= 8)>, Ts...>
{
  static_assert(PackNumBytes<Ts...>::value <= 8, "");
  static_assert(sizeof(unsigned) == 4, "");
  static_assert(sizeof(unsigned long long) == 8, "");

  static constexpr bool packable = true;
  using type = std::conditional_t< (PackNumBytes<Ts...>::value <= 4),
                                   unsigned, unsigned long long >;

  RAJA_PERFSUITE_HOST_DEVICE
  static inline type pack(Ts const&... args)
  {
    type packed_val = 0;
    auto ptr = reinterpret_cast<char*>(&packed_val);
    int in_order_expansion[]{(
        memcpy(ptr, &args, sizeof(Ts)),
        ptr += sizeof(Ts),
    0)...};
    ignore_unused(in_order_expansion);
    return packed_val;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  static inline void unpack(type const& packed_val, Ts&... args)
  {
    auto ptr = reinterpret_cast<const char*>(&packed_val);
    int in_order_expansion[]{(
        memcpy(&args, ptr, sizeof(Ts)),
        ptr += sizeof(Ts),
    0)...};
    ignore_unused(in_order_expansion);
  }
};
///
template < >
struct AtomicDataPackerImpl<void, double, char> :
    DoubleChar2Bits
{ };
///
template < >
struct AtomicDataPackerImpl<void, long, char> :
    IntegralChar2bits<long>
{ };
///
template < >
struct AtomicDataPackerImpl<void, unsigned long, char> :
    IntegralChar2bits<unsigned long>
{ };
///
template < >
struct AtomicDataPackerImpl<void, long long, char> :
    IntegralChar2bits<long long>
{ };
///
template < >
struct AtomicDataPackerImpl<void, unsigned long long, char> :
    IntegralChar2bits<unsigned long long>
{ };

template < typename ... Ts >
using AtomicDataPacker = AtomicDataPackerImpl<void, Ts...>;


/*!
 * \brief Trivial mapping of positive indices with no reordering.
 */
struct NonReorder
{
  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t range(size_t domain) const noexcept
  {
    return domain;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t operator()(size_t i) const noexcept
  {
    return i;
  }
};

/*!
 * \brief Map positive indices to [0, range).
 * Map groups of indices (i) to i % range.
 *   {0, 1, 2, ..., range-1,
 *    0, 1, 2, ..., range-1, ...}
 */
template < size_t t_max_range >
struct ModReorderStatic
{
  static_assert(t_max_range == next_pow2(t_max_range),
                "max_range must be a power of 2");

  static constexpr size_t max_range = t_max_range;
  static constexpr size_t max_range_mask = max_range - 1;

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t range(size_t domain) const noexcept
  {
    return (domain > max_range) ? max_range : domain;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t operator()(size_t i) const noexcept
  {
    return i & max_range_mask;
  }
};

/*!
 * \brief Map positive indices to [0, size) reordering such that most are
 * separated by stride.
 * Map groups of indices (i) to i * stride + group,
 * where group = (i / (size / stride)) % stride.
 *   [0, stride,   2*stride, ...,
 *    1, stride+1, 2*stride+1, ...)
 */
template < size_t t_stride, size_t t_max_domain, size_t t_max_range >
struct StridedReorderStatic
{
  static_assert(t_stride == next_pow2(t_stride),
                "stride must be a power of 2");
  static_assert(t_max_domain == next_pow2(t_max_domain),
                "max_domain must be a power of 2");
  static_assert(t_max_range == next_pow2(t_max_range),
                "max_range must be a power of 2");

  static constexpr size_t max_domain = t_max_domain;
  static constexpr size_t domain_mask = max_domain - 1;

  static constexpr size_t max_range = t_max_range;
  static constexpr size_t stride = (t_max_range > t_stride) ? t_stride : t_max_range;

  static constexpr size_t stride_shift = log2(stride);
  static constexpr size_t stride_mask = stride - 1;

  static constexpr size_t num_groups = (max_range > stride) ? (max_range / stride) : 1;
  static constexpr size_t group_shift = log2(num_groups);
  static constexpr size_t group_mask = num_groups - 1;

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t range(size_t domain) const noexcept
  {
    if (domain > max_domain) {
      domain = max_domain;
    }
    // round up to a multiple of stride
    if (domain >= num_groups) {
      return max_range;
    } else if (domain > 0) {
      const size_t group = (domain-1) & group_mask;
      return (group + 1) * stride;
    } else {
      return 0;
    }
  }

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t operator()(size_t i) const noexcept
  {
    if constexpr(max_domain < max_range) {
      i &= domain_mask;
    }
    return ((i >> group_shift) & stride_mask) |
           ((i & group_mask) << stride_shift);
  }
};
///
template < size_t stride >
struct StridedReorder
{
  static_assert(stride == next_pow2(stride),
                "stride must be a power of 2");

  static constexpr size_t stride_shift = log2(stride);
  static constexpr size_t stride_mask = stride - 1;

  size_t group_shift;
  size_t group_mask;

  constexpr void setup(size_t max_range)
  {
    assert(max_range == next_pow2(max_range));
    assert(max_range >= stride);

    size_t num_groups = (max_range > stride) ? (max_range / stride) : 1;
    group_shift = log2(num_groups);
    group_mask = num_groups - 1;
  }

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t range(size_t domain) const noexcept
  {
    // round up to a multiple of stride
    const size_t num_groups = group_mask+1;
    const size_t max_range = num_groups * stride;
    if (domain >= num_groups) {
      return max_range;
    } else if (domain > 0) {
      const size_t group = (domain-1) & group_mask;
      return (group + 1) * stride;
    } else {
      return 0;
    }
  }

  RAJA_PERFSUITE_HOST_DEVICE
  constexpr size_t operator()(size_t i) const noexcept
  {
    return ((i >> group_shift) & stride_mask) |
           ((i & group_mask) << stride_shift);
  }
};

struct GetModReorderStatic
{
  template < size_t max_range >
  using type = ModReorderStatic<max_range>;
  static const char* name() { return "mod"; }
};

template < size_t stride, size_t min_range >
struct GetStridedReorderStatic
{
  template < size_t max_domain >
  using type = StridedReorderStatic<stride, max_domain, std::max(max_domain, min_range)>;
  static const char* name() { return "stride"; }
};

// store block and grid counts in separate places
// because they can not be stored together with the ready flag
template < typename DataType >
struct AtomicDeviceStorageUnpackable
{
  DataType* block_counts;
  DataType* grid_counts;
  unsigned* block_readys;

  template < typename Allocator >
  void allocate(void*& storage_to_zero, size_t& storage_size,
                size_t grid_size, Allocator&& aloc)
  {
    aloc(block_counts, grid_size);
    aloc(grid_counts, grid_size);
    aloc(block_readys, grid_size);
    storage_to_zero = block_readys;
    storage_size = sizeof(unsigned)*grid_size;
  }

  template < typename Deallocator >
  void deallocate(Deallocator&& dealoc)
  {
    dealoc(block_counts);
    dealoc(grid_counts);
    dealoc(block_readys);
  }
};

// store the block/grid count together with the ready flag
template < typename DataType, size_t atomic_destructive_interference_size >
struct AtomicDeviceStoragePackable
{
  static constexpr size_t atomic_stride =
      (atomic_destructive_interference_size > sizeof(DataType))
      ? atomic_destructive_interference_size / sizeof(DataType)
      : 1;

  struct strided_pointer
  {
    DataType* ptr;
    StridedReorder<atomic_stride> reorder;

    RAJA_PERFSUITE_HOST_DEVICE
    constexpr DataType& operator[](size_t i) noexcept
    {
      return ptr[reorder(i)];
    }
  };

  strided_pointer count_readys;

  template < typename Allocator >
  void allocate(void*& storage_to_zero, size_t& storage_size,
                size_t grid_size, Allocator&& aloc)
  {
    size_t aloc_size = next_pow2(grid_size);

    aloc(count_readys.ptr, aloc_size);
    count_readys.reorder.setup(aloc_size);

    storage_to_zero = count_readys.ptr;
    storage_size = sizeof(DataType)*aloc_size;
  }

  template < typename Deallocator >
  void deallocate(Deallocator&& dealoc)
  {
    dealoc(count_readys.ptr);
  }
};

}  // closing brace for detail namespace

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
