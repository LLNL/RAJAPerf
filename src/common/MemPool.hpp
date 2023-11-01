//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// This file is taken from RAJA/util/basic_mempool.hpp
// and modified to support resources
//

#ifndef RAJAPerf_BASIC_MEMPOOL_HPP
#define RAJAPerf_BASIC_MEMPOOL_HPP

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>

#include "RAJA/util/align.hpp"
#include "RAJA/util/mutex.hpp"

namespace rajaperf
{

namespace basic_mempool
{

namespace detail
{


/*! \class MemoryArena
 ******************************************************************************
 *
 * \brief  MemoryArena is a map based subclass for class MemPool
 * provides book-keeping to divy a large chunk of pre-allocated memory to avoid
 * the overhead of  malloc/free or cudaMalloc/cudaFree, etc
 *
 * get/give are the primary calls used by class MemPool to get aligned memory
 * from the pool or give it back
 *
 *
 ******************************************************************************
 */
class MemoryArena
{
public:
  using free_type = std::map<void*, void*>;
  using free_value_type = typename free_type::value_type;
  using used_type = std::map<void*, void*>;
  using used_value_type = typename used_type::value_type;

  MemoryArena(void* ptr, size_t size)
    : m_allocation{ ptr, static_cast<char*>(ptr)+size },
      m_free_space(),
      m_used_space()
  {
     m_free_space[ptr] = static_cast<char*>(ptr)+size ;
    if (m_allocation.begin == nullptr) {
      fprintf(stderr, "Attempt to create MemoryArena with no memory");
      std::abort();
    }
  }

  MemoryArena(MemoryArena const&) = delete;
  MemoryArena& operator=(MemoryArena const&) = delete;

  MemoryArena(MemoryArena&&) = default;
  MemoryArena& operator=(MemoryArena&&) = default;

  size_t capacity()
  {
    return static_cast<char*>(m_allocation.end) -
           static_cast<char*>(m_allocation.begin);
  }

  bool unused() { return m_used_space.empty(); }

  void* get_allocation() { return m_allocation.begin; }

  void* get(size_t nbytes, size_t alignment)
  {
    void* ptr_out = nullptr;
    if (capacity() >= nbytes) {
      free_type::iterator end = m_free_space.end();
      for (free_type::iterator iter = m_free_space.begin(); iter != end;
           ++iter) {

        void* adj_ptr = iter->first;
        size_t cap =
            static_cast<char*>(iter->second) - static_cast<char*>(adj_ptr);

        if (::RAJA::align(alignment, nbytes, adj_ptr, cap)) {

          ptr_out = adj_ptr;

          remove_free_chunk(iter,
                            adj_ptr,
                            static_cast<char*>(adj_ptr) + nbytes);

          add_used_chunk(adj_ptr, static_cast<char*>(adj_ptr) + nbytes);

          break;
        }
      }
    }
    return ptr_out;
  }

  bool give(void* ptr)
  {
    if (m_allocation.begin <= ptr && ptr < m_allocation.end) {

      used_type::iterator found = m_used_space.find(ptr);

      if (found != m_used_space.end()) {

        add_free_chunk(found->first, found->second);

        m_used_space.erase(found);

      } else {
        fprintf(stderr, "Invalid free %p", ptr);
        std::abort();
      }

      return true;
    } else {
      return false;
    }
  }

private:
  struct memory_chunk {
    void* begin;
    void* end;
  };

  void add_free_chunk(void* begin, void* end)
  {
    // integrates a chunk of memory into free_space
    free_type::iterator invl = m_free_space.end();
    free_type::iterator next = m_free_space.lower_bound(begin);

    // check if prev exists
    if (next != m_free_space.begin()) {
      // check if prev can cover [begin, end)
      free_type::iterator prev = next;
      --prev;
      if (prev->second == begin) {
        // extend prev to cover [begin, end)
        prev->second = end;

        // check if prev can cover next too
        if (next != invl) {
          assert(next->first != begin);

          if (next->first == end) {
            // extend prev to cover next too
            prev->second = next->second;

            // remove redundant next
            m_free_space.erase(next);
          }
        }
        return;
      }
    }

    if (next != invl) {
      assert(next->first != begin);

      if (next->first == end) {
        // extend next to cover [begin, end)
        m_free_space.insert(next, free_value_type{begin, next->second});
        m_free_space.erase(next);

        return;
      }
    }

    // no free space adjacent to this chunk, add seperate free chunk [begin,
    // end)
    m_free_space.insert(next, free_value_type{begin, end});
  }

  void remove_free_chunk(free_type::iterator iter, void* begin, void* end)
  {

    void* ptr = iter->first;
    void* ptr_end = iter->second;

    // fixup m_free_space, shrinking and adding chunks as needed
    if (ptr != begin) {

      // shrink end of current free region to [ptr, begin)
      iter->second = begin;

      if (end != ptr_end) {

        // insert free region [end, ptr_end) after current free region
        free_type::iterator next = iter;
        ++next;
        m_free_space.insert(next, free_value_type{end, ptr_end});
      }

    } else if (end != ptr_end) {

      // shrink beginning of current free region to [end, ptr_end)
      free_type::iterator next = iter;
      ++next;
      m_free_space.insert(next, free_value_type{end, ptr_end});
      m_free_space.erase(iter);

    } else {

      // can not reuse current region, erase
      m_free_space.erase(iter);
    }
  }

  void add_used_chunk(void* begin, void* end)
  {
    // simply inserts a chunk of memory into used_space
    m_used_space.insert(used_value_type{begin, end});
  }

  memory_chunk m_allocation;
  free_type m_free_space;
  used_type m_used_space;
};

} /* end namespace detail */


/*! \class MemPool
 ******************************************************************************
 *
 * \brief  MemPool pre-allocates a large chunk of memory and provides generic
 * malloc/free for the user to allocate aligned data within the pool
 *
 * MemPool uses MemoryArena to do the heavy lifting of maintaining access to
 * the used/free space.
 *
 * MemPool provides an example generic_allocator which can guide more
 *specialized
 * allocators. The following are some examples
 *
 * using device_mempool_type = basic_mempool::MemPool<cuda::DeviceAllocator>;
 * using device_zeroed_mempool_type =
 *basic_mempool::MemPool<cuda::DeviceZeroedAllocator>;
 * using pinned_mempool_type = basic_mempool::MemPool<cuda::PinnedAllocator>;
 *
 * The user provides the specialized allocator, for example :
 * struct DeviceAllocator {
 *
 *  // returns a valid pointer on success, nullptr on failure
 *  void* malloc(size_t nbytes)
 *  {
 *    void* ptr;
 *    cudaErrchk(cudaMalloc(&ptr, nbytes));
 *    return ptr;
 *  }
 *
 *  // returns true on success, false on failure
 *  bool free(void* ptr)
 *  {
 *    cudaErrchk(cudaFree(ptr));
 *    return true;
 *  }
 * };
 *
 *
 ******************************************************************************
 */
template <typename allocator_t, typename lagged_res>
class MemPool
{
public:
  using allocator_type = allocator_t;

  static inline MemPool<allocator_t, lagged_res>& getInstance()
  {
    static MemPool<allocator_t, lagged_res> pool{};
    return pool;
  }

  static const size_t default_default_arena_size = 32ull * 1024ull * 1024ull;

  MemPool(size_t default_arena_size = default_default_arena_size,
          lagged_res const& lag_res = lagged_res::get_default())
      : m_arenas(), m_default_arena_size(default_arena_size),
        m_alloc(), m_lagged_frees(), m_lag_res(lag_res)
  {
  }

  ~MemPool()
  {
    // This is here for the case that MemPool is used as a static object.
    // If allocator_t uses cudaFree then it will fail with
    // cudaErrorCudartUnloading when called after main.
  }


  void free_chunks()
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    while (!m_arenas.empty()) {
      void* allocation_ptr = m_arenas.front().get_allocation();
      m_alloc.free(allocation_ptr);
      m_arenas.pop_front();
    }
  }

  size_t arena_size()
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    return m_default_arena_size;
  }

  size_t arena_size(size_t new_size)
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    size_t prev_size = m_default_arena_size;
    m_default_arena_size = new_size;
    return prev_size;
  }

  template <typename T>
  T* malloc(size_t nTs, size_t alignment = alignof(T))
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    auto get_from_existing_arena = [&](size_t size, size_t alignment) {
      void* ptr = nullptr;
      for (detail::MemoryArena& arena : m_arenas) {
        ptr = arena.get(size, alignment);
        if (ptr != nullptr) {
          break;
        }
      }
      return ptr;
    };

    auto get_from_new_arena = [&](size_t size, size_t alignment) {
      void* ptr = nullptr;
      const size_t alloc_size =
          std::max(size + alignment, m_default_arena_size);
      void* arena_ptr = m_alloc.malloc(alloc_size);
      if (arena_ptr != nullptr) {
        m_arenas.emplace_front(arena_ptr, alloc_size);
        ptr = m_arenas.front().get(size, alignment);
      }
      return ptr;
    };

    const size_t size = nTs * sizeof(T);

    void* ptr = get_from_existing_arena(size, alignment);

    if (ptr == nullptr) {
      free_lagged_memory_impl();
      ptr = get_from_existing_arena(size, alignment);
    }

    if (ptr == nullptr) {
      ptr = get_from_new_arena(size, alignment);
    }

    return static_cast<T*>(ptr);
  }

  void free(const void* cptr)
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    m_lagged_frees.emplace_back(const_cast<void*>(cptr));
  }

  void free_lagged_memory()
  {
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::lock_guard<RAJA::omp::mutex> lock(m_mutex);
#endif

    free_lagged_memory_impl();
  }

private:
  using arena_container_type = std::list<detail::MemoryArena>;

#if defined(RAJA_ENABLE_OPENMP)
  RAJA::omp::mutex m_mutex;
#endif

  arena_container_type m_arenas;
  size_t m_default_arena_size;
  allocator_t m_alloc;
  std::vector<void*> m_lagged_frees;
  lagged_res m_lag_res;


  void free_lagged_memory_impl()
  {
    if (!m_lagged_frees.empty()) {
      m_lag_res.wait();
      for (void* ptr : m_lagged_frees) {
        for (detail::MemoryArena& arena : m_arenas) {
          if (arena.give(ptr)) {
            ptr = nullptr;
            break;
          }
        }
        if (ptr != nullptr) {
          fprintf(stderr, "Unknown pointer %p", ptr);
        }
      }
      m_lagged_frees.clear();
    }
  }
};

//! example allocator for basic_mempool using malloc/free
struct generic_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes) { return std::malloc(nbytes); }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    std::free(ptr);
    return true;
  }
};

} /* end namespace basic_mempool */

} /* end namespace rajaperf */


#endif /* RAJAPerf_BASIC_MEMPOOL_HPP */
