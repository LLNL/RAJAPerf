/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_CUDA_HPP
#define __CAMP_CUDA_HPP

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_CUDA
#include <cuda_runtime.h>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class CudaEvent
    {
    public:
      CudaEvent(cudaStream_t stream)
      {
        cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming);
        cudaEventRecord(m_event, stream);
      }
      bool check() const { return (cudaEventQuery(m_event) == cudaSuccess); }
      void wait() const { cudaEventSynchronize(m_event); }
      cudaEvent_t getCudaEvent_t() const { return m_event; }

    private:
      cudaEvent_t m_event;
    };

    class Cuda
    {
      static cudaStream_t get_a_stream(int num)
      {
        static cudaStream_t streams[16] = {};
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        std::call_once(m_onceFlag, [] {
          if (streams[0] == nullptr) {
            for (auto &s : streams) {
              cudaStreamCreate(&s);
            }
          }
        });

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return streams[previous];
        }

        return streams[num % 16];
      }

    public:
      Cuda(int group = -1) : stream(get_a_stream(group)) {}

      // Methods
      Platform get_platform() { return Platform::cuda; }
      static Cuda &get_default()
      {
        static Cuda h;
        return h;
      }
      CudaEvent get_event() { return CudaEvent(get_stream()); }
      Event get_event_erased() { return Event{CudaEvent(get_stream())}; }
      void wait() { cudaStreamSynchronize(stream); }
      void wait_for(Event *e)
      {
        auto *cuda_event = e->try_get<CudaEvent>();
        if (cuda_event) {
          cudaStreamWaitEvent(get_stream(),
                              cuda_event->getCudaEvent_t(),
                              0);
        } else {
          e->wait();
        }
      }

      // Memory
      template <typename T>
      T *allocate(size_t size)
      {
        T *ret = nullptr;
        if (size > 0) {
          cudaMallocManaged(&ret, sizeof(T) * size);
        }
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p)
      { 
        cudaFree(p);
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          cudaMemsetAsync(p, val, size, stream);
        }
      }

      cudaStream_t get_stream() { return stream; }

    private:
      cudaStream_t stream;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_CUDA

#endif /* __CAMP_CUDA_HPP */
