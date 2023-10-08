#include "caffe/core/syncedmemory.hpp"
#include "caffe/core/common.hpp"
#include <string.h>

namespace facethink {
  
  SyncedMemory::SyncedMemory()
    : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED),
      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
  }

  SyncedMemory::SyncedMemory(size_t size)
    : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
  }

    
  SyncedMemory::~SyncedMemory() {
    if (cpu_ptr_ && own_cpu_data_) {
      CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    }

#ifndef CPU_ONLY
    if (gpu_ptr_ && own_gpu_data_) {
      CHECK_CUDA(cudaFree(gpu_ptr_));
    }
#endif 
  }

  
  inline void SyncedMemory::to_cpu() {
    switch (head_) {
    case UNINITIALIZED:
      CaffeMallocHost(&cpu_ptr_,  size_, &cpu_malloc_use_cuda_);
      memset(cpu_ptr_, 0, size_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case HEAD_AT_GPU:
#ifndef CPU_ONLY
      if (cpu_ptr_ == nullptr) {
	CaffeMallocHost(&cpu_ptr_,  size_, &cpu_malloc_use_cuda_);
	own_cpu_data_ = true;
      }
      CHECK_CUDA(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost));
      head_ = SYNCED;
#else
      NO_GPU;
#endif
      break;
    case HEAD_AT_CPU:
    case SYNCED:
      break;
    }
  }

#ifndef CPU_ONLY
  inline void SyncedMemory::to_gpu() {
    switch (head_) {
    case UNINITIALIZED:
      CHECK_CUDA(cudaMalloc(&gpu_ptr_, size_));
      CHECK_CUDA(cudaMemset(gpu_ptr_, 0, size_));
      head_ = HEAD_AT_GPU;
      own_gpu_data_ = true;
      break;
    case HEAD_AT_CPU:
      if (gpu_ptr_ == nullptr) {
	CHECK_CUDA(cudaMalloc(&gpu_ptr_, size_));
	own_gpu_data_ = true;
      }
      CHECK_CUDA(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
      head_ = SYNCED;
      break;
    case HEAD_AT_GPU:
    case SYNCED:
      break;
    }
  }
#endif

  const void* SyncedMemory::cpu_data() {
    to_cpu();
    return (const void*)cpu_ptr_;
  }

  void* SyncedMemory::mutable_cpu_data() {
    to_cpu();
    head_ = HEAD_AT_CPU;
    return cpu_ptr_;
  }


#ifndef CPU_ONLY
  const void* SyncedMemory::gpu_data() {
    to_gpu();
    return (const void*)gpu_ptr_;
  }

  void* SyncedMemory::mutable_gpu_data() {
    to_gpu();
    head_ = HEAD_AT_GPU;
    return gpu_ptr_;
  }
#endif 

}
