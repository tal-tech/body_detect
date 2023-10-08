#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y){
    CUDA_KERNEL_LOOP(index, n){
      y[index] = alpha;
    }
  }

  template <>
  void caffe_gpu_set<float>(const int N, const float alpha, float* Y){
    if (alpha == 0) {
      CHECK_CUDA(cudaMemset(Y, 0, sizeof(float) * N));  // NOLINT(caffe/alt_fn)
      return;
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    set_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
  }


  template <typename Dtype>
  __global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = abs(a[index]);
    }
  }

  template <>
  void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
								       N, a, y);
  }
  

  template <typename Dtype>
  __global__ void powx_kernel(const int n, const Dtype* a,
			      const Dtype alpha, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = pow(a[index], alpha);
    }
  }

  template <>
  void caffe_gpu_powx<float>(const int N, const float* a,
			     const float alpha, float* y){
    // NOLINT_NEXT_LINE(whitespace/operators)
    powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, alpha, y);
  }


  
  template <typename Dtype>
  __global__ void div_kernel(const int n, const Dtype* a,
			     const Dtype* b, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = a[index] / b[index];
    }
  }

  template <>
  void caffe_gpu_div<float>(const int N, const float* a,
			    const float* b, float* y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
								       N, a, b, y);
  }


  template <typename Dtype>
  __global__ void mul_kernel(const int n, const Dtype* a,
			     const Dtype* b, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = a[index] * b[index];
    }
  }

  template <>
  void caffe_gpu_mul<float>(const int N, const float* a,
			    const float* b, float* y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
								       N, a, b, y);
  }
  

  template <typename Dtype>
  __global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] += alpha;
    }
  }

  template <>
  void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
									      N, alpha, Y);
  }




} // namespace facethink
