#ifndef __FACETHINK_CAFFE_UTIL_MATH_FUNC_HPP__
#define __FACETHINK_CAFFE_UTIL_MATH_FUNC_HPP__


#include "caffe/core/common.hpp"

namespace facethink {
  
  template <typename Dtype>
  void caffe_cpu_set(const int N, const Dtype alpha, Dtype* Y);

  template <typename Dtype>
  void caffe_cpu_abs(const int n, const Dtype* a, Dtype* y);
  
  template <typename Dtype>
  void caffe_cpu_sqr(const int N, const Dtype* a, Dtype* y);

  template <typename Dtype>
  void caffe_cpu_exp(const int n, const Dtype* a, Dtype* y);

  template <typename Dtype>
  void caffe_cpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);


  template <typename Dtype>
  void caffe_cpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  template <typename Dtype>
  void caffe_cpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  template <typename Dtype>
  void caffe_cpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

  template <typename Dtype>
  void caffe_cpu_copy(const int N, const Dtype *X, Dtype *Y);


#ifdef CPU_ONLY
  //Caffe gemm provides a simpler interface to the gemm functions, with the
  // limitation that the data has to be contiguous in memory.
  template <typename Dtype>
  void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
		      const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		      const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		      Dtype* C);

  template <typename Dtype>
  void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
		      const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
		      Dtype* y);

  template <typename Dtype>
  void caffe_cpu_axpy(const int N, const Dtype alpha, const Dtype* X,
		      Dtype* Y);
  
  // Returns the sum of the absolute values of the elements of vector x
  template <typename Dtype>
  Dtype caffe_cpu_asum(const int n, const Dtype* x);

  template <typename Dtype>
  void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

  template <typename Dtype>
  void caffe_cpu_scal(const int N, const Dtype alpha, Dtype *X);

#endif
  
#ifndef CPU_ONLY

  enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

  template <typename Dtype>
  void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y);

  template <typename Dtype>
  void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_copy(const int N, const Dtype *X, Dtype *Y);

  template <typename Dtype>
  void caffe_gpu_powx(const int N, const Dtype* a, const Dtype alpha, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

  template <typename Dtype>
  void caffe_gpu_asum(cublasHandle_t handle,
		      const int n, const Dtype* x, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_scal(cublasHandle_t handle,
		      const int N, const Dtype alpha, Dtype *X);
  
  template <typename Dtype>
  void caffe_gpu_scale(cublasHandle_t handle,
		       const int n, const Dtype alpha, const Dtype *x, Dtype* y);

  template <typename Dtype>
  void caffe_gpu_axpy(cublasHandle_t handle,
		      const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
  
  template <typename Dtype>
  void caffe_gpu_gemm(cublasHandle_t handle,
		      const CBLAS_TRANSPOSE TransA,
		      const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		      const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		      Dtype* C);

  template <typename Dtype>
  void caffe_gpu_gemv(cublasHandle_t handle,
		      const CBLAS_TRANSPOSE TransA, const int M, const int N,
		      const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
		      Dtype* y);
#endif 
  
} // namespace facethink

#endif
