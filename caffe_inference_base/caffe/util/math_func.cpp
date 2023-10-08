#include "caffe/util/math_func.hpp"
#include <string.h>

namespace facethink {

  template <typename Dtype>
  void caffe_cpu_set(const int N, const Dtype alpha, Dtype* Y) {
    if (alpha == 0) {
      memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
      return;
    }
    for (int i = 0; i < N; ++i) {
      Y[i] = alpha;
    }
  }

  template void caffe_cpu_set<int>(const int N, const int alpha, int* Y);
  template void caffe_cpu_set<float>(const int N, const float alpha, float* Y);


  template <>
  void caffe_cpu_abs<float>(const int n, const float* a, float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = fabs(a[i]);
    }
  }
  
  template <>
  void caffe_cpu_sqr<float>(const int n, const float* a, float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i]*a[i];
    }
  }

  
  template <>
  void caffe_cpu_exp<float>(const int n, const float* a, float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = exp(a[i]);
    }
  }

  template <>
  void caffe_cpu_powx<float>(const int n, const float* a, const float b,
			 float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = pow(a[i], b);
    }
  }
  
  template <>
  void caffe_cpu_div<float>(const int n, const float* a, const float* b,
			    float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] / b[i];
    }
  }

    
  template <>
  void caffe_cpu_mul<float>(const int n, const float* a, const float* b,
			    float* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = a[i] * b[i];
    }
  }

  template <>
  void caffe_cpu_add_scalar(const int N, const float alpha, float* Y) {
    for (int i = 0; i < N; ++i) {
      Y[i] += alpha;
    }
  }

  template <typename Dtype>
  void caffe_cpu_copy(const int N, const Dtype* X, Dtype* Y) {
    if (X != Y) {
      memcpy(Y, X, sizeof(Dtype) * N); 
    }
  }

  template void caffe_cpu_copy<float>(const int N, const float* X, float* Y);

  
#ifdef CPU_ONLY
  template<>
  void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
			     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			     const float alpha, const float* A, const float* B, const float beta,
			     float* C) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
  }

  template <>
  void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
			     const int N, const float alpha, const float* A, const float* x,
			     const float beta, float* y) {
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
  }

  template <>
  void caffe_cpu_axpy<float>(const int N, const float alpha, const float* X,
			     float* Y) {
    cblas_saxpy(N, alpha, X, 1, Y, 1);
  }
  
  template <>
  float caffe_cpu_asum<float>(const int n, const float* x) {
    return cblas_sasum(n, x, 1);
  }

  
  template <>
  void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
			      float* y) {
    cblas_scopy(n, x, 1, y, 1);
    cblas_sscal(n, alpha, y, 1);
  }

  template <>
  void caffe_cpu_scal<float>(const int N, const float alpha, float *X) {
    cblas_sscal(N, alpha, X, 1);
  }

  
#endif

  
#ifndef CPU_ONLY

  
  template <typename Dtype>
  void caffe_gpu_copy(const int N, const Dtype* X, Dtype* Y) {
    if (X != Y) {
      CHECK_CUDA(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
    }
  }
  template void caffe_gpu_copy<float>(const int N, const float* X, float* Y);
  
  template <>
  void caffe_gpu_asum<float>(cublasHandle_t handle,
			     const int n, const float* x, float* y) {
    CHECK_CUBLAS(cublasSasum(handle, n, x, 1, y));
  }


  template <>
  void caffe_gpu_scal<float>(cublasHandle_t handle,
			     const int N, const float alpha, float *X) {
    CHECK_CUBLAS(cublasSscal(handle, N, &alpha, X, 1));
  }

  template <>
  void caffe_gpu_scale<float>(cublasHandle_t handle,
			      const int n, const float alpha, const float *x,
			      float* y) {
    CHECK_CUBLAS(cublasScopy(handle, n, x, 1, y, 1));
    CHECK_CUBLAS(cublasSscal(handle, n, &alpha, y, 1));
  }

  template <>
  void caffe_gpu_axpy<float>(cublasHandle_t handle,
			     const int N, const float alpha, const float* X,
			     float* Y) {
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, X, 1, Y, 1));
  }

  template <>
  void caffe_gpu_gemm<float>(cublasHandle_t handle,
			     const CBLAS_TRANSPOSE TransA,
			     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			     const float alpha, const float* A, const float* B, const float beta,
			     float* C) {
    // Note that cublas follows fortran order.
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CHECK_CUBLAS(cublasSgemm(handle, cuTransB, cuTransA,
			     N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  }

  template <>
  void caffe_gpu_gemv<float>(cublasHandle_t handle,
			     const CBLAS_TRANSPOSE TransA, const int M,
			     const int N, const float alpha, const float* A, const float* x,
			     const float beta, float* y) {
    cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    CHECK_CUBLAS(cublasSgemv(handle, cuTransA, N, M, &alpha,
			     A, N, x, 1, &beta, y, 1));
  }

  
#endif
  
}
