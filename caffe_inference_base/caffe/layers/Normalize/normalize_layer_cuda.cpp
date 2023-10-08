#include "caffe/layers/Normalize/normalize_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDANormalizeLayer<Dtype>::InitLayer() {
    if (!cublas_handle_)
      CHECK_CUBLAS(cublasCreate(&cublas_handle_));
  }

  template <typename Dtype>
  CUDANormalizeLayer<Dtype>::~CUDANormalizeLayer() {
    if (cublas_handle_){
      CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
  }

  
  INSTANTIATE_CLASS(CUDANormalizeLayer);
  
} // namespace facethink
