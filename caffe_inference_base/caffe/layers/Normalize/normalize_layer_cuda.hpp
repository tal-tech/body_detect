#ifndef __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_CUDA_HPP__

#include "caffe/layers/Normalize/normalize_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDANormalizeLayer: public BaseNormalizeLayer<Dtype> {
  public:
    explicit CUDANormalizeLayer(bool across_spatial = true,
				bool channel_shared = true,
				Dtype eps = Dtype(1e-10),
				const std::string& name="")
      :BaseNormalizeLayer<Dtype>(across_spatial,
				 channel_shared,
				 eps,
				 name),
      cublas_handle_(nullptr) {
      InitLayer();
    }

    ~CUDANormalizeLayer();

    virtual void InitLayer();
    //virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const { return "Normalize(CUDA)"; }
  private:
    cublasHandle_t cublas_handle_;

    DISABLE_COPY_AND_ASSIGN(CUDANormalizeLayer);
  }; //class CUDANormalizeLayer

} // namespace facethink


#endif
